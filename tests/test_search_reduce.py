"""The thresholds in `search.reduce` decide the whole space cut, so pin them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.search.reduce import (
    DEFAULT_BAND_FRACTION,
    PIN_FP32,
    PIN_INT8,
    SEARCH,
    Band,
    ReducedSpace,
    band_for,
    classify,
    greedy_order,
    group_fingerprint,
    reduce_space,
)

COST_BENEFIT = Path("artifacts/reports_pi/cost_benefit.json")

# ResNet-18's measured anchor, and the band a 2% rule cuts it at.
RESNET_REFERENCE_MS = 3.41
RESNET_BAND_MS = 0.02 * RESNET_REFERENCE_MS


def row(group: str, share: float, cost: float) -> dict:
    return {
        "group": group,
        "recovery_share": share,
        "cost_ms": cost,
        "latency_ms": RESNET_REFERENCE_MS + cost,
        "share_per_ms": (share / cost) if cost > 0.02 else None,
    }


def test_a_cost_inside_the_band_that_buys_accuracy_is_searchable() -> None:
    # resnet18's `fc` measured -0.040 ms, well inside a 0.068 ms band. We cannot
    # tell that from zero, so the enumeration decides it rather than the pin.
    assert classify(row("fc", 0.028, -0.040), 1.45, RESNET_BAND_MS) == SEARCH


def test_a_cost_inside_the_band_that_buys_nothing_is_pinned_int8() -> None:
    # Nothing to buy at any price, so there is no reason to spend a vector on it.
    assert classify(row("avgpool", 0.0, -0.053), 1.45, RESNET_BAND_MS) == PIN_INT8


def test_a_cost_measurably_below_the_band_is_pinned_fp32() -> None:
    # Genuinely faster when spared, and it buys accuracy: free either way.
    assert classify(row("hypothetical", 0.20, -0.400), 1.45, RESNET_BAND_MS) == PIN_FP32


def test_negative_share_is_pinned_int8() -> None:
    # Sparing resnet18's `relu` breaks Conv+Relu QDQ fusion and hurts accuracy.
    assert classify(row("relu", -0.205, 0.090), 1.45, RESNET_BAND_MS) == PIN_INT8


def test_expensive_group_below_the_relative_floor_is_pinned_int8() -> None:
    # layer4: 6.5 ms to recover 4% is 0.006 share/ms against a best of 1.45.
    assert classify(row("layer4", 0.042, 6.516), 1.45, RESNET_BAND_MS) == PIN_INT8


def test_group_within_a_tenth_of_the_best_stays_searchable() -> None:
    assert classify(row("layer1", 0.433, 2.378), 1.45, RESNET_BAND_MS) == SEARCH


def test_reduce_space_partitions_every_group_exactly_once() -> None:
    rows = [
        row("conv1", 0.203, 0.140),
        row("layer1", 0.433, 2.378),
        row("layer4", 0.042, 6.516),
        row("fc", 0.028, -0.040),
    ]
    space = reduce_space("resnet18_cifar", rows)
    placed = space.pinned_fp32 + space.pinned_int8 + space.searchable
    assert sorted(placed) == sorted(r["group"] for r in rows)
    assert space.full_size == 16
    assert space.reduced_size == 2 ** len(space.searchable)


def test_greedy_puts_groups_inside_the_band_first_then_descending_ratio() -> None:
    rows = [
        row("layer1", 0.433, 2.378),
        row("conv1", 0.203, 0.140),
        row("fc", 0.028, -0.040),
    ]
    assert [r["group"] for r in greedy_order(rows, RESNET_BAND_MS)] == ["fc", "conv1", "layer1"]


def test_the_band_defaults_to_the_documented_fraction_without_a_sentinel() -> None:
    band = band_for([row("conv1", 0.2, 0.14)], None)
    assert band.fraction == DEFAULT_BAND_FRACTION
    assert band.source == "default"
    assert band.reference_ms == pytest.approx(RESNET_REFERENCE_MS)


def test_the_band_comes_from_the_sentinel_when_one_was_measured() -> None:
    band = band_for([row("conv1", 0.2, 0.14)], {"spread_fraction": 0.0255})
    assert band.fraction == pytest.approx(0.0255)
    assert band.source == "sentinel"
    assert band.ms == pytest.approx(0.0255 * RESNET_REFERENCE_MS)


@pytest.mark.parametrize("model", ["custom_cnn", "mobilenetv2_cifar", "resnet18_cifar"])
@pytest.mark.parametrize("fraction", [0.02, 0.03])
def test_the_measured_partition_is_the_same_at_two_and_three_percent(model: str, fraction: float) -> None:
    """The reason for the rule: the absolute floor flips groups between plausible
    band choices, and the banded rule does not."""
    if not COST_BENEFIT.exists():
        pytest.skip(f"{COST_BENEFIT} is not in this checkout")
    rows = json.loads(COST_BENEFIT.read_text(encoding="utf-8"))[model]

    expected = {
        "custom_cnn": ["classifier", "stage2", "stem"],
        "mobilenetv2_cifar": ["classifier.1", "features.0", "features.1", "features.18"],
        "resnet18_cifar": ["conv1", "fc", "layer1"],
    }[model]

    band = Band(fraction, "test", band_for(rows, None).reference_ms)
    space = reduce_space(model, rows, band)
    assert sorted(space.searchable) == expected
    # The collapse was always coming from pinned INT8, not from pinned FP32.
    assert space.pinned_fp32 == ()
    assert len(space.pinned_int8) == len(rows) - len(expected)


GROUPS = ("conv1", "fc", "layer2", "layer3")


def complete_space(**overrides) -> ReducedSpace:
    """A valid partition of GROUPS, fingerprinted against the graph as a real one is."""
    fields = {
        "model": "resnet18_cifar",
        "pinned_fp32": (),
        "pinned_int8": ("layer2", "layer3"),
        "searchable": ("conv1", "fc"),
        "fingerprint": group_fingerprint(GROUPS),
    }
    fields.update(overrides)
    return ReducedSpace(**fields)


def test_a_complete_partition_validates() -> None:
    complete_space().validate("resnet18_cifar", GROUPS)


def test_a_partition_for_another_model_is_refused() -> None:
    with pytest.raises(SystemExit, match="is for"):
        complete_space().validate("custom_cnn", GROUPS)


def test_a_group_in_two_buckets_is_refused() -> None:
    space = complete_space(pinned_int8=("layer2", "conv1"))
    with pytest.raises(SystemExit, match="both"):
        space.validate("resnet18_cifar", GROUPS)


def test_a_group_the_graph_does_not_expose_is_refused() -> None:
    space = complete_space(pinned_int8=("layer2", "layer3", "layer9"))
    with pytest.raises(SystemExit, match="does not expose"):
        space.validate("resnet18_cifar", GROUPS)


def test_a_group_with_no_decision_is_refused() -> None:
    space = complete_space(pinned_int8=("layer2",))
    with pytest.raises(SystemExit, match="decides nothing"):
        space.validate("resnet18_cifar", GROUPS)


def test_a_fingerprint_from_a_different_group_set_is_refused() -> None:
    space = complete_space(fingerprint=group_fingerprint(("a", "b")))
    with pytest.raises(SystemExit, match="different group set"):
        space.validate("resnet18_cifar", GROUPS)


def test_the_artifact_round_trips() -> None:
    space = complete_space(band=Band(0.02, "sentinel", 3.41))
    restored = ReducedSpace.from_dict(space.as_dict())
    assert restored == space


def test_nothing_searchable_is_a_one_vector_enumeration_not_an_error() -> None:
    space = complete_space(pinned_int8=GROUPS, searchable=())
    space.validate("resnet18_cifar", GROUPS)
    assert space.reduced_size == 1


def test_everything_pinned_fp32_is_a_valid_answer_meaning_no_quantization() -> None:
    space = complete_space(pinned_fp32=GROUPS, pinned_int8=(), searchable=())
    space.validate("resnet18_cifar", GROUPS)
    assert space.reduced_size == 1
