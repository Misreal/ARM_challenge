"""The thresholds in `search.reduce` decide the whole space cut, so pin them."""

from __future__ import annotations

from src.search.reduce import NOISE_MS, PIN_FP32, PIN_INT8, SEARCH, classify, greedy_order, reduce_space


def row(group: str, share: float, cost: float) -> dict:
    return {
        "group": group,
        "recovery_share": share,
        "cost_ms": cost,
        "share_per_ms": (share / cost) if cost > 0.02 else None,
    }


def test_free_group_that_buys_accuracy_is_pinned_fp32() -> None:
    # resnet18's `fc` measured at -0.040 ms for 2.8% of the distortion.
    assert classify(row("fc", 0.028, -0.040), best_ratio=1.45) == PIN_FP32


def test_free_group_that_buys_nothing_is_pinned_int8() -> None:
    assert classify(row("avgpool", 0.0, -0.053), best_ratio=1.45) == PIN_INT8


def test_negative_share_is_pinned_int8() -> None:
    # Sparing resnet18's `relu` breaks Conv+Relu QDQ fusion and hurts accuracy.
    assert classify(row("relu", -0.205, 0.090), best_ratio=1.45) == PIN_INT8


def test_expensive_group_below_the_relative_floor_is_pinned_int8() -> None:
    # layer4: 6.5 ms to recover 4% is 0.006 share/ms against a best of 1.45.
    assert classify(row("layer4", 0.042, 6.516), best_ratio=1.45) == PIN_INT8


def test_group_within_a_tenth_of_the_best_stays_searchable() -> None:
    assert classify(row("layer1", 0.433, 2.378), best_ratio=1.45) == SEARCH


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


def test_greedy_puts_free_groups_first_then_descending_ratio() -> None:
    rows = [
        row("layer1", 0.433, 2.378),
        row("conv1", 0.203, 0.140),
        row("fc", 0.028, -0.040),
    ]
    assert [r["group"] for r in greedy_order(rows)] == ["fc", "conv1", "layer1"]


def test_noise_band_is_the_measured_one() -> None:
    # The Pi sweep produced costs down to -0.053 ms for graph-identical work.
    assert NOISE_MS > 0.053
