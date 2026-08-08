"""The probe configs decide what each measurement actually means, and the ranking
sign decides whether the result is inverted. Both fail silently if wrong."""

from __future__ import annotations

from collections import OrderedDict

import pytest

from src.quant.groups import UNSCOPED_GROUP, refine_group
from src.sensitivity.analyze import rank_groups
from src.sensitivity.probes import (
    ISOLATE,
    LEAVE_ONE_OUT,
    SCHEMES,
    Probe,
    anchor_configs,
    probe_config,
    probes_for,
)

# Mirrors custom_cnn's exported graph, which is the model the answer key covers.
CUSTOM_CNN_MAP: "OrderedDict[str, tuple[str, ...]]" = OrderedDict(
    stem=("/stem/conv/Conv", "/stem/relu/Relu"),
    stage1=("/stage1/conv1/conv/Conv", "/stage1/pool/MaxPool"),
    stage3=(
        "/stage3/bottleneck/expand/conv/Conv",
        "/stage3/bottleneck/project/conv/Conv",
        "/stage3/bottleneck/Add",
        "/stage3/pool/MaxPool",
    ),
    classifier=("/classifier/Gemm",),
)
CUSTOM_CNN_MAP[UNSCOPED_GROUP] = ("/Flatten",)


def test_leave_one_out_excludes_only_its_own_group() -> None:
    config = probe_config(Probe("stage3", LEAVE_ONE_OUT, "per_channel"), tuple(CUSTOM_CNN_MAP))
    assert config.excluded_groups == ("stage3",)
    assert config.quant_type == "static"
    assert config.per_channel is True


def test_isolate_excludes_everything_else_including_the_unscoped_bucket() -> None:
    # Leaving the unscoped nodes quantized would mean the probe is not isolating
    # anything: GlobalAveragePool still carries QDQ and contributes divergence.
    config = probe_config(Probe("stage3", ISOLATE, "per_tensor"), tuple(CUSTOM_CNN_MAP))
    assert set(config.excluded_groups) == set(CUSTOM_CNN_MAP) - {"stage3"}
    assert UNSCOPED_GROUP in config.excluded_groups
    assert config.per_channel is False


def test_the_two_probe_kinds_partition_the_graph() -> None:
    groups = tuple(CUSTOM_CNN_MAP)
    leave_one_out = probe_config(Probe("stage1", LEAVE_ONE_OUT, "per_tensor"), groups)
    isolate = probe_config(Probe("stage1", ISOLATE, "per_tensor"), groups)
    assert set(leave_one_out.excluded_groups) | set(isolate.excluded_groups) == set(groups)
    assert not set(leave_one_out.excluded_groups) & set(isolate.excluded_groups)


def test_probe_config_rejects_an_unknown_group() -> None:
    with pytest.raises(KeyError):
        probe_config(Probe("stage9", LEAVE_ONE_OUT, "per_tensor"), tuple(CUSTOM_CNN_MAP))


def test_probe_rejects_an_unknown_kind_or_scheme() -> None:
    with pytest.raises(ValueError):
        Probe("stem", "guess", "per_tensor")
    with pytest.raises(ValueError):
        Probe("stem", LEAVE_ONE_OUT, "per_layer")


def test_probes_for_covers_every_combination_in_a_stable_order() -> None:
    groups = ("stem", "stage1")
    first = probes_for(groups)
    assert len(first) == len(groups) * 2 * len(SCHEMES)
    assert first == probes_for(groups)
    assert len({probe.key for probe in first}) == len(first)


def test_probe_configs_differ_between_schemes() -> None:
    # If both schemes produced the same hash the sweep would silently measure one
    # of them twice and report it as two independent results.
    groups = tuple(CUSTOM_CNN_MAP)
    per_tensor = probe_config(Probe("stage3", LEAVE_ONE_OUT, "per_tensor"), groups)
    per_channel = probe_config(Probe("stage3", LEAVE_ONE_OUT, "per_channel"), groups)
    assert per_tensor.hash != per_channel.hash


def test_anchor_configs_cover_fp32_and_both_full_int8_corners() -> None:
    anchors = anchor_configs()
    assert list(anchors) == ["fp32", "static_per_tensor", "static_per_channel"]
    assert anchors["fp32"].quant_type == "none"
    assert all(not config.excluded_groups for config in anchors.values())


def _metrics(kl: float, top1_delta: float = 0.0) -> dict[str, object]:
    return {
        "status": "ok",
        "metrics": {"kl_mean": kl, "flip_rate": 0.0, "top1_delta": top1_delta, "top1": 70.0},
    }


def test_ranking_puts_the_group_that_heals_the_most_first() -> None:
    # stage3 is the sensitive one: excluding it drops divergence from 1.0 to 0.1,
    # so it must outrank stem, whose exclusion barely helps. Ranking on the raw
    # leave-one-out value instead would invert this exactly.
    anchors = {"static_per_tensor": _metrics(1.0)}
    probes = {
        "per_tensor/leave_one_out/stage3": _metrics(0.1),
        "per_tensor/isolate/stage3": _metrics(0.9),
        "per_tensor/leave_one_out/stem": _metrics(0.95),
        "per_tensor/isolate/stem": _metrics(0.05),
    }

    ranking = rank_groups(("stem", "stage3"), anchors, probes, "per_tensor")
    assert [row["group"] for row in ranking] == ["stage3", "stem"]
    assert ranking[0]["recovery_kl"] == pytest.approx(0.9)
    assert ranking[1]["recovery_kl"] == pytest.approx(0.05)


def test_ranking_skips_groups_whose_probes_have_not_run_yet() -> None:
    anchors = {"static_per_tensor": _metrics(1.0)}
    probes = {"per_tensor/leave_one_out/stem": _metrics(0.9)}
    assert rank_groups(("stem",), anchors, probes, "per_tensor") == []


def test_refine_group_qualifies_names_so_sibling_pools_stay_apart() -> None:
    # Depth-2 naming alone would call both `/stage1/pool` and `/stage3/pool`
    # "pool" and merge two stages' damage into one number.
    refined = refine_group(CUSTOM_CNN_MAP, "stage3", depth=3)
    assert "/stage3/pool/MaxPool" in refined["stage3.pool"]
    assert "/stage1/pool/MaxPool" in refined["stage1"]
    assert "stage3.bottleneck.project" in refined
    assert refined["stage3.bottleneck.project"] == ("/stage3/bottleneck/project/conv/Conv",)


def test_refine_group_keeps_a_parent_owned_node_under_the_parent_name() -> None:
    # The residual Add belongs to the bottleneck itself, not to a submodule.
    refined = refine_group(CUSTOM_CNN_MAP, "stage3", depth=3)
    assert refined["stage3.bottleneck"] == ("/stage3/bottleneck/Add",)


def test_refine_group_preserves_every_node_and_leaves_other_groups_alone() -> None:
    refined = refine_group(CUSTOM_CNN_MAP, "stage3", depth=3)
    assert sum(len(nodes) for nodes in refined.values()) == sum(
        len(nodes) for nodes in CUSTOM_CNN_MAP.values()
    )
    assert refined["stem"] == CUSTOM_CNN_MAP["stem"]
    assert "stage3" not in refined


def test_refine_group_rejects_an_unknown_parent() -> None:
    with pytest.raises(KeyError):
        refine_group(CUSTOM_CNN_MAP, "stage9", depth=2)
