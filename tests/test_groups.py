"""Node-to-block grouping decides what `nodes_to_exclude` actually excludes, so
a silent mistake here would make every mixed-precision result meaningless while
still producing plausible-looking numbers."""

from __future__ import annotations

from collections import OrderedDict

import pytest

from src.quant.groups import (
    UNSCOPED_GROUP,
    group_depth_for,
    group_of_node,
    nodes_for_groups,
    quantizable_groups,
)


@pytest.mark.parametrize(
    ("node_name", "depth", "expected"),
    [
        # Real names taken from the exported graphs.
        ("/stem/conv/Conv", 1, "stem"),
        ("/stage2/dw1/dw/conv/Conv", 1, "stage2"),
        ("/classifier/Gemm", 1, "classifier"),
        ("/layer1/layer1.0/conv1/Conv", 1, "layer1"),
        ("/conv1/Conv", 1, "conv1"),
        # MobileNetV2 needs depth 2, or 167 of 170 nodes land in one group.
        ("/features/features.0/features.0.0/Conv", 2, "features.0"),
        ("/features/features.7/features.7.conv/Conv", 2, "features.7"),
        # Module path shorter than the requested depth falls back to what exists.
        ("/classifier/classifier.1/Gemm", 2, "classifier.1"),
        ("/classifier/Gemm", 2, "classifier"),
    ],
)
def test_group_of_node(node_name: str, depth: int, expected: str) -> None:
    assert group_of_node(node_name, depth) == expected


def test_functional_nodes_land_in_the_unscoped_bucket() -> None:
    # These come from functional calls, not named modules, and carry no weights.
    assert group_of_node("/Flatten", 1) == UNSCOPED_GROUP
    assert group_of_node("/GlobalAveragePool", 2) == UNSCOPED_GROUP


def test_configured_depths_match_the_three_models() -> None:
    assert group_depth_for("resnet18_cifar") == 1
    assert group_depth_for("custom_cnn") == 1
    assert group_depth_for("mobilenetv2_cifar") == 2
    # An unregistered model gets the default rather than a KeyError; the
    # MAX_GROUP_SHARE guard in build_group_map catches it if that is wrong.
    assert group_depth_for("some_future_model") == 1


def test_quantizable_groups_drops_the_unscoped_bucket() -> None:
    group_map: OrderedDict[str, tuple[str, ...]] = OrderedDict(
        [("stem", ("/stem/conv/Conv",)), (UNSCOPED_GROUP, ("/Flatten",))]
    )
    assert quantizable_groups(group_map) == ("stem",)


def test_nodes_for_groups_flattens_in_request_order() -> None:
    group_map: OrderedDict[str, tuple[str, ...]] = OrderedDict(
        [("stem", ("/stem/conv/Conv",)), ("stage1", ("/stage1/a/Conv", "/stage1/b/Relu"))]
    )
    assert nodes_for_groups(group_map, ("stem", "stage1")) == [
        "/stem/conv/Conv",
        "/stage1/a/Conv",
        "/stage1/b/Relu",
    ]


def test_unknown_group_is_rejected() -> None:
    # Guards against replaying a stale config against a re-exported model.
    group_map: OrderedDict[str, tuple[str, ...]] = OrderedDict([("stem", ("/stem/conv/Conv",))])
    with pytest.raises(KeyError, match="stage9"):
        nodes_for_groups(group_map, ("stage9",))
