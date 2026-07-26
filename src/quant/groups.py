"""Map ONNX nodes back to named PyTorch blocks.

PLAN.md schedules this for Phase 5, but Phase 4 needs it first: mixed precision
is implemented through `nodes_to_exclude`, and a config that excludes a *block*
cannot be translated without this map. Phase 5 consumes it rather than
introducing it.

How the names survive
---------------------
`torch.onnx.export` prefixes each node with its module path, so a node arrives
as `/stage2/dw1/dw/conv/Conv` -- module path components, then the op type. The
registry's stable-naming contract (`src/models/registry.py:15`) is what makes
this reliable, and it was verified to hold for all three exported graphs: every
node is named, nothing anonymous.

Grouping depth is per-model on purpose
--------------------------------------
ResNet-18 and custom_cnn group correctly at depth 1 (`layer1`, `stem`, ...).
MobileNetV2 does not: 167 of its 170 nodes live under a single `features`
container, so depth 1 yields one useless mega-group and its block structure is
at depth 2 (`features.0`, `features.1`, ...). Three models is too few to be
clever about inferring this, so it is declared. `MAX_GROUP_SHARE` catches the
mistake if a new model is added without the corresponding entry.

custom_cnn deliberately groups to exactly the keys in its `EXPECTED_SENSITIVITY`
answer key, so Phase 5's validation compares them with no name translation.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import onnx

# Nodes produced by functional calls rather than named modules -- Flatten and
# GlobalAveragePool, one or two per model. They carry no parameters and are
# never quantization targets, but they need a home so the map stays total.
UNSCOPED_GROUP = "_unscoped"

# Depth of the module path that defines a block, per model.
GROUP_DEPTH: dict[str, int] = {
    "resnet18_cifar": 1,
    "mobilenetv2_cifar": 2,
    "custom_cnn": 1,
}
DEFAULT_GROUP_DEPTH = 1

# If one group swallows more than this share of the graph, the depth is wrong.
MAX_GROUP_SHARE = 0.5


def group_depth_for(model: str) -> int:
    return GROUP_DEPTH.get(model, DEFAULT_GROUP_DEPTH)


def group_of_node(node_name: str, depth: int) -> str:
    """Block a node belongs to, from its ONNX name.

    The last path component is the op type, so the module path is everything
    before it. A node with no module path at all (`/Flatten`) is unscoped.
    """
    components = [part for part in node_name.split("/") if part]
    module_path = components[:-1]
    if not module_path:
        return UNSCOPED_GROUP
    return module_path[min(depth, len(module_path)) - 1]


def build_group_map(
    model_path: Path, model_name: str, depth: int | None = None
) -> "OrderedDict[str, tuple[str, ...]]":
    """Group name -> node names, in graph order.

    Raises if the graph contains unnamed nodes (the naming contract is broken,
    and every downstream exclusion would be silently unreliable) or if one group
    dominates the graph (the configured depth is too shallow).
    """
    resolved_depth = group_depth_for(model_name) if depth is None else depth
    graph = onnx.load(str(model_path)).graph

    groups: OrderedDict[str, list[str]] = OrderedDict()
    anonymous = 0
    for node in graph.node:
        if not node.name:
            anonymous += 1
            continue
        groups.setdefault(group_of_node(node.name, resolved_depth), []).append(node.name)

    if anonymous:
        raise ValueError(
            f"{model_path} has {anonymous} unnamed nodes. The registry's stable-naming "
            "contract is broken, so block exclusions cannot be trusted. Re-export before "
            "running any sensitivity or search work."
        )

    total = sum(len(nodes) for nodes in groups.values())
    for name, nodes in groups.items():
        if name != UNSCOPED_GROUP and len(nodes) / total > MAX_GROUP_SHARE:
            raise ValueError(
                f"Group {name!r} holds {len(nodes)}/{total} nodes in {model_path.name}. "
                f"Grouping depth {resolved_depth} is too shallow for {model_name!r}; "
                f"add it to GROUP_DEPTH in {__name__}."
            )

    return OrderedDict((name, tuple(nodes)) for name, nodes in groups.items())


def quantizable_groups(group_map: "OrderedDict[str, tuple[str, ...]]") -> tuple[str, ...]:
    """Groups the search may assign precision to.

    Excludes the unscoped bucket: those nodes carry no weights, so pinning them
    FP32 would be a search dimension with no effect -- a wasted axis in an
    already-sparse NSGA-II front.
    """
    return tuple(name for name in group_map if name != UNSCOPED_GROUP)


def nodes_for_groups(
    group_map: "OrderedDict[str, tuple[str, ...]]", groups: tuple[str, ...]
) -> list[str]:
    """Flatten selected groups into the node-name list ORT's API expects."""
    unknown = sorted(set(groups) - set(group_map))
    if unknown:
        raise KeyError(
            f"Unknown group(s) {unknown}; graph has {sorted(group_map)}. "
            "A stale config is being replayed against a re-exported model."
        )
    return [node for name in groups for node in group_map[name]]
