"""Named selective-FP32 candidates for the Phase 5 head-to-head.

Exclusion sets name real graph groups, so they are model-specific and cannot
live alongside the model-agnostic BASELINE_CONFIGS.
"""

from __future__ import annotations

from pathlib import Path

from src.quant.config import QuantConfig
from src.quant.groups import build_group_map, quantizable_groups
from src.quant.quantize import DEFAULT_ONNX_DIR, ModelPaths

PER_GROUP_PREFIX = "spare_"

# Standard practice everywhere: spare the first conv and the classifier.
COOKBOOK_EXCLUSIONS: dict[str, tuple[str, ...]] = {
    "resnet18_cifar": ("conv1", "fc"),
    "custom_cnn": ("stem", "classifier"),
    "mobilenetv2_cifar": ("features.0", "classifier.1"),
}

# Top two groups by per-channel recovery share in artifacts/reports_pi. Frozen
# here so the comparison cannot be retuned after seeing the latency it produces.
MEASURED_EXCLUSIONS: dict[str, tuple[str, ...]] = {
    "resnet18_cifar": ("layer1", "conv1"),
    "custom_cnn": ("stage2", "stem"),
    "mobilenetv2_cifar": ("features.0", "features.1"),
}

CANDIDATE_EXCLUSIONS: dict[str, dict[str, tuple[str, ...]]] = {
    "mixed_cookbook": COOKBOOK_EXCLUSIONS,
    "mixed_measured": MEASURED_EXCLUSIONS,
}


def per_group_configs(model: str, onnx_dir: Path = DEFAULT_ONNX_DIR) -> dict[str, QuantConfig]:
    """One config per group, sparing only that group, everything else INT8.

    These are byte-identical to the Phase 5 leave-one-out probes, so measuring
    their latency completes an existing accuracy row instead of starting a
    second table that would have to be reconciled with the first.
    """
    group_map = build_group_map(ModelPaths.resolve(model, onnx_dir).quant_ready, model)
    return {
        f"{PER_GROUP_PREFIX}{group}": QuantConfig(
            quant_type="static", per_channel=True, excluded_groups=(group,)
        )
        for group in quantizable_groups(group_map)
    }


def candidate_configs(model: str) -> dict[str, QuantConfig]:
    """The selective-FP32 configs defined for `model`, keyed by candidate name."""
    configs: dict[str, QuantConfig] = {}
    for name, exclusions in CANDIDATE_EXCLUSIONS.items():
        groups = exclusions.get(model)
        if groups is None:
            continue
        configs[name] = QuantConfig(
            quant_type="static",
            per_channel=True,
            excluded_groups=tuple(sorted(groups)),
        )
    return configs
