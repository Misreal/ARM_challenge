"""Named selective-FP32 candidates for the Phase 5 head-to-head.

Exclusion sets name real graph groups, so they are model-specific and cannot
live alongside the model-agnostic BASELINE_CONFIGS.
"""

from __future__ import annotations

from src.quant.config import QuantConfig

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
