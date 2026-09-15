# Map an Optuna trial onto a DeploymentConfig: artifact knobs plus runtime knobs.
#
# Both reductions here are measured, from `*_runtime_sweep.json` on the Pi.

from __future__ import annotations

from dataclasses import replace
from typing import Any

from src.quant.config import CALIBRATION_METHODS, DeploymentConfig, QuantConfig, RunConfig

# Capped at the 1000 images `export_pi_data` actually writes: a larger value is
# not a slow candidate, it is an unsatisfiable one that spends a trial to fail.
CALIBRATION_SIZES = (128, 256, 512, 1000)

# Dynamic was sampled half the time and measured 9-19x slower than static on all
# three models, reaching the front only through top-1 differences smaller than the
# noise floor. It stays a reported baseline in `*_quant_baselines.json`; searching
# it just halved the budget for the per-layer decision the method is about.
QUANT_TYPES = ("static",)

# Entropy and percentile accumulate a histogram per tensor across the whole
# calibration set and OOM the 4 GB Pi at 1000 images. MinMax keeps a running
# min/max, so its cost does not grow with the set.
HISTOGRAM_METHODS = ("entropy", "percentile")
HISTOGRAM_MAX_CALIBRATION_SIZE = 512

# Models whose graph has too many activation tensors for a histogram at any
# size. The cap above bounds the calibration set; it cannot bound the tensor
# count, and for `vit_cifar` that is what exhausts the device: 1037 QDQ nodes
# over 65x192 activations per block. Measured on the Pi 5 -- entropy was
# OOM-killed at 3.96 GB for 128, 256 and 512 images alike, while minmax peaked
# at 346 MB. A histogram trial here is not a slow candidate, it is one that
# kills the agent and costs the study its remaining budget.
NO_HISTOGRAM_MODELS = ("vit_cifar",)

# The runtime sweep measured 4 threads fastest at every optimization level with
# peak RSS flat, and spinning on faster with no memory saving. Neither is a
# trade-off, so searching them would only dilute the trial budget.
FIXED_THREADS = 4
FIXED_SPINNING = True

# "basic" measured 7x slower than "all" on a quantized graph, and "disabled" is
# worse by construction: both leave QDQ pairs unfused.
SEARCHED_OPT_LEVELS = ("extended", "all")


def calibration_sizes_for(method: str) -> tuple[int, ...]:
    if method in HISTOGRAM_METHODS:
        return tuple(size for size in CALIBRATION_SIZES if size <= HISTOGRAM_MAX_CALIBRATION_SIZE)
    return CALIBRATION_SIZES


def calibration_methods_for(model: str | None) -> tuple[str, ...]:
    """Calibration methods the device can actually build for this model."""
    if model in NO_HISTOGRAM_MODELS:
        return tuple(m for m in CALIBRATION_METHODS if m not in HISTOGRAM_METHODS)
    return tuple(CALIBRATION_METHODS)


def suggest_run_config(trial: Any) -> RunConfig:
    """The execution half of the space: no effect on the artifact bytes."""
    return RunConfig(
        intra_op_num_threads=FIXED_THREADS,
        graph_optimization_level=trial.suggest_categorical(
            "graph_optimization_level", list(SEARCHED_OPT_LEVELS)
        ),
        # The one runtime knob that trades: off saved 8.1 MB of peak RSS for a
        # latency change inside the noise floor.
        enable_cpu_mem_arena=trial.suggest_categorical("enable_cpu_mem_arena", [True, False]),
        allow_intra_op_spinning=FIXED_SPINNING,
    )


def suggest_within(trial: Any, space: Any, model: str | None = None) -> DeploymentConfig:
    """One point inside a reduced space: pins composed, searchable groups sampled.

    The pins are composed rather than sampled, so no trial can spend device time
    on a precision vector the measurements already decided.
    """
    config = suggest_config(trial, tuple(sorted(space.searchable)), model=model)
    excluded = tuple(sorted(set(config.quant.excluded_groups) | set(space.pinned_fp32)))
    return DeploymentConfig(
        quant=replace(config.quant, excluded_groups=excluded), run=config.run
    )


def suggest_config(
    trial: Any, groups: tuple[str, ...], model: str | None = None
) -> DeploymentConfig:
    """One point in the space: a per-group precision vector plus global settings.

    The precision vector is sampled as one independent boolean per group rather
    than as a subset, so NSGA-II's crossover can recombine per-group decisions
    instead of treating every subset as an unrelated categorical level.
    """
    quant_type = trial.suggest_categorical("quant_type", list(QUANT_TYPES))
    per_channel = trial.suggest_categorical("per_channel", [True, False])
    excluded = tuple(
        group for group in groups if trial.suggest_categorical(f"fp32__{group}", [False, True])
    )

    if quant_type == "dynamic":
        # Dynamic computes activation ranges at inference time, so calibration
        # and activation type are not its parameters and must not be sampled:
        # they would be recorded on the trial while changing nothing.
        quant = QuantConfig(
            quant_type="dynamic", per_channel=per_channel, excluded_groups=excluded
        )
    else:
        method = trial.suggest_categorical(
            "calibration_method", list(calibration_methods_for(model))
        )
        quant = QuantConfig(
            quant_type="static",
            per_channel=per_channel,
            activation_type=trial.suggest_categorical("activation_type", ["uint8", "int8"]),
            calibration_method=method,
            # Conditional on the method so the sampler cannot draw a combination
            # the device has already been measured unable to build.
            calibration_size=trial.suggest_categorical(
                f"calibration_size__{method}", list(calibration_sizes_for(method))
            ),
            excluded_groups=excluded,
        )

    return DeploymentConfig(quant=quant, run=suggest_run_config(trial))
