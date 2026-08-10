"""Map an Optuna trial onto a DeploymentConfig: artifact knobs plus runtime knobs.

Both reductions here are measured, from `*_runtime_sweep.json` on the Pi.
"""

from __future__ import annotations

from typing import Any

from src.quant.config import CALIBRATION_METHODS, DeploymentConfig, QuantConfig, RunConfig

# Capped at the 1000 images `export_pi_data` actually writes: a larger value is
# not a slow candidate, it is an unsatisfiable one that spends a trial to fail.
CALIBRATION_SIZES = (128, 256, 512, 1000)
QUANT_TYPES = ("static", "dynamic")

# Entropy and percentile accumulate a histogram per tensor across the whole
# calibration set and OOM the 4 GB Pi at 1000 images. MinMax keeps a running
# min/max, so its cost does not grow with the set.
HISTOGRAM_METHODS = ("entropy", "percentile")
HISTOGRAM_MAX_CALIBRATION_SIZE = 512

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


def suggest_config(trial: Any, groups: tuple[str, ...]) -> DeploymentConfig:
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
        method = trial.suggest_categorical("calibration_method", list(CALIBRATION_METHODS))
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
