"""Map an Optuna trial onto a DeploymentConfig.

Only the artifact is searched: `intra_op_num_threads` is frozen at 4 by the
Phase 3 thread sweep and graph optimization at "all", so neither is a dimension.
"""

from __future__ import annotations

from typing import Any

from src.quant.config import CALIBRATION_METHODS, DeploymentConfig, QuantConfig, RunConfig

# Capped at the 1000 images `export_pi_data` actually writes: a larger value is
# not a slow candidate, it is an unsatisfiable one that spends a trial to fail.
CALIBRATION_SIZES = (128, 256, 512, 1000)
QUANT_TYPES = ("static", "dynamic")

# Frozen, not searched. See ROADMAP §4 #6 for the thread evidence.
FROZEN_RUN = RunConfig(intra_op_num_threads=4, graph_optimization_level="all")


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
        quant = QuantConfig(
            quant_type="static",
            per_channel=per_channel,
            activation_type=trial.suggest_categorical("activation_type", ["uint8", "int8"]),
            calibration_method=trial.suggest_categorical(
                "calibration_method", list(CALIBRATION_METHODS)
            ),
            calibration_size=trial.suggest_categorical(
                "calibration_size", list(CALIBRATION_SIZES)
            ),
            excluded_groups=excluded,
        )

    return DeploymentConfig(quant=quant, run=FROZEN_RUN)
