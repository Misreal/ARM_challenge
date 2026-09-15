# Learning-rate schedule: linear warmup into cosine decay.
#
# Computed per *optimizer step*, not per epoch. Warmup exists to stop the first
# few large-gradient updates from destabilising BatchNorm statistics; stepping it
# once per epoch would leave the entire first epoch at full learning rate, which
# is exactly the window it is meant to protect. MobileNetV2 trained from scratch
# is the model in this project most sensitive to getting that wrong.
#
# Pure functions of `step` -- no optimizer state, no hidden counters -- so the
# schedule can be unit-tested and plotted without constructing a training run.

from __future__ import annotations

import math
from dataclasses import dataclass


def warmup_cosine_lr(
    step: int,
    base_lr: float,
    total_steps: int,
    warmup_steps: int,
    final_lr_ratio: float = 0.0,
) -> float:
    """Learning rate at `step` (0-indexed).

    Warmup ramps linearly from `base_lr / warmup_steps` to `base_lr` (never
    starting at exactly 0, which would waste the first update entirely), then
    cosine-decays to `base_lr * final_lr_ratio` at `total_steps`.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError(
            f"warmup_steps must satisfy 0 <= warmup_steps < total_steps, "
            f"got warmup_steps={warmup_steps}, total_steps={total_steps}"
        )
    if not 0.0 <= final_lr_ratio <= 1.0:
        raise ValueError(f"final_lr_ratio must be in [0, 1], got {final_lr_ratio}")

    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps

    # Clamp so an extra step past the end (e.g. a partial final batch) returns
    # the floor rather than turning the cosine back upward.
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (final_lr_ratio + (1.0 - final_lr_ratio) * cosine)


@dataclass(frozen=True)
class WarmupCosine:
    """Callable wrapper so the loop can stay ignorant of the schedule's shape."""

    base_lr: float
    total_steps: int
    warmup_steps: int
    final_lr_ratio: float = 0.0

    def __call__(self, step: int) -> float:
        return warmup_cosine_lr(
            step,
            base_lr=self.base_lr,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            final_lr_ratio=self.final_lr_ratio,
        )
