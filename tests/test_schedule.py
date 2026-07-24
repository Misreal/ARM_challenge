"""Tests for the LR schedule.

Worth testing because a schedule bug is invisible: training still runs, the
loss still falls, and you find out three days later that the model is two
points short and cannot tell whether the architecture or the schedule is to
blame. These assertions pin the shape.
"""

from __future__ import annotations

import pytest

from src.schedule import WarmupCosine, warmup_cosine_lr

BASE_LR = 0.1
TOTAL = 1000
WARMUP = 100


def test_warmup_ramps_linearly_to_base_lr() -> None:
    # Arrange / Act
    first = warmup_cosine_lr(0, BASE_LR, TOTAL, WARMUP)
    mid = warmup_cosine_lr(WARMUP // 2 - 1, BASE_LR, TOTAL, WARMUP)
    last = warmup_cosine_lr(WARMUP - 1, BASE_LR, TOTAL, WARMUP)

    # Assert
    assert first == pytest.approx(BASE_LR / WARMUP)  # never exactly zero
    assert mid == pytest.approx(BASE_LR / 2)
    assert last == pytest.approx(BASE_LR)


def test_cosine_decays_from_peak_to_floor() -> None:
    at_peak = warmup_cosine_lr(WARMUP, BASE_LR, TOTAL, WARMUP)
    halfway = warmup_cosine_lr(WARMUP + (TOTAL - WARMUP) // 2, BASE_LR, TOTAL, WARMUP)
    at_end = warmup_cosine_lr(TOTAL, BASE_LR, TOTAL, WARMUP)

    assert at_peak == pytest.approx(BASE_LR, rel=1e-3)
    assert halfway == pytest.approx(BASE_LR / 2, rel=1e-2)
    assert at_end == pytest.approx(0.0, abs=1e-12)


def test_schedule_is_monotonic_after_warmup() -> None:
    values = [warmup_cosine_lr(s, BASE_LR, TOTAL, WARMUP) for s in range(WARMUP, TOTAL + 1)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_overshooting_total_steps_clamps_instead_of_rising() -> None:
    """A partial final batch can push the step counter past total_steps; the
    cosine must not turn back upward."""
    assert warmup_cosine_lr(TOTAL + 50, BASE_LR, TOTAL, WARMUP) == pytest.approx(0.0, abs=1e-12)


def test_final_lr_ratio_sets_a_floor() -> None:
    lr = warmup_cosine_lr(TOTAL, BASE_LR, TOTAL, WARMUP, final_lr_ratio=0.01)
    assert lr == pytest.approx(BASE_LR * 0.01)


def test_zero_warmup_starts_at_base_lr() -> None:
    assert warmup_cosine_lr(0, BASE_LR, TOTAL, warmup_steps=0) == pytest.approx(BASE_LR)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"total_steps": 0, "warmup_steps": 0},
        {"total_steps": 100, "warmup_steps": 100},  # warmup consumes the whole run
        {"total_steps": 100, "warmup_steps": -1},
    ],
)
def test_invalid_configurations_raise(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        warmup_cosine_lr(0, BASE_LR, **kwargs)


def test_callable_wrapper_matches_the_function() -> None:
    schedule = WarmupCosine(base_lr=BASE_LR, total_steps=TOTAL, warmup_steps=WARMUP)
    assert [schedule(s) for s in (0, 50, 500, 999)] == [
        warmup_cosine_lr(s, BASE_LR, TOTAL, WARMUP) for s in (0, 50, 500, 999)
    ]
