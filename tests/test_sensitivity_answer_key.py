"""Check the analyzer against custom_cnn's planted answer key.

The thresholds here are a translation of the wording in `EXPECTED_SENSITIVITY`,
written before the device sweep ran. Neither the key nor these assertions may be
adjusted to match observed output (PLAN.md DO-NOT #14) -- a failure is a finding
about the plants or the analyzer, and gets reported as one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.models.custom_cnn import EXPECTED_SENSITIVITY

REPORT = Path("artifacts/reports_pi/custom_cnn_sensitivity.json")

# "largely recovered by per-channel": per-channel must own at most half the
# share of total damage that per-tensor does.
RECOVERY_FACTOR = 0.5


def _ranking(scheme: str) -> list[dict[str, Any]]:
    if not REPORT.exists():
        pytest.skip(f"No device sweep at {REPORT}; run src.sensitivity.analyze on the Pi")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if not report.get("on_target"):
        pytest.skip("Report was not measured on the Pi; x86 INT8 kernels saturate on this host")
    return report["ranking"][scheme]


def _position(scheme: str, group: str) -> int:
    ranking = _ranking(scheme)
    for index, row in enumerate(ranking):
        if row["group"] == group:
            return index
    raise AssertionError(f"{group!r} missing from the {scheme} ranking")


def _share(scheme: str, group: str) -> float:
    return _ranking(scheme)[_position(scheme, group)]["recovery_share"]


def test_the_answer_key_still_says_what_these_assertions_encode() -> None:
    # Guards against the key drifting under the test rather than the reverse.
    assert EXPECTED_SENSITIVITY["stem"] == "low"
    assert EXPECTED_SENSITIVITY["stage1"] == "low"
    assert EXPECTED_SENSITIVITY["stage2"].startswith("high under per-tensor weights")
    assert EXPECTED_SENSITIVITY["stage3"].startswith("high regardless")


@pytest.mark.parametrize("scheme", ["per_tensor", "per_channel"])
def test_stage3_is_fragile_under_both_weight_schemes(scheme: str) -> None:
    assert _position(scheme, "stage3") < 2


def test_stage2_is_fragile_under_per_tensor() -> None:
    assert _position("per_tensor", "stage2") < 2


def test_stage2_is_largely_recovered_by_per_channel() -> None:
    assert _share("per_channel", "stage2") <= RECOVERY_FACTOR * _share("per_tensor", "stage2")


@pytest.mark.parametrize("scheme", ["per_tensor", "per_channel"])
@pytest.mark.parametrize("robust", ["stem", "stage1"])
def test_the_robust_groups_rank_below_the_fragile_ones(scheme: str, robust: str) -> None:
    assert _position(scheme, robust) > _position(scheme, "stage3")
