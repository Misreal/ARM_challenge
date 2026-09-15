# Staged evaluation of one candidate, cheap gates first.
#
# Ordering is the point: a candidate that fails to build, or that a 500-image
# screen already rejects, must never reach a latency measurement. Running an
# expensive stage before a cheap one that would have rejected the candidate is
# the main performance trap in the search loop.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from src.bench.agent import BenchSpec

# Binomial sigma at n=500 is about 2 points, so screening exactly at the
# threshold false-rejects good candidates roughly half the time.
SCREEN_LIMIT = 500
SCREEN_MARGIN_PT = 2.0

OK = "ok"
BUILD_FAILED = "build_failed"
TRANSPORT_FAILED = "transport_failed"
SCREEN_REJECTED = "screen_rejected"
BELOW_THRESHOLD = "below_threshold"
MEASUREMENT_FAILED = "measurement_failed"


class DeviceUnavailable(RuntimeError):
    """The link to the Pi died, so nothing was learned about this candidate.

    Distinct from a build failure on purpose: recording a dropped connection as
    an infeasible trial teaches the sampler that a perfectly good region of the
    space is bad, and every later trial inherits that lie.
    """


class Runner(Protocol):
    """What the evaluator needs from a device. `RemoteBenchmarker` satisfies it."""

    def score(self, spec: BenchSpec, limit: int | None = None) -> dict[str, Any]: ...

    def measure(self, spec: BenchSpec, use_cache: bool = True) -> dict[str, Any]: ...


@dataclass(frozen=True)
class TrialResult:
    """Everything one candidate produced, feasible or not."""

    status: str
    latency_ms: float | None = None
    size_bytes: int | None = None
    peak_rss_mb: float | None = None
    top1: float | None = None
    screen_top1: float | None = None
    admissible: bool = False
    error: str | None = None

    @property
    def feasible(self) -> bool:
        return self.status == OK

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "latency_ms": self.latency_ms,
            "size_bytes": self.size_bytes,
            "peak_rss_mb": self.peak_rss_mb,
            "top1": self.top1,
            "screen_top1": self.screen_top1,
            "admissible": self.admissible,
            "error": self.error,
        }


def _stage(result: dict[str, Any]) -> dict[str, Any]:
    """Pass a stage result through, unless the device was never reached."""
    if result.get("status") == TRANSPORT_FAILED:
        raise DeviceUnavailable(result.get("error", "the device could not be reached"))
    return result


def evaluate(
    runner: Runner,
    spec: BenchSpec,
    threshold_top1: float,
    screen_limit: int = SCREEN_LIMIT,
    screen_margin_pt: float = SCREEN_MARGIN_PT,
) -> TrialResult:
    """Run the staged gates for one candidate and report what came back."""
    screen = _stage(runner.score(spec, limit=screen_limit))
    if screen["status"] != "ok":
        return TrialResult(status=BUILD_FAILED, error=screen.get("error"))

    screen_top1 = screen["accuracy"]["top1"]
    if screen_top1 < threshold_top1 - screen_margin_pt:
        return TrialResult(status=SCREEN_REJECTED, screen_top1=screen_top1)

    measured = _stage(runner.measure(spec))
    if measured["status"] != "ok":
        return TrialResult(
            status=MEASUREMENT_FAILED, screen_top1=screen_top1, error=measured.get("error")
        )

    full = _stage(runner.score(spec, limit=None))
    if full["status"] != "ok":
        return TrialResult(
            status=MEASUREMENT_FAILED, screen_top1=screen_top1, error=full.get("error")
        )

    top1 = full["accuracy"]["top1"]
    result = TrialResult(
        status=OK if top1 >= threshold_top1 else BELOW_THRESHOLD,
        latency_ms=measured["latency"]["median_ms"],
        size_bytes=full["bytes"],
        peak_rss_mb=measured["peak_rss_mb"],
        top1=top1,
        screen_top1=screen_top1,
        admissible=bool(measured.get("admissible")),
    )
    return result
