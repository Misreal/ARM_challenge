# Inference latency and memory measurement, shared verbatim between PC and Pi.
#
# Same reason as `onnx_eval`: the Pi never gets PyTorch, and a timing
# methodology that differs between the two machines makes PC-vs-Pi comparison
# meaningless -- Phase 3's device agent imports this module unchanged. Reports
# the median, not the mean, since the Pi throttles and a handful of slow
# iterations while it heats would drag the mean somewhere that describes
# neither the hot nor the cold steady state (p95/p99 are reported separately so
# the tail stays visible). Also reports a stability verdict: the timed samples
# split in half, their medians compared, and if the second half is slower by
# more than `stability_tolerance` the run was still heating and `stable` is
# False -- a fact about the measurement, not the model, so callers are expected
# to re-measure rather than record it. Peak RSS is process-wide (`ru_maxrss`
# never decreases for the life of the process), so benchmark one artifact per
# process or the second inherits the first one's peak.

from __future__ import annotations

import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import onnxruntime as ort

# `resource` is POSIX-only. It is present on the Pi (the machine whose RAM
# actually matters) and absent on the Windows dev box, where None is reported
# rather than a fabricated substitute.
try:
    import resource
except ImportError:  # pragma: no cover - platform-dependent
    resource = None  # type: ignore[assignment]

DEFAULT_WARMUP = 20
DEFAULT_ITERATIONS = 100
DEFAULT_STABILITY_TOLERANCE = 0.05

# Linux exposes SoC temperature in millidegrees Celsius here. The Pi 5 populates
# it; a generic x86 box may not, and may not have thermal_zone0 at all.
THERMAL_ZONE = Path("/sys/class/thermal/thermal_zone0/temp")


class BenchmarkError(RuntimeError):
    """Raised when a graph cannot be benchmarked as exported.

    Raised rather than worked around: every case that triggers it means an
    upstream invariant was violated, and guessing a shape would hide the bug
    behind a plausible-looking number.
    """


@dataclass(frozen=True)
class LatencyStats:
    """Timing distribution of one artifact on one host."""

    iterations: int
    warmup: int
    batch_size: int
    mean_ms: float
    median_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float
    throughput_ips: float
    drift_ratio: float
    stable: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "iterations": self.iterations,
            "warmup": self.warmup,
            "batch_size": self.batch_size,
            "mean_ms": round(self.mean_ms, 4),
            "median_ms": round(self.median_ms, 4),
            "p90_ms": round(self.p90_ms, 4),
            "p95_ms": round(self.p95_ms, 4),
            "p99_ms": round(self.p99_ms, 4),
            "min_ms": round(self.min_ms, 4),
            "max_ms": round(self.max_ms, 4),
            "stdev_ms": round(self.stdev_ms, 4),
            "throughput_ips": round(self.throughput_ips, 2),
            "drift_ratio": round(self.drift_ratio, 4),
            "stable": self.stable,
        }


@dataclass(frozen=True)
class HostProbe:
    """Where a measurement was taken, and under what thermal conditions."""

    platform: str
    machine: str
    processor: str
    temperature_start_c: float | None
    temperature_end_c: float | None
    peak_rss_mb: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "machine": self.machine,
            "processor": self.processor,
            "temperature_start_c": self.temperature_start_c,
            "temperature_end_c": self.temperature_end_c,
            "peak_rss_mb": self.peak_rss_mb,
        }


def read_cpu_temperature() -> float | None:
    """SoC temperature in Celsius, or None where the host does not expose it."""
    try:
        return int(THERMAL_ZONE.read_text().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def peak_rss_mb() -> float | None:
    """Process high-water resident memory in MB, or None on non-POSIX hosts.

    Linux reports `ru_maxrss` in kilobytes; macOS reports bytes. Only the Linux
    reading is used in this project, since the Pi is the RAM-constrained target.
    """
    if resource is None:
        return None
    kilobytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kilobytes / 1024.0


def probe_host(
    temperature_start_c: float | None = None,
    temperature_end_c: float | None = None,
) -> HostProbe:
    return HostProbe(
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor(),
        temperature_start_c=temperature_start_c,
        temperature_end_c=temperature_end_c,
        peak_rss_mb=peak_rss_mb(),
    )


def resolve_input_shape(session: ort.InferenceSession) -> tuple[int, ...]:
    """Static input shape of the graph, refusing anything dynamic.

    Every artifact in this project is exported at batch size 1 with static axes
    (a hard invariant -- dynamic axes break static INT8 calibration and the Pi
    latency methodology). A symbolic dimension here means that invariant was
    broken upstream, so this raises instead of substituting a batch size and
    producing a number that silently benchmarks the wrong graph.
    """
    spec = session.get_inputs()[0]
    dims = spec.shape
    resolved: list[int] = []
    for axis, dim in enumerate(dims):
        if not isinstance(dim, int) or dim <= 0:
            raise BenchmarkError(
                f"Input '{spec.name}' has non-static axis {axis} (={dim!r}); "
                "artifacts must be exported with static shapes at batch size 1"
            )
        resolved.append(dim)
    return tuple(resolved)


def summarize(
    samples_ms: Sequence[float],
    warmup: int,
    batch_size: int,
    stability_tolerance: float = DEFAULT_STABILITY_TOLERANCE,
) -> LatencyStats:
    """Turn raw per-iteration timings into the reported distribution.

    Split out from the timing loop so the statistics -- including the drift
    verdict, which is the part most likely to be wrong -- are testable without
    running a model.
    """
    if len(samples_ms) < 2:
        raise BenchmarkError(f"Need at least 2 timed samples, got {len(samples_ms)}")

    ordered = np.asarray(samples_ms, dtype=np.float64)
    mean_ms = float(ordered.mean())

    # Compare medians of the first and second half in *arrival* order. Sorting
    # first would destroy the time ordering the drift check depends on.
    midpoint = len(ordered) // 2
    first_median = float(np.median(ordered[:midpoint]))
    second_median = float(np.median(ordered[midpoint:]))
    drift_ratio = second_median / first_median if first_median > 0 else float("inf")

    return LatencyStats(
        iterations=len(ordered),
        warmup=warmup,
        batch_size=batch_size,
        mean_ms=mean_ms,
        median_ms=float(np.median(ordered)),
        p90_ms=float(np.percentile(ordered, 90)),
        p95_ms=float(np.percentile(ordered, 95)),
        p99_ms=float(np.percentile(ordered, 99)),
        min_ms=float(ordered.min()),
        max_ms=float(ordered.max()),
        stdev_ms=float(statistics.stdev(ordered.tolist())),
        throughput_ips=batch_size * 1000.0 / mean_ms if mean_ms > 0 else float("inf"),
        drift_ratio=drift_ratio,
        stable=drift_ratio <= 1.0 + stability_tolerance,
    )


def benchmark_session(
    session: ort.InferenceSession,
    warmup: int = DEFAULT_WARMUP,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = 0,
    stability_tolerance: float = DEFAULT_STABILITY_TOLERANCE,
) -> tuple[LatencyStats, HostProbe]:
    """Time `iterations` single-input runs after `warmup` discarded ones.

    Warmup is not optional padding: the first calls pay for lazy kernel
    selection, memory arena growth, and a cold instruction cache, and on the Pi
    they can be several times the steady-state cost.

    Input is seeded standard-normal float32, which matches the statistics of the
    normalized CIFAR tensors the graph sees in deployment. Convolution timing is
    data-independent, so content does not affect the result -- but a fixed seed
    keeps runs byte-reproducible, and normal-distributed values avoid the
    denormal-float slow paths that all-zeros input can trigger.
    """
    if warmup < 0:
        raise BenchmarkError(f"warmup must be >= 0, got {warmup}")
    if iterations < 2:
        raise BenchmarkError(f"iterations must be >= 2, got {iterations}")

    shape = resolve_input_shape(session)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    rng = np.random.default_rng(seed)
    payload = rng.standard_normal(shape, dtype=np.float32)
    feed = {input_name: payload}
    fetch = [output_name]

    for _ in range(warmup):
        session.run(fetch, feed)

    temperature_start = read_cpu_temperature()
    samples_ms: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        session.run(fetch, feed)
        samples_ms.append((time.perf_counter() - started) * 1000.0)
    temperature_end = read_cpu_temperature()

    stats = summarize(
        samples_ms,
        warmup=warmup,
        batch_size=shape[0],
        stability_tolerance=stability_tolerance,
    )
    return stats, probe_host(temperature_start, temperature_end)
