"""Measure one candidate per process on the Pi, so ru_maxrss is attributable."""

# Host contract is file-based (spec JSON in, result JSON out): stdout is unusable
# over SSH, where MOTD banners land in the same stream.
#
#     python -m src.bench.agent --spec spec.json --out result.json
#     python -m src.bench.agent --print-device

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import onnxruntime as ort

from src.bench.protocol import RESULT_SCHEMA, SPEC_SCHEMA
from src.portable.benchmark import (
    DEFAULT_ITERATIONS,
    DEFAULT_STABILITY_TOLERANCE,
    DEFAULT_WARMUP,
    BenchmarkError,
    benchmark_session,
    peak_rss_mb,
)
from src.portable.device import (
    ReadinessPolicy,
    check_readiness,
    probe_device,
    read_process_cpu_seconds,
    throttle_events_between,
)
from src.quant.config import DeploymentConfig, QuantConfig, RunConfig
from src.quant.quantize import QuantizationFailure, build_artifact, build_session


@dataclass(frozen=True)
class BenchSpec:
    """One measurement request, as it travels from host to device."""

    model: str
    config: DeploymentConfig = field(default_factory=DeploymentConfig)
    warmup: int = DEFAULT_WARMUP
    iterations: int = DEFAULT_ITERATIONS
    seed: int = 0
    stability_tolerance: float = DEFAULT_STABILITY_TOLERANCE
    policy: ReadinessPolicy = field(default_factory=ReadinessPolicy)
    # Let the host take a number off a device that fails the guards. It is
    # still marked inadmissible; this only stops the agent refusing outright.
    enforce_guards: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SPEC_SCHEMA,
            "model": self.model,
            "config": self.config.as_dict(),
            "warmup": self.warmup,
            "iterations": self.iterations,
            "seed": self.seed,
            "stability_tolerance": self.stability_tolerance,
            "policy": self.policy.as_dict(),
            "enforce_guards": self.enforce_guards,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchSpec:
        schema = data.get("schema")
        if schema != SPEC_SCHEMA:
            raise ValueError(
                f"spec schema is {schema!r}, expected {SPEC_SCHEMA!r} -- host and device "
                "are running different versions of this code"
            )
        return cls(
            model=data["model"],
            config=DeploymentConfig.from_dict(data["config"]),
            warmup=data.get("warmup", DEFAULT_WARMUP),
            iterations=data.get("iterations", DEFAULT_ITERATIONS),
            seed=data.get("seed", 0),
            stability_tolerance=data.get("stability_tolerance", DEFAULT_STABILITY_TOLERANCE),
            policy=ReadinessPolicy(**data.get("policy", {})),
            enforce_guards=data.get("enforce_guards", True),
        )

    @property
    def quant(self) -> QuantConfig:
        return self.config.quant

    @property
    def run(self) -> RunConfig:
        return self.config.run


def _envelope(spec: BenchSpec, status: str) -> dict[str, Any]:
    """Fields every result carries, whatever happened to it."""
    return {
        "schema": RESULT_SCHEMA,
        "status": status,
        "model": spec.model,
        "config": spec.config.as_dict(),
        "quant_hash": spec.quant.hash,
        "config_hash": spec.config.hash,
        "onnxruntime": ort.__version__,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }


def build_only(spec: BenchSpec) -> dict[str, Any]:
    """Quantize and cache the artifact, without timing anything."""
    # Separate process so the calibrator's memory never lands in the timed RSS.
    result = _envelope(spec, "ok")
    started = time.perf_counter()
    try:
        artifact = build_artifact(spec.model, spec.quant)
    except (QuantizationFailure, FileNotFoundError) as error:
        return {**result, "status": "build_failed", "error": f"{type(error).__name__}: {error}"}

    return {
        **result,
        "phase": "build",
        "bytes": artifact.bytes,
        "sha256": artifact.sha256,
        "from_cache": artifact.from_cache,
        "int8_op_count": artifact.int8_op_count,
        "build_seconds": round(time.perf_counter() - started, 3),
        "peak_rss_mb_build": peak_rss_mb(),
    }


def measure(spec: BenchSpec) -> dict[str, Any]:
    """Gate the device, time one candidate, and report why the number is (in)valid."""
    result = _envelope(spec, "ok")

    readiness = check_readiness(spec.policy)
    result["readiness"] = readiness.as_dict()
    if readiness.violations and spec.enforce_guards:
        listed = "\n  - ".join(readiness.violations)
        return {
            **result,
            "status": "device_not_ready",
            "error": f"Device is not fit to benchmark:\n  - {listed}",
        }

    try:
        artifact = build_artifact(spec.model, spec.quant)
    except (QuantizationFailure, FileNotFoundError) as error:
        return {**result, "status": "build_failed", "error": f"{type(error).__name__}: {error}"}

    device_before = probe_device()
    cpu_before = read_process_cpu_seconds()
    wall_before = time.perf_counter()

    try:
        load_started = time.perf_counter()
        session = build_session(artifact.path, spec.run)
        session_load_seconds = time.perf_counter() - load_started

        stats, host = benchmark_session(
            session,
            warmup=spec.warmup,
            iterations=spec.iterations,
            seed=spec.seed,
            stability_tolerance=spec.stability_tolerance,
        )
    except (BenchmarkError, RuntimeError) as error:
        return {**result, "status": "benchmark_failed", "error": f"{type(error).__name__}: {error}"}

    wall_elapsed = time.perf_counter() - wall_before
    cpu_after = read_process_cpu_seconds()
    device_after = probe_device()
    events = throttle_events_between(device_before.throttle, device_after.throttle)

    admissible, reason = _judge(device_after, events, stats.stable)

    return {
        **result,
        "admissible": admissible,
        "admissibility_reason": reason,
        "bytes": artifact.bytes,
        "sha256": artifact.sha256,
        "int8_op_count": artifact.int8_op_count,
        "latency": stats.as_dict(),
        "session_load_seconds": round(session_load_seconds, 4),
        "peak_rss_mb": host.peak_rss_mb,
        # True only on a cache hit: nothing here allocated on the quant path.
        "rss_attributable": artifact.from_cache,
        "rss_note": (
            "peak RSS covers session creation and inference only"
            if artifact.from_cache
            else "artifact was quantized in this process, so peak RSS includes the "
            "calibrator; run --build-only first for an attributable RAM number"
        ),
        "cpu_cores_busy": _cores_busy(cpu_before, cpu_after, wall_elapsed),
        "throttle_events": list(events),
        "device_before": device_before.as_dict(),
        "device_after": device_after.as_dict(),
    }


def _judge(device, events: tuple[str, ...], stable: bool) -> tuple[bool, str]:
    """Decide whether this timing may enter a results table, and say why."""
    # Ordered by severity: wrong ISA, then thermodynamics, then still warming.
    if not device.is_target:
        return False, (
            f"host machine '{device.machine}' is not the aarch64 target -- latency does "
            "not transfer to the Pi and must not enter any results table"
        )
    if events:
        return False, f"throttle events during the run ({', '.join(events)}); re-measure when cool"
    if not stable:
        return False, "timings drifted between the first and second half; device still warming"
    return True, "aarch64 target, no throttling, timings stable"


def _cores_busy(before: float | None, after: float | None, wall_seconds: float) -> float | None:
    """Average cores kept busy: CPU-seconds consumed per wall-second."""
    # Catches a 4-thread session silently running on one core, which would make
    # the thread-count search dimension measure nothing.
    if before is None or after is None or wall_seconds <= 0:
        return None
    return round((after - before) / wall_seconds, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, help="Path to a spec JSON written by the host")
    parser.add_argument("--out", type=Path, help="Where to write the result JSON")
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Quantize and cache the artifact, then exit without timing it",
    )
    parser.add_argument(
        "--print-device",
        action="store_true",
        help="Dump device state (governor, clock, temperature, throttle mask) and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.print_device:
        print(json.dumps(probe_device().as_dict(), indent=2))
        return

    if args.spec is None:
        raise SystemExit("--spec is required (or use --print-device)")

    spec = BenchSpec.from_dict(json.loads(args.spec.read_text(encoding="utf-8")))
    result = build_only(spec) if args.build_only else measure(spec)

    payload = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    else:
        print(payload)

    # Non-zero exit plus a result file means the candidate failed; no result
    # file means the agent never ran. The host tells them apart without parsing
    # SSH stderr.
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
