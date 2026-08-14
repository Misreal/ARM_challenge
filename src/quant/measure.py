"""Measure latency, throughput and size of quantized candidates on this host.

Admissibility
-------------
This is the first module in the project that produces numbers which are **not
automatically valid**. Accuracy transfers between x86 and ARM to within about
+/-0.1 pt (modulo the saturation caveat in `saturation_probe`). Latency does
not transfer at all: Zen 3 with AVX2 and Cortex-A76 with NEON are different
instruction sets driving different MLAS kernels over different memory systems,
and "a smaller model is frequently not faster on ARM" is the premise the whole
project rests on.

So every report carries an `admissible` flag that is True only on aarch64, and
the filename is suffixed with the machine type. A PC report and a Pi report can
therefore never overwrite one another, and a PC latency cannot reach a results
table without someone deliberately ignoring a field that says it must not.

Run it on the dev box to exercise the harness. Run it on the Pi to get results.

Example:
    python -m src.quant.measure --model resnet18_cifar
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import onnxruntime as ort

from src.model_index import known_models
from src.portable.benchmark import (
    DEFAULT_ITERATIONS,
    DEFAULT_WARMUP,
    BenchmarkError,
    benchmark_session,
    peak_rss_mb,
)
from src.quant.baselines import BASELINE_CONFIGS
from src.quant.config import QuantConfig, RunConfig
from src.quant.quantize import QuantizationFailure, build_artifact, build_session

DEFAULT_REPORT_DIR = Path("artifacts/reports")

# Latency measured anywhere else is a harness smoke test, not evidence.
ADMISSIBLE_MACHINES = ("aarch64", "arm64")

FP32_KEY = "fp32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=known_models())
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=sorted(BASELINE_CONFIGS),
        default=sorted(BASELINE_CONFIGS),
        help="Baseline configs to measure (default: all)",
    )
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def measure_one(
    model: str,
    config: QuantConfig,
    run: RunConfig,
    warmup: int,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Build (or reuse) one artifact and time it, recording failures inline.

    A failure is recorded rather than raised so one unbenchmarkable candidate
    does not discard the measurements already taken -- the same contract Phase 6
    needs when a trial turns out infeasible.
    """
    try:
        artifact = build_artifact(model, config)
    except (QuantizationFailure, FileNotFoundError) as error:
        return {"status": "build_failed", "error": str(error), "config": config.as_dict()}

    try:
        session = build_session(artifact.path, run)
        stats, host = benchmark_session(
            session, warmup=warmup, iterations=iterations, seed=seed
        )
    except (BenchmarkError, ort.capi.onnxruntime_pybind11_state.Fail, RuntimeError) as error:
        return {
            "status": "benchmark_failed",
            "error": f"{type(error).__name__}: {error}",
            "config": config.as_dict(),
            "quant_hash": config.hash,
        }

    return {
        "status": "ok",
        "config": config.as_dict(),
        "quant_hash": config.hash,
        "bytes": artifact.bytes,
        "latency": stats.as_dict(),
        # Process-wide high-water mark, so it accumulates across the configs
        # measured in this run. Named to prevent it being read as per-model RAM;
        # Phase 3's device agent must fork per artifact to attribute memory.
        "peak_rss_mb_cumulative": host.peak_rss_mb,
        "temperature_start_c": host.temperature_start_c,
        "temperature_end_c": host.temperature_end_c,
    }


def main() -> None:
    args = parse_args()
    run = RunConfig(intra_op_num_threads=args.threads)
    machine = platform.machine().lower()
    admissible = machine in ADMISSIBLE_MACHINES

    if not admissible:
        print(
            f"NOTE: machine is '{platform.machine()}', not ARM. These timings exercise "
            "the harness only and are marked inadmissible in the report.\n"
        )

    results: dict[str, dict[str, Any]] = {}
    for name in args.configs:
        print(f"[{name}]")
        result = measure_one(
            args.model,
            BASELINE_CONFIGS[name],
            run,
            warmup=args.warmup,
            iterations=args.iterations,
            seed=args.seed,
        )
        results[name] = result

        if result["status"] != "ok":
            print(f"  {result['status']}: {result['error']}")
            continue

        latency = result["latency"]
        flag = "" if latency["stable"] else "  [UNSTABLE: still warming]"
        print(
            f"  median {latency['median_ms']:.3f} ms   p95 {latency['p95_ms']:.3f} ms   "
            f"{latency['throughput_ips']:.1f} img/s   {result['bytes'] / 1e6:.2f} MB{flag}"
        )

    # Speedup and size ratio are the numbers the campaign actually ranks on, so
    # derive them once here rather than leaving every consumer to recompute.
    reference = results.get(FP32_KEY)
    if reference and reference["status"] == "ok":
        base_ms = reference["latency"]["median_ms"]
        base_bytes = reference["bytes"]
        for result in results.values():
            if result["status"] == "ok":
                result["speedup_vs_fp32"] = round(base_ms / result["latency"]["median_ms"], 3)
                result["size_ratio_vs_fp32"] = round(result["bytes"] / base_bytes, 4)

        print("\n  config              median ms   speedup   size")
        for name, result in results.items():
            if result["status"] == "ok":
                print(
                    f"  {name:<18} {result['latency']['median_ms']:>9.3f}   "
                    f"{result['speedup_vs_fp32']:>6.2f}x   "
                    f"{result['size_ratio_vs_fp32']:>5.2f}x"
                )

    report = {
        "model": args.model,
        "admissible": admissible,
        "admissibility_reason": (
            "aarch64 host; latency is a measured result"
            if admissible
            else f"host machine '{platform.machine()}' is not ARM -- latency does not "
            "transfer to the Pi and these numbers must not enter any results table"
        ),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "run_config": run.as_dict(),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "seed": args.seed,
        "peak_rss_mb_final": peak_rss_mb(),
        "results": results,
        "onnxruntime": ort.__version__,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    # Machine suffix keeps PC and Pi reports from ever overwriting each other.
    report_path = args.report_dir / f"{args.model}_latency_{machine}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
