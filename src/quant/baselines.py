"""Build and score the global quantization baselines (PLAN.md Phase 4 + 6.6)."""

# These are the bar the optimizer has to beat, so they must exist before any
# search runs and must go through the identical toolkit the search uses -- a
# baseline measured by a different code path proves nothing.
#
#     fp32                 the deployable export, unquantized
#     dynamic              dynamic INT8; ORT recommends it for RNN/transformers
#                          rather than CNNs, kept because it is a required baseline
#     static_per_tensor    diagnostic exhibit only. Presenting this as "the INT8
#                          baseline" would be a strawman: per-tensor weights are
#                          known-bad for depthwise convolutions, so beating it
#                          says nothing.
#     static_per_channel   THE headline comparison. Per-channel weights plus
#                          static activations is standard cookbook practice, and
#                          the contribution is whatever the search finds beyond it.
#
# Running this also answers, retroactively, the MobileNetV2 question the Phase 1
# pilot gate was meant to settle: if per-channel recovers what per-tensor loses,
# the architecture was never the problem.
#
# Example:
#     python -m src.quant.baselines --model mobilenetv2_cifar

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import onnxruntime as ort

from src.model_index import known_models
from src.quant.config import QuantConfig, RunConfig
from src.quant.evaluate import score_artifact
from src.quant.quantize import (
    DEFAULT_CACHE_DIR,
    DEFAULT_ONNX_DIR,
    QuantizationFailure,
    build_artifact,
)

DEFAULT_REPORT_DIR = Path("artifacts/reports")

BASELINE_CONFIGS: dict[str, QuantConfig] = {
    "fp32": QuantConfig(quant_type="none"),
    "dynamic": QuantConfig(quant_type="dynamic"),
    "static_per_tensor": QuantConfig(quant_type="static", per_channel=False),
    "static_per_channel": QuantConfig(quant_type="static", per_channel=True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=known_models())
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="evaluate only the first N optval images (default: all 3000)",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = RunConfig(intra_op_num_threads=args.threads)

    results: dict[str, Any] = {}
    fp32_top1: float | None = None

    for label, config in BASELINE_CONFIGS.items():
        print(f"\n=== {label}: {config.describe()} ===")
        try:
            artifact = build_artifact(
                args.model,
                config,
                onnx_dir=args.onnx_dir,
                cache_dir=args.cache_dir,
                use_cache=not args.no_cache,
            )
        except QuantizationFailure as error:
            # An infeasible baseline is itself a result worth recording.
            print(f"  FAILED: {error}")
            results[label] = {"status": "failed", "error": str(error), "config": config.as_dict()}
            continue

        accuracy = score_artifact(artifact, run=run, limit=args.limit)
        if label == "fp32":
            fp32_top1 = accuracy.top1

        delta = None if fp32_top1 is None else round(accuracy.top1 - fp32_top1, 2)
        results[label] = {
            "status": "ok",
            "config": config.as_dict(),
            "quant_hash": config.hash,
            "bytes": artifact.bytes,
            "qdq_nodes": artifact.int8_op_count,
            "build_seconds": round(artifact.build_seconds, 2),
            "accuracy": accuracy.as_dict(),
            "top1_delta_vs_fp32": delta,
        }
        size_mb = artifact.bytes / 1e6
        print(
            f"  top-1 {accuracy.top1:.2f}%  top-5 {accuracy.top5:.2f}%  "
            f"{size_mb:.2f} MB  QDQ nodes {artifact.int8_op_count}"
            + (f"  ({delta:+.2f} pt vs fp32)" if delta is not None else "")
        )

    # Accuracy, unlike latency, is not gated on the host machine -- an x86 run is
    # still a real measurement. But INT8 kernels differ between x86 and ARM, so the
    # report has to say which one produced it rather than assuming the PC.
    on_target = platform.machine().lower() in ("aarch64", "arm64")
    note = (
        "Measured on the aarch64 target (Raspberry Pi 5). These are deployment "
        "numbers, not screening numbers."
        if on_target
        else "PC (x86) screening numbers. INT8 kernels differ between x86 and ARM "
        "by roughly +/-0.1 pt; finalists are re-verified on the Pi."
    )

    report = {
        "model": args.model,
        "eval_split": "optval",
        "eval_limit": args.limit,
        "on_target": on_target,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "run_config": run.as_dict(),
        "baselines": results,
        "onnxruntime": ort.__version__,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "note": note,
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / f"{args.model}_quant_baselines.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
