"""Diagnose INT8 accumulator saturation on the host CPU.

Per-channel INT8 scored worse than per-tensor on all three models here, which
is impossible as a property of quantization -- per-channel is strictly finer
grained. This dev CPU (Ryzen 7 5800H, Zen 3) has AVX2 but no VNNI, so ONNX
Runtime's U8S8 path accumulates in 16 bits and saturates, and per-channel
makes that worse by using more of the int8 range per channel. reduce_range
avoids the overflow but changes the weights, so it's not screening-only.
Re-run this on the Pi (Cortex-A76 has ARMv8.2 dot-product, should not
saturate) before trusting any PC accuracy screen.

    python -m src.quant.saturation_probe --model resnet18_cifar
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
from src.quant.config import QuantConfig, RunConfig
from src.quant.evaluate import score_artifact
from src.quant.quantize import build_artifact

DEFAULT_REPORT_DIR = Path("artifacts/reports")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=known_models())
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = RunConfig(intra_op_num_threads=args.threads)

    cells: list[dict[str, Any]] = []
    scores: dict[tuple[bool, bool], float] = {}
    for per_channel in (False, True):
        for reduce_range in (False, True):
            config = QuantConfig(
                quant_type="static", per_channel=per_channel, reduce_range=reduce_range
            )
            artifact = build_artifact(args.model, config)
            accuracy = score_artifact(artifact, run=run, limit=args.limit)
            scores[(per_channel, reduce_range)] = accuracy.top1
            cells.append(
                {
                    "per_channel": per_channel,
                    "reduce_range": reduce_range,
                    "top1": accuracy.top1,
                    "quant_hash": config.hash,
                }
            )
            print(
                f"  per_channel={per_channel!s:<5} reduce_range={reduce_range!s:<5} "
                f"top-1 {accuracy.top1:.2f}%"
            )

    # The tell: how much does reduce_range rescue per-channel? On a saturating
    # host this is large and positive; on a clean one it should be ~0.
    per_channel_recovery = scores[(True, True)] - scores[(True, False)]
    per_tensor_recovery = scores[(False, True)] - scores[(False, False)]
    saturating = per_channel_recovery > 1.0

    print(
        f"\nreduce_range recovers {per_channel_recovery:+.2f} pt on per-channel, "
        f"{per_tensor_recovery:+.2f} pt on per-tensor"
    )
    print(
        "VERDICT: host appears to SATURATE -- per-channel accuracy measured here is "
        "not trustworthy."
        if saturating
        else "VERDICT: no saturation signature; per-channel accuracy on this host looks sound."
    )

    report = {
        "model": args.model,
        "eval_split": "optval",
        "eval_limit": args.limit,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "run_config": run.as_dict(),
        "cells": cells,
        "per_channel_reduce_range_recovery": round(per_channel_recovery, 2),
        "per_tensor_reduce_range_recovery": round(per_tensor_recovery, 2),
        "saturation_suspected": saturating,
        "onnxruntime": ort.__version__,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / f"{args.model}_saturation_probe.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
