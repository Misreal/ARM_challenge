"""Diagnose INT8 accumulator saturation on the host CPU.

Why this exists
---------------
Phase 4's first baseline run produced an impossible result: static per-channel
INT8 scored *worse* than per-tensor on all three models (-5.37 / -2.37 /
-10.53 pt vs FP32, against -2.00 / -0.10 / -3.97 for per-tensor). Per-channel
quantization is strictly finer-grained -- one scale per output channel instead
of one for the whole tensor -- so it cannot genuinely lose accuracy to
per-tensor. Something other than the models was being measured.

The mechanism
-------------
ONNX Runtime's U8S8 path (uint8 activations, int8 weights) accumulates products
in 16 bits on x86 CPUs without VNNI. Large products saturate. Per-channel makes
this *more* likely, not less: rescaling every channel to use the full int8
range raises typical magnitudes, whereas one coarse per-tensor scale leaves most
channels using only part of the range. `reduce_range=True` quantizes weights to
7 bits and avoids the overflow.

The dev machine is an AMD Ryzen 7 5800H (Zen 3): AVX2, no VNNI. AMD gained VNNI
with Zen 4.

Why it matters to the campaign
------------------------------
The Pi 5's Cortex-A76 implements the ARMv8.2 dot-product instructions, so its
MLAS kernels accumulate into int32 and should not saturate at all. If that
holds, PC accuracy screening is biased by 3-5 pt *specifically against
per-channel candidates* -- 30-50x the +/-0.1 pt PLAN.md budgets for x86-vs-ARM
kernel differences, and enough to make Phase 6's hard accuracy constraint reject
exactly the candidates most likely to win.

Note `reduce_range` is not a screening-only workaround: it changes the weights,
so a model screened with it is not the model deployed without it.

**This must be re-run on the Pi before trusting any PC accuracy screen.** Same
artifacts, same evaluation code (`src.portable.onnx_eval`), different machine.
If the Pi shows a flat per-channel/per-tensor relationship, the effect is
confirmed as host-specific and accuracy screening has to move on-device.

Example:
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

from src.models.registry import available_models
from src.quant.config import QuantConfig, RunConfig
from src.quant.evaluate import score_artifact
from src.quant.quantize import build_artifact

DEFAULT_REPORT_DIR = Path("artifacts/reports")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=available_models())
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
