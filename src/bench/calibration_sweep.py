"""Vary only the static calibration recipe, on the Pi, with the graph held fixed.

The Phase 6 trials cannot answer this: each one also had different FP32 blocks.

    python -m src.bench.calibration_sweep --model resnet18_cifar
"""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.bench.agent import BenchSpec
from src.bench.remote import PiConnection, RemoteBenchmarker
from src.quant.config import (
    CALIBRATION_METHODS,
    EXPORTED_MODELS,
    DeploymentConfig,
    QuantConfig,
    RunConfig,
)
from src.search.space import CALIBRATION_SIZES

DEFAULT_REPORT_DIR = Path("artifacts/reports_pi")

# The runtime the rest of the campaign was measured under, so these join to it.
FIXED_RUN = RunConfig(intra_op_num_threads=4, graph_optimization_level="all")


def sweep_configs(
    per_channel: bool,
    methods: tuple[str, ...] = CALIBRATION_METHODS,
    sizes: tuple[int, ...] = CALIBRATION_SIZES,
) -> list[QuantConfig]:
    """Every calibration recipe, with the graph and precision held constant."""
    return [
        QuantConfig(
            quant_type="static",
            per_channel=per_channel,
            calibration_method=method,
            calibration_size=size,
        )
        for method, size in itertools.product(methods, sizes)
    ]


def measure_one(
    runner: RemoteBenchmarker, model: str, quant: QuantConfig
) -> dict[str, Any]:
    """Full accuracy plus latency for one calibration recipe."""
    spec = BenchSpec(model=model, config=DeploymentConfig(quant=quant, run=FIXED_RUN))
    row: dict[str, Any] = {
        "calibration_method": quant.calibration_method,
        "calibration_size": quant.calibration_size,
        "per_channel": quant.per_channel,
        "quant_hash": quant.hash,
    }

    scored = runner.score(spec, limit=None)
    if scored["status"] != "ok":
        return {**row, "status": scored["status"], "error": scored.get("error", "")[:200]}

    measured = runner.measure(spec, use_cache=False)
    if measured["status"] != "ok":
        return {**row, "status": measured["status"], "error": measured.get("error", "")[:200]}

    return {
        **row,
        "status": "ok",
        "top1": scored["accuracy"]["top1"],
        "top5": scored["accuracy"]["top5"],
        "samples": scored["accuracy"]["samples"],
        "bytes": scored["bytes"],
        "median_ms": measured["latency"]["median_ms"],
        "peak_rss_mb": measured["peak_rss_mb"],
        "admissible": measured["admissible"],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if row["status"] == "ok"]
    if not ok:
        return {"measured": 0}

    by_method: dict[str, list[float]] = {}
    for row in ok:
        by_method.setdefault(row["calibration_method"], []).append(row["top1"])

    best = max(ok, key=lambda row: row["top1"])
    return {
        "measured": len(ok),
        "best_top1": best["top1"],
        "best_recipe": f"{best['calibration_method']}/{best['calibration_size']}",
        "top1_span": round(max(r["top1"] for r in ok) - min(r["top1"] for r in ok), 3),
        "method_best": {
            method: round(max(scores), 3) for method, scores in sorted(by_method.items())
        },
        "method_mean": {
            method: round(sum(scores) / len(scores), 3)
            for method, scores in sorted(by_method.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=EXPORTED_MODELS)
    parser.add_argument(
        "--per-channel", nargs="+", type=int, default=[1], help="1, 0, or both"
    )
    parser.add_argument("--methods", nargs="+", default=list(CALIBRATION_METHODS))
    parser.add_argument("--sizes", nargs="+", type=int, default=list(CALIBRATION_SIZES))
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = RemoteBenchmarker(PiConnection.load())
    runner.push_code()

    state = runner.device_state()
    if not state.get("governor_is_pinned"):
        raise SystemExit("governor is not pinned; run: sudo bash scripts/pi_prepare.sh")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    path = args.report_dir / f"{args.model}_calibration_sweep.json"

    # Merge on the recipe key so a retry of a few failed cells keeps the rest.
    report: dict[str, Any] = (
        json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    )
    report.update(
        model=args.model,
        device_state=state,
        run_config=FIXED_RUN.as_dict(),
        created_at_utc=datetime.now(UTC).isoformat(),
    )
    report.setdefault("schemes", {})

    for flag in args.per_channel:
        per_channel = bool(flag)
        scheme = "per_channel" if per_channel else "per_tensor"
        configs = sweep_configs(per_channel, tuple(args.methods), tuple(args.sizes))
        print(f"{scheme}: {len(configs)} calibration recipes\n")

        merged: dict[tuple[str, int], dict[str, Any]] = {
            (row["calibration_method"], row["calibration_size"]): row
            for row in report["schemes"].get(scheme, {}).get("rows", [])
        }
        for quant in configs:
            row = measure_one(runner, args.model, quant)
            label = f"{row['calibration_method']}/{row['calibration_size']}"
            if row["status"] == "ok":
                print(
                    f"  {label:<20} top1 {row['top1']:6.2f}   "
                    f"{row['median_ms']:>7.3f} ms   {row['bytes'] / 1e6:6.2f} MB"
                )
            else:
                print(f"  {label:<20} {row['status']}: {row.get('error', '')[:90]}")
            merged[(row["calibration_method"], row["calibration_size"])] = row

        rows = [merged[key] for key in sorted(merged)]
        summary = summarize(rows)
        report["schemes"][scheme] = {"rows": rows, "summary": summary}
        print(f"\n  best {summary.get('best_recipe')} at {summary.get('best_top1')}, "
              f"span {summary.get('top1_span')} pt\n")

    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
