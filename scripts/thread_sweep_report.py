"""Build the thread-sweep table from cached Pi measurements.

Reads artifacts/bench_cache/ rather than any transcribed number, so the table
cannot drift from what was measured (PLAN.md forbids hand-copied figures).

    python scripts/thread_sweep_report.py
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CACHE = Path("artifacts/bench_cache")
MODELS = ("resnet18_cifar", "mobilenetv2_cifar", "custom_cnn")
CONFIG_ORDER = ("fp32", "dynamic", "static_per_tensor", "static_per_channel")


def label_for(config: dict[str, Any]) -> str:
    quant = config["quant"]
    if quant["quant_type"] == "none":
        return "fp32"
    if quant["quant_type"] == "dynamic":
        return "dynamic"
    return "static_per_channel" if quant["per_channel"] else "static_per_tensor"


def collect(model: str) -> dict[tuple[str, int], dict[str, Any]]:
    measurements: dict[tuple[str, int], dict[str, Any]] = {}
    for path in (CACHE / model).glob("*.json"):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "ok" or not result.get("admissible"):
            continue
        threads = result["config"]["run"]["intra_op_num_threads"]
        measurements[(label_for(result["config"]), threads)] = result
    return measurements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("artifacts/reports_pi/thread_sweep.json"))
    args = parser.parse_args()

    report: dict[str, Any] = {
        "schema": "thread_sweep/1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "models": {},
    }

    for model in MODELS:
        found = collect(model)
        if not found:
            continue
        rows: dict[str, Any] = {}
        for config in CONFIG_ORDER:
            per_thread = {
                str(threads): found[(config, threads)]
                for threads in (1, 2, 4)
                if (config, threads) in found
            }
            if not per_thread:
                continue
            latencies = {t: r["latency"]["median_ms"] for t, r in per_thread.items()}
            rows[config] = {
                "median_ms": {t: round(v, 4) for t, v in latencies.items()},
                "peak_rss_mb": {
                    t: round(r["peak_rss_mb"], 1) for t, r in per_thread.items()
                },
                "rss_attributable": all(r["rss_attributable"] for r in per_thread.values()),
                "bytes": next(iter(per_thread.values()))["bytes"],
                "cores_busy_at_4t": per_thread.get("4", {}).get("cpu_cores_busy"),
                # Speedup from 1 to 4 threads; 4.0 would be perfect scaling.
                "thread_scaling_1_to_4": (
                    round(latencies["1"] / latencies["4"], 2)
                    if "1" in latencies and "4" in latencies
                    else None
                ),
            }
        fp32_4t = rows.get("fp32", {}).get("median_ms", {}).get("4")
        for config, row in rows.items():
            best = min(row["median_ms"].values())
            row["best_median_ms"] = best
            row["best_threads"] = min(row["median_ms"], key=row["median_ms"].get)
            row["speedup_vs_fp32_4t"] = round(fp32_4t / best, 3) if fp32_4t else None
        report["models"][model] = rows

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for model, rows in report["models"].items():
        print(f"\n{model}")
        print(f"  {'config':<20}{'1t':>9}{'2t':>9}{'4t':>9}{'scale':>7}{'RAM@4t':>9}{'MB':>8}")
        for config, row in rows.items():
            times = row["median_ms"]
            print(
                f"  {config:<20}"
                + "".join(f"{times.get(t, float('nan')):>9.3f}" for t in ("1", "2", "4"))
                + f"{row['thread_scaling_1_to_4'] or 0:>6.2f}x"
                + f"{row['peak_rss_mb'].get('4', 0):>9.1f}"
                + f"{row['bytes'] / 1e6:>8.2f}"
            )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
