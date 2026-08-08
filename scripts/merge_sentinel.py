"""Merge sentinel part-reports into one drift summary.

Long sentinels are run in parts because a single unattended invocation is not
reliable here; the spread must still be computed across all parts at once.

    python scripts/merge_sentinel.py artifacts/reports_pi/sentinel_part*.json \
        --out artifacts/reports_pi/sentinel_resnet18_cifar_20260808.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("parts", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trials: list[dict] = []
    model = None
    config = None
    for path in sorted(args.parts):
        part = json.loads(path.read_text(encoding="utf-8"))
        model = model or part["model"]
        config = config or part["config"]
        trials.extend(part["results"])

    ok = [t for t in trials if t.get("status") == "ok"]
    medians = [t["latency"]["median_ms"] for t in ok]
    if len(medians) < 2:
        raise SystemExit("need at least 2 successful trials to report a spread")

    median = statistics.median(medians)
    spread = (max(medians) - min(medians)) / median
    temps = [t["device_before"]["temperature_c"] for t in ok if t.get("device_before")]

    summary = {
        "schema": "sentinel/1",
        "model": model,
        "config": config,
        "parts": [str(path) for path in sorted(args.parts)],
        "trials": len(trials),
        "admissible_trials": sum(1 for t in trials if t.get("admissible")),
        "medians_ms": [round(value, 4) for value in medians],
        "median_of_medians_ms": round(median, 4),
        "min_ms": round(min(medians), 4),
        "max_ms": round(max(medians), 4),
        "stdev_ms": round(statistics.stdev(medians), 4),
        "spread_fraction": round(spread, 5),
        "spread_percent": round(spread * 100, 2),
        "tolerance_percent": round(args.tolerance * 100, 2),
        "passes_tolerance": spread <= args.tolerance,
        "start_temperatures_c": temps,
        "first_trial_utc": ok[0].get("created_at_utc"),
        "last_trial_utc": ok[-1].get("created_at_utc"),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "results": trials,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    verdict = "PASS" if summary["passes_tolerance"] else "FAIL"
    print(
        f"{summary['trials']} trials  spread {summary['spread_percent']}%  "
        f"(bar {summary['tolerance_percent']}%)  -> {verdict}"
    )
    print(f"  median {summary['median_of_medians_ms']} ms  "
          f"min {summary['min_ms']}  max {summary['max_ms']}  sd {summary['stdev_ms']}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
