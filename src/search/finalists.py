"""Re-measure the shortlist, so no ranking rests on a single noisy measurement.

The front is built from one measurement per candidate. Repeat-to-repeat spread
on byte-equivalent work reaches 1.9% at p90, which is larger than most gaps
between neighbouring front members -- so the finalists are measured again, with
the cache off, and compared inside the band rather than on printed digits.

    python -m src.search.finalists --model resnet18_cifar --study <name>
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.bench.agent import BenchSpec
from src.quant.config import DeploymentConfig, EXPORTED_MODELS
from src.search import select
from src.search.evaluate import DeviceUnavailable
from src.search.study import DEFAULT_STUDY_DIR
from src.search.version import study_name

FINALIST_SCHEMA = "finalists/1"

DEFAULT_REPEATS = 5
DEFAULT_LIMIT = 5


def median_absolute_deviation(values: list[float]) -> float:
    """Spread that one thermal outlier cannot inflate, unlike a standard deviation."""
    if len(values) < 2:
        return 0.0
    middle = statistics.median(values)
    return statistics.median([abs(value - middle) for value in values])


def repeat_rounds(
    runner: Any, model: str, rows: list[dict[str, Any]], repeats: int
) -> dict[str, list[dict[str, Any]]]:
    """Measure every finalist once per round, reversing order on alternate rounds.

    Interleaving is the point: measuring one candidate five times in a row aligns
    thermal drift with candidate identity, so the last candidate measured would
    carry the warmest die rather than the worst design.
    """
    raw: dict[str, list[dict[str, Any]]] = {row["config_hash"]: [] for row in rows}

    for round_index in range(repeats):
        ordered = rows if round_index % 2 == 0 else list(reversed(rows))
        for row in ordered:
            config = DeploymentConfig.from_dict(row["config"])
            spec = BenchSpec(model=model, config=config)
            try:
                measured = runner.measure(spec, use_cache=False)
            except DeviceUnavailable as error:
                raw[row["config_hash"]].append({"round": round_index, "status": "transport_failed",
                                                "error": str(error)})
                return raw
            device = measured.get("readiness", {}).get("device", {})
            raw[row["config_hash"]].append(
                {
                    "round": round_index,
                    "status": measured.get("status"),
                    "latency_ms": measured.get("latency", {}).get("median_ms"),
                    "peak_rss_mb": measured.get("peak_rss_mb"),
                    "admissible": measured.get("admissible"),
                    "temperature_c": device.get("temperature_c"),
                    "throttled": device.get("throttled"),
                }
            )
    return raw


def aggregate(row: dict[str, Any], measurements: list[dict[str, Any]]) -> dict[str, Any]:
    """One finalist's repeats reduced to the numbers a ranking may use."""
    good = [m for m in measurements if m.get("status") == "ok" and m.get("latency_ms") is not None]
    latencies = [float(m["latency_ms"]) for m in good]
    rss = [float(m["peak_rss_mb"]) for m in good if m.get("peak_rss_mb") is not None]
    temperatures = [m["temperature_c"] for m in good if m.get("temperature_c") is not None]

    repeated: dict[str, Any] = {
        "repeats": len(measurements),
        "succeeded": len(good),
        "failed": len(measurements) - len(good),
        "median_ms": round(statistics.median(latencies), 4) if latencies else None,
        "mad_ms": round(median_absolute_deviation(latencies), 4) if latencies else None,
        "min_ms": round(min(latencies), 4) if latencies else None,
        "max_ms": round(max(latencies), 4) if latencies else None,
        "peak_rss_mb": round(max(rss), 1) if rss else None,
        "admissible": all(m.get("admissible") for m in good) if good else False,
        "throttled": any(m.get("throttled") for m in measurements),
        "temperature_c": {"min": min(temperatures), "max": max(temperatures)} if temperatures else None,
        "measurements": measurements,
    }
    return {**row, "repeated": repeated}


def run_finalists(
    runner: Any,
    model: str,
    summary: dict[str, Any],
    repeats: int = DEFAULT_REPEATS,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Pick the shortlist, re-measure it, and name the deployment choices."""
    band = summary.get("band") or {}
    band_ms = band.get("ms")
    max_rss_mb = (summary.get("search_plan") or {}).get("max_rss_mb")

    evaluated = summary.get("evaluated_configs") or summary.get("pareto") or []
    shortlist = select.finalists(evaluated, band_ms, limit)

    raw = repeat_rounds(runner, model, shortlist, repeats) if shortlist else {}
    measured = [aggregate(row, raw.get(row["config_hash"], [])) for row in shortlist]

    # Only latency needed re-measuring, so the named choices are drawn from every
    # candidate with the repeated medians folded in. Picking them from the
    # shortlist alone would report the most accurate *finalist* as the most
    # accurate candidate, which is a different and wrong claim.
    repeated_by_hash = {row["config_hash"]: row for row in measured}
    merged = [repeated_by_hash.get(row["config_hash"], row) for row in evaluated]
    chosen = select.selections(merged, band_ms, max_rss_mb)
    return {
        "schema": FINALIST_SCHEMA,
        "model": model,
        "study": summary.get("study"),
        "band": summary.get("band"),
        "repeats": repeats,
        "max_rss_mb": max_rss_mb,
        # Named from the repeated medians, so the winner is the one that is
        # faster than the others by more than the harness can misreport.
        "selections": {
            choice: None if row is None else row["config_hash"] for choice, row in chosen.items()
        },
        "ties": ties_within_band(measured, band_ms),
        "finalists": measured,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }


def ties_within_band(rows: list[dict[str, Any]], band_ms: float | None) -> list[list[str]]:
    """Groups of finalists whose repeated medians the band cannot separate."""
    if not band_ms:
        return []

    groups: list[list[str]] = []
    leader: float | None = None
    for row in sorted(rows, key=select.latency_of):
        latency = select.latency_of(row)
        if leader is None or latency - leader > band_ms:
            groups.append([row["config_hash"]])
            leader = latency
        else:
            groups[-1].append(row["config_hash"])
    return [group for group in groups if len(group) > 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=EXPORTED_MODELS)
    parser.add_argument("--study", default=None, help="study name; derived from --budget-pt if absent")
    parser.add_argument("--budget-pt", type=float, default=0.2)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--report-dir", type=Path, default=Path("artifacts/reports_pi"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    name = args.study or study_name(args.model, args.budget_pt, args.mock)
    summary = json.loads((args.study_dir / f"{name}.json").read_text(encoding="utf-8"))

    if args.mock:
        from src.model_index import entry_for
        from src.search.mock import MockRunner

        runner = MockRunner(entry_for(args.model).groups)
    else:
        from src.bench.remote import PiConnection, RemoteBenchmarker

        runner = RemoteBenchmarker(PiConnection.load())
        runner.push_code()

    report = run_finalists(runner, args.model, summary, args.repeats, args.limit)

    print(f"{name}: {len(report['finalists'])} finalists x {args.repeats} repeats")
    for row in report["finalists"]:
        repeated = row["repeated"]
        print(
            f"  {repeated['median_ms']:8.4f} ms  MAD {repeated['mad_ms']:7.4f}  "
            f"{repeated['succeeded']}/{repeated['repeats']} ok  "
            f"[{row['describe']} | {row['describe_run']}]"
        )
    for choice, config_hash in report["selections"].items():
        print(f"  {choice:16s} {config_hash}")
    for group in report["ties"]:
        print(f"  tie inside the band: {', '.join(group)}")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    path = args.report_dir / f"{args.model}_finalists.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
