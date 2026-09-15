"""Sweep the ONNX Runtime session knobs against a fixed artifact, on the Pi."""

# Nothing here is requantized: every cell measures the same bytes run differently.
#
#     python -m src.bench.runtime_sweep --model resnet18_cifar

from __future__ import annotations

import argparse
import itertools
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.bench.agent import BenchSpec
from src.bench.remote import PiConnection, RemoteBenchmarker, configs_for
from src.model_index import known_models
from src.quant.config import DeploymentConfig, RunConfig

DEFAULT_REPORT_DIR = Path("artifacts/reports_pi")
DEFAULT_CONFIGS = ("fp32", "static_per_channel")

THREADS = (1, 2, 4)
# "disabled" leaves QDQ pairs unfused, which measures a known-bad graph.
OPT_LEVELS = ("basic", "extended", "all")
ARENA = (True, False)
SPINNING = (True, False)

REFERENCE = RunConfig()


def runtime_grid() -> list[RunConfig]:
    return [
        RunConfig(
            intra_op_num_threads=threads,
            graph_optimization_level=level,
            enable_cpu_mem_arena=arena,
            allow_intra_op_spinning=spinning,
        )
        for threads, level, arena, spinning in itertools.product(
            THREADS, OPT_LEVELS, ARENA, SPINNING
        )
    ]


def sweep_one(
    runner: RemoteBenchmarker, model: str, config_name: str, use_cache: bool
) -> list[dict[str, Any]]:
    """Measure one artifact once per runtime cell."""
    quant = configs_for(model)[config_name]
    rows: list[dict[str, Any]] = []

    for run in runtime_grid():
        spec = BenchSpec(model=model, config=DeploymentConfig(quant=quant, run=run))
        result = runner.measure(spec, use_cache=use_cache)
        row: dict[str, Any] = {"run": run.as_dict(), "describe": run.describe()}

        if result["status"] != "ok":
            row.update(status=result["status"], error=result.get("error", "")[:200])
            print(f"  {run.describe():<34} {result['status']}")
        else:
            row.update(
                status="ok",
                median_ms=result["latency"]["median_ms"],
                peak_rss_mb=result["peak_rss_mb"],
                admissible=result["admissible"],
                stable=result["latency"]["stable"],
            )
            flag = "" if result["admissible"] else "  [INADMISSIBLE]"
            print(
                f"  {run.describe():<34} {row['median_ms']:>8.3f} ms "
                f"{row['peak_rss_mb']:>7.1f} MB{flag}"
            )
        rows.append(row)

    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Rank the cells and report each cell relative to the frozen reference."""
    ok = [row for row in rows if row["status"] == "ok"]
    if not ok:
        return {"measured": 0}

    reference = next(
        (row for row in ok if row["run"] == REFERENCE.as_dict()),
        min(ok, key=lambda row: row["median_ms"]),
    )
    for row in ok:
        row["latency_vs_reference"] = round(row["median_ms"] / reference["median_ms"], 4)
        row["rss_delta_mb"] = round(row["peak_rss_mb"] - reference["peak_rss_mb"], 2)

    fastest = min(ok, key=lambda row: row["median_ms"])
    leanest = min(ok, key=lambda row: row["peak_rss_mb"])
    return {
        "measured": len(ok),
        "reference": reference["describe"],
        "reference_ms": reference["median_ms"],
        "fastest": fastest["describe"],
        "fastest_ms": fastest["median_ms"],
        "reference_is_fastest": fastest["describe"] == reference["describe"],
        "leanest": leanest["describe"],
        "leanest_rss_mb": leanest["peak_rss_mb"],
        "rss_span_mb": round(
            max(row["peak_rss_mb"] for row in ok) - min(row["peak_rss_mb"] for row in ok), 2
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=known_models())
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = RemoteBenchmarker(PiConnection.load())
    runner.push_code()

    state = runner.device_state()
    if not state.get("governor_is_pinned"):
        raise SystemExit("governor is not pinned; run: sudo bash scripts/pi_prepare.sh")

    cells = len(runtime_grid())
    print(f"{args.model}: {cells} runtime cells x {len(args.configs)} config(s)\n")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    path = args.report_dir / f"{args.model}_runtime_sweep.json"

    # Merge rather than replace: sweeping one config at a time is the normal way
    # to run this, and a plain write would drop the configs swept earlier.
    report: dict[str, Any] = (
        json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    )
    report.update(
        model=args.model,
        device_state=state,
        grid={
            "threads": list(THREADS),
            "graph_optimization_level": list(OPT_LEVELS),
            "enable_cpu_mem_arena": list(ARENA),
            "allow_intra_op_spinning": list(SPINNING),
        },
        created_at_utc=datetime.now(UTC).isoformat(),
    )
    report.setdefault("configs", {})

    for name in args.configs:
        print(f"{name}")
        rows = sweep_one(runner, args.model, name, use_cache=not args.no_cache)
        summary = summarize(rows)
        report["configs"][name] = {"rows": rows, "summary": summary}
        print(f"  -> fastest {summary.get('fastest')} at {summary.get('fastest_ms')} ms, "
              f"RSS span {summary.get('rss_span_mb')} MB\n")

    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {path} ({', '.join(sorted(report['configs']))})")


if __name__ == "__main__":
    main()
