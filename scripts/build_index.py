"""Build the landing page that lists every run.

Headline figures are extracted from each run's own stage files, so this page
cannot disagree with the page it links to.

    python -m scripts.build_index
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.build_dashboard import REFERENCE_RUN, measurements_by_config
from src.pipeline import STAGES, produced_path
from src.quant.baselines import BASELINE_CONFIGS
from src.quant.config import DeploymentConfig
from src.runs import RUNS_DIR, Run, list_runs, read_manifest

CACHE_DIR = Path("artifacts/bench_cache")

# What a card needs before it can show figures at all. Everything else is detail,
# and adopted campaigns legitimately lack some of it.
ESSENTIAL = ("quant_baselines", "study")

TEMPLATE = Path("scripts/index_template.html")
DEFAULT_OUT = Path("artifacts/dashboard/index.html")
DATA_PLACEHOLDER = "/*__INDEX_DATA__*/"

LABELS = {
    "resnet18_cifar": "ResNet-18",
    "mobilenetv2_cifar": "MobileNetV2",
    "custom_cnn": "Custom CNN",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fp32_latency_ms(run: Run, cache_dir: Path) -> float | None:
    """What the unquantized export measured, for the speedup on each card.

    The quantization report carries accuracy and bytes but not time; latency
    lives in the device cache, joined by deployment hash exactly as the run's
    own page joins it.
    """
    if run.venue != "pi":
        return None
    measured = measurements_by_config(run.model, cache_dir)
    reference = DeploymentConfig(quant=BASELINE_CONFIGS["fp32"], run=REFERENCE_RUN).hash
    found = measured.get(reference)
    return found["latency"]["median_ms"] if found else None


def day(stamp: str) -> str:
    """`2026-08-09T18:25:23+00:00` as `9 Aug 2026`, for a card anyone can read."""
    try:
        moment = datetime.fromisoformat(stamp)
    except ValueError:
        return stamp[:10]
    return f"{moment.day} {moment:%b %Y}"


def satisfied_stages(run: Run) -> dict[str, bool]:
    """Which stages this run can be said to have done, judged from what is on disk.

    Two cases the file list alone gets wrong. An adopted campaign never ran the
    orchestrator, so an intermediate like `group_cost` has no file even though the
    stage that consumed it landed. And a stage recorded `n/a` was deliberately
    skipped, not missed.
    """
    entries = read_manifest(run).get("stages", {})
    tracked = [stage for stage in STAGES if produced_path(stage, run) is not None]
    downstream = {stage.name: tuple(o.name for o in STAGES if stage.name in o.needs) for stage in STAGES}

    ok: dict[str, bool] = {}
    for stage in reversed(tracked):
        if produced_path(stage, run).exists() or entries.get(stage.name, {}).get("status") == "n/a":
            ok[stage.name] = True
            continue
        feeds = [name for name in downstream[stage.name] if name in ok]
        ok[stage.name] = bool(feeds) and all(ok[name] for name in feeds)
    return {stage.name: ok[stage.name] for stage in tracked}


def measured_on(run: Run) -> str:
    """When the search actually ran, which is not when the directory was made.

    The three adopted campaigns were copied in together, so every run.json
    carries the same adoption stamp and none of them is a measurement date.
    """
    study = run.paths.stage("study")
    if study.exists():
        stamp = read_json(study).get("created_at_utc")
        if stamp:
            return day(stamp)
    return day(run.created_at_utc)


def progress(run: Run) -> dict[str, Any]:
    """How far this run got, and what it never produced."""
    ok = satisfied_stages(run)
    missing = [name for name, done in ok.items() if not done]
    manifest = read_manifest(run)
    entries = manifest.get("stages", {})
    failed = [name for name, entry in entries.items() if entry.get("status") == "failed"]

    return {
        "stages_done": sum(ok.values()),
        "stages_total": len(ok),
        "missing": missing,
        "next_stage": missing[0] if missing else None,
        "failed_stage": failed[0] if failed else None,
        "failed_error": entries[failed[0]].get("error", "") if failed else "",
        "updated_at_utc": manifest.get("updated_at_utc", run.created_at_utc),
    }


def figures(run: Run, cache_dir: Path) -> dict[str, Any] | None:
    """The headline numbers, or None when the run has not produced any yet."""
    if any(not run.paths.stage(name).exists() for name in ESSENTIAL):
        return None

    summary = read_json(run.paths.stage("study"))
    front = [member["result"] for member in summary["pareto"] if member["result"]["status"] == "ok"]
    if not front:
        return None

    fp32 = read_json(run.paths.stage("quant_baselines"))["baselines"]["fp32"]
    fp32_latency = fp32.get("latency_ms") or fp32_latency_ms(run, cache_dir)
    fastest = min(result["latency_ms"] for result in front)

    return {
        "trials": summary["trials"],
        "front_size": len(front),
        "best_latency_ms": fastest,
        "best_size_mb": min(result["size_bytes"] for result in front) / 1e6,
        "best_top1": max(result["top1"] for result in front),
        "baseline_top1": summary["baseline_top1"],
        "fp32_size_mb": fp32["bytes"] / 1e6,
        # Against the FP32 export measured through the same harness. Zero when
        # that measurement is not in the cache, and the card drops the note.
        "speedup": (fp32_latency / fastest) if fp32_latency else 0.0,
    }


def card(run: Run, out: Path, cache_dir: Path = CACHE_DIR) -> dict[str, Any]:
    """One card. Every run gets one, whether or not it has finished."""
    found = figures(run, cache_dir)
    walked = progress(run)
    page = run.paths.dashboard

    # A card's job is showing results, so having them is what decides the shape.
    # `missing` then qualifies them, rather than hiding a finished search behind
    # a progress bar because one optional stage never ran.
    if walked["failed_stage"]:
        state = "failed"
    elif found:
        state = "done"
    else:
        # Nothing to show yet. Says nothing about whether a process is still
        # alive, which is not knowable from disk.
        state = "running"

    return {
        "run_id": run.run_id,
        "label": LABELS.get(run.model, run.model),
        "model": run.model,
        "venue": run.venue,
        "state": state,
        "measured_on": measured_on(run),
        "updated_on": day(walked["updated_at_utc"]),
        **walked,
        **(found or {}),
        # Relative, so the file:// link works wherever the repo is cloned.
        "page": Path(os.path.relpath(page, out.parent)).as_posix() if page.exists() else None,
    }


def sort_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    """Unfinished first because that is the live news, then fastest, then failures."""
    rank = {"running": 0, "done": 1, "failed": 2}[entry["state"]]
    return (rank, entry["venue"] != "pi", entry.get("best_latency_ms", 0.0))


def latest_per_model(runs: list[Run]) -> list[Run]:
    """One card per model: the most recently created run that actually finished.

    A run that died partway through (a stopped SSH session, a device timeout)
    never reaches its own dashboard stage, so it has no page to link to and
    would show as a broken card. A superseded earlier campaign has a page but
    is not the story the landing page should tell once a newer one exists.
    """
    newest: dict[str, Run] = {}
    for run in runs:
        if not run.paths.dashboard.exists():
            continue
        current = newest.get(run.model)
        if current is None or run.created_at_utc > current.created_at_utc:
            newest[run.model] = run
    return list(newest.values())


def build_payload(runs_dir: Path, out: Path, include_mock: bool = False) -> dict[str, Any]:
    # The committed landing page lists measured runs only. A simulated card is
    # useful locally and misleading to a visitor who did not run it themselves.
    runs = [run for run in list_runs(runs_dir) if include_mock or run.is_measured]
    if not include_mock:
        runs = latest_per_model(runs)
    cards = sorted((card(run, out) for run in runs), key=sort_key)
    return {
        "built_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "runs": cards,
    }


def render(payload: dict[str, Any], template: Path) -> str:
    text = template.read_text(encoding="utf-8")
    if DATA_PLACEHOLDER not in text:
        raise SystemExit(f"{template} has no {DATA_PLACEHOLDER} to fill")
    return text.replace(DATA_PLACEHOLDER, json.dumps(payload, indent=1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--include-mock", action="store_true", help="also list simulated runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    payload = build_payload(args.runs_dir, args.out, args.include_mock)
    if not payload["runs"]:
        # Not fatal: this runs as the last stage of a campaign, and an empty tree
        # should leave a page saying so rather than fail a run that succeeded.
        print(f"No runs to list under {args.runs_dir}.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(payload, args.template), encoding="utf-8")

    for entry in payload["runs"]:
        if entry["state"] == "done":
            detail = (f"{entry['best_latency_ms']:7.2f} ms  {entry['best_top1']:6.2f}%  "
                      f"{entry['front_size']:2d} on the front")
            if entry["missing"]:
                detail += f"   (no {', '.join(entry['missing'])})"
        elif entry["state"] == "failed":
            detail = f"stopped at {entry['failed_stage']}"
        else:
            detail = f"{entry['stages_done']}/{entry['stages_total']} stages, next {entry['next_stage']}"
        print(f"  {entry['run_id']:34s} {entry['venue']:5s} {entry['state']:7s} {detail}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
