"""Adopt a finished campaign into a run directory.

The three Phase 6 campaigns predate the run layout, so their outputs are filed
by phase with the model name in the filename. This copies those JSONs into a run
without touching the originals, and records where each one came from.

    python -m src.adopt --all
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.runs import Run, RunPaths, create_run, record_stage

SEARCH_DIR = Path("artifacts/search")
REPORT_DIR = Path("artifacts/reports")
DEVICE_REPORT_DIR = Path("artifacts/reports_pi")

# Studies keep the model, budget and space version in their filename.
LEGACY_STUDIES = {
    "resnet18_cifar": "resnet18_cifar_budget0.2_v2",
    "mobilenetv2_cifar": "mobilenetv2_cifar_budget0.2_v2",
    "custom_cnn": "custom_cnn_budget0.2_v2",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def legacy_sources(model: str, study: str) -> dict[str, tuple[Path, str | None]]:
    """Stage name -> (source file, key to slice out of it if it holds every model)."""
    sentinels = sorted(DEVICE_REPORT_DIR.glob(f"sentinel_{model}_*.json"))
    sources: dict[str, tuple[Path, str | None]] = {
        "baseline": (REPORT_DIR / f"{model}_baseline.json", None),
        "quant_baselines": (DEVICE_REPORT_DIR / f"{model}_quant_baselines.json", None),
        "sensitivity": (DEVICE_REPORT_DIR / f"{model}_sensitivity.json", None),
        "cost_benefit": (DEVICE_REPORT_DIR / "cost_benefit.json", model),
        "search_space": (DEVICE_REPORT_DIR / "search_space.json", model),
        "study": (SEARCH_DIR / f"{study}.json", None),
        "final_test": (DEVICE_REPORT_DIR / f"{model}_final_test.json", None),
    }
    if sentinels:
        sources["sentinel"] = (sentinels[-1], None)
    return sources


def adopt(model: str, study: str, runs_dir: Path, force: bool = False) -> Run:
    """Copy one finished campaign's stage outputs into a run of its own."""
    summary = read_json(SEARCH_DIR / f"{study}.json")
    # Date the run when the campaign ran, not when it was adopted.
    when = datetime.fromisoformat(summary["created_at_utc"])
    venue = "mock" if summary.get("mock") else "pi"

    run = create_run(
        model=model,
        budget_pt=summary["accuracy_budget_pt"],
        trials=summary["trials"],
        venue=venue,
        study=study,
        source="adopted",
        notes="Measured before the run layout existed; stage outputs copied from artifacts/.",
        run_id=f"{model}_b{summary['accuracy_budget_pt']:g}_{when:%Y%m%d}",
        runs_dir=runs_dir,
    )

    copied: dict[str, Any] = {}
    for stage, (source, key) in legacy_sources(model, study).items():
        target = run.paths.stage(stage)
        if not source.exists():
            continue
        if target.exists() and not force:
            copied[stage] = "already present"
            continue

        document = read_json(source)
        if key is not None:
            # These two files hold every model at once; a run carries only its own.
            document = document[key]
        target.write_text(json.dumps(document, indent=2), encoding="utf-8")
        copied[stage] = {"from": source.as_posix(), "sha256": digest(source)}

    record_stage(run, "adopt", "ok", 0.0, {"sources": copied})
    return run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(LEGACY_STUDIES), default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--force", action="store_true", help="overwrite stage files already copied")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.all and args.model is None:
        raise SystemExit("Pass --model <name> or --all")

    models = sorted(LEGACY_STUDIES) if args.all else [args.model]
    for model in models:
        run = adopt(model, LEGACY_STUDIES[model], args.runs_dir, args.force)
        stages = sorted(path.stem for path in run.paths.stages_dir.glob("*.json"))
        print(f"{run.run_id:34s} venue {run.venue:5s} {len(stages)} stages: {', '.join(stages)}")


if __name__ == "__main__":
    main()
