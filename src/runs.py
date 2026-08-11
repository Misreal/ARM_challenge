"""Where one campaign's artifacts live.

A run is a directory: the recipe that was asked for, every stage's output, its
own dashboard, and a manifest of what actually happened. Nothing else in the
tree composes a run's paths by hand.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RUNS_DIR = Path("runs")
RUN_SCHEMA = "run/1"
MANIFEST_SCHEMA = "manifest/1"

# Simulated device or the real Pi. The dashboard prints this, so a mock run can
# never be mistaken for a measured one.
VENUES = ("pi", "mock")


def run_id_for(model: str, budget_pt: float, when: datetime | None = None) -> str:
    """`<model>_b<budget>_<YYYYMMDD>`, which sorts sensibly and reads plainly."""
    stamp = (when or datetime.now(UTC)).strftime("%Y%m%d")
    return f"{model}_b{budget_pt:g}_{stamp}"


@dataclass(frozen=True)
class RunPaths:
    """Every path in one run, derived from its id."""

    run_id: str
    root: Path

    @classmethod
    def under(cls, run_id: str, runs_dir: Path = RUNS_DIR) -> RunPaths:
        return cls(run_id=run_id, root=runs_dir / run_id)

    @property
    def config(self) -> Path:
        return self.root / "run.json"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def stages_dir(self) -> Path:
        return self.root / "stages"

    @property
    def dashboard(self) -> Path:
        return self.root / "dashboard" / "index.html"

    def stage(self, name: str) -> Path:
        return self.stages_dir / f"{name}.json"

    def exists(self) -> bool:
        return self.config.exists()


@dataclass(frozen=True)
class Run:
    """The recipe a run was created from, plus where it lives."""

    paths: RunPaths
    model: str
    budget_pt: float
    trials: int
    venue: str
    study: str
    source: str
    created_at_utc: str
    notes: str = ""

    @property
    def run_id(self) -> str:
        return self.paths.run_id

    @property
    def is_measured(self) -> bool:
        return self.venue == "pi"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": RUN_SCHEMA,
            "run_id": self.run_id,
            "model": self.model,
            "budget_pt": self.budget_pt,
            "trials": self.trials,
            "venue": self.venue,
            "study": self.study,
            "source": self.source,
            "created_at_utc": self.created_at_utc,
            "notes": self.notes,
        }


def create_run(
    model: str,
    budget_pt: float,
    trials: int,
    venue: str,
    study: str,
    source: str = "pipeline",
    notes: str = "",
    run_id: str | None = None,
    runs_dir: Path = RUNS_DIR,
) -> Run:
    """Make the directory and write the recipe. Re-creating an existing run reuses it."""
    if venue not in VENUES:
        raise ValueError(f"venue must be one of {VENUES}, got {venue!r}")

    paths = RunPaths.under(run_id or run_id_for(model, budget_pt), runs_dir)
    if paths.exists():
        existing = load_run(paths.run_id, runs_dir)
        # Resuming a mock run into a device run, or vice versa, would mix
        # simulated and measured numbers inside one directory.
        if existing.venue != venue:
            raise SystemExit(
                f"{paths.run_id} already exists with venue {existing.venue!r}, not {venue!r}. "
                "Use a different run id."
            )
        return existing

    run = Run(
        paths=paths,
        model=model,
        budget_pt=budget_pt,
        trials=trials,
        venue=venue,
        study=study,
        source=source,
        created_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        notes=notes,
    )
    paths.stages_dir.mkdir(parents=True, exist_ok=True)
    paths.dashboard.parent.mkdir(parents=True, exist_ok=True)
    paths.config.write_text(json.dumps(run.as_dict(), indent=2), encoding="utf-8")
    return run


def load_run(run_id: str, runs_dir: Path = RUNS_DIR) -> Run:
    paths = RunPaths.under(run_id, runs_dir)
    if not paths.exists():
        raise FileNotFoundError(f"No run at {paths.root}. List them with: python -m src.app list")

    document = json.loads(paths.config.read_text(encoding="utf-8"))
    if document.get("schema") != RUN_SCHEMA:
        raise SystemExit(f"{paths.config} is schema {document.get('schema')!r}, expected {RUN_SCHEMA}")

    return Run(
        paths=paths,
        model=document["model"],
        budget_pt=document["budget_pt"],
        trials=document["trials"],
        venue=document["venue"],
        study=document["study"],
        source=document["source"],
        created_at_utc=document["created_at_utc"],
        notes=document.get("notes", ""),
    )


def list_runs(runs_dir: Path = RUNS_DIR) -> list[Run]:
    if not runs_dir.exists():
        return []
    runs = [load_run(path.name, runs_dir) for path in sorted(runs_dir.iterdir()) if (path / "run.json").exists()]
    return sorted(runs, key=lambda run: (run.model, run.created_at_utc))


# ------------------------------------------------------------------ manifest


def read_manifest(run: Run) -> dict[str, Any]:
    if not run.paths.manifest.exists():
        return {"schema": MANIFEST_SCHEMA, "stages": {}}
    return json.loads(run.paths.manifest.read_text(encoding="utf-8"))


def record_stage(run: Run, name: str, status: str, seconds: float, detail: dict[str, Any] | None = None) -> None:
    """Append one stage's outcome, so an interrupted run knows where it stopped."""
    manifest = read_manifest(run)
    manifest["schema"] = MANIFEST_SCHEMA
    manifest.setdefault("stages", {})[name] = {
        "status": status,
        "seconds": round(seconds, 2),
        "completed_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **(detail or {}),
    }
    manifest["host"] = {"platform": platform.platform(), "machine": platform.machine()}
    manifest["updated_at_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    run.paths.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def completed_stages(run: Run) -> set[str]:
    return {
        name for name, entry in read_manifest(run).get("stages", {}).items() if entry.get("status") == "ok"
    }
