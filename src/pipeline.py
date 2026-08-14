"""The stages of one campaign, declared as data so a run can be resumed.

A stage knows what it needs, what it produces and whether it touches the device.
The orchestrator skips whatever a run has already completed, which is what makes
an interrupted overnight campaign restartable rather than repeatable.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.runs import RUNS_DIR, Run, completed_stages, record_stage

DEFAULT_INDEX = Path("artifacts/dashboard/index.html")


@dataclass(frozen=True)
class Stage:
    """One step of the campaign."""

    name: str
    needs: tuple[str, ...]
    summary: str
    # Stages that need a Pi. With --mock these are simulated instead, and the
    # run is stamped so the page can never present the result as measured.
    device: bool = False


STAGES: tuple[Stage, ...] = (
    Stage("baseline", (), "validate the graph and freeze its FP32 accuracy"),
    Stage("quant_baselines", ("baseline",), "fp32, dynamic and both static INT8 recipes", device=True),
    Stage("sensitivity", ("baseline",), "per-group INT8 damage probes", device=True),
    Stage("group_cost", ("sensitivity",), "latency price of excluding each group", device=True),
    Stage("cost_benefit", ("sensitivity", "group_cost"), "join each group's benefit to its price"),
    # Before the cut, not after it. The band this measures is what decides which
    # group costs are real, and it never read the study in the first place.
    Stage("sentinel", ("quant_baselines",), "repeat one config to measure the noise floor", device=True),
    Stage("search_space", ("cost_benefit",), "shrink the space and take a greedy reference"),
    Stage("study", ("quant_baselines", "search_space"), "the multi-objective campaign", device=True),
    Stage("finalists", ("study",), "re-measure the shortlist and name the choices", device=True),
    Stage("final_test", ("study",), "score the front once on the sealed test set", device=True),
    Stage("dashboard", ("study",), "render this run's page"),
    Stage("index", ("dashboard",), "refresh the landing page that lists every run"),
)

STAGE_NAMES = tuple(stage.name for stage in STAGES)
BY_NAME = {stage.name: stage for stage in STAGES}


def resolve_order(only: tuple[str, ...] | None) -> list[Stage]:
    """The stages to attempt, in declaration order."""
    if not only:
        return list(STAGES)
    unknown = set(only) - set(STAGE_NAMES)
    if unknown:
        raise SystemExit(f"Unknown stage(s): {', '.join(sorted(unknown))}. Known: {', '.join(STAGE_NAMES)}")
    return [stage for stage in STAGES if stage.name in only]


def produced_path(stage: Stage, run: Run) -> Path | None:
    """Where this stage's output lands, or None when it writes outside the run."""
    if stage.name == "dashboard":
        return run.paths.dashboard
    if stage.name == "index":
        # The landing page belongs to the whole tree, so there is nothing here to
        # skip on: every finished run refreshes it.
        return None
    return run.paths.stage(stage.name)


def index_out(runs_dir: Path) -> Path:
    """The landing page for this tree. A scratch tree gets its own, not the committed one."""
    if runs_dir.resolve() == RUNS_DIR.resolve():
        return DEFAULT_INDEX
    return runs_dir.parent / "dashboard" / "index.html"


def missing_needs(stage: Stage, run: Run) -> list[str]:
    """Prerequisites whose output is not on disk yet."""
    blocked = []
    for need in stage.needs:
        path = produced_path(BY_NAME[need], run)
        if path is not None and not path.exists():
            blocked.append(need)
    return blocked


def local_command(stage: str, run: Run) -> list[str] | None:
    """The subprocess that produces this stage on the PC, if it is a local one."""
    model = run.model
    if stage == "baseline":
        return [sys.executable, "-m", "src.export_onnx", "--model", model, "--device", "cpu"]
    if stage == "cost_benefit":
        return [sys.executable, "-m", "src.sensitivity.cost_benefit", "--model", model]
    if stage == "search_space":
        argv = [sys.executable, "-m", "src.search.reduce", "--model", model]
        # Absent on a simulator and on any run whose sentinel did not apply, and
        # `reduce` then falls back to the documented default band and says so.
        sentinel = run.paths.stage("sentinel")
        if sentinel.exists():
            argv += ["--sentinel", str(sentinel)]
        return argv
    if stage == "dashboard":
        # The runs directory travels with the run: a campaign built under a
        # scratch tree must not write its page into the committed one.
        return [sys.executable, "-m", "scripts.build_dashboard", "--run", run.run_id,
                "--runs-dir", str(run.paths.root.parent)]
    if stage == "index":
        runs_dir = run.paths.root.parent
        out = index_out(runs_dir)
        argv = [sys.executable, "-m", "scripts.build_index",
                "--runs-dir", str(runs_dir), "--out", str(out)]
        # A scratch tree is where simulated runs are tested, so list them there.
        # The committed page stays measured-only whoever triggers the rebuild.
        if out != DEFAULT_INDEX:
            argv.append("--include-mock")
        return argv
    return None


def execute(argv: list[str], echo: bool = True) -> None:
    if echo:
        print(f"    $ {' '.join(argv[2:] if argv[:2] == [sys.executable, '-m'] else argv)}")
    completed = subprocess.run(argv, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"exited {completed.returncode}")


def run_pipeline(
    run: Run,
    executor: Callable[[Stage, Run], dict[str, Any]],
    only: tuple[str, ...] | None = None,
    force: bool = False,
) -> dict[str, str]:
    """Walk the stages, skipping what is done. Returns each stage's outcome."""
    done = completed_stages(run)
    outcomes: dict[str, str] = {}

    for stage in resolve_order(only):
        produced = produced_path(stage, run)
        if not force and stage.name in done and produced is not None and produced.exists():
            print(f"  {stage.name:16s} skip      already done")
            outcomes[stage.name] = "skipped"
            continue

        blocked = missing_needs(stage, run)
        if blocked:
            print(f"  {stage.name:16s} blocked   needs {', '.join(blocked)}")
            outcomes[stage.name] = "blocked"
            continue

        print(f"  {stage.name:16s} running   {stage.summary}")
        started = time.perf_counter()
        try:
            detail = executor(stage, run)
        except Exception as error:  # a failed stage stops this run, not the tool
            seconds = time.perf_counter() - started
            record_stage(run, stage.name, "failed", seconds, {"error": str(error)})
            print(f"  {stage.name:16s} FAILED    {error}")
            outcomes[stage.name] = "failed"
            break

        seconds = time.perf_counter() - started
        if detail is None:
            # The stage does not apply here, e.g. a noise floor on a simulator
            # that repeats itself exactly. Not a failure, and not a result.
            record_stage(run, stage.name, "n/a", seconds)
            print(f"  {stage.name:16s} n/a       nothing to measure in this venue")
            outcomes[stage.name] = "n/a"
            continue

        record_stage(run, stage.name, "ok", seconds, detail)
        print(f"  {stage.name:16s} done      {seconds:.1f}s")
        outcomes[stage.name] = "ok"

    return outcomes


def summarize(outcomes: dict[str, str]) -> str:
    counts: dict[str, int] = {}
    for outcome in outcomes.values():
        counts[outcome] = counts.get(outcome, 0) + 1
    return ", ".join(f"{count} {name}" for name, count in sorted(counts.items()))
