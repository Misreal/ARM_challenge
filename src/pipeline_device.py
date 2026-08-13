"""The same campaign, measured on the Raspberry Pi.

Three of these stages run *on* the device over SSH because their reports are
only meaningful there: the x86 host's INT8 kernels saturate (see CLAUDE.md), so
a quantization accuracy number measured here would be wrong in a direction that
targets exactly the candidates most likely to win.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

from src.bench.remote import PiConnection, RemoteBenchmarker
from src.pipeline import Stage, execute, index_out, local_command
from src.runs import Run

REPORT_DIR = Path("artifacts/reports")
DEVICE_REPORT_DIR = Path("artifacts/reports_pi")
SEARCH_DIR = Path("artifacts/search")

SENTINEL_REPEATS = 8
SENTINEL_SPACING_S = 240.0


def adopt_file(source: Path, run: Run, stage: str, key: str | None = None) -> dict[str, Any]:
    """Copy a stage's report into the run, slicing it if it holds every model."""
    if not source.exists():
        raise RuntimeError(f"the stage did not write {source}")
    document = json.loads(source.read_text(encoding="utf-8"))
    if key is not None:
        document = document[key]
    target = run.paths.stage(stage)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return {"from": source.as_posix()}


def on_device(connection: PiConnection, module: str, arguments: str, produces: str, local: Path) -> None:
    """Run one module on the Pi and bring its report back.

    The device-side reports land in artifacts/reports_pi *on the Pi*, which is
    why they have to be fetched rather than just written.
    """
    from src.bench.remote import SshTransport

    transport = SshTransport(connection)
    command = f"cd {connection.remote_root} && python3 -m {module} {arguments}"
    result = transport.run(command, timeout=7200)
    if not result.ok:
        raise RuntimeError(f"{module} on the device: {result.stderr.strip()[:400]}")

    local.parent.mkdir(parents=True, exist_ok=True)
    fetched = transport.fetch(f"{connection.remote_root}/{produces}", local)
    if not fetched.ok:
        raise RuntimeError(f"could not fetch {produces}: {fetched.stderr.strip()[:400]}")


def make_executor(run: Run):
    """One callable the orchestrator drives, holding the SSH connection open."""
    connection = PiConnection.load()
    benchmarker = RemoteBenchmarker(connection)
    # A stale checkout fails every stage identically at its first gate, which
    # reads as a broken pipeline rather than a missing file.
    benchmarker.push_code()

    model = run.model

    def executor(stage: Stage, run: Run) -> dict[str, Any] | None:
        name = stage.name

        if name == "baseline":
            execute(local_command("baseline", run))
            return adopt_file(REPORT_DIR / f"{model}_baseline.json", run, "baseline")

        if name == "quant_baselines":
            local = DEVICE_REPORT_DIR / f"{model}_quant_baselines.json"
            on_device(connection, "src.quant.baselines", f"--model {model}",
                      f"artifacts/reports_pi/{model}_quant_baselines.json", local)
            return adopt_file(local, run, "quant_baselines")

        if name == "sensitivity":
            local = DEVICE_REPORT_DIR / f"{model}_sensitivity.json"
            on_device(connection, "src.sensitivity.analyze", f"--model {model}",
                      f"artifacts/reports_pi/{model}_sensitivity.json", local)
            return adopt_file(local, run, "sensitivity")

        if name == "group_cost":
            # Driven from the PC: one measurement per single-group exclusion,
            # each cached by config hash so a resumed run pays for none of them.
            execute([sys.executable, "-m", "src.bench.remote", "--model", model,
                     "--exclude-each", "--threads", "4"])
            return {"measured": "per-group exclusions, cached under artifacts/bench_cache"}

        if name == "cost_benefit":
            execute(local_command("cost_benefit", run))
            return adopt_file(DEVICE_REPORT_DIR / "cost_benefit.json", run, "cost_benefit", key=model)

        if name == "search_space":
            execute(local_command("search_space", run))
            return adopt_file(DEVICE_REPORT_DIR / "search_space.json", run, "search_space", key=model)

        if name == "study":
            argv = [sys.executable, "-m", "src.search.study", "--model", model,
                    "--budget-pt", str(run.budget_pt),
                    "--population-size", str(run.population_size),
                    # This run's own reduction, not the shared artifact: two runs
                    # of one model may have been cut against different bands.
                    "--space", str(run.paths.stage("search_space"))]
            if run.trials is not None:
                argv += ["--trials", str(run.trials)]
            if run.max_rss_mb is not None:
                argv += ["--max-rss-mb", str(run.max_rss_mb)]
            execute(argv)
            return adopt_file(SEARCH_DIR / f"{run.study}.json", run, "study")

        if name == "sentinel":
            report = DEVICE_REPORT_DIR / f"sentinel_{model}_{run.created_at_utc[:10].replace('-', '')}.json"
            execute([sys.executable, "-m", "src.bench.remote", "--model", model,
                     "--configs", "static_per_channel", "--threads", "4",
                     "--repeat", str(SENTINEL_REPEATS), "--spacing-s", str(SENTINEL_SPACING_S),
                     "--no-cache", "--report", str(report)])
            return adopt_file(report, run, "sentinel")

        if name == "finalists":
            execute([sys.executable, "-m", "src.search.finalists", "--model", model,
                     "--study", run.study, "--budget-pt", str(run.budget_pt)])
            return adopt_file(DEVICE_REPORT_DIR / f"{model}_finalists.json", run, "finalists")

        if name == "final_test":
            # --confirm is the module's own guard against an accidental second
            # look at the sealed split. The pipeline reaching this stage is that
            # confirmation: every front member is scored in one pass.
            execute([sys.executable, "-m", "src.search.final_test", "--model", model,
                     "--study", run.study, "--budget-pt", str(run.budget_pt), "--confirm"])
            return adopt_file(DEVICE_REPORT_DIR / f"{model}_final_test.json", run, "final_test")

        if name == "dashboard":
            execute(local_command("dashboard", run))
            return {"page": run.paths.dashboard.as_posix()}

        if name == "index":
            execute(local_command("index", run))
            return {"index": index_out(run.paths.root.parent).as_posix()}

        raise RuntimeError(f"no device runner for stage {name!r}")

    return executor
