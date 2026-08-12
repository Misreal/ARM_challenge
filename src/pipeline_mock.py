"""A simulated campaign, so the whole app runs with no Raspberry Pi attached.

Every number here is invented. It exists to prove the plumbing end to end and to
let someone without the hardware see the app work; the run is stamped `mock` and
the page says so in a banner.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.bench.agent import BenchSpec
from src.model_index import entry_for
from src.pipeline import Stage, execute, index_out, local_command
from src.quant.baselines import BASELINE_CONFIGS
from src.quant.config import DeploymentConfig, QuantConfig, RunConfig
from src.runs import Run
from src.search.mock import MockRunner

REPORT_DIR = Path("artifacts/reports")
SEARCH_DIR = Path("artifacts/search")

REFERENCE_RUN = RunConfig(intra_op_num_threads=4, graph_optimization_level="all")
MOCK_ONNXRUNTIME = "1.27.0 (simulated)"
MOCK_SPLIT = "optval"
MOCK_SAMPLES = 3000


def write(path: Path, document: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return {"wrote": path.as_posix()}


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def spec_for(model: str, quant: QuantConfig) -> BenchSpec:
    return BenchSpec(model=model, config=DeploymentConfig(quant=quant, run=REFERENCE_RUN))


def group_weight(group: str) -> float:
    """A stable pretend fragility for one group, so the ranking is not flat.

    The real analyzer measures this. A simulator has nothing to measure, and a
    ranking where every group scores identically would hide bugs in everything
    downstream that consumes an ordering.
    """
    digest = hashlib.sha256(group.encode()).digest()
    return 0.05 + (digest[0] / 255) * 0.95


# --------------------------------------------------------------------- stages


def mock_baseline(run: Run) -> dict[str, Any]:
    """Adopt the real PC-side baseline report; nothing about it needs a device."""
    source = REPORT_DIR / f"{run.model}_baseline.json"
    if not source.exists():
        raise RuntimeError(f"No {source}. Export the model first, or import an ONNX.")
    return write(run.paths.stage("baseline"), json.loads(source.read_text(encoding="utf-8")))


def mock_quant_baselines(run: Run, runner: MockRunner) -> dict[str, Any]:
    baselines: dict[str, Any] = {}
    for name, quant in BASELINE_CONFIGS.items():
        spec = spec_for(run.model, quant)
        scored = runner.score(spec, limit=MOCK_SAMPLES)
        measured = runner.measure(spec)
        baselines[name] = {
            "status": "ok",
            "config": quant.as_dict(),
            "quant_hash": quant.hash,
            "bytes": scored["bytes"],
            "accuracy": scored["accuracy"],
            "latency_ms": measured["latency"]["median_ms"],
            "peak_rss_mb": measured["peak_rss_mb"],
        }
    return write(
        run.paths.stage("quant_baselines"),
        {
            "model": run.model,
            "eval_split": MOCK_SPLIT,
            "eval_limit": None,
            "on_target": False,
            "simulated": True,
            "host": {"platform": f"{platform.system()} (simulator)"},
            "run_config": REFERENCE_RUN.as_dict(),
            "baselines": baselines,
            "onnxruntime": MOCK_ONNXRUNTIME,
            "created_at_utc": now(),
        },
    )


def mock_sensitivity(run: Run, runner: MockRunner, groups: tuple[str, ...]) -> dict[str, Any]:
    baseline = json.loads(run.paths.stage("baseline").read_text(encoding="utf-8"))
    weights = {group: group_weight(group) for group in groups}
    total = sum(weights.values())

    ranking: dict[str, list[dict[str, Any]]] = {}
    for scheme in ("per_channel", "per_tensor"):
        # Per-tensor damage is larger, which is the one qualitative fact the
        # simulator does carry over from the measured campaigns.
        scale = 1.0 if scheme == "per_channel" else 2.4
        rows = [
            {
                "group": group,
                "recovery_share": round(weight / total, 6),
                "recovery_kl": round(weight * scale * 1e-3, 8),
                "isolate_kl": round(weight * scale * 1.2e-3, 8),
                "leave_one_out_kl": round(weight * scale * 1.1e-3, 8),
                "recovery_top1": round(weight * scale * 0.3, 4),
                "isolate_flip_rate": round(weight * scale, 4),
            }
            for group, weight in weights.items()
        ]
        ranking[scheme] = sorted(rows, key=lambda row: -row["recovery_share"])

    return write(
        run.paths.stage("sensitivity"),
        {
            "model": run.model,
            "eval_split": MOCK_SPLIT,
            "samples": MOCK_SAMPLES,
            "split_fingerprint": baseline["split_fingerprint"],
            "on_target": False,
            "simulated": True,
            "group_sizes": {group: 1 for group in groups},
            "onnxruntime": MOCK_ONNXRUNTIME,
            "created_at_utc": now(),
            "ranking": ranking,
        },
    )


def mock_group_cost(run: Run, runner: MockRunner, groups: tuple[str, ...]) -> dict[str, Any]:
    """What excluding each single group costs in latency, one measurement each."""
    reference = runner.measure(spec_for(run.model, BASELINE_CONFIGS["static_per_channel"]))
    base_ms = reference["latency"]["median_ms"]

    costs = {}
    for group in groups:
        quant = replace(BASELINE_CONFIGS["static_per_channel"], excluded_groups=(group,))
        measured = runner.measure(spec_for(run.model, quant))
        costs[group] = {
            "latency_ms": measured["latency"]["median_ms"],
            "cost_ms": round(measured["latency"]["median_ms"] - base_ms, 4),
        }
    return write(
        run.paths.stage("group_cost"),
        {"model": run.model, "reference_ms": base_ms, "simulated": True, "groups": costs},
    )


def mock_cost_benefit(run: Run) -> dict[str, Any]:
    """Pure join: each group's share of the damage against what it costs to keep FP32."""
    sensitivity = json.loads(run.paths.stage("sensitivity").read_text(encoding="utf-8"))
    costs = json.loads(run.paths.stage("group_cost").read_text(encoding="utf-8"))["groups"]

    rows = []
    for row in sensitivity["ranking"]["per_channel"]:
        group = row["group"]
        cost_ms = costs[group]["cost_ms"]
        rows.append(
            {
                "group": group,
                "recovery_share": row["recovery_share"],
                "latency_ms": costs[group]["latency_ms"],
                "cost_ms": cost_ms,
                # A free exclusion has no price to divide by, so it ranks above
                # anything that costs something rather than dividing by zero.
                "share_per_ms": round(row["recovery_share"] / cost_ms, 6) if cost_ms > 0 else 999.0,
            }
        )
    rows.sort(key=lambda row: -row["share_per_ms"])
    return write(run.paths.stage("cost_benefit"), rows)


def mock_search_space(run: Run) -> dict[str, Any]:
    rows = json.loads(run.paths.stage("cost_benefit").read_text(encoding="utf-8"))
    # Anything worth more than its price stays searchable; the rest is pinned.
    keep = [row["group"] for row in rows if row["share_per_ms"] >= 1.0]
    return write(
        run.paths.stage("search_space"),
        {
            "model": run.model,
            "simulated": True,
            "searchable": keep,
            "pinned_int8": [row["group"] for row in rows if row["group"] not in keep],
        },
    )


def mock_study(run: Run) -> dict[str, Any]:
    """The real optuna search, against the fake device it already supports."""
    argv = [
        sys.executable, "-m", "src.search.study",
        "--model", run.model,
        "--trials", str(run.trials),
        "--budget-pt", str(run.budget_pt),
        "--mock",
    ]
    execute(argv)

    produced = SEARCH_DIR / f"{run.study}.json"
    if not produced.exists():
        raise RuntimeError(f"the study did not write {produced}")
    return write(run.paths.stage("study"), json.loads(produced.read_text(encoding="utf-8")))


def mock_final_test(run: Run, runner: MockRunner) -> dict[str, Any]:
    """Score the front once, on a pretend sealed split."""
    summary = json.loads(run.paths.stage("study").read_text(encoding="utf-8"))

    results = []
    for member in summary["pareto"]:
        if member["result"]["status"] != "ok":
            continue
        # A held-out split scores a little below the split the search steered on.
        optval = member["result"]["top1"]
        results.append(
            {
                "label": f"{member['describe']} | {member['describe_run']}",
                "status": "ok",
                "quant_hash": member["quant_hash"],
                "test": {"samples": 10_000, "top1": round(optval - 0.51, 2)},
            }
        )
    results.append(
        {
            "label": "fp32 (reference)",
            "status": "ok",
            "test": {"samples": 10_000, "top1": round(summary["baseline_top1"] - 0.51, 2)},
        }
    )
    return write(
        run.paths.stage("final_test"),
        {
            "schema": "final-test/1",
            "model": run.model,
            "study": run.study,
            "eval_split": "test",
            "on_target": False,
            "simulated": True,
            "results": results,
            "created_at_utc": now(),
        },
    )


# ------------------------------------------------------------------ executor


def make_executor(run: Run):
    """One callable the orchestrator drives, holding the fake device between stages."""
    groups = entry_for(run.model).groups
    runner = MockRunner(groups)

    def executor(stage: Stage, run: Run) -> dict[str, Any] | None:
        name = stage.name
        if name == "baseline":
            return mock_baseline(run)
        if name == "quant_baselines":
            return mock_quant_baselines(run, runner)
        if name == "sensitivity":
            return mock_sensitivity(run, runner, groups)
        if name == "group_cost":
            return mock_group_cost(run, runner, groups)
        if name == "cost_benefit":
            return mock_cost_benefit(run)
        if name == "search_space":
            return mock_search_space(run)
        if name == "study":
            return mock_study(run)
        if name == "sentinel":
            # A deterministic simulator repeats itself exactly, so there is no
            # thermal drift to measure. The page falls back to printed-digit ties.
            return None
        if name == "final_test":
            return mock_final_test(run, runner)
        if name == "dashboard":
            execute(local_command("dashboard", run))
            return {"page": run.paths.dashboard.as_posix()}
        if name == "index":
            execute(local_command("index", run))
            return {"index": index_out(run.paths.root.parent).as_posix()}
        raise RuntimeError(f"no simulator for stage {name!r}")

    return executor
