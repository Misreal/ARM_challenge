# A simulated campaign, so the whole app runs with no Raspberry Pi attached.
#
# Every number here is invented. It exists to prove the plumbing end to end and to
# let someone without the hardware see the app work; the run is stamped `mock` and
# the page says so in a banner.

from __future__ import annotations

import hashlib
import json
import platform
import statistics
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
from src.search.finalists import run_finalists
from src.search.mock import MockRunner
from src.search.reduce import band_for, greedy_order, reduce_space

REPORT_DIR = Path("artifacts/reports")
SEARCH_DIR = Path("artifacts/search")

# Matches the device stage, so the mock exercises the same repeat count.
SENTINEL_REPEATS = 8

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


# Measured per-group sensitivity is concentrated: the top group owns 21-49% of
# the total recovery across the three models. A uniform draw would leave every
# group worth searching, so the reduction would never reduce and the mock would
# never exercise the path a real campaign takes.
SENSITIVITY_CONCENTRATION = 3


def group_weight(group: str) -> float:
    """A stable pretend fragility for one group, so the ranking is not flat.

    The real analyzer measures this. A simulator has nothing to measure, and a
    ranking where every group scores identically would hide bugs in everything
    downstream that consumes an ordering.
    """
    digest = hashlib.sha256(group.encode()).digest()
    return 0.02 + (digest[0] / 255) ** SENSITIVITY_CONCENTRATION * 0.98


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


def mock_search_space(run: Run, groups: tuple[str, ...]) -> dict[str, Any]:
    """The real reduction, run on simulated costs.

    Reusing `reduce` rather than inventing a cutoff is the point: the mock has to
    produce a partition the study will actually accept, so the acceptance run
    exercises the validation instead of routing around it.
    """
    rows = json.loads(run.paths.stage("cost_benefit").read_text(encoding="utf-8"))
    sentinel = run.paths.stage("sentinel")
    band = band_for(rows, json.loads(sentinel.read_text(encoding="utf-8")) if sentinel.exists() else None)
    space = reduce_space(run.model, rows, band, groups)
    space.validate(run.model, groups)
    return write(
        run.paths.stage("search_space"),
        {
            **space.as_dict(),
            "simulated": True,
            "greedy_order": [row["group"] for row in greedy_order(rows, band.ms)],
        },
    )


def mock_study(run: Run) -> dict[str, Any]:
    """The real optuna search, against the fake device it already supports."""
    argv = [
        sys.executable, "-m", "src.search.study",
        "--model", run.model,
        "--budget-pt", str(run.budget_pt),
        "--population-size", str(run.population_size),
        # This run's own reduction, not the shared artifact: two runs of one
        # model may have been cut against different bands.
        "--space", str(run.paths.stage("search_space")),
        "--mock",
    ]
    if run.trials is not None:
        argv += ["--trials", str(run.trials)]
    if run.max_rss_mb is not None:
        argv += ["--max-rss-mb", str(run.max_rss_mb)]
    execute(argv)

    produced = SEARCH_DIR / f"{run.study}.json"
    if not produced.exists():
        raise RuntimeError(f"the study did not write {produced}")
    return write(run.paths.stage("study"), json.loads(produced.read_text(encoding="utf-8")))


def mock_sentinel(run: Run, runner: MockRunner) -> dict[str, Any]:
    """Repeat one config with the cache off, exactly as the device stage does.

    The simulator jitters per repeat, so this produces a real spread fraction and
    the reduction downstream is banded on a measured number rather than the
    fallback default.
    """
    spec = spec_for(run.model, BASELINE_CONFIGS["static_per_channel"])
    results = [runner.measure(spec, use_cache=False) for _ in range(SENTINEL_REPEATS)]
    medians = [result["latency"]["median_ms"] for result in results]
    middle = statistics.median(medians)
    spread = (max(medians) - min(medians)) / middle

    return write(
        run.paths.stage("sentinel"),
        {
            "schema": "sentinel/1",
            "model": run.model,
            "simulated": True,
            "config": spec.config.as_dict(),
            "trials": len(results),
            "admissible_trials": sum(1 for result in results if result["admissible"]),
            "medians_ms": medians,
            "median_of_medians_ms": round(middle, 4),
            "min_ms": min(medians),
            "max_ms": max(medians),
            "spread_fraction": round(spread, 5),
            "spread_percent": round(spread * 100, 2),
            "results": results,
            "created_at_utc": now(),
        },
    )


def mock_finalists(run: Run, runner: MockRunner) -> dict[str, Any]:
    summary = json.loads(run.paths.stage("study").read_text(encoding="utf-8"))
    report = run_finalists(runner, run.model, summary)
    return write(run.paths.stage("finalists"), {**report, "simulated": True})


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
            "schema": "final_test/1",
            "model": run.model,
            "study": run.study,
            "eval_split": "test",
            "on_target": False,
            "simulated": True,
            "results": results,
            "created_at_utc": now(),
        },
    )


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
            return mock_search_space(run, groups)
        if name == "study":
            return mock_study(run)
        if name == "sentinel":
            return mock_sentinel(run, runner)
        if name == "finalists":
            return mock_finalists(run, runner)
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
