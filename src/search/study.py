"""Multi-objective deployment search over the reduced space, measured on the Pi."""

# Minimizes {median latency, size, peak RAM} and maximizes top-1. Accuracy is both
# an objective and a hard filter: the filter rejects anything under the budget, and
# the objective ranks what survives. Accuracy as a filter alone made the front
# degenerate, because the other three improve together and nothing opposed them.
#
# The space is whatever `src.search.reduce` left searchable, so the sensitivity
# measurements decide the per-layer precision and the search spends its budget on
# what they could not settle.
#
#     python -m src.search.study --model resnet18_cifar --trials 20 --mock
#     python -m src.search.study --model resnet18_cifar --trials 40

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.bench.agent import BenchSpec
from src.bench.remote import PiConnection, RemoteBenchmarker
from src.model_index import known_models
from src.quant.baselines import BASELINE_CONFIGS
from src.quant.candidates import candidate_configs
from src.quant.config import DeploymentConfig, QuantConfig
from src.quant.groups import build_group_map, quantizable_groups
from src.quant.quantize import DEFAULT_ONNX_DIR, ModelPaths
from src.search import select
from src.search.evaluate import DeviceUnavailable, TrialResult, evaluate
from src.search.mock import MockRunner
from src.search.plan import CANONICAL_RUN, ENUMERATE, SearchPlan, load_space, make_plan
from src.sensitivity.analyze import DEVICE_REPORT_DIR
from src.search.version import DEFAULT_POPULATION_SIZE, SPACE_VERSION, study_name

DEFAULT_STUDY_DIR = Path("artifacts/search")
DEFAULT_ACCURACY_BUDGET_PT = 1.0
DEFAULT_SPACE = DEVICE_REPORT_DIR / "search_space.json"

STUDY_SCHEMA = "study/2"

# The runtime the references are measured under: the Phase 3 sweep's best thread
# count and ORT's own defaults, so they join to the Phase 4 reports.
SEED_RUN = CANONICAL_RUN

# Infeasible trials still need finite objective values: NSGA-II ranks on them,
# and non-finite values propagate into the crowding distance and poison it. The
# last entry is a top-1 of zero, which no real candidate can match from below.
PENALTY = (1.0e4, 1.0e11, 1.0e5, 0.0)
DIRECTIONS = ["minimize", "minimize", "minimize", "maximize"]


def baseline_top1(model: str, report_dir: Path = Path("artifacts/reports")) -> float:
    """The FP32 optval accuracy the constraint is measured against."""
    report = json.loads((report_dir / f"{model}_baseline.json").read_text(encoding="utf-8"))
    return float(report["metrics"]["optval_top1"])


def objectives_for(result: TrialResult) -> tuple[float, float, float, float]:
    if not result.feasible:
        return PENALTY
    return (
        float(result.latency_ms),
        float(result.size_bytes),
        float(result.peak_rss_mb),
        float(result.top1),
    )


def record_for(label: str, config: DeploymentConfig, result: TrialResult) -> dict[str, Any]:
    """One evaluated candidate, complete enough to never need reconstructing."""
    return {
        "round": label,
        "quant_hash": config.quant.hash,
        "config_hash": config.hash,
        "config": config.as_dict(),
        "describe": config.quant.describe(),
        "describe_run": config.run.describe(),
        "result": result.as_dict(),
    }


def report_line(counter: str, label: str, config: DeploymentConfig, result: TrialResult) -> str:
    numbers = (
        f"{result.latency_ms:7.3f} ms  {result.size_bytes / 1e6:6.2f} MB  "
        f"{result.peak_rss_mb:6.1f} MB  top1 {result.top1:.2f}"
        if result.feasible
        else " " * 48
    )
    return (
        f"  {counter:>7s} {label:<14s} {result.status:16s} {numbers}"
        f"  [{config.quant.describe()} | {config.run.describe()}]"
    )


def run_enumeration(runner: Any, plan: SearchPlan) -> list[dict[str, Any]]:
    """Measure every planned candidate in order, cheap gates first."""
    evaluated: list[dict[str, Any]] = []
    total = len(plan.candidates)

    for index, (label, config) in enumerate(plan.candidates, start=1):
        spec = BenchSpec(model=plan.model, config=config)
        try:
            result = evaluate(runner, spec, plan.threshold_top1)
        except DeviceUnavailable as error:
            # Nothing was learned about this candidate, so record nothing about
            # it. Writing it down as infeasible would teach the report a lie.
            print(f"  device unavailable after {len(evaluated)} candidates: {error}", flush=True)
            break
        print(report_line(f"{index}/{total}", label, config, result), flush=True)
        evaluated.append(record_for(label, config, result))

    return evaluated


def run_nsga2(runner: Any, plan: SearchPlan, name: str, study_dir: Path, seed: int) -> list[dict[str, Any]]:
    """Sample the reduced space when it is too large to enumerate.

    Unreachable for every model bundled here, whose reduced spaces are 8 to 16
    vectors. It exists for a model brought in through `src.import_onnx`.
    """
    import optuna

    from src.search.space import suggest_within

    storage = f"sqlite:///{(study_dir / f'{name}.db').as_posix()}"
    study = optuna.create_study(
        study_name=name,
        storage=storage,
        load_if_exists=True,
        directions=DIRECTIONS,
        sampler=optuna.samplers.NSGAIISampler(
            seed=seed,
            population_size=plan.population_size,
            constraints_func=lambda trial: tuple(trial.user_attrs.get("constraints", (1.0,))),
        ),
    )

    evaluated: list[dict[str, Any]] = []
    seen: set[str] = set()

    def objective(trial: optuna.Trial) -> tuple[float, float, float, float]:
        config = suggest_within(trial, plan.space, model=plan.model)
        if config.hash in seen:
            # A duplicate costs a 500-image screen plus a full pass on the
            # device and teaches the sampler nothing it does not already know.
            trial.set_user_attr("duplicate", True)
            return PENALTY
        seen.add(config.hash)

        try:
            result = evaluate(runner, BenchSpec(model=plan.model, config=config), plan.threshold_top1)
        except DeviceUnavailable as error:
            print(f"  trial {trial.number:3d} device unavailable: {error}", flush=True)
            trial.study.stop()
            raise

        trial.set_user_attr("constraints", [1.0 if result.top1 is None else plan.threshold_top1 - result.top1])
        print(report_line(f"{trial.number + 1}/{plan.budget}", "sampled", config, result), flush=True)
        evaluated.append(record_for("sampled", config, result))
        return objectives_for(result)

    study.optimize(objective, n_trials=plan.budget, catch=(Exception,))
    return evaluated


def reference_configs(model: str, space) -> dict[str, QuantConfig]:
    """The comparators the method has to beat, whether or not the space allows them.

    Under enforced pins most of these are inadmissible as candidates -- every
    global baseline violates `pinned_fp32`, and `mixed_measured` misses the
    pinned classifier on all three models. Measuring them as references rather
    than enqueueing them keeps the comparison without corrupting the front.
    """
    references: dict[str, QuantConfig] = dict(BASELINE_CONFIGS)
    references.update(candidate_configs(model))

    greedy = tuple(sorted(space.pinned_fp32 + space.searchable))
    if greedy:
        references["greedy_prefix"] = QuantConfig(
            quant_type="static", per_channel=True, excluded_groups=greedy
        )
    return {
        name: config
        for name, config in references.items()
        if not set(config.excluded_groups) - set(space.groups)
    }


def measure_references(runner: Any, model: str, space, threshold: float) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name, quant in sorted(reference_configs(model, space).items()):
        config = DeploymentConfig(quant=quant, run=SEED_RUN)
        try:
            result = evaluate(runner, BenchSpec(model=model, config=config), threshold)
        except DeviceUnavailable as error:
            print(f"  reference {name}: device unavailable ({error})", flush=True)
            break
        print(report_line("ref", name, config, result), flush=True)
        rows[name] = record_for("reference", config, result)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=known_models())
    parser.add_argument("--trials", type=int, default=None, help="max unique candidate evaluations")
    parser.add_argument("--budget-pt", type=float, default=DEFAULT_ACCURACY_BUDGET_PT)
    parser.add_argument("--max-rss-mb", type=float, default=None)
    parser.add_argument("--mock", action="store_true", help="dry run against a fake device")
    parser.add_argument("--space", type=Path, default=DEFAULT_SPACE, help="the reduction to enforce")
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR)
    parser.add_argument("--seed", type=int, default=0)
    # NSGA-II samples randomly until one full population exists and only breeds
    # after that, so at Optuna's default of 50 a 40-trial campaign is all random
    # search. Only reachable when the reduced space is too large to enumerate.
    parser.add_argument("--population-size", type=int, default=DEFAULT_POPULATION_SIZE)
    return parser.parse_args()


def groups_for(model: str, mock: bool, onnx_dir: Path) -> tuple[str, ...]:
    if mock:
        # The graphs are gitignored, so a fresh clone has none to read groups
        # from. A mock run never quantizes anything, so the recorded group list
        # is all it needs, and it is the same list the graph would produce.
        from src.model_index import entry_for

        return entry_for(model).groups
    return quantizable_groups(build_group_map(ModelPaths.resolve(model, onnx_dir).quant_ready, model))


def main() -> None:
    args = parse_args()
    args.study_dir.mkdir(parents=True, exist_ok=True)

    groups = groups_for(args.model, args.mock, args.onnx_dir)
    space = load_space(args.space, args.model, groups)
    baseline = baseline_top1(args.model)
    plan = make_plan(
        model=args.model,
        space=space,
        baseline_top1=baseline,
        accuracy_budget_pt=args.budget_pt,
        trials=args.trials,
        max_rss_mb=args.max_rss_mb,
    )

    name = study_name(args.model, args.budget_pt, args.mock, args.population_size)
    band = plan.band
    print(
        f"{name}: {len(groups)} groups -> {space.reduced_size} precision vectors "
        f"({len(space.pinned_int8)} pinned INT8, {len(space.pinned_fp32)} pinned FP32)\n"
        f"  band     {band.fraction:.2%} of {band.reference_ms:.3f} ms (from {band.source})\n"
        f"  budget   {plan.budget} evaluations, baseline {baseline:.2f}%, "
        f"threshold {plan.threshold_top1:.2f}%\n"
        f"  strategy {plan.strategy}: {plan.strategy_reason}\n"
    )

    if args.mock:
        runner = MockRunner(groups)
    else:
        runner = RemoteBenchmarker(PiConnection.load())
        # Unconditional: a stale device checkout fails every trial identically at
        # the first gate, which reads as a bad search space rather than a missing
        # file. An scp of src/ is cheaper than one wasted trial.
        runner.push_code()

    print("references (measured outside the space, as comparators):")
    references = measure_references(runner, args.model, space, plan.threshold_top1)

    print(f"\ncandidates ({plan.strategy}):")
    if plan.strategy == ENUMERATE:
        evaluated = run_enumeration(runner, plan)
    else:
        evaluated = run_nsga2(runner, plan, name, args.study_dir, args.seed)

    front = select.non_dominated(select.feasible(evaluated, plan.max_rss_mb))
    chosen = select.selections(evaluated, band.ms if band else None, plan.max_rss_mb)

    print(f"\nPareto set: {len(front)} of {len(evaluated)} evaluated")
    for row in sorted(front, key=lambda row: row["result"]["latency_ms"]):
        result = row["result"]
        print(
            f"  {result['latency_ms']:7.3f} ms  {result['size_bytes'] / 1e6:6.2f} MB  "
            f"{result['peak_rss_mb']:6.1f} MB RSS  top1 {result['top1']:.2f}  "
            f"[{row['describe']} | {row['describe_run']}]"
        )
    for choice, row in chosen.items():
        if row is not None:
            print(f"  {choice:16s} {row['describe']} | {row['describe_run']}")

    summary = {
        "schema": STUDY_SCHEMA,
        "study": name,
        "space_version": SPACE_VERSION,
        "model": args.model,
        "mock": args.mock,
        "baseline_top1": baseline,
        "accuracy_budget_pt": args.budget_pt,
        "threshold_top1": plan.threshold_top1,
        "groups": list(groups),
        "search_plan": plan.as_dict(),
        "strategy": plan.strategy,
        "band": band.as_dict() if band else None,
        "trials": len(evaluated),
        "references": references,
        "evaluated_configs": evaluated,
        "pareto": front,
        "selections": {
            choice: None if row is None else row["config_hash"] for choice, row in chosen.items()
        },
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    report = args.study_dir / f"{name}.json"
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
