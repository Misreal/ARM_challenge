"""Multi-objective deployment search: NSGA-II over configs, measured on the Pi.

Minimizes {median latency, size, peak RAM} and maximizes top-1. Accuracy is both
an objective and a hard filter: the filter rejects anything under the budget, and
the objective ranks what survives. Accuracy as a filter alone made the front
degenerate, because the other three improve together and nothing opposed them.

    python -m src.search.study --model resnet18_cifar --trials 20 --mock
    python -m src.search.study --model resnet18_cifar --trials 30
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import optuna

from src.bench.agent import BenchSpec
from src.bench.remote import PiConnection, RemoteBenchmarker
from src.quant.baselines import BASELINE_CONFIGS
from src.quant.candidates import candidate_configs
from src.quant.config import EXPORTED_MODELS, QuantConfig, RunConfig
from src.quant.groups import build_group_map, quantizable_groups
from src.quant.quantize import DEFAULT_ONNX_DIR, ModelPaths
from src.search.evaluate import DeviceUnavailable, TrialResult, evaluate
from src.search.mock import MockRunner
from src.search.space import suggest_config

DEFAULT_STUDY_DIR = Path("artifacts/search")
DEFAULT_ACCURACY_BUDGET_PT = 1.0

# The runtime the seeded baselines are measured under: the Phase 3 sweep's best
# thread count and ORT's own defaults, so they join to the Phase 4 reports.
SEED_RUN = RunConfig(intra_op_num_threads=4, graph_optimization_level="all")

# Infeasible trials still need finite objective values: NSGA-II ranks on them,
# and non-finite values propagate into the crowding distance and poison it. The
# last entry is a top-1 of zero, which no real candidate can match from below.
PENALTY = (1.0e4, 1.0e11, 1.0e5, 0.0)
DIRECTIONS = ["minimize", "minimize", "minimize", "maximize"]

# Bumped when the objectives or the searched knobs change, so a resumed study can
# never mix trials drawn from two different spaces. v2 added top-1 as a fourth
# objective and unfroze the four runtime knobs.
SPACE_VERSION = "v2"


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


def constraint_for(result: TrialResult, threshold: float) -> float:
    """Optuna reads a constraint as feasible when it is <= 0."""
    if result.top1 is None:
        return 1.0
    return threshold - result.top1


def _constraints(trial: optuna.trial.FrozenTrial) -> tuple[float, ...]:
    return tuple(trial.user_attrs.get("constraints", (1.0,)))


def build_objective(runner: Any, model: str, groups: tuple[str, ...], threshold: float):
    def objective(trial: optuna.Trial) -> tuple[float, float, float, float]:
        config = suggest_config(trial, groups)
        try:
            result = evaluate(runner, BenchSpec(model=model, config=config), threshold)
        except DeviceUnavailable as error:
            # Halt rather than grind: with the link down every remaining trial
            # would be an identical fake rejection, and Optuna's `catch` would
            # bury them. The FAILED state keeps them out of the NSGA-II
            # population, and the study resumes from the sqlite file.
            print(f"  trial {trial.number:3d} device unavailable: {error}", flush=True)
            trial.study.stop()
            raise

        trial.set_user_attr("result", result.as_dict())
        trial.set_user_attr("quant_hash", config.quant.hash)
        trial.set_user_attr("describe", config.quant.describe())
        trial.set_user_attr("describe_run", config.run.describe())
        trial.set_user_attr("constraints", [constraint_for(result, threshold)])

        print(
            f"  trial {trial.number:3d} {result.status:16s} "
            + (
                f"{result.latency_ms:7.3f} ms  {result.size_bytes / 1e6:6.2f} MB  "
                f"{result.peak_rss_mb:6.1f} MB  top1 {result.top1:.2f}"
                if result.feasible
                else ""
            )
            + f"  [{config.quant.describe()} | {config.run.describe()}]",
            flush=True,  # a device campaign is watched live; block buffering hides it
        )
        return objectives_for(result)

    return objective


def _params_for(quant: QuantConfig, groups: tuple[str, ...]) -> dict[str, Any]:
    """Spell one config in the parameter names `suggest_config` uses.

    Seeds pin the runtime to `SEED_RUN` rather than leaving it to the sampler, so
    the baselines land under the exact conditions Phases 3-4 measured them under
    and stay comparable to those reports.
    """
    params: dict[str, Any] = {
        "quant_type": quant.quant_type,
        "per_channel": quant.per_channel,
        "graph_optimization_level": SEED_RUN.graph_optimization_level,
        "enable_cpu_mem_arena": SEED_RUN.enable_cpu_mem_arena,
        **{f"fp32__{group}": group in quant.excluded_groups for group in groups},
    }
    if quant.quant_type == "static":
        params["activation_type"] = quant.activation_type
        params["calibration_method"] = quant.calibration_method
        params[f"calibration_size__{quant.calibration_method}"] = quant.calibration_size
    return params


def seed_trials(study: optuna.Study, model: str, groups: tuple[str, ...]) -> int:
    """Enqueue the baselines and the Phase 5 candidates before sampling starts.

    Two reasons, and both matter. PLAN.md requires every global baseline to go
    through the identical harness, and with one independent boolean per group a
    random draw reaches the no-exclusions config with probability 2^-len(groups)
    -- so the very candidate most likely to win would otherwise never be tried.
    """
    seeds = {name: config for name, config in BASELINE_CONFIGS.items() if config.is_quantized}
    seeds.update(candidate_configs(model))

    enqueued = 0
    for quant in seeds.values():
        if set(quant.excluded_groups) - set(groups):
            continue  # names a group this graph does not expose
        study.enqueue_trial(_params_for(quant, groups), skip_if_exists=True)
        enqueued += 1
    return enqueued


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=EXPORTED_MODELS)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--budget-pt", type=float, default=DEFAULT_ACCURACY_BUDGET_PT)
    parser.add_argument("--mock", action="store_true", help="dry run against a fake device")
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.study_dir.mkdir(parents=True, exist_ok=True)

    group_map = build_group_map(ModelPaths.resolve(args.model, args.onnx_dir).quant_ready, args.model)
    groups = quantizable_groups(group_map)

    baseline = baseline_top1(args.model)
    threshold = baseline - args.budget_pt
    if args.mock:
        runner = MockRunner(groups)
    else:
        runner = RemoteBenchmarker(PiConnection.load())
        # Unconditional: a stale device checkout fails every trial identically at
        # the first gate, which reads as a bad search space rather than a missing
        # file. An scp of src/ is cheaper than one wasted trial.
        runner.push_code()

    # The name carries the budget, the venue and the space version: re-running at
    # a different budget must start a new study, a mock run must never resume a
    # device one, and `load_if_exists` raises outright if it finds a study whose
    # directions differ -- which every pre-v2 study does.
    suffix = "_mock" if args.mock else ""
    name = f"{args.model}_budget{args.budget_pt:g}_{SPACE_VERSION}{suffix}"
    storage = f"sqlite:///{(args.study_dir / f'{name}.db').as_posix()}"

    study = optuna.create_study(
        study_name=name,
        storage=storage,
        load_if_exists=True,
        directions=DIRECTIONS,
        sampler=optuna.samplers.NSGAIISampler(seed=args.seed, constraints_func=_constraints),
    )

    seeded = seed_trials(study, args.model, groups)
    print(
        f"{name}: {len(groups)} groups, baseline {baseline:.2f}%, "
        f"threshold {threshold:.2f}%, {args.trials} trials ({seeded} seeded)\n"
    )
    study.optimize(
        build_objective(runner, args.model, groups, threshold),
        n_trials=args.trials,
        catch=(Exception,),  # an infeasible candidate is a trial, never a dead study
    )

    feasible = [
        trial
        for trial in study.best_trials
        if trial.user_attrs.get("result", {}).get("status") == "ok"
    ]
    print(f"\nPareto set: {len(feasible)} of {len(study.trials)} trials")
    for trial in sorted(feasible, key=lambda t: t.values[0]):
        result = trial.user_attrs["result"]
        print(
            f"  {result['latency_ms']:7.3f} ms  {result['size_bytes'] / 1e6:6.2f} MB  "
            f"{result['peak_rss_mb']:6.1f} MB RSS  top1 {result['top1']:.2f}  "
            f"[{trial.user_attrs['describe']} | {trial.user_attrs.get('describe_run', '')}]"
        )

    summary = {
        "study": name,
        "model": args.model,
        "mock": args.mock,
        "baseline_top1": baseline,
        "accuracy_budget_pt": args.budget_pt,
        "threshold_top1": threshold,
        "groups": list(groups),
        "trials": len(study.trials),
        "pareto": [
            {"values": trial.values, **trial.user_attrs} for trial in feasible
        ],
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    report = args.study_dir / f"{name}.json"
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
