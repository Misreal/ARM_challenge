"""Score the measured Pareto front on the sealed CIFAR-100 test set, on the Pi.

The front JSON stores only a label per member, so configs are rebuilt from the
study's trial params through `suggest_config` and checked against the recorded
quant hash -- a silent reconstruction error would move every headline number.

    python -m src.search.final_test --model resnet18_cifar --confirm
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
from src.data.export_test_bundle import DEFAULT_OUTPUT_DIR as TEST_BUNDLE_DIR, SPLIT_NAME
from src.quant.baselines import BASELINE_CONFIGS
from src.quant.config import DeploymentConfig, EXPORTED_MODELS
from src.quant.groups import build_group_map, quantizable_groups
from src.quant.quantize import DEFAULT_ONNX_DIR, ModelPaths
from src.search.space import suggest_config
from src.search.study import DEFAULT_STUDY_DIR, SEED_RUN, SPACE_VERSION

DEFAULT_REPORT_DIR = Path("artifacts/reports_pi")

# FP32 is scored alongside the front rather than read from the baseline report:
# that number came from the PC, and the whole point of this run is that the Pi's
# kernels are the ones that decide.
FP32_LABEL = "fp32 (reference)"


def load_front(study_path: Path) -> list[dict[str, Any]]:
    return json.loads(study_path.read_text(encoding="utf-8"))["pareto"]


def rebuild_configs(
    study_name: str, study_dir: Path, groups: tuple[str, ...], front: list[dict[str, Any]]
) -> list[tuple[str, DeploymentConfig]]:
    """Recover each front member's full config, keyed back by its quant hash."""
    storage = f"sqlite:///{(study_dir / f'{study_name}.db').as_posix()}"
    study = optuna.load_study(study_name=study_name, storage=storage)

    by_hash: dict[str, DeploymentConfig] = {}
    for trial in study.trials:
        recorded = trial.user_attrs.get("quant_hash")
        if recorded is None:
            continue
        config = suggest_config(optuna.trial.FixedTrial(trial.params), groups)
        if config.quant.hash != recorded:
            raise SystemExit(
                f"Trial {trial.number} rebuilt to quant hash {config.quant.hash} but the study "
                f"recorded {recorded}. The search space changed since the run; refusing to "
                "attribute test numbers to configs that may not be the ones measured."
            )
        by_hash[_identity(recorded, trial.user_attrs.get("describe_run", ""))] = config

    rebuilt: list[tuple[str, DeploymentConfig]] = []
    for member in front:
        key = _identity(member["quant_hash"], member.get("describe_run", ""))
        if key not in by_hash:
            raise SystemExit(f"No trial in the study matches front member {key}")
        rebuilt.append((f"{member['describe']} | {member.get('describe_run', '')}", by_hash[key]))
    return rebuilt


def _identity(quant_hash: str, describe_run: str) -> str:
    """Front members share quant hashes across runtimes, so both halves are the key."""
    return f"{quant_hash}@{describe_run}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=EXPORTED_MODELS)
    parser.add_argument("--budget-pt", type=float, default=0.2)
    parser.add_argument(
        "--confirm", action="store_true", help="required: this consumes the sealed test set"
    )
    parser.add_argument("--limit", type=int, default=None, help="first N test images (debug only)")
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR)
    parser.add_argument("--bundle-dir", type=Path, default=TEST_BUNDLE_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--push-code", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm:
        raise SystemExit(
            "Refusing to run without --confirm. This is the first and only use of the sealed "
            "test set; every front member is scored in one pass so no selection happens here."
        )

    study_name = f"{args.model}_budget{args.budget_pt:g}_{SPACE_VERSION}"
    front = load_front(args.study_dir / f"{study_name}.json")

    group_map = build_group_map(ModelPaths.resolve(args.model, args.onnx_dir).quant_ready, args.model)
    groups = quantizable_groups(group_map)

    candidates = rebuild_configs(study_name, args.study_dir, groups, front)
    # The reference runs under the runtime the seeded baselines used, so its test
    # number joins to the Phase 3-4 reports rather than to an arbitrary trial.
    candidates.append(
        (FP32_LABEL, DeploymentConfig(quant=BASELINE_CONFIGS["fp32"], run=SEED_RUN))
    )

    runner = RemoteBenchmarker(PiConnection.load())
    if args.push_code:
        runner.push_code()
    remote_bundle = runner.push_bundle(args.bundle_dir)
    print(f"Bundle at {remote_bundle}\n{len(candidates)} candidates on the sealed test set\n")

    rows: list[dict[str, Any]] = []
    for label, config in candidates:
        spec = BenchSpec(model=args.model, config=config)
        result = runner.score(
            spec, limit=args.limit, split=SPLIT_NAME, bundle_dir=remote_bundle
        )
        if result.get("status") != "ok":
            print(f"  {label:<70} {result.get('status')}: {result.get('error', '')[:100]}")
            rows.append({"label": label, "status": result.get("status"), "error": result.get("error")})
            continue

        accuracy = result["accuracy"]
        print(f"  {label:<70} top1 {accuracy['top1']:6.2f}  top5 {accuracy['top5']:6.2f}")
        rows.append(
            {
                "label": label,
                "status": "ok",
                "quant_hash": config.quant.hash,
                "config": config.as_dict(),
                "test": accuracy,
                "seconds": result.get("seconds"),
            }
        )

    report = {
        "schema": "final_test/1",
        "model": args.model,
        "study": study_name,
        "eval_split": SPLIT_NAME,
        "limit": args.limit,
        "on_target": True,
        "note": (
            "First and only use of the sealed test set. Every Pareto member is scored, so no "
            "candidate was selected using test-set signal."
        ),
        "results": rows,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    path = args.report_dir / f"{args.model}_final_test.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
