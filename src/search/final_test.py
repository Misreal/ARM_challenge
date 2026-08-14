"""Score the measured Pareto front on the sealed CIFAR-100 test set, on the Pi.

Every front member plus any named choice not already on it, in one pass, so no
candidate is ever selected using test-set signal. Configs are read back from the
study report rather than replayed out of a gitignored sqlite file.

    python -m src.search.final_test --model resnet18_cifar --confirm
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.bench.agent import BenchSpec
from src.bench.remote import PiConnection, RemoteBenchmarker
from src.data.export_test_bundle import DEFAULT_OUTPUT_DIR as TEST_BUNDLE_DIR, SPLIT_NAME
from src.model_index import known_models
from src.quant.baselines import BASELINE_CONFIGS
from src.quant.config import DeploymentConfig
from src.quant.quantize import DEFAULT_ONNX_DIR
from src.search.study import DEFAULT_STUDY_DIR, SEED_RUN
from src.search.version import study_name

DEFAULT_REPORT_DIR = Path("artifacts/reports_pi")

# FP32 is scored alongside the front rather than read from the baseline report:
# that number came from the PC, and the whole point of this run is that the Pi's
# kernels are the ones that decide.
FP32_LABEL = "fp32 (reference)"


def load_summary(study_path: Path) -> dict[str, Any]:
    return json.loads(study_path.read_text(encoding="utf-8"))


def label_for(member: dict[str, Any]) -> str:
    return f"{member['describe']} | {member.get('describe_run', '')}"


def scored_candidates(summary: dict[str, Any]) -> list[tuple[str, DeploymentConfig]]:
    """The front, plus any named choice not already on it.

    A named choice that the front does not dominate into itself -- the lowest-RAM
    pick under a ceiling, say -- still has to carry a test number, or the headline
    table would quote a candidate nobody measured on the sealed split.
    """
    members = {member["config_hash"]: member for member in summary["pareto"]}
    for member in summary.get("evaluated_configs", []):
        if member["config_hash"] in set(summary.get("selections", {}).values()):
            members.setdefault(member["config_hash"], member)

    candidates: list[tuple[str, DeploymentConfig]] = []
    for member in members.values():
        stored = member.get("config")
        if not stored:
            raise SystemExit(
                f"{label_for(member)} carries no config. This study predates study/2, whose "
                "summaries store them in full; re-run the search rather than reconstructing it."
            )
        config = DeploymentConfig.from_dict(stored)
        if config.quant.hash != member["quant_hash"]:
            raise SystemExit(
                f"{label_for(member)} reads back as quant hash {config.quant.hash} but the study "
                f"recorded {member['quant_hash']}. Refusing to attribute test numbers to configs "
                "that may not be the ones measured."
            )
        candidates.append((label_for(member), config))
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=known_models())
    parser.add_argument("--budget-pt", type=float, default=0.2)
    # The pipeline knows the exact study it produced, including any population
    # suffix. Rebuilding the name from --budget-pt alone silently scores a
    # different campaign's front whenever the two disagree.
    parser.add_argument("--study", default=None, help="study name; derived from --budget-pt if absent")
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

    name = args.study or study_name(args.model, args.budget_pt)
    summary = load_summary(args.study_dir / f"{name}.json")
    candidates = scored_candidates(summary)
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
        "study": name,
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
