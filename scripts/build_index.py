"""Build the landing page that lists every run.

Headline figures are extracted from each run's own stage files, so this page
cannot disagree with the page it links to.

    python -m scripts.build_index
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.build_dashboard import REFERENCE_RUN, measurements_by_config
from src.quant.baselines import BASELINE_CONFIGS
from src.quant.config import DeploymentConfig
from src.runs import RUNS_DIR, Run, list_runs

CACHE_DIR = Path("artifacts/bench_cache")

TEMPLATE = Path("scripts/index_template.html")
DEFAULT_OUT = Path("artifacts/dashboard/index.html")
DATA_PLACEHOLDER = "/*__INDEX_DATA__*/"

LABELS = {
    "resnet18_cifar": "ResNet-18",
    "mobilenetv2_cifar": "MobileNetV2",
    "custom_cnn": "Custom CNN",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fp32_latency_ms(run: Run, cache_dir: Path) -> float | None:
    """What the unquantized export measured, for the speedup on each card.

    The quantization report carries accuracy and bytes but not time; latency
    lives in the device cache, joined by deployment hash exactly as the run's
    own page joins it.
    """
    if run.venue != "pi":
        return None
    measured = measurements_by_config(run.model, cache_dir)
    reference = DeploymentConfig(quant=BASELINE_CONFIGS["fp32"], run=REFERENCE_RUN).hash
    found = measured.get(reference)
    return found["latency"]["median_ms"] if found else None


def headline(run: Run, out: Path, cache_dir: Path = CACHE_DIR) -> dict[str, Any] | None:
    """One card's figures, or None if the run has not got far enough to have any."""
    study_file = run.paths.stage("study")
    quant_file = run.paths.stage("quant_baselines")
    if not study_file.exists() or not quant_file.exists():
        return None

    summary = read_json(study_file)
    front = [member["result"] for member in summary["pareto"] if member["result"]["status"] == "ok"]
    if not front:
        return None

    fp32 = read_json(quant_file)["baselines"]["fp32"]
    fp32_latency = fp32.get("latency_ms") or fp32_latency_ms(run, cache_dir)
    fastest = min(result["latency_ms"] for result in front)

    page = run.paths.dashboard
    return {
        "run_id": run.run_id,
        "label": LABELS.get(run.model, run.model),
        "model": run.model,
        "venue": run.venue,
        "trials": summary["trials"],
        "front_size": len(front),
        "best_latency_ms": fastest,
        "best_size_mb": min(result["size_bytes"] for result in front) / 1e6,
        "best_top1": max(result["top1"] for result in front),
        "baseline_top1": summary["baseline_top1"],
        "fp32_size_mb": fp32["bytes"] / 1e6,
        # Against the FP32 export measured through the same harness. Zero when
        # that measurement is not in the cache, and the card drops the note.
        "speedup": (fp32_latency / fastest) if fp32_latency else 0.0,
        # Relative, so the file:// link works wherever the repo is cloned.
        "page": Path(os.path.relpath(page, out.parent)).as_posix() if page.exists() else None,
    }


def build_payload(runs_dir: Path, out: Path) -> dict[str, Any]:
    cards = [card for card in (headline(run, out) for run in list_runs(runs_dir)) if card]
    # Measured runs first, then by how fast their best candidate is.
    cards.sort(key=lambda card: (card["venue"] != "pi", card["best_latency_ms"]))
    return {
        "built_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "runs": cards,
    }


def render(payload: dict[str, Any], template: Path) -> str:
    text = template.read_text(encoding="utf-8")
    if DATA_PLACEHOLDER not in text:
        raise SystemExit(f"{template} has no {DATA_PLACEHOLDER} to fill")
    return text.replace(DATA_PLACEHOLDER, json.dumps(payload, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    payload = build_payload(args.runs_dir, args.out)
    if not payload["runs"]:
        raise SystemExit(f"No runs with results under {args.runs_dir}. Run one first.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(payload, args.template), encoding="utf-8")

    for card in payload["runs"]:
        print(f"  {card['run_id']:34s} {card['venue']:5s} "
              f"{card['best_latency_ms']:7.2f} ms  {card['best_top1']:6.2f}%  "
              f"{card['front_size']:2d} on the front")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
