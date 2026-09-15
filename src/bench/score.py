"""Score one candidate's accuracy on the device, in its own process."""

# Deliberately not part of `agent.py`: evaluating 3000 images allocates arrays that
# would land in the `ru_maxrss` the agent exists to attribute to inference alone.
#
#     python -m src.bench.score --spec spec.json --out result.json --limit 500

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import onnxruntime as ort

from src.bench.agent import BenchSpec
from src.portable.bundle import DEFAULT_BUNDLE_DIR, load_bundle
from src.portable.onnx_eval import evaluate_accuracy
from src.quant.quantize import QuantizationFailure, build_artifact, build_session

# Local rather than in protocol.py: this is a second, independent contract, and
# bumping it must never invalidate cached bench results keyed on RESULT_SCHEMA.
SCORE_SCHEMA = "score/1"
DEFAULT_EVAL_SPLIT = "optval"


def score(
    spec: BenchSpec,
    limit: int | None,
    bundle_dir: Path,
    split: str = DEFAULT_EVAL_SPLIT,
) -> dict[str, Any]:
    """Top-1/top-5 over the first `limit` images of `split`, or all of them."""
    envelope: dict[str, Any] = {
        "schema": SCORE_SCHEMA,
        "status": "ok",
        "model": spec.model,
        "config": spec.config.as_dict(),
        "quant_hash": spec.quant.hash,
        "config_hash": spec.config.hash,
        "eval_split": split,
        "limit": limit,
        "onnxruntime": ort.__version__,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }

    started = time.perf_counter()
    try:
        artifact = build_artifact(spec.model, spec.quant, bundle_dir=bundle_dir)
    except (QuantizationFailure, FileNotFoundError) as error:
        return {**envelope, "status": "build_failed", "error": f"{type(error).__name__}: {error}"}

    bundle = load_bundle(split, bundle_dir=bundle_dir)
    session = build_session(artifact.path, spec.run)
    accuracy = evaluate_accuracy(session, bundle.images_uint8, bundle.labels, limit=limit)

    return {
        **envelope,
        "accuracy": accuracy.as_dict(),
        "bytes": artifact.bytes,
        "split_fingerprint": bundle.split_fingerprint,
        "seconds": round(time.perf_counter() - started, 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="first N images (default: all)")
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--split", default=DEFAULT_EVAL_SPLIT, help="split name inside the bundle")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = BenchSpec.from_dict(json.loads(args.spec.read_text(encoding="utf-8")))
    result = score(spec, args.limit, args.bundle_dir, args.split)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if result["status"] != "ok":
        raise SystemExit(result.get("error", result["status"]))


if __name__ == "__main__":
    main()
