"""Bring an ONNX model that was not trained here into the app.

Torch-free on purpose: an imported graph has no checkpoint, so everything the
pipeline needs -- the quant-ready twin, the FP32 baseline, the group map -- is
derived from the file itself.

    python -m src.import_onnx --onnx model.onnx --name hf_resnet20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort
from onnxruntime.quantization import quant_pre_process

from src.model_index import INDEX_PATH, ModelEntry, register
from src.portable.bundle import load_bundle
from src.portable.onnx_eval import evaluate_accuracy
from src.quant.groups import MAX_GROUP_SHARE, UNSCOPED_GROUP, build_group_map, quantizable_groups
from src.quant.quantize import DEFAULT_ONNX_DIR

REPORT_DIR = Path("artifacts/reports")
OPTVAL_SPLIT = "optval"
TEST_SPLIT = "test"
TEST_BUNDLE_DIR = Path("artifacts/pi_test_data")

# The eval bundles are 32x32 CIFAR-100. Anything else would need resizing, and
# resizing is where PIL-versus-OpenCV interpolation silently costs accuracy.
REQUIRED_SHAPE = (1, 3, 32, 32)
REQUIRED_CLASSES = 100
MIN_OPSET = 13  # per-channel QDQ needs 13 or later


class ImportRejected(SystemExit):
    """The graph cannot be run through this pipeline, with the reason why."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ------------------------------------------------------------------- checking


def tensor_shape(value: onnx.ValueInfoProto) -> tuple[Any, ...]:
    dims = []
    for dim in value.type.tensor_type.shape.dim:
        dims.append(dim.dim_value if dim.HasField("dim_value") else dim.dim_param or "?")
    return tuple(dims)


def check_graph(model: onnx.ModelProto, path: Path) -> dict[str, Any]:
    """Refuse anything the rest of the pipeline cannot honestly handle."""
    onnx.checker.check_model(model)

    opset = max((entry.version for entry in model.opset_import if entry.domain in ("", "ai.onnx")), default=0)
    if opset < MIN_OPSET:
        raise ImportRejected(
            f"{path} is opset {opset}; per-channel static quantization needs {MIN_OPSET} or later. "
            "Re-export it with a newer opset."
        )

    inputs = [value for value in model.graph.input if value.name not in {i.name for i in model.graph.initializer}]
    if len(inputs) != 1:
        raise ImportRejected(f"{path} has {len(inputs)} inputs; this pipeline runs single-input classifiers.")

    shape = tensor_shape(inputs[0])
    if shape != REQUIRED_SHAPE:
        raise ImportRejected(
            f"{path} takes {shape}, and this pipeline needs {REQUIRED_SHAPE}.\n"
            "It reuses a fixed CIFAR-100 bundle of 32x32 uint8 images, and a dynamic or larger "
            "input would need resizing, which is where preprocessing differences quietly cost "
            "accuracy. Re-export at batch 1 with static 32x32 input, or train a 32x32 model."
        )

    outputs = tensor_shape(model.graph.output[0])
    if outputs[-1] != REQUIRED_CLASSES:
        raise ImportRejected(
            f"{path} emits {outputs[-1]} classes; the committed split is CIFAR-100, so it needs "
            f"{REQUIRED_CLASSES}."
        )

    return {
        "opset": opset,
        "input_name": inputs[0].name,
        "input_shape": list(shape),
        "output_name": model.graph.output[0].name,
        "operators": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
    }


def check_groups(quant_ready: Path, name: str) -> tuple[str, ...]:
    """The block map sensitivity analysis will read, or a clear refusal.

    Nodes carry their module path only when the graph came out of
    `torch.onnx.export`. A graph exported some other way can arrive with names
    that carry no structure, and then per-group sensitivity has nothing real to
    group on. Better to say so than to produce a ranking over one mega-group.
    """
    group_map = build_group_map(quant_ready, name)
    groups = quantizable_groups(group_map)
    if not groups:
        raise ImportRejected(f"{quant_ready} produced no quantizable groups; its nodes carry no module paths.")

    counts = Counter(group_map.values())
    largest, share = counts.most_common(1)[0]
    fraction = share / sum(counts.values())
    if fraction > MAX_GROUP_SHARE:
        raise ImportRejected(
            f"Group {largest!r} holds {fraction:.0%} of the graph, over the {MAX_GROUP_SHARE:.0%} bar.\n"
            "Sensitivity analysis over one mega-group measures nothing. This usually means the "
            "graph's nodes are not named by module path, or that its blocks live one level "
            "deeper -- add an entry to GROUP_DEPTH in src/quant/groups.py and re-import."
        )
    if largest == UNSCOPED_GROUP:
        raise ImportRejected(f"Most nodes fell into {UNSCOPED_GROUP!r}; this graph has no usable block names.")
    return groups


# ------------------------------------------------------------------ importing


def score(path: Path, split: str, bundle_dir: Path) -> dict[str, Any]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    bundle = load_bundle(split, bundle_dir=bundle_dir)
    result = evaluate_accuracy(session, bundle.images_uint8, bundle.labels)
    return {"top1": result.top1, "top5": result.top5, "samples": len(bundle), "fingerprint": bundle.split_fingerprint}


def import_model(
    source: Path,
    name: str,
    onnx_dir: Path,
    report_dir: Path,
    seal_test: bool,
    index: Path = INDEX_PATH,
) -> ModelEntry:
    model = onnx.load(str(source))
    facts = check_graph(model, source)

    deployable = onnx_dir / f"{name}.onnx"
    quant_ready = onnx_dir / f"{name}_quant_ready.onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    if source.resolve() != deployable.resolve():
        shutil.copy2(source, deployable)

    print(f"  preprocessing for quantization -> {quant_ready.name}")
    quant_pre_process(
        input_model=str(deployable),
        output_model_path=str(quant_ready),
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=False,
    )
    onnx.checker.check_model(str(quant_ready))

    groups = check_groups(quant_ready, name)
    print(f"  {len(groups)} groups: {', '.join(groups)}")

    print("  scoring on optval")
    optval = score(deployable, OPTVAL_SPLIT, Path("artifacts/pi_data"))
    # The two graphs must agree, or the optimizer would be tuning a different
    # network from the one whose baseline it is held to.
    ready = score(quant_ready, OPTVAL_SPLIT, Path("artifacts/pi_data"))
    if abs(ready["top1"] - optval["top1"]) > 1e-6:
        raise ImportRejected(
            f"The quant-ready graph scores {ready['top1']:.2f}% against the deployable graph's "
            f"{optval['top1']:.2f}%. Preprocessing changed the model; do not proceed."
        )

    metrics = {"optval_top1": optval["top1"], "optval_top5": optval["top5"]}
    if seal_test:
        print("  scoring on the sealed test split, once")
        metrics["test_top1_sealed"] = score(deployable, TEST_SPLIT, TEST_BUNDLE_DIR)["top1"]

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{name}_baseline.json"
    if report_path.exists() and "test_top1_sealed" in json.loads(report_path.read_text())["metrics"]:
        raise ImportRejected(f"{report_path} already carries a sealed test score; refusing to overwrite it.")

    report_path.write_text(
        json.dumps(
            {
                "model": name,
                "source": "imported",
                "checkpoint": {"sha256": sha256_file(source), "path": str(source)},
                "split_fingerprint": optval["fingerprint"],
                "onnx": {
                    "path": str(deployable),
                    "sha256": sha256_file(deployable),
                    "bytes": deployable.stat().st_size,
                    **facts,
                },
                "onnx_quant_ready": {"path": str(quant_ready), "sha256": sha256_file(quant_ready)},
                "metrics": metrics,
                "verification": {
                    "onnx_vs_quant_ready": {"samples": optval["samples"], "top1_delta": 0.0}
                },
                "package_versions": {"onnx": onnx.__version__, "onnxruntime": ort.__version__, "torch": None},
                "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    entry = ModelEntry(
        name=name,
        source="imported",
        groups=groups,
        group_depth=1,
        baseline_top1=optval["top1"],
        fp32_bytes=deployable.stat().st_size,
        classes=REQUIRED_CLASSES,
    )
    register(entry, index)
    return entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--name", required=True, help="the slug every later command refers to")
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument(
        "--seal-test",
        action="store_true",
        help="score the sealed test split once, now, so it is never touched during the search",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.onnx.exists():
        raise SystemExit(f"No such file: {args.onnx}")

    print(f"Importing {args.onnx} as {args.name!r}")
    entry = import_model(
        args.onnx, args.name, args.onnx_dir, args.report_dir, args.seal_test, args.index
    )

    print(f"\n{entry.name}: {entry.baseline_top1:.2f}% top-1 on optval, "
          f"{entry.fp32_bytes / 1e6:.1f} MB, {len(entry.groups)} groups")
    print(f"Run it with: python -m src.app run --model {entry.name}")


if __name__ == "__main__":
    main()
