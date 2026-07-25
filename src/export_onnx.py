"""Export a trained CIFAR-100 checkpoint to static ONNX and verify it.

Example:
    python -m src.export_onnx --model custom_cnn --tag pilot
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import quant_pre_process

import src.models  # noqa: F401 -- register model builders
from src.checkpoint import load_checkpoint
from src.data.loaders import CIFAR100_MEAN, CIFAR100_STD, DEFAULT_DATA_ROOT, build_loader, load_base_dataset
from src.data.splits import DEFAULT_SPLIT_PATH, load_or_create_split, split_fingerprint
from src.models.registry import available_models, build_model
from src.utils import resolve_device

DEFAULT_CHECKPOINT_DIR = Path("artifacts/checkpoints")
DEFAULT_ONNX_DIR = Path("artifacts/onnx")
DEFAULT_REPORT_DIR = Path("artifacts/reports")
OPSET_VERSION = 17
INPUT_NAME = "images"
OUTPUT_NAME = "logits"
INPUT_SHAPE = (1, 3, 32, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=available_models())
    parser.add_argument("--tag", default=None, help="checkpoint suffix, for example: pilot")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--parity-samples", type=int, default=512)
    parser.add_argument(
        "--seal-test",
        action="store_true",
        help="evaluate the test set once; use only for a real final baseline",
    )
    return parser.parse_args()


def artifact_stem(model: str, tag: str | None) -> str:
    return model + (f"_{tag}" if tag else "")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_model(model: torch.nn.Module, output_path: Path) -> None:
    """Export an evaluation-mode model with the project's fixed input shape."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(INPUT_SHAPE, dtype=torch.float32)
    torch.onnx.export(
        model.cpu().eval(),
        example,
        str(output_path),
        export_params=True,
        do_constant_folding=True,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        opset_version=OPSET_VERSION,
        dynamo=False,
    )
    onnx.checker.check_model(str(output_path))


def preprocess_for_quantization(source_path: Path, output_path: Path) -> None:
    """Write the shape-inferred, fused graph that Phase 4 must quantize.

    Running `quantize_static` against the raw export instead of this file is the
    single most common ONNX Runtime quantization mistake (PLAN.md DO-NOT #4):
    anything left unfused gets wrapped in QDQ nodes, making the result both
    slower and less accurate. Static calibration also needs the inferred shapes
    this pass adds. The deployable FP32 graph is kept separately -- this variant
    is an optimizer input, not a deliverable.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quant_pre_process(
        input_model=str(source_path),
        output_model_path=str(output_path),
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=False,
    )
    onnx.checker.check_model(str(output_path))


@torch.inference_mode()
def compare_sessions(
    reference: ort.InferenceSession,
    candidate: ort.InferenceSession,
    loader: torch.utils.data.DataLoader,
    limit: int,
) -> dict[str, float | int]:
    """Compare two ONNX graphs fed identical inputs.

    Fusion and shape inference are supposed to be numerically transparent, so a
    real delta here means preprocessing changed the model. Catching that now
    keeps it from being misread in Phase 4 as quantization damage.
    """
    if limit <= 0:
        raise ValueError("parity-samples must be positive")

    seen = 0
    agreements = 0
    max_logit_delta = 0.0
    ref_input = reference.get_inputs()[0].name
    ref_output = reference.get_outputs()[0].name
    cand_input = candidate.get_inputs()[0].name
    cand_output = candidate.get_outputs()[0].name

    for images, _ in loader:
        remaining = limit - seen
        if remaining <= 0:
            break
        batch = images[:remaining].numpy()
        ref_logits = torch.from_numpy(reference.run([ref_output], {ref_input: batch})[0])
        cand_logits = torch.from_numpy(candidate.run([cand_output], {cand_input: batch})[0])

        agreements += int((ref_logits.argmax(dim=1) == cand_logits.argmax(dim=1)).sum().item())
        max_logit_delta = max(max_logit_delta, float((ref_logits - cand_logits).abs().max().item()))
        seen += batch.shape[0]

    if seen == 0:
        raise ValueError("Parity loader produced zero images")
    return {
        "samples": seen,
        "top1_agreement": agreements / seen,
        "max_logit_delta": max_logit_delta,
    }


@torch.inference_mode()
def compare_pytorch_and_onnx(
    model: torch.nn.Module,
    session: ort.InferenceSession,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    limit: int,
) -> dict[str, float | int]:
    """Compare logits and top-1 predictions on at most `limit` optval images."""
    if limit <= 0:
        raise ValueError("parity-samples must be positive")

    model.eval()
    seen = 0
    agreements = 0
    max_logit_delta = 0.0
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    for images, _ in loader:
        remaining = limit - seen
        if remaining <= 0:
            break
        images = images[:remaining]
        torch_logits = model(images.to(device)).float().cpu()
        ort_logits = torch.from_numpy(session.run([output_name], {input_name: images.numpy()})[0])

        agreements += int((torch_logits.argmax(dim=1) == ort_logits.argmax(dim=1)).sum().item())
        max_logit_delta = max(max_logit_delta, float((torch_logits - ort_logits).abs().max().item()))
        seen += images.size(0)

    if seen == 0:
        raise ValueError("Parity loader produced zero images")
    return {
        "samples": seen,
        "top1_agreement": agreements / seen,
        "max_logit_delta": max_logit_delta,
    }


@torch.inference_mode()
def accuracy(
    session: ort.InferenceSession, loader: torch.utils.data.DataLoader
) -> float:
    """Return ONNX Runtime top-1 accuracy as a percentage."""
    correct = 0
    seen = 0
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    for images, targets in loader:
        logits = torch.from_numpy(session.run([output_name], {input_name: images.numpy()})[0])
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        seen += targets.size(0)
    if seen == 0:
        raise ValueError("Accuracy loader produced zero images")
    return 100.0 * correct / seen


def operator_inventory(model: onnx.ModelProto) -> dict[str, int]:
    return dict(sorted(Counter(node.op_type for node in model.graph.node).items()))


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Write the Stage 0 to Stage 1 baseline contract."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        sealed = previous.get("metrics", {}).get("test_top1_sealed")
        if sealed is not None and report["metrics"]["test_top1_sealed"] != sealed:
            raise ValueError(
                f"{path} already contains sealed test accuracy. Refusing to replace it. "
                "Use a new --tag for a different checkpoint."
            )
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    stem = artifact_stem(args.model, args.tag)
    checkpoint_path = args.checkpoint_path or args.checkpoint_dir / f"{stem}.pt"
    onnx_path = args.onnx_dir / f"{stem}.onnx"
    quant_ready_path = args.onnx_dir / f"{stem}_quant_ready.onnx"
    report_path = args.report_dir / f"{stem}_baseline.json"

    labels = load_base_dataset(args.data_root, train=True).targets
    split = load_or_create_split(labels, path=args.split_path)
    fingerprint = split_fingerprint(split)
    checkpoint = load_checkpoint(checkpoint_path, expected_split_fingerprint=fingerprint)
    if checkpoint.recipe.model != args.model:
        raise ValueError(
            f"Checkpoint is for {checkpoint.recipe.model!r}, not requested model {args.model!r}"
        )

    model = build_model(args.model, num_classes=100).float()
    model.load_state_dict(checkpoint.state_dict, strict=True)
    export_model(model, onnx_path)
    preprocess_for_quantization(onnx_path, quant_ready_path)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    quant_ready_session = ort.InferenceSession(
        str(quant_ready_path), providers=["CPUExecutionProvider"]
    )
    device = resolve_device(args.device)
    model = model.to(device).eval()
    # The exported graph is intentionally static at batch size 1, so ORT must
    # receive one image per call during verification too.
    optval_loader = build_loader(split, "optval", root=args.data_root, batch_size=1, num_workers=0)
    parity = compare_pytorch_and_onnx(model, session, optval_loader, device, args.parity_samples)
    if parity["top1_agreement"] < 0.998 or parity["max_logit_delta"] >= 1e-3:
        raise RuntimeError(
            "ONNX parity failed: expected >=99.8% top-1 agreement and max logit delta < 1e-3, "
            f"got {parity['top1_agreement']:.2%} and {parity['max_logit_delta']:.6g}"
        )

    quant_ready_parity = compare_sessions(
        session, quant_ready_session, optval_loader, args.parity_samples
    )
    if quant_ready_parity["top1_agreement"] < 0.998 or quant_ready_parity["max_logit_delta"] >= 1e-3:
        raise RuntimeError(
            "Quantization-preprocessed graph diverges from the deployable export: expected "
            ">=99.8% top-1 agreement and max logit delta < 1e-3, got "
            f"{quant_ready_parity['top1_agreement']:.2%} and "
            f"{quant_ready_parity['max_logit_delta']:.6g}"
        )

    optval_top1 = accuracy(session, optval_loader)
    test_top1: float | None = None
    if args.seal_test:
        test_loader = build_loader(split, "test", root=args.data_root, batch_size=1, num_workers=0)
        test_top1 = accuracy(session, test_loader)

    onnx_model = onnx.load(str(onnx_path))
    quant_ready_model = onnx.load(str(quant_ready_path))
    report: dict[str, Any] = {
        "model": args.model,
        "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
        "split_fingerprint": fingerprint,
        "onnx": {
            "path": str(onnx_path),
            "sha256": sha256_file(onnx_path),
            "bytes": onnx_path.stat().st_size,
            "opset": OPSET_VERSION,
            "input_name": INPUT_NAME,
            "input_shape": list(INPUT_SHAPE),
            "output_name": OUTPUT_NAME,
            "operators": operator_inventory(onnx_model),
        },
        "onnx_quant_ready": {
            "path": str(quant_ready_path),
            "sha256": sha256_file(quant_ready_path),
            "bytes": quant_ready_path.stat().st_size,
            "operators": operator_inventory(quant_ready_model),
            "note": (
                "Shape-inferred and fused graph. Phase 4 must quantize THIS file, "
                "never the deployable export (PLAN.md DO-NOT #4)."
            ),
        },
        "preprocessing": {
            "input": "normalized_float32_nchw",
            "mean": list(CIFAR100_MEAN),
            "std": list(CIFAR100_STD),
        },
        "metrics": {"optval_top1": optval_top1, "test_top1_sealed": test_top1},
        "verification": {
            "torch_vs_onnx": parity,
            "onnx_vs_quant_ready": quant_ready_parity,
        },
        "package_versions": {"torch": str(torch.__version__), "onnxruntime": ort.__version__},
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    write_report(report_path, report)

    print(f"Exported {onnx_path}")
    print(f"Quantization-ready graph: {quant_ready_path}")
    print(
        f"Parity: {parity['top1_agreement']:.2%} agreement, "
        f"max logit delta {parity['max_logit_delta']:.6g} over {parity['samples']} optval images"
    )
    print(
        f"Quant-ready parity: {quant_ready_parity['top1_agreement']:.2%} agreement, "
        f"max logit delta {quant_ready_parity['max_logit_delta']:.6g}"
    )
    print(f"ONNX Runtime optval top-1: {optval_top1:.2f}%")
    if test_top1 is None:
        print("Test set not evaluated (use --seal-test only for the final baseline).")
    else:
        print(f"Sealed test top-1: {test_top1:.2f}%")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
