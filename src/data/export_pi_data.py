"""Export CIFAR-100 evaluation data as .npy bundles for the Raspberry Pi agent.

The Pi agent is torch-free (see requirements-pi.txt), so it cannot reuse
`loaders.py`. That creates the project's most dangerous silent failure mode:
preprocessing that differs between PC and Pi shows up as an unexplained
accuracy drop that looks like quantization damage (PLAN.md DO-NOT #7).

Two defences, both implemented here:

1. Images ship as raw **uint8**, never as pre-normalized floats. Normalization
   happens on device, from the constants recorded in the manifest, using
   `normalize_uint8_nchw` below -- a numpy-only function the agent copies
   verbatim. Shipping floats would instead bake one machine's arithmetic into
   the artifact and hide any divergence.
2. Before writing anything, the numpy path is checked against the torchvision
   path image-by-image. If they disagree, the export fails rather than handing
   the Pi a subtly wrong dataset.

CIFAR-100 is natively 32x32, so there is no resize step and therefore no
PIL-vs-OpenCV interpolation divergence to worry about. Keep it that way.

The test split is deliberately NOT exported. It stays sealed on the PC until
Phase 7 (PLAN.md DO-NOT #3); putting it on the benchmark device would make it
casually queryable during the search campaign.

Example:
    python -m src.data.export_pi_data
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.loaders import (
    CIFAR100_MEAN,
    CIFAR100_STD,
    DEFAULT_DATA_ROOT,
    build_loader,
    load_base_dataset,
)
from src.data.splits import DEFAULT_SPLIT_PATH, SplitIndices, load_or_create_split, split_fingerprint

DEFAULT_OUTPUT_DIR = Path("artifacts/pi_data")

# optval drives on-device accuracy verification; calib is the only legal source
# of static-quantization calibration data. test is excluded on purpose.
EXPORTED_SPLITS: tuple[str, ...] = ("optval", "calib")

# Tolerance for the numpy-vs-torchvision check. Both paths do the same float32
# operations in the same order, so anything above float noise means the
# documented recipe is not what the PC actually does.
MAX_ALLOWED_DELTA = 1e-6


def normalize_uint8_nchw(images: np.ndarray) -> np.ndarray:
    """Convert uint8 NCHW images to the normalized float32 the ONNX graph expects.

    This is the preprocessing contract. The Pi agent must apply exactly this,
    with the constants from the manifest -- it is reproduced here in pure numpy
    precisely so it can be copied to a torch-free device.

    Mirrors torchvision `ToTensor()` followed by `Normalize(mean, std)`:
    scale to [0, 1], then subtract the per-channel mean and divide by the
    per-channel standard deviation.
    """
    if images.dtype != np.uint8:
        raise TypeError(f"Expected uint8 images, got {images.dtype}")
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected NCHW images with 3 channels, got shape {images.shape}")

    mean = np.asarray(CIFAR100_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(CIFAR100_STD, dtype=np.float32).reshape(1, 3, 1, 1)
    return (images.astype(np.float32) / 255.0 - mean) / std


def extract_uint8_nchw(data_hwc: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    """Select split members from CIFAR's native HWC array and convert to NCHW.

    Transposing here rather than on device keeps the agent's hot path to a
    single arithmetic expression, and keeps the axis order identical to the
    ONNX input signature.
    """
    selected = data_hwc[np.asarray(indices, dtype=np.int64)]
    return np.ascontiguousarray(selected.transpose(0, 3, 1, 2))


def verify_against_torchvision(
    images_uint8: np.ndarray,
    labels: np.ndarray,
    split: SplitIndices,
    name: str,
    data_root: Path,
) -> float:
    """Prove the numpy recipe reproduces the PC data pipeline exactly.

    Returns the largest absolute difference found, and raises if the two paths
    disagree beyond float noise or if the label ordering does not line up.
    """
    loader = build_loader(split, name, root=data_root, batch_size=256, num_workers=0)
    normalized = normalize_uint8_nchw(images_uint8)

    max_delta = 0.0
    offset = 0
    for reference_images, reference_targets in loader:
        count = reference_targets.size(0)
        candidate = torch.from_numpy(normalized[offset : offset + count])

        if not torch.equal(reference_targets, torch.from_numpy(labels[offset : offset + count])):
            raise ValueError(
                f"Label order mismatch in split {name!r} at offset {offset}. The .npy bundle "
                "must follow the same index order as the dataloader."
            )
        max_delta = max(max_delta, float((reference_images - candidate).abs().max().item()))
        offset += count

    if offset != images_uint8.shape[0]:
        raise ValueError(
            f"Split {name!r}: dataloader yielded {offset} images but the bundle holds "
            f"{images_uint8.shape[0]}"
        )
    if max_delta > MAX_ALLOWED_DELTA:
        raise ValueError(
            f"Split {name!r}: numpy preprocessing diverges from torchvision by {max_delta:.3g} "
            f"(limit {MAX_ALLOWED_DELTA:.0e}). Fix normalize_uint8_nchw before shipping to the Pi."
        )
    return max_delta


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = load_base_dataset(args.data_root, train=True, augment=False)
    split = load_or_create_split(base.targets, path=args.split_path)
    fingerprint = split_fingerprint(split)

    data_hwc = np.asarray(base.data, dtype=np.uint8)
    all_targets = np.asarray(base.targets, dtype=np.int64)

    arrays: dict[str, Any] = {}
    for name in EXPORTED_SPLITS:
        indices = split.get(name)
        images_uint8 = extract_uint8_nchw(data_hwc, indices)
        labels = all_targets[np.asarray(indices, dtype=np.int64)]

        max_delta = verify_against_torchvision(images_uint8, labels, split, name, args.data_root)

        image_path = args.output_dir / f"{name}_images_uint8.npy"
        label_path = args.output_dir / f"{name}_labels.npy"
        np.save(image_path, images_uint8)
        np.save(label_path, labels)

        arrays[name] = {
            "images": {
                "file": image_path.name,
                "sha256": sha256_file(image_path),
                "shape": list(images_uint8.shape),
                "dtype": "uint8",
            },
            "labels": {
                "file": label_path.name,
                "sha256": sha256_file(label_path),
                "shape": list(labels.shape),
                "dtype": "int64",
            },
            "max_abs_delta_vs_torchvision": max_delta,
        }
        print(f"{name}: {images_uint8.shape[0]} images, max delta vs torchvision {max_delta:.3g}")

    manifest = {
        "split_fingerprint": fingerprint,
        "splits": arrays,
        "preprocessing": {
            "layout": "NCHW",
            "stored_dtype": "uint8",
            "stored_range": "[0, 255]",
            "mean": list(CIFAR100_MEAN),
            "std": list(CIFAR100_STD),
            "recipe": "x = images.astype(float32) / 255.0; x = (x - mean) / std",
            "reference_implementation": "src.data.export_pi_data.normalize_uint8_nchw",
            "note": (
                "Normalization is NOT in the ONNX graph; the consumer applies it. "
                "Any divergence from this recipe shows up as unexplained accuracy loss."
            ),
        },
        "excluded": {
            "test": "Sealed until Phase 7 final evaluation (PLAN.md DO-NOT #3); never ship to the Pi.",
            "train": "Not needed on device; Stage 0 training is frozen.",
            "trainval": "Checkpoint-selection split; has no role in deployment measurement.",
        },
        "package_versions": {"numpy": np.__version__, "torch": str(torch.__version__)},
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
