"""Export the sealed CIFAR-100 test set as a Pi bundle, for final evaluation only.

Separate from `export_pi_data` on purpose: that module must keep excluding the
test set, so breaking the seal stays a deliberate, auditable act rather than a
flag someone flips during the search campaign.

    python -m src.data.export_test_bundle --confirm
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.export_pi_data import extract_uint8_nchw, sha256_file, verify_against_torchvision
from src.data.loaders import CIFAR100_MEAN, CIFAR100_STD, DEFAULT_DATA_ROOT, load_base_dataset
from src.data.splits import DEFAULT_SPLIT_PATH, load_split, split_fingerprint

# A directory of its own, never artifacts/pi_data: a stray --bundle-dir default
# must not be able to serve test images to a search run.
DEFAULT_OUTPUT_DIR = Path("artifacts/pi_test_data")

SPLIT_NAME = "test"
EXPECTED_IMAGES = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="required: acknowledges this is the final evaluation, not a screen",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing bundle")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm:
        raise SystemExit(
            "Refusing to export the sealed test set without --confirm. The test set is for "
            "final evaluation only; querying it repeatedly to steer optimization overfits it."
        )

    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        raise SystemExit(
            f"{manifest_path} already exists. The bundle is built once; pass --force only if "
            "you are certain the previous export was wrong."
        )

    # From the split file rather than the dataset: the test set is not part of the
    # partition, but recording the fingerprint ties this bundle to one campaign.
    split = load_split(args.split_path)
    fingerprint = split_fingerprint(split)

    base = load_base_dataset(args.data_root, train=False, augment=False)
    data_hwc = np.asarray(base.data, dtype=np.uint8)
    if data_hwc.shape[0] != EXPECTED_IMAGES:
        raise SystemExit(f"Expected {EXPECTED_IMAGES} test images, got {data_hwc.shape[0]}")

    images_uint8 = extract_uint8_nchw(data_hwc, tuple(range(data_hwc.shape[0])))
    labels = np.asarray(base.targets, dtype=np.int64)

    max_delta = verify_against_torchvision(
        images_uint8, labels, split, SPLIT_NAME, args.data_root
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.output_dir / f"{SPLIT_NAME}_images_uint8.npy"
    label_path = args.output_dir / f"{SPLIT_NAME}_labels.npy"
    np.save(image_path, images_uint8)
    np.save(label_path, labels)

    manifest: dict[str, Any] = {
        "split_fingerprint": fingerprint,
        "sealed_split": True,
        "splits": {
            SPLIT_NAME: {
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
        },
        "preprocessing": {
            "layout": "NCHW",
            "stored_dtype": "uint8",
            "stored_range": "[0, 255]",
            "mean": list(CIFAR100_MEAN),
            "std": list(CIFAR100_STD),
            "recipe": "x = images.astype(float32) / 255.0; x = (x - mean) / std",
            "reference_implementation": "src.portable.preprocess.normalize_uint8_nchw",
        },
        "purpose": (
            "Phase 7 final evaluation of the measured Pareto front. Never a search or "
            "calibration input; the front was selected without any test-set signal."
        ),
        "package_versions": {"numpy": np.__version__, "torch": str(torch.__version__)},
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"{SPLIT_NAME}: {images_uint8.shape[0]} images, max delta vs torchvision {max_delta:.3g}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
