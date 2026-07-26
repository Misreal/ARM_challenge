"""Load the uint8 evaluation bundles written by `src.data.export_pi_data`.

Shared with the Pi agent. The checksum verification matters more on the device
than on the PC: files get there over SSH, and a truncated or stale transfer
would otherwise surface as a mysterious accuracy drop attributed to INT8.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_BUNDLE_DIR = Path("artifacts/pi_data")


@dataclass(frozen=True)
class EvalBundle:
    """Raw uint8 images plus labels for one named split."""

    name: str
    images_uint8: np.ndarray  # (N, 3, 32, 32) uint8
    labels: np.ndarray  # (N,) int64
    split_fingerprint: str

    def __len__(self) -> int:
        return int(self.images_uint8.shape[0])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bundle(name: str, bundle_dir: Path = DEFAULT_BUNDLE_DIR, verify: bool = True) -> EvalBundle:
    """Load one split's arrays, checking them against the manifest checksums.

    `verify=False` exists only for throwaway experiments on a machine where the
    bundle was just written; any measurement that ends up in a report should
    keep it on.
    """
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest at {manifest_path}. Run: python -m src.data.export_pi_data"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if name not in manifest["splits"]:
        available = sorted(manifest["splits"])
        raise KeyError(
            f"Split {name!r} is not in the bundle (have {available}). "
            "The test split is excluded on purpose and must not be added here."
        )

    entry = manifest["splits"][name]
    image_path = bundle_dir / entry["images"]["file"]
    label_path = bundle_dir / entry["labels"]["file"]

    if verify:
        for path, expected in ((image_path, entry["images"]), (label_path, entry["labels"])):
            actual = sha256_file(path)
            if actual != expected["sha256"]:
                raise ValueError(
                    f"{path} does not match the manifest checksum "
                    f"(expected {expected['sha256'][:16]}..., got {actual[:16]}...). "
                    "Re-copy the bundle or regenerate it; do not measure against it."
                )

    return EvalBundle(
        name=name,
        images_uint8=np.load(image_path),
        labels=np.load(label_path),
        split_fingerprint=manifest["split_fingerprint"],
    )
