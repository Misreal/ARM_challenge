"""Fixed, stratified four-way partition of the CIFAR-100 *training* set.

Computed once and persisted rather than regenerated per process: Stage 0
(training) and Stage 1 (the optimizer) run separately, often days apart, and a
fresh split each time would let calibration or optimization-validation images
leak into training. The CIFAR-100 test set is never touched here -- reserved
for final evaluation only.

    train     44,000   440/class   fit model weights
    trainval   2,000    20/class   epoch-level model selection during Stage 0
    optval     3,000    30/class   Stage 1 search signal / accuracy constraint
    calib      1,000    10/class   static INT8 quantization calibration

`trainval` and `optval` stay separate because reusing one set to both pick
checkpoints and score the optimizer's candidates would bias the headline result.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Per-class counts; they sum to the 500 images CIFAR-100 provides per class.
SPLIT_SIZES: dict[str, int] = {
    "train": 440,
    "trainval": 20,
    "optval": 30,
    "calib": 10,
}

SPLIT_NAMES = tuple(SPLIT_SIZES)
NUM_CLASSES = 100
IMAGES_PER_CLASS = 500
DEFAULT_SPLIT_PATH = Path("artifacts/splits/cifar100_split.json")


@dataclass(frozen=True)
class SplitIndices:
    """Immutable index sets into the torchvision CIFAR-100 train dataset."""

    train: tuple[int, ...]
    trainval: tuple[int, ...]
    optval: tuple[int, ...]
    calib: tuple[int, ...]
    seed: int

    def get(self, name: str) -> tuple[int, ...]:
        if name not in SPLIT_NAMES:
            raise KeyError(f"Unknown split {name!r}; expected one of {SPLIT_NAMES}")
        return getattr(self, name)

    def as_dict(self) -> dict[str, tuple[int, ...]]:
        return {name: self.get(name) for name in SPLIT_NAMES}


def _fingerprint(indices: dict[str, tuple[int, ...]], seed: int) -> str:
    """Stable hash of the split, stored alongside it to detect edited files."""
    payload = json.dumps(
        {name: list(indices[name]) for name in SPLIT_NAMES} | {"seed": seed},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _assert_disjoint_and_complete(indices: dict[str, tuple[int, ...]], total: int) -> None:
    """Guard the property the whole design depends on: no overlap, no gaps."""
    seen: set[int] = set()
    for name in SPLIT_NAMES:
        current = set(indices[name])
        if len(current) != len(indices[name]):
            raise ValueError(f"Split {name!r} contains duplicate indices")
        overlap = seen & current
        if overlap:
            raise ValueError(
                f"Split {name!r} overlaps earlier splits on {len(overlap)} indices "
                f"(e.g. {sorted(overlap)[:5]}) - this would leak data across stages"
            )
        seen |= current
    if len(seen) != total:
        raise ValueError(f"Splits cover {len(seen)} indices, expected {total}")


def create_split(labels: list[int], seed: int = 42) -> SplitIndices:
    """Build the stratified partition from the dataset's label list.

    Stratifying matters most for `calib`: 1,000 unstratified samples could miss
    classes entirely, skewing INT8 activation ranges toward whatever happened to
    be sampled.
    """
    labels_array = np.asarray(labels)
    if labels_array.size != NUM_CLASSES * IMAGES_PER_CLASS:
        raise ValueError(
            f"Expected {NUM_CLASSES * IMAGES_PER_CLASS} training labels, "
            f"got {labels_array.size}. Is this really the CIFAR-100 train set?"
        )

    rng = np.random.default_rng(seed)
    buckets: dict[str, list[int]] = {name: [] for name in SPLIT_NAMES}

    for class_id in range(NUM_CLASSES):
        class_indices = np.flatnonzero(labels_array == class_id)
        if class_indices.size != IMAGES_PER_CLASS:
            raise ValueError(
                f"Class {class_id} has {class_indices.size} images, expected {IMAGES_PER_CLASS}"
            )
        shuffled = rng.permutation(class_indices)
        offset = 0
        for name in SPLIT_NAMES:
            take = SPLIT_SIZES[name]
            buckets[name].extend(int(i) for i in shuffled[offset : offset + take])
            offset += take

    indices = {name: tuple(sorted(buckets[name])) for name in SPLIT_NAMES}
    _assert_disjoint_and_complete(indices, labels_array.size)
    return SplitIndices(seed=seed, **indices)


def split_fingerprint(split: SplitIndices) -> str:
    """Public accessor for the split's content hash.

    Every trained checkpoint records this, so a model can never be silently
    evaluated against a split it was not trained under (see src/checkpoint.py).
    """
    return _fingerprint(split.as_dict(), split.seed)


def save_split(split: SplitIndices, path: Path = DEFAULT_SPLIT_PATH) -> None:
    indices = split.as_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "seed": split.seed,
        "fingerprint": _fingerprint(indices, split.seed),
        "per_class_counts": SPLIT_SIZES,
        "counts": {name: len(indices[name]) for name in SPLIT_NAMES},
        "indices": {name: list(indices[name]) for name in SPLIT_NAMES},
    }
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")


def load_split(path: Path = DEFAULT_SPLIT_PATH) -> SplitIndices:
    document = json.loads(path.read_text(encoding="utf-8"))
    indices = {name: tuple(document["indices"][name]) for name in SPLIT_NAMES}
    seed = int(document["seed"])

    expected = document.get("fingerprint")
    actual = _fingerprint(indices, seed)
    if expected is not None and expected != actual:
        raise ValueError(
            f"Split file {path} failed its integrity check "
            f"(stored {expected}, computed {actual}). It was edited or truncated; "
            f"delete it and re-run to regenerate, then RETRAIN any model that used it."
        )

    total = sum(len(v) for v in indices.values())
    _assert_disjoint_and_complete(indices, total)
    return SplitIndices(seed=seed, **indices)


def load_or_create_split(
    labels: list[int], path: Path = DEFAULT_SPLIT_PATH, seed: int = 42
) -> SplitIndices:
    """Load the persisted split, creating and saving it on first run."""
    if path.exists():
        return load_split(path)
    split = create_split(labels, seed=seed)
    save_split(split, path)
    return split


def _main() -> None:
    """CLI entry point: `python -m src.data.splits [--verify|--create]`.

    Verification is the default. Without this block the documented check
    `python -m src.data.splits --verify` would exit 0 having checked nothing.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Verify or create the CIFAR-100 split file.")
    parser.add_argument(
        "--verify", action="store_true", help="verify the persisted split (default behavior)"
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="create and save the split if missing (downloads CIFAR-100 on first use)",
    )
    parser.add_argument("--path", type=Path, default=DEFAULT_SPLIT_PATH)
    args = parser.parse_args()

    if not args.path.exists():
        if not args.create:
            raise SystemExit(
                f"{args.path} does not exist. Run with --create to generate it "
                f"(this downloads CIFAR-100 into ./data on first use)."
            )
        # Heavy import kept out of module scope so verify-only runs stay light.
        from torchvision.datasets import CIFAR100

        labels = CIFAR100(root="data", train=True, download=True).targets
        save_split(create_split(labels), args.path)
        print(f"Created {args.path}")

    # load_split runs the fingerprint, disjointness and completeness checks.
    split = load_split(args.path)
    counts = {name: len(split.get(name)) for name in SPLIT_NAMES}
    print(f"OK: {args.path} (seed={split.seed}) counts={counts}")


if __name__ == "__main__":
    _main()
