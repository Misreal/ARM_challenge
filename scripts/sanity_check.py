"""Phase 0 sanity check: does everything wire together before any GPU time?

    python scripts/sanity_check.py           # models only, no dataset needed
    python scripts/sanity_check.py --data    # also builds the split + loaders

Checks the contract every later stage assumes (src/models/registry.py:8):
each registered builder takes `num_classes`, accepts a (N, 3, 32, 32) tensor,
and returns raw logits of shape (N, num_classes) with no softmax tail.

The softmax check is not pedantry. A softmax on the end of the graph would be
quantized like any other op, its output range would be clamped to [0, 1] at
INT8 resolution, and every accuracy number in Stage 1 would be measured through
that distortion -- while still looking plausible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Running `python scripts/sanity_check.py` puts scripts/ on sys.path, not the
# repo root, so `import src.*` would fail. Prepend the root before any src
# import. (train.py needs no such bootstrap -- it already lives at the root.)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

import src.models  # noqa: F401,E402  -- runs the @register_model decorators
from src.models.registry import available_models, build_model  # noqa: E402
from src.utils import count_parameters, resolve_device  # noqa: E402

BATCH = 4
NUM_CLASSES = 100


def check_models(device: torch.device) -> None:
    dummy = torch.randn(BATCH, 3, 32, 32, device=device)
    print(f"{'model':24} {'params':>12} {'MB fp32':>9}  output")
    for name in available_models():
        model = build_model(name, num_classes=NUM_CLASSES).to(device).eval()
        with torch.inference_mode():
            logits = model(dummy)

        if logits.shape != (BATCH, NUM_CLASSES):
            raise SystemExit(f"{name}: expected {(BATCH, NUM_CLASSES)}, got {tuple(logits.shape)}")
        # Raw logits sum to an arbitrary value; probabilities would sum to ~1.
        if torch.allclose(logits.sum(dim=1), torch.ones(BATCH, device=device), atol=1e-3):
            raise SystemExit(f"{name}: output rows sum to 1 -- softmax must not be in the graph")

        params = count_parameters(model)
        print(
            f"{name:24} {params:>12,} {params * 4 / 1e6:>8.1f}  "
            f"{tuple(logits.shape)} logits in [{logits.min():.2f}, {logits.max():.2f}]"
        )


def check_data() -> None:
    from src.data.loaders import build_loader, build_train_loader, load_base_dataset
    from src.data.splits import DEFAULT_SPLIT_PATH, load_or_create_split, split_fingerprint

    labels = load_base_dataset(train=True).targets
    split = load_or_create_split(labels, path=DEFAULT_SPLIT_PATH)
    counts = {name: len(split.get(name)) for name in ("train", "trainval", "optval", "calib")}
    print(f"\nsplit {split_fingerprint(split)} counts={counts} -> {DEFAULT_SPLIT_PATH}")

    # num_workers=0: this is a one-shot check, and worker spawn on Windows costs
    # more than the check itself.
    images, targets = next(iter(build_train_loader(split, batch_size=8, num_workers=0)))
    print(f"train batch  {tuple(images.shape)} {images.dtype} labels {targets[:8].tolist()}")
    images, _ = next(iter(build_loader(split, "calib", batch_size=8, num_workers=0)))
    print(
        f"calib batch  {tuple(images.shape)} normalized range "
        f"[{images.min():.2f}, {images.max():.2f}] (not [0,1] -- normalization is applied)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", action="store_true", help="also check split + loaders")
    parser.add_argument("--device", default="cpu", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"torch {torch.__version__}  device {device}  cuda={torch.cuda.is_available()}\n")
    check_models(device)
    if args.data:
        check_data()
    print("\nOK")


if __name__ == "__main__":
    main()
