"""Stage 0 training entry point.

    python train.py --model resnet18_cifar
    python train.py --model custom_cnn --epochs 30 --tag pilot
    python train.py --model mobilenetv2_cifar --smoke

Stage 0's only job is to produce credible, honestly-measured FP32 baselines and
ONNX inputs for the Stage 1 optimizer. The recipe is deliberately plain and,
once a model is trained, frozen: tuning it in response to downstream
quantization results would make the "post-training optimization" claim false.

Checkpoint selection uses `trainval` (2k images), which the split reserves for
exactly this. `optval` is Stage 1's search signal and the test set is sealed
until Phase 7 -- neither is read here.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn as nn

import src.models  # noqa: F401  -- import for its @register_model side effects
from src.checkpoint import DEFAULT_CHECKPOINT_DIR, TrainingRecipe, save_checkpoint
from src.data.loaders import DEFAULT_DATA_ROOT, build_loader, build_train_loader, load_base_dataset
from src.data.splits import DEFAULT_SPLIT_PATH, load_or_create_split, split_fingerprint
from src.engine import build_param_groups, evaluate, select_amp_dtype, train_one_epoch
from src.models.registry import available_models, build_model
from src.schedule import WarmupCosine
from src.utils import count_parameters, resolve_device, set_seed


@dataclass(frozen=True)
class ModelDefaults:
    """Model-specific training defaults."""

    epochs: int
    lr: float


# MobileNetV2 gets 300 epochs: it is known to still be improving at 200, and
# GPU time is not the binding constraint (training runs on a rented A6000).
MODEL_DEFAULTS: dict[str, ModelDefaults] = {
    "resnet18_cifar": ModelDefaults(epochs=200, lr=0.1),
    "mobilenetv2_cifar": ModelDefaults(epochs=300, lr=0.1),
    "custom_cnn": ModelDefaults(epochs=200, lr=0.1),
}
FALLBACK_DEFAULTS = ModelDefaults(epochs=200, lr=0.1)

MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.1
WARMUP_EPOCHS = 5
SMOKE_EPOCHS = 2
SMOKE_BATCHES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, choices=available_models())
    parser.add_argument("--epochs", type=int, default=None, help="default is per-model")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--lr", type=float, default=None, help="peak LR after warmup; default is per-model"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--warmup-epochs", type=int, default=WARMUP_EPOCHS, help="linear LR warmup length"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="2 epochs x 5 batches: proves the pipeline runs, trains nothing",
    )
    parser.add_argument(
        "--compile", action="store_true", help="torch.compile the model (needs Triton; Linux)"
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="checkpoint name suffix, e.g. --tag pilot, so short runs do not "
        "overwrite the full-length ones",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--amp", default="auto", choices=["auto", "bf16", "off"])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    return parser.parse_args()


def resolve_recipe(args: argparse.Namespace) -> TrainingRecipe:
    """Combine command-line options with model defaults."""
    defaults = MODEL_DEFAULTS.get(args.model, FALLBACK_DEFAULTS)
    epochs = SMOKE_EPOCHS if args.smoke else (args.epochs or defaults.epochs)
    # A 5-epoch warmup inside a 2-epoch smoke run is not a schedule; shrink it.
    warmup = min(args.warmup_epochs, max(1, epochs - 1))
    return TrainingRecipe(
        model=args.model,
        epochs=epochs,
        batch_size=args.batch_size,
        lr=args.lr or defaults.lr,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=warmup,
        label_smoothing=LABEL_SMOOTHING,
        seed=args.seed,
        amp_dtype="pending",  # replaced once the device is known
    )


def maybe_compile(model: nn.Module, enabled: bool) -> nn.Module:
    """Compile the model when requested; fall back to normal mode on failure."""
    if not enabled:
        return model
    try:
        return torch.compile(model)
    except Exception as error:  # Triton is unavailable on Windows
        print(f"[warn] torch.compile unavailable ({error}); continuing eager.")
        return model


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    amp_dtype = select_amp_dtype(device, args.amp)
    if device.type == "cuda":
        # TF32 for the FP32 matmul/conv paths that autocast does not cover.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    recipe = replace(
        resolve_recipe(args),
        amp_dtype="fp32" if amp_dtype is None else str(amp_dtype),
    )

    # The split is created on first run and reused verbatim forever after; its
    # fingerprint is stamped into the checkpoint.
    labels = load_base_dataset(args.data_root, train=True).targets
    split = load_or_create_split(labels, path=args.split_path, seed=args.seed)
    fingerprint = split_fingerprint(split)

    workers = 0 if args.smoke else args.workers
    train_loader = build_train_loader(
        split, root=args.data_root, batch_size=recipe.batch_size, num_workers=workers
    )
    trainval_loader = build_loader(
        split, "trainval", root=args.data_root, batch_size=256, num_workers=workers
    )

    model = build_model(args.model, num_classes=100).to(device)
    # channels_last is a small but free win for conv nets on tensor-core GPUs.
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    trainable = model
    model = maybe_compile(model, args.compile)

    criterion = nn.CrossEntropyLoss(label_smoothing=recipe.label_smoothing)
    optimizer = torch.optim.SGD(
        build_param_groups(trainable, recipe.weight_decay),
        lr=recipe.lr,
        momentum=recipe.momentum,
        nesterov=True,
    )

    max_batches = SMOKE_BATCHES if args.smoke else None
    steps_per_epoch = min(len(train_loader), max_batches or len(train_loader))
    schedule = WarmupCosine(
        base_lr=recipe.lr,
        total_steps=steps_per_epoch * recipe.epochs,
        warmup_steps=steps_per_epoch * recipe.warmup_epochs,
    )

    print(
        f"model={args.model} params={count_parameters(trainable):,} device={device} "
        f"amp={recipe.amp_dtype} epochs={recipe.epochs} bs={recipe.batch_size} "
        f"lr={recipe.lr} steps/epoch={steps_per_epoch} split={fingerprint}"
    )

    history: list[dict[str, float]] = []
    best_top1 = -1.0
    best_top5 = -1.0
    best_epoch = -1
    global_step = 0
    out_name = args.model + (f"_{args.tag}" if args.tag else "")
    checkpoint_path = args.out_dir / f"{out_name}.pt"

    for epoch in range(recipe.epochs):
        started = time.perf_counter()
        epoch_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            lr_schedule=schedule,
            global_step=global_step,
            amp_dtype=amp_dtype,
            max_batches=max_batches,
        )
        global_step = epoch_result.next_step
        val = evaluate(model, trainval_loader, device, criterion, max_batches=max_batches)
        elapsed = time.perf_counter() - started

        history.append(
            {
                "epoch": epoch,
                "lr": epoch_result.last_lr,
                "train_loss": epoch_result.loss,
                "train_top1": epoch_result.top1,
                "trainval_loss": val.loss,
                "trainval_top1": val.top1,
                "trainval_top5": val.top5,
                "seconds": elapsed,
            }
        )

        # Selection on trainval only -- optval belongs to Stage 1, and the test
        # set is sealed until the final report.
        is_best = val.top1 > best_top1
        if is_best:
            best_top1, best_top5, best_epoch = val.top1, val.top5, epoch
            save_checkpoint(
                checkpoint_path,
                trainable,
                recipe,
                split_fingerprint=fingerprint,
                best_epoch=best_epoch,
                trainval_top1=best_top1,
                trainval_top5=best_top5,
                history=history,
            )

        print(
            f"epoch {epoch + 1:3d}/{recipe.epochs}  lr {epoch_result.last_lr:.4f}  "
            f"train {epoch_result.loss:.3f}/{epoch_result.top1:5.2f}%  "
            f"trainval {val.top1:5.2f}%/{val.top5:5.2f}%  "
            f"best {best_top1:5.2f}%{'  *' if is_best else ''}  {elapsed:5.1f}s"
        )

    # Rewrite the sidecar so `history` covers the whole run, not just up to the
    # last improvement. The weights stay those of the best epoch.
    sidecar = checkpoint_path.with_suffix(".json")
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    document["history"] = history
    sidecar.write_text(json.dumps(document, indent=2), encoding="utf-8")

    print(f"\nbest trainval top-1 {best_top1:.2f}% at epoch {best_epoch + 1} -> {checkpoint_path}")
    if args.smoke:
        print("(smoke run: accuracy is meaningless, this only proves the pipeline runs)")


if __name__ == "__main__":
    main()
