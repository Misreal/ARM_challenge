"""The two loops: one training epoch, and evaluation.

Kept separate from `train.py` because Stage 0 is not their only caller --
Phase 2's export-parity gate and Phase 7's final test-set evaluation both need
`evaluate()` with byte-identical semantics. `evaluate()` always runs full FP32
even though training uses bf16 autocast, since it produces the baseline the
Stage 1 accuracy constraint is measured against and autocast shifts accuracy
in the third decimal place. bf16 rather than fp16 throughout: bf16 keeps
FP32's exponent range so gradients can't underflow and no GradScaler is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import AverageMeter, topk_correct


@dataclass(frozen=True)
class EvalResult:
    """Accuracies as percentages, so they print and compare like the papers."""

    loss: float
    top1: float
    top5: float
    n_samples: int


@dataclass(frozen=True)
class EpochResult:
    loss: float
    top1: float
    n_samples: int
    last_lr: float
    next_step: int


def select_amp_dtype(device: torch.device, requested: str = "auto") -> torch.dtype | None:
    """Resolve the autocast dtype, or None to train in full FP32.

    `auto` picks bf16 on CUDA hardware that supports it and FP32 everywhere
    else. CPU bf16 autocast exists but is slower than FP32 for these model
    sizes, so it is not used.
    """
    if requested == "off":
        return None
    if requested == "bf16" or requested == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if requested == "bf16":
            raise RuntimeError("bf16 requested but this device does not support it")
        return None
    raise ValueError(f"Unknown amp mode {requested!r}; expected auto|bf16|off")


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, object]]:
    """Split parameters into decayed and non-decayed groups.

    Weight decay is applied to conv/linear weight matrices only. Applying it to
    BatchNorm scales and biases shrinks them toward zero, which fights the
    normalisation itself and costs roughly half a point of top-1 on CIFAR. This
    is standard practice in the recipes whose published accuracies this project
    expects to reproduce, not an optimisation experiment.
    """
    decayed: list[nn.Parameter] = []
    plain: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # BN weights/biases and all biases are 1-D; conv/linear weights are not.
        if param.ndim <= 1 or name.endswith(".bias"):
            plain.append(param)
        else:
            decayed.append(param)
    return [
        {"params": decayed, "weight_decay": weight_decay},
        {"params": plain, "weight_decay": 0.0},
    ]


def _batches(loader: DataLoader, max_batches: int | None) -> Iterator[tuple[int, tuple]]:
    """Yield (index, batch), stopping early when `max_batches` is set (smoke runs)."""
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            return
        yield index, batch


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    lr_schedule: Callable[[int], float],
    global_step: int,
    amp_dtype: torch.dtype | None = None,
    max_batches: int | None = None,
    grad_clip: float | None = None,
) -> EpochResult:
    """Run one pass over `loader`, stepping the LR schedule per batch."""
    model.train()
    loss_meter = AverageMeter()
    correct = 0
    seen = 0
    step = global_step
    last_lr = 0.0

    for _, (images, targets) in _batches(loader, max_batches):
        last_lr = lr_schedule(step)
        for group in optimizer.param_groups:
            group["lr"] = last_lr

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
        ):
            logits = model(images)
            loss = criterion(logits, targets)

        # set_to_none frees the gradient buffers rather than filling them with
        # zeros: slightly faster, and it makes an un-stepped parameter obvious.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = targets.size(0)
        loss_meter.update(loss.item(), batch_size)
        # Accuracy from the autocast logits is fine here -- it is a progress
        # readout, not a reported number.
        correct += topk_correct(logits.detach().float(), targets, ks=(1,))[1]
        seen += batch_size
        step += 1

    return EpochResult(
        loss=loss_meter.avg,
        top1=100.0 * correct / seen if seen else 0.0,
        n_samples=seen,
        last_lr=last_lr,
        next_step=step,
    )


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
    max_batches: int | None = None,
) -> EvalResult:
    """Full-FP32 evaluation. See the module docstring for why no autocast."""
    model.eval()
    loss_meter = AverageMeter()
    correct = {1: 0, 5: 0}
    seen = 0

    for _, (images, targets) in _batches(loader, max_batches):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        if criterion is not None:
            loss_meter.update(criterion(logits, targets).item(), targets.size(0))

        hits = topk_correct(logits, targets, ks=(1, 5))
        for k in correct:
            correct[k] += hits[k]
        seen += targets.size(0)

    if seen == 0:
        raise ValueError("evaluate() saw zero samples; is the loader empty?")

    return EvalResult(
        loss=loss_meter.avg,
        top1=100.0 * correct[1] / seen,
        top5=100.0 * correct[5] / seen,
        n_samples=seen,
    )
