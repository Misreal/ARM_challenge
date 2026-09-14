"""Small shared helpers: seeding, metric accumulation, model introspection."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and Torch RNGs.

    `deterministic=True` also pins cuDNN algorithm selection. It costs
    throughput, so it is off by default -- training runs do not need to be
    bit-reproducible, but the dataset split does (that one is seeded
    separately and persisted to disk).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def resolve_device(requested: str = "auto") -> torch.device:
    """Map 'auto'/'cuda'/'mps'/'cpu' to a concrete device, falling back gracefully.

    CUDA outranks MPS under `auto` because the campaign's recorded runs were
    trained on a rented GPU; MPS is the Apple Silicon path, which is slower but
    an order of magnitude better than this machine's CPU.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False")
    return torch.device(requested)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Total parameter count. Reported in baselines, never used as a speed proxy."""
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


class AverageMeter:
    """Running mean of a scalar, weighted by batch size."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0


def topk_correct(
    logits: torch.Tensor, targets: torch.Tensor, ks: tuple[int, ...] = (1, 5)
) -> dict[int, int]:
    """Count top-k correct predictions for each k. Returns counts, not rates,
    so callers can accumulate across batches of differing size."""
    maxk = max(ks)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    hits = pred.eq(targets.view(-1, 1).expand_as(pred))
    return {k: int(hits[:, :k].any(dim=1).sum().item()) for k in ks}
