"""Checkpoint I/O that binds weights to the exact split and recipe that made them.

Mirrors the `src/data/splits.py` pattern: a self-describing document plus an
integrity check on load. The check that matters is the **split fingerprint**.
Stage 1 measures every candidate's accuracy on `optval` and compares it to a
frozen baseline; if a checkpoint were ever trained under a different split,
`optval` might contain images that model was trained on, and the accuracy
constraint -- the one hard filter in the whole system -- would be measuring
memorisation. Recording the fingerprint makes that failure loud instead of
silent.

Two files are written per checkpoint:

    <name>.pt     weights + full metadata (needs torch to read)
    <name>.json   the same metadata minus weights, plus the .pt's SHA-256

The sidecar exists so Phase 2 can stamp a checkpoint hash into the ONNX
metadata, and so training provenance stays greppable without loading torch.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

DEFAULT_CHECKPOINT_DIR = Path("artifacts/checkpoints")


@dataclass(frozen=True)
class TrainingRecipe:
    """Everything needed to reproduce a training run.

    Frozen by invariant once a model is trained: retuning the recipe after
    seeing quantization results turns "post-training optimization" into
    quantization-aware training by the back door (PLAN.md DO-NOT #11).
    """

    model: str
    epochs: int
    batch_size: int
    lr: float
    momentum: float
    weight_decay: float
    warmup_epochs: int
    label_smoothing: float
    seed: int
    amp_dtype: str
    optimizer: str = "sgd_nesterov"
    schedule: str = "warmup_cosine"
    augmentation: str = "randomcrop32_pad4_reflect+hflip"


@dataclass(frozen=True)
class CheckpointDocument:
    """A loaded checkpoint: weights plus the provenance that validates them."""

    state_dict: dict[str, torch.Tensor]
    recipe: TrainingRecipe
    split_fingerprint: str
    best_epoch: int
    trainval_top1: float
    trainval_top5: float
    history: list[dict[str, float]] = field(default_factory=list)
    package_versions: dict[str, str] = field(default_factory=dict)


def unwrap_compiled(model: nn.Module) -> nn.Module:
    """Return the eager module behind a `torch.compile` wrapper.

    `torch.compile` returns an OptimizedModule whose state_dict keys are all
    prefixed `_orig_mod.`. Saving those keys produces a checkpoint that will
    not load into a plain model -- and the failure surfaces days later, at
    export time. Always save through this.
    """
    return getattr(model, "_orig_mod", model)


def _package_versions() -> dict[str, str]:
    # Newer PyTorch represents __version__ as TorchVersion, a str subclass
    # that weights_only=True will not deserialize unless it is allowlisted.
    # Store plain strings so checkpoints contain only ordinary metadata.
    versions = {"torch": str(torch.__version__)}
    try:  # torchvision is a training-only dependency; the Pi never has it.
        import torchvision

        versions["torchvision"] = str(torchvision.__version__)
    except ImportError:
        pass
    return versions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    path: Path,
    model: nn.Module,
    recipe: TrainingRecipe,
    split_fingerprint: str,
    best_epoch: int,
    trainval_top1: float,
    trainval_top5: float,
    history: list[dict[str, float]] | None = None,
) -> Path:
    """Write `<path>` and its `.json` sidecar. Returns the sidecar path."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # .float().cpu() is not cosmetic: exporting from a half-precision state dict
    # silently degrades the FP32 baseline every later number is compared to.
    state_dict = {
        key: value.detach().float().cpu()
        for key, value in unwrap_compiled(model).state_dict().items()
    }

    metadata: dict[str, Any] = {
        "recipe": asdict(recipe),
        "split_fingerprint": split_fingerprint,
        "best_epoch": best_epoch,
        "trainval_top1": trainval_top1,
        "trainval_top5": trainval_top5,
        "history": history or [],
        "package_versions": _package_versions(),
    }

    torch.save({"state_dict": state_dict, **metadata}, path)

    sidecar = path.with_suffix(".json")
    sidecar.write_text(
        json.dumps({"checkpoint": path.name, "sha256": _sha256(path), **metadata}, indent=2),
        encoding="utf-8",
    )
    return sidecar


def load_checkpoint(
    path: Path, expected_split_fingerprint: str | None = None
) -> CheckpointDocument:
    """Load a checkpoint, refusing one trained under a different data split."""
    # weights_only=True: the document holds only tensors and plain Python types,
    # so there is no reason to allow arbitrary pickle execution.
    # Older checkpoints may contain PyTorch's TorchVersion object in their
    # package-version metadata. It is a trusted PyTorch built-in, so allow it
    # while retaining weights_only=True for every other object.
    try:
        from torch.torch_version import TorchVersion

        with torch.serialization.safe_globals([TorchVersion]):
            payload = torch.load(path, map_location="cpu", weights_only=True)
    except (ImportError, AttributeError):  # PyTorch versions before safe_globals
        payload = torch.load(path, map_location="cpu", weights_only=True)

    fingerprint = payload["split_fingerprint"]
    if expected_split_fingerprint is not None and fingerprint != expected_split_fingerprint:
        raise ValueError(
            f"{path} was trained under split {fingerprint}, but the split file on "
            f"disk is {expected_split_fingerprint}. Its 'train' images may now be "
            f"inside optval or calib, which would invalidate every accuracy number "
            f"downstream. Restore the original split file or RETRAIN."
        )

    return CheckpointDocument(
        state_dict=payload["state_dict"],
        recipe=TrainingRecipe(**payload["recipe"]),
        split_fingerprint=fingerprint,
        best_epoch=int(payload["best_epoch"]),
        trainval_top1=float(payload["trainval_top1"]),
        trainval_top5=float(payload["trainval_top5"]),
        history=list(payload.get("history", [])),
        package_versions=dict(payload.get("package_versions", {})),
    )
