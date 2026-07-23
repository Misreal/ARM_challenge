"""CIFAR-100 transforms and dataloader construction.

Normalization is applied *here*, in the data pipeline, and deliberately NOT
baked into the exported ONNX graph. The graph therefore expects an already
normalized float tensor. The constants below are written into every baseline
report so the Raspberry Pi benchmark agent preprocesses identically -- a
mismatch here shows up as an unexplained accuracy drop on device.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100

from src.data.splits import SplitIndices

# Channel statistics of the CIFAR-100 training set.
CIFAR100_MEAN: tuple[float, float, float] = (0.5071, 0.4865, 0.4409)
CIFAR100_STD: tuple[float, float, float] = (0.2673, 0.2564, 0.2762)

DEFAULT_DATA_ROOT = Path("data")


def build_transforms(train: bool) -> transforms.Compose:
    """Standard CIFAR augmentation. Kept intentionally plain -- Stage 0's job is
    a credible baseline, not a SOTA accuracy chase (see CLAUDE.md)."""
    normalize = transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    if not train:
        return transforms.Compose([transforms.ToTensor(), normalize])
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )


def load_base_dataset(
    root: Path = DEFAULT_DATA_ROOT, train: bool = True, augment: bool = False
) -> CIFAR100:
    return CIFAR100(
        root=str(root),
        train=train,
        download=True,
        transform=build_transforms(train=augment),
    )


def build_train_loader(
    split: SplitIndices,
    root: Path = DEFAULT_DATA_ROOT,
    batch_size: int = 128,
    num_workers: int = 4,
) -> DataLoader:
    """Augmented loader over the 44k training subset."""
    dataset = load_base_dataset(root, train=True, augment=True)
    return DataLoader(
        Subset(dataset, list(split.train)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=num_workers > 0,
    )


def build_loader(
    split: SplitIndices,
    name: str,
    root: Path = DEFAULT_DATA_ROOT,
    batch_size: int = 256,
    num_workers: int = 4,
) -> DataLoader:
    """Non-augmented, non-shuffled loader for any named split.

    `name="test"` bypasses the split entirely and returns the full held-out test
    set. Reserve it for final evaluation.
    """
    if name == "test":
        dataset = load_base_dataset(root, train=False, augment=False)
        subset: torch.utils.data.Dataset = dataset
    else:
        dataset = load_base_dataset(root, train=True, augment=False)
        subset = Subset(dataset, list(split.get(name)))

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
