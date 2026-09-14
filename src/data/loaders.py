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

# Re-exported from the torch-free package that ships to the Pi. The constants
# have to live on that side so the device is never asked to duplicate them.
from src.portable.preprocess import CIFAR100_MEAN, CIFAR100_STD

__all__ = [
    "AUGMENTATION_IDS",
    "CIFAR100_MEAN",
    "CIFAR100_STD",
    "DEFAULT_DATA_ROOT",
    "augmentation_id",
    "build_loader",
    "build_train_loader",
    "build_transforms",
    "load_base_dataset",
]

DEFAULT_DATA_ROOT = Path("data")


# Recipe name -> the string stamped into the checkpoint's TrainingRecipe, so a
# trained model always carries the augmentation it actually saw.
AUGMENTATION_IDS: dict[str, str] = {
    "standard": "randomcrop32_pad4_reflect+hflip",
    "heavy": "randomcrop32_pad4_reflect+hflip+randaugment2x9+erasing0.25",
}


def augmentation_id(recipe: str) -> str:
    """Provenance string for an augmentation recipe name."""
    if recipe not in AUGMENTATION_IDS:
        raise ValueError(f"Unknown augmentation {recipe!r}; known: {sorted(AUGMENTATION_IDS)}")
    return AUGMENTATION_IDS[recipe]


def build_transforms(train: bool, recipe: str = "standard") -> transforms.Compose:
    """CIFAR augmentation. `standard` is kept intentionally plain -- Stage 0's job
    is a credible baseline, not a SOTA accuracy chase (see CLAUDE.md).

    `heavy` adds RandAugment and random erasing, and exists for `vit_cifar`
    alone. That is not a SOTA chase either: a from-scratch transformer has no
    convolutional prior, and under the plain recipe it overfits 44k images
    badly enough that its FP32 baseline would not be credible. The CNNs keep
    `standard`, so nothing already trained is affected.
    """
    augmentation_id(recipe)  # reject an unknown name before building anything
    normalize = transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    if not train:
        return transforms.Compose([transforms.ToTensor(), normalize])
    if recipe == "heavy":
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                normalize,
                # After normalize, so erased pixels are the channel mean (zero
                # in normalized space) rather than a raw-pixel constant.
                transforms.RandomErasing(p=0.25),
            ]
        )
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )


def load_base_dataset(
    root: Path = DEFAULT_DATA_ROOT,
    train: bool = True,
    augment: bool = False,
    recipe: str = "standard",
) -> CIFAR100:
    return CIFAR100(
        root=str(root),
        train=train,
        download=True,
        transform=build_transforms(train=augment, recipe=recipe),
    )


def build_train_loader(
    split: SplitIndices,
    root: Path = DEFAULT_DATA_ROOT,
    batch_size: int = 128,
    num_workers: int = 4,
    recipe: str = "standard",
) -> DataLoader:
    """Augmented loader over the 44k training subset."""
    dataset = load_base_dataset(root, train=True, augment=True, recipe=recipe)
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
