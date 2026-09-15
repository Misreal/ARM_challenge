# The CIFAR-100 preprocessing contract, in numpy.
#
# This module is the single source of truth for the normalization constants.
# `src.data.loaders` imports them from here rather than the other way round: the
# torch-free side has to own them, because it is the side that ships to the Pi.
#
# Normalization is deliberately NOT baked into the exported ONNX graph, so every
# consumer of a model artifact must apply exactly this transform first.

from __future__ import annotations

import numpy as np

# Channel statistics of the CIFAR-100 training set.
CIFAR100_MEAN: tuple[float, float, float] = (0.5071, 0.4865, 0.4409)
CIFAR100_STD: tuple[float, float, float] = (0.2673, 0.2564, 0.2762)


def normalize_uint8_nchw(images: np.ndarray) -> np.ndarray:
    """Convert uint8 NCHW images to the normalized float32 the graphs expect.

    Mirrors torchvision `ToTensor()` then `Normalize(mean, std)`: scale to
    [0, 1], subtract the per-channel mean, divide by the per-channel standard
    deviation. `src.data.export_pi_data` asserts this reproduces the torchvision
    pipeline bit-for-bit before writing any bundle.
    """
    if images.dtype != np.uint8:
        raise TypeError(f"Expected uint8 images, got {images.dtype}")
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected NCHW images with 3 channels, got shape {images.shape}")

    mean = np.asarray(CIFAR100_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(CIFAR100_STD, dtype=np.float32).reshape(1, 3, 1, 1)
    return (images.astype(np.float32) / 255.0 - mean) / std
