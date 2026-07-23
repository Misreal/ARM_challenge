"""ResNet-18 adapted for 32x32 CIFAR input.

Torchvision's ImageNet ResNet-18 opens with a 7x7 stride-2 convolution followed
by a stride-2 max-pool, downsampling 224 -> 56 before the first residual block.
Applied to a 32x32 image that collapses the map to 8x8 immediately and throws
away most of the spatial signal, costing roughly 10 accuracy points on CIFAR.

The standard fix -- a 3x3 stride-1 stem with the max-pool removed -- keeps the
map at 32x32 into layer1 and yields the familiar 32 -> 16 -> 8 -> 4 progression.

Role in this project: the quantization-ROBUST anchor. It is heavily
over-parameterized for 100 classes, so INT8 barely dents it. That makes it the
control against which MobileNetV2's INT8 sensitivity is measured.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18, resnet34

from src.models.registry import register_model


def _adapt_stem_for_cifar(model: nn.Module) -> nn.Module:
    """Replace the ImageNet stem with a CIFAR-appropriate one."""
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Identity rather than deletion: keeps the forward() call sites intact and
    # exports as a no-op that constant folding removes from the ONNX graph.
    model.maxpool = nn.Identity()
    return model


@register_model("resnet18_cifar")
def build_resnet18_cifar(num_classes: int = 100, **_: object) -> nn.Module:
    """~11.2M parameters, ~45 MB FP32. Expect ~76-78% top-1 at 200 epochs."""
    return _adapt_stem_for_cifar(resnet18(weights=None, num_classes=num_classes))


@register_model("resnet34_cifar")
def build_resnet34_cifar(num_classes: int = 100, **_: object) -> nn.Module:
    """~21.3M parameters. Registered as a ready-made capacity comparison; not
    part of the core three, but costs nothing to keep available."""
    return _adapt_stem_for_cifar(resnet34(weights=None, num_classes=num_classes))
