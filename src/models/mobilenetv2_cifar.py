# MobileNetV2 adapted for 32x32 CIFAR input.
#
# Two changes versus the ImageNet configuration:
#
#     * stem convolution stride 2 -> 1
#     * the second inverted-residual stage stride 2 -> 1
#
# Stock MobileNetV2 downsamples 32x total (224 -> 7). On a 32x32 input that ends
# at 1x1 before the classifier, destroying spatial information. The edits above
# reduce total downsampling to 8x, giving a 4x4 final feature map that matches the
# ResNet-CIFAR variant.
#
# Role in this project: the quantization-SENSITIVE probe, and the reason this
# architecture earns its slot. Depthwise convolutions have wildly different weight
# ranges per channel, and the linear (unactivated) bottlenecks produce
# large-dynamic-range activations. Per-TENSOR INT8 typically collapses this model
# by tens of accuracy points, while per-CHANNEL INT8 recovers most of it. That gap
# is precisely the phenomenon the Stage 1 sensitivity analysis exists to detect and
# exploit, so this model is the strongest single piece of evidence for the thesis.

from __future__ import annotations

import torch.nn as nn
from torchvision.models.mobilenetv2 import MobileNetV2

from src.models.registry import register_model

# (expansion factor t, output channels c, repeats n, first-block stride s)
# Stride of the c=24 stage relaxed to 1 for CIFAR; total downsampling becomes 8x.
_CIFAR_INVERTED_RESIDUAL_SETTING: list[list[int]] = [
    [1, 16, 1, 1],
    [6, 24, 2, 1],  # ImageNet uses stride 2 here
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]


@register_model("mobilenetv2_cifar")
def build_mobilenetv2_cifar(
    num_classes: int = 100, width_mult: float = 1.0, **_: object
) -> nn.Module:
    """~2.4M parameters at width_mult=1.0, ~9 MB FP32. Expect ~68-72% top-1."""
    model = MobileNetV2(
        num_classes=num_classes,
        width_mult=width_mult,
        inverted_residual_setting=_CIFAR_INVERTED_RESIDUAL_SETTING,
    )

    # The stem's stride is hardcoded inside MobileNetV2.__init__ and cannot be
    # passed as an argument, so patch it after construction. features[0] is the
    # Conv2dNormActivation block; index [0] within it is the Conv2d itself.
    stem_conv = model.features[0][0]
    assert isinstance(stem_conv, nn.Conv2d), (
        "torchvision changed MobileNetV2's stem layout; re-check this patch"
    )
    stem_conv.stride = (1, 1)

    return model
