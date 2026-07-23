"""A sensitivity TESTBED: a CNN with planted fragile and robust blocks.

The original design ("quantization-friendly by construction") is retired: a
model that quantizes cleanly everywhere gives the sensitivity analyzer nothing
to find, so it validated nothing. This redesign plants known-robust and
known-fragile blocks deliberately, which turns the model into an answer key:
Stage 1's analyzer is correct if and only if it rediscovers the plants.

The plants and their mechanisms:

    * stem / stage1 -- dense 3x3 Conv+BN+ReLU. Folded weights are
      well-conditioned and activations one-sided: expected LOW sensitivity.
    * stage2 -- depthwise-separable convolutions. Depthwise weight ranges vary
      wildly per channel, so expected HIGH sensitivity under per-TENSOR
      weights, largely RECOVERED by per-channel. Probes MobileNetV2's failure
      mechanism in isolation.
    * stage3 -- linear bottleneck with a residual add: the projection conv has
      BN but NO activation, so its output range is wide and two-sided, and the
      add mixes mismatched scales. Activation quantization is per-tensor
      regardless of the weight scheme, so this block stays fragile EVEN with
      per-channel weights -- the plant that survives the honest baseline.
    * classifier -- the standard-cookbook "keep FP32" layer; moderate
      sensitivity expected.

EXPECTED_SENSITIVITY below is the machine-readable answer key; Stage 1's
validation test asserts the analyzer's ranking against it.

Named modules throughout (stage2.dw1.dw.conv, ...), never anonymous Sequential
indices: checkpoint keys and ONNX node scopes inherit these names, and
renaming after training would orphan every saved checkpoint, so the names are
settled now, before the first training run.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

from src.models.registry import register_model

# Ground truth the Stage 1 sensitivity analyzer must rediscover.
EXPECTED_SENSITIVITY: dict[str, str] = {
    "stem": "low",
    "stage1": "low",
    "stage2": "high under per-tensor weights; largely recovered by per-channel",
    "stage3": "high regardless of weight scheme (activation-range fragility)",
    "classifier": "moderate (standard-advice layer)",
}


def _conv_bn(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    groups: int = 1,
    relu: bool = True,
) -> nn.Sequential:
    """Conv -> BN (-> ReLU). `relu=False` is the deliberate fragility switch:
    it leaves the output range wide and two-sided."""
    layers: OrderedDict[str, nn.Module] = OrderedDict(
        conv=nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=groups,
            bias=False,
        ),
        bn=nn.BatchNorm2d(out_channels),
    )
    if relu:
        layers["relu"] = nn.ReLU(inplace=True)
    return nn.Sequential(layers)


class LinearBottleneck(nn.Module):
    """Expand 1x1 -> dense 3x3 -> LINEAR 1x1 projection, plus residual add.

    The dense middle conv keeps the weights robust on purpose, so any
    sensitivity measured here is attributable to the activation mechanism
    (linear projection range + mismatched-scale add), not to weights.
    """

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.expand = _conv_bn(channels, hidden, kernel_size=1)
        self.conv = _conv_bn(hidden, hidden, kernel_size=3)
        self.project = _conv_bn(hidden, channels, kernel_size=1, relu=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.project(self.conv(self.expand(x)))


class SensitivityTestbedCNN(nn.Module):
    """Stem + three heterogeneous stages, 32 -> 16 -> 8 -> 4 like the other
    CIFAR models so latency comparisons are not confounded by resolution."""

    def __init__(self, num_classes: int = 100, dropout: float = 0.1) -> None:
        super().__init__()

        self.stem = _conv_bn(3, 64)
        self.stage1 = nn.Sequential(
            OrderedDict(
                conv1=_conv_bn(64, 128),
                conv2=_conv_bn(128, 128),
                pool=nn.MaxPool2d(kernel_size=2, stride=2),
            )
        )
        self.stage2 = nn.Sequential(
            OrderedDict(
                dw1=nn.Sequential(
                    OrderedDict(
                        dw=_conv_bn(128, 128, kernel_size=3, groups=128),
                        pw=_conv_bn(128, 192, kernel_size=1),
                    )
                ),
                dw2=nn.Sequential(
                    OrderedDict(
                        dw=_conv_bn(192, 192, kernel_size=3, groups=192),
                        pw=_conv_bn(192, 192, kernel_size=1),
                    )
                ),
                pool=nn.MaxPool2d(kernel_size=2, stride=2),
            )
        )
        self.stage3 = nn.Sequential(
            OrderedDict(
                bottleneck=LinearBottleneck(192, expansion=2),
                pool=nn.MaxPool2d(kernel_size=2, stride=2),
            )
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(192, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        # torch.flatten rather than x.view(x.size(0), -1): view() on a traced
        # graph emits Shape/Gather/Unsqueeze nodes that clutter the ONNX output
        # and can block operator fusion.
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)


@register_model("custom_cnn")
def build_custom_cnn(num_classes: int = 100, **kwargs: object) -> nn.Module:
    """~1.8M parameters, ~7 MB FP32. Expect ~64-69% top-1 (estimate; the model
    is an instrument first, an accuracy contender second)."""
    return SensitivityTestbedCNN(num_classes=num_classes, **kwargs)  # type: ignore[arg-type]
