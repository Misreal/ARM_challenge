"""Name -> builder registry for CIFAR-100 architectures.

Adding an architecture is two steps:

    1. Create `src/models/<name>.py` with a builder decorated `@register_model("<name>")`
    2. Import it in `src/models/__init__.py` so the decorator runs

Every registered builder must honour the same contract, because Stage 1 assumes
it uniformly across all exported models:

    * accepts `num_classes: int`
    * returns an `nn.Module` taking a (N, 3, 32, 32) normalized float tensor
    * returns raw logits of shape (N, num_classes) -- no softmax; the ONNX graph
      ends at the linear layer so quantization sees a clean tail
    * uses modules with stable, meaningful names, so per-layer sensitivity
      analysis can map ONNX nodes back to blocks
"""

from __future__ import annotations

from typing import Callable, Protocol

import torch.nn as nn


class ModelBuilder(Protocol):
    def __call__(self, num_classes: int = ..., **kwargs: object) -> nn.Module: ...


_REGISTRY: dict[str, ModelBuilder] = {}


def register_model(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    """Decorator registering a builder under `name`."""

    def decorator(builder: ModelBuilder) -> ModelBuilder:
        if name in _REGISTRY:
            raise ValueError(
                f"Model {name!r} is already registered by "
                f"{_REGISTRY[name].__module__}. Pick a distinct name."
            )
        _REGISTRY[name] = builder
        return builder

    return decorator


def available_models() -> list[str]:
    return sorted(_REGISTRY)


def build_model(name: str, num_classes: int = 100, **kwargs: object) -> nn.Module:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {available_models()}")
    return _REGISTRY[name](num_classes=num_classes, **kwargs)
