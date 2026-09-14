"""Model registry.

Importing the architecture modules here is what runs their `@register_model`
decorators. A new architecture file MUST be imported below or `build_model`
will not find it.
"""

from src.models import custom_cnn, mobilenetv2_cifar, resnet_cifar, vit_cifar  # noqa: F401
from src.models.registry import available_models, build_model, register_model

__all__ = ["available_models", "build_model", "register_model"]
