"""Deployment configuration objects and their content hashes.

`QuantConfig` determines the artifact -- the bytes of the .onnx file.
`RunConfig` determines the execution -- how ONNX Runtime is told to run it.
Splitting them is what makes the search affordable: thread count is a search
dimension, so one config object would re-quantize an identical file once per
thread setting, whereas the artifact cache can key on the quant hash alone.
`canonical_payload` drops fields that don't affect the artifact's bytes before
hashing, so two configs that produce the same file also hash the same --
otherwise the cache would miss and the Pi could measure one file twice under
different names.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import onnxruntime.quantization as ortq

HASH_LENGTH = 16

# Model names, mirrored here rather than read from `src.models.registry`.
#
# The registry imports torch for its `nn.Module` annotations, and the Pi runs
# torch-free by design (see `requirements-pi.txt`). Importing it just to fill in
# argparse `choices=` would make every device-side entry point -- measure,
# baselines, saturation_probe -- unimportable on the machine they exist to run
# on. Duplication is the cheaper cost, and `tests/test_pi_entrypoints.py` fails
# if this list ever drifts from the registry.
EXPORTED_MODELS = ("custom_cnn", "mobilenetv2_cifar", "resnet18_cifar", "vit_cifar")

QUANT_TYPES = ("none", "dynamic", "static")
TENSOR_TYPES = ("int8", "uint8")
CALIBRATION_METHODS = ("minmax", "entropy", "percentile")
GRAPH_OPT_LEVELS = ("disabled", "basic", "extended", "all")

# The fields `RunConfig.hash` covered before the runtime knobs were searchable.
# Frozen so digests minted in Phases 3-6 keep resolving. See `canonical_payload`.
_LEGACY_FIELDS = ("intra_op_num_threads", "graph_optimization_level")

_ORT_TENSOR_TYPE = {
    "int8": ortq.QuantType.QInt8,
    "uint8": ortq.QuantType.QUInt8,
}
_ORT_CALIBRATION_METHOD = {
    "minmax": ortq.CalibrationMethod.MinMax,
    "entropy": ortq.CalibrationMethod.Entropy,
    "percentile": ortq.CalibrationMethod.Percentile,
}


def _content_hash(payload: dict[str, Any]) -> str:
    """Stable short hash of a JSON-serializable payload.

    Mirrors the fingerprint pattern in `src/data/splits.py`: sorted keys so the
    digest does not depend on dict insertion order, and a truncated hex digest
    for readable filenames.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:HASH_LENGTH]


@dataclass(frozen=True)
class QuantConfig:
    """Everything that affects the produced .onnx bytes.

    `excluded_groups` names *blocks* (see `src.quant.groups`), not ONNX nodes.
    Configs therefore stay readable and portable across models, and the
    translation to `nodes_to_exclude` happens once, at quantization time,
    against the graph actually being quantized.
    """

    quant_type: str = "static"
    per_channel: bool = False
    activation_type: str = "uint8"
    weight_type: str = "int8"
    calibration_method: str = "minmax"
    calibration_size: int = 512
    excluded_groups: tuple[str, ...] = ()
    # Halves the weight range to avoid INT8 accumulator saturation on x86 CPUs
    # without VNNI. The Pi 5's Cortex-A76 has dot-product instructions, so this
    # should stay False for the campaign; it is recorded rather than searched so
    # artifacts remain self-describing.
    reduce_range: bool = False

    def __post_init__(self) -> None:
        if self.quant_type not in QUANT_TYPES:
            raise ValueError(f"quant_type must be one of {QUANT_TYPES}, got {self.quant_type!r}")
        if self.activation_type not in TENSOR_TYPES:
            raise ValueError(f"activation_type must be one of {TENSOR_TYPES}")
        if self.weight_type not in TENSOR_TYPES:
            raise ValueError(f"weight_type must be one of {TENSOR_TYPES}")
        if self.calibration_method not in CALIBRATION_METHODS:
            raise ValueError(f"calibration_method must be one of {CALIBRATION_METHODS}")
        if self.calibration_size <= 0:
            raise ValueError("calibration_size must be positive")
        if not isinstance(self.excluded_groups, tuple):
            raise TypeError("excluded_groups must be a tuple so the config stays hashable")
        if len(set(self.excluded_groups)) != len(self.excluded_groups):
            raise ValueError(f"excluded_groups contains duplicates: {self.excluded_groups}")

    @property
    def is_quantized(self) -> bool:
        return self.quant_type != "none"

    def canonical_payload(self) -> dict[str, Any]:
        """Only the fields that actually change this config's output bytes."""
        if self.quant_type == "none":
            return {"quant_type": "none"}

        payload: dict[str, Any] = {
            "quant_type": self.quant_type,
            "per_channel": self.per_channel,
            "weight_type": self.weight_type,
            "reduce_range": self.reduce_range,
            # Sorted so group order can never split one artifact into two hashes.
            "excluded_groups": sorted(self.excluded_groups),
        }
        if self.quant_type == "static":
            # Activation scales and calibration only exist for static quantization;
            # dynamic computes activation ranges at inference time.
            payload["activation_type"] = self.activation_type
            payload["calibration_method"] = self.calibration_method
            payload["calibration_size"] = self.calibration_size
        return payload

    @property
    def hash(self) -> str:
        return _content_hash(self.canonical_payload())

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["excluded_groups"] = list(self.excluded_groups)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuantConfig:
        payload = dict(data)
        payload["excluded_groups"] = tuple(payload.get("excluded_groups", ()))
        return cls(**payload)

    def ort_activation_type(self) -> ortq.QuantType:
        return _ORT_TENSOR_TYPE[self.activation_type]

    def ort_weight_type(self) -> ortq.QuantType:
        return _ORT_TENSOR_TYPE[self.weight_type]

    def ort_calibration_method(self) -> ortq.CalibrationMethod:
        return _ORT_CALIBRATION_METHOD[self.calibration_method]

    def describe(self) -> str:
        """Short human-readable label for logs and result tables."""
        if self.quant_type == "none":
            return "fp32"
        parts = [self.quant_type]
        if self.quant_type == "static":
            parts.append("per-channel" if self.per_channel else "per-tensor")
            parts.append(f"act:{self.activation_type}")
            parts.append(f"{self.calibration_method}/{self.calibration_size}")
        elif self.per_channel:
            parts.append("per-channel")
        if self.excluded_groups:
            parts.append(f"fp32:{'+'.join(sorted(self.excluded_groups))}")
        return " ".join(parts)


@dataclass(frozen=True)
class RunConfig:
    """Everything that affects execution but not the artifact bytes."""

    intra_op_num_threads: int = 4
    graph_optimization_level: str = "all"
    # ORT's reusable memory pool. On trades RAM for allocator latency, off trades
    # it back, which is the only knob here that moves RAM and speed in opposite
    # directions -- every quantization knob moves them together.
    enable_cpu_mem_arena: bool = True
    # Intra-op threads busy-wait between inferences instead of sleeping. Always
    # flattering in a tight benchmark loop, so a win here is not automatically a
    # win in a duty-cycled deployment.
    allow_intra_op_spinning: bool = True

    def __post_init__(self) -> None:
        if self.intra_op_num_threads <= 0:
            raise ValueError("intra_op_num_threads must be positive")
        if self.graph_optimization_level not in GRAPH_OPT_LEVELS:
            raise ValueError(f"graph_optimization_level must be one of {GRAPH_OPT_LEVELS}")

    def canonical_payload(self) -> dict[str, Any]:
        """The two original fields, plus any newer knob set away from its default.

        Hashing every field unconditionally would change the digest of a plain
        4-thread config each time a knob is added, orphaning every measurement
        already cached on the Pi and every result already joined to a hash. The
        asymmetry is the price of that continuity, so `_LEGACY_FIELDS` is frozen:
        it records what the hash meant before the runtime knobs existed.
        """
        defaults = RunConfig()
        payload = {name: getattr(self, name) for name in _LEGACY_FIELDS}
        payload.update(
            {
                name: value
                for name, value in asdict(self).items()
                if name not in _LEGACY_FIELDS and value != getattr(defaults, name)
            }
        )
        return payload

    @property
    def hash(self) -> str:
        return _content_hash(self.canonical_payload())

    def describe(self) -> str:
        """Short label for logs and result tables, mirroring `QuantConfig.describe`."""
        parts = [f"{self.intra_op_num_threads}t", f"opt:{self.graph_optimization_level}"]
        if not self.enable_cpu_mem_arena:
            parts.append("no-arena")
        if not self.allow_intra_op_spinning:
            parts.append("no-spin")
        return " ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunConfig:
        # Tolerate fields this version does not know: the Pi is code-synced
        # before every campaign, but a cached result written by a newer PC must
        # not crash an older reader.
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


@dataclass(frozen=True)
class DeploymentConfig:
    """One point in the search space: an artifact plus how it is run."""

    quant: QuantConfig = field(default_factory=QuantConfig)
    run: RunConfig = field(default_factory=RunConfig)

    @property
    def hash(self) -> str:
        """Identity for *measurements*. The artifact cache uses `quant.hash`."""
        return _content_hash({"quant": self.quant.hash, "run": self.run.hash})

    def as_dict(self) -> dict[str, Any]:
        return {
            "quant": self.quant.as_dict(),
            "run": self.run.as_dict(),
            "quant_hash": self.quant.hash,
            "config_hash": self.hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeploymentConfig:
        return cls(
            quant=QuantConfig.from_dict(data["quant"]),
            run=RunConfig.from_dict(data["run"]),
        )
