"""Per-group INT8 probes: what one block costs alone, and what it costs in context.

Every probe is an ordinary `QuantConfig` with a different `excluded_groups`, so
the sweep reuses the Phase 4 toolkit verbatim rather than introducing a second
quantization path whose results would not be comparable.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.portable.onnx_eval import collect_logits
from src.quant.config import QuantConfig, RunConfig
from src.quant.quantize import (
    DEFAULT_CACHE_DIR,
    DEFAULT_ONNX_DIR,
    QuantizationFailure,
    build_artifact,
    build_session,
)
from src.sensitivity.metrics import ComparisonResult, compare

# Weight schemes are probed in both settings because custom_cnn's answer key
# predicts a group that is fragile under one and recovered by the other; a
# single-scheme sweep could not observe that at all.
SCHEMES: dict[str, bool] = {"per_tensor": False, "per_channel": True}

LEAVE_ONE_OUT = "leave_one_out"
ISOLATE = "isolate"
PROBE_KINDS = (LEAVE_ONE_OUT, ISOLATE)


@dataclass(frozen=True)
class Probe:
    """One (group, question, weight scheme) measurement."""

    group: str
    kind: str
    scheme: str

    def __post_init__(self) -> None:
        if self.kind not in PROBE_KINDS:
            raise ValueError(f"kind must be one of {PROBE_KINDS}, got {self.kind!r}")
        if self.scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {sorted(SCHEMES)}, got {self.scheme!r}")

    @property
    def key(self) -> str:
        return f"{self.scheme}/{self.kind}/{self.group}"

    def as_dict(self) -> dict[str, str]:
        return {"group": self.group, "kind": self.kind, "scheme": self.scheme}


def probe_config(probe: Probe, all_groups: tuple[str, ...]) -> QuantConfig:
    """The static-INT8 config that answers `probe`.

    `all_groups` must be every key in the group map, including the unscoped
    bucket: an isolate probe that left those nodes quantized would not actually
    be isolating anything.
    """
    if probe.group not in all_groups:
        raise KeyError(f"Group {probe.group!r} is not in {sorted(all_groups)}")

    if probe.kind == LEAVE_ONE_OUT:
        excluded: tuple[str, ...] = (probe.group,)
    else:
        excluded = tuple(sorted(set(all_groups) - {probe.group}))

    return QuantConfig(
        quant_type="static",
        per_channel=SCHEMES[probe.scheme],
        excluded_groups=excluded,
    )


def anchor_configs() -> "OrderedDict[str, QuantConfig]":
    """FP32 reference plus the two full-INT8 corners the probes sit between."""
    anchors: OrderedDict[str, QuantConfig] = OrderedDict()
    anchors["fp32"] = QuantConfig(quant_type="none")
    for scheme, per_channel in SCHEMES.items():
        anchors[f"static_{scheme}"] = QuantConfig(quant_type="static", per_channel=per_channel)
    return anchors


def probes_for(groups: tuple[str, ...], schemes: tuple[str, ...] = tuple(SCHEMES)) -> list[Probe]:
    """Every probe in a stable order, so a resumed sweep continues where it stopped."""
    return [
        Probe(group=group, kind=kind, scheme=scheme)
        for scheme in schemes
        for kind in PROBE_KINDS
        for group in groups
    ]


@dataclass(frozen=True)
class ProbeOutcome:
    """One probe's measurement, or the reason it could not be made."""

    status: str
    config: QuantConfig
    metrics: ComparisonResult | None = None
    artifact_bytes: int | None = None
    qdq_nodes: int | None = None
    seconds: float = 0.0
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "config": self.config.as_dict(),
            "quant_hash": self.config.hash,
            "seconds": round(self.seconds, 2),
        }
        if self.metrics is not None:
            payload["metrics"] = self.metrics.as_dict()
        if self.artifact_bytes is not None:
            payload["bytes"] = self.artifact_bytes
        if self.qdq_nodes is not None:
            payload["qdq_nodes"] = self.qdq_nodes
        if self.error is not None:
            payload["error"] = self.error
        return payload


def logits_for_config(
    model: str,
    config: QuantConfig,
    images_uint8: np.ndarray,
    run: RunConfig,
    onnx_dir: Path = DEFAULT_ONNX_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    bundle_dir: Path | None = None,
    group_map: "OrderedDict[str, tuple[str, ...]] | None" = None,
    keep_artifact: bool = False,
) -> tuple[np.ndarray, int, int]:
    """Build `config`, score it, and return (logits, bytes, qdq_nodes).

    The artifact is deleted afterwards by default. A full sweep materializes
    over a hundred graphs of up to 45 MB, which would fill the Pi's card; the
    metrics are the durable output, not the .onnx files.
    """
    artifact = build_artifact(
        model,
        config,
        onnx_dir=onnx_dir,
        cache_dir=cache_dir,
        bundle_dir=bundle_dir,
        group_map=group_map,
    )
    try:
        session = build_session(artifact.path, run)
        logits = collect_logits(session, images_uint8)
        return logits, artifact.bytes, artifact.int8_op_count
    finally:
        if not keep_artifact:
            artifact.path.unlink(missing_ok=True)


def evaluate_config(
    model: str,
    config: QuantConfig,
    images_uint8: np.ndarray,
    labels: np.ndarray,
    reference_logits: np.ndarray,
    reference_top1: float,
    run: RunConfig,
    **kwargs: Any,
) -> ProbeOutcome:
    """Score one config against the reference, recording a failure as a result."""
    started = time.perf_counter()
    try:
        logits, artifact_bytes, qdq_nodes = logits_for_config(
            model, config, images_uint8, run, **kwargs
        )
    except QuantizationFailure as error:
        # An infeasible config is a finding about the search space, not a crash.
        return ProbeOutcome(
            status="failed",
            config=config,
            seconds=time.perf_counter() - started,
            error=str(error),
        )

    return ProbeOutcome(
        status="ok",
        config=config,
        metrics=compare(reference_logits, logits, labels, reference_top1=reference_top1),
        artifact_bytes=artifact_bytes,
        qdq_nodes=qdq_nodes,
        seconds=time.perf_counter() - started,
    )
