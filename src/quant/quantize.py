"""Turn a QuantConfig into a validated ONNX artifact.

Source graph selection matters
------------------------------
Quantized variants are always built from `<model>_quant_ready.onnx`, the
shape-inferred and fused graph. Quantizing the deployable export instead leaves
unfused subgraphs to be wrapped in QDQ nodes -- slower *and* less accurate, and
PLAN.md's DO-NOT #4. The FP32 baseline is the opposite: it uses the deployable
export, because that is the file one would actually ship. Export verified the
two are numerically identical (max logit delta 0), so the asymmetry costs
nothing in comparability.

Caching
-------
Artifacts are keyed by `QuantConfig.hash` alone, not by the full deployment
config: runtime settings do not change the bytes. A cache hit skips both the
quantization and its gates, which is what makes a multi-hundred-trial search
affordable.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path

import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, quantize_static

from src.quant.calibration import build_calibration_reader
from src.quant.config import QuantConfig, RunConfig
from src.quant.groups import build_group_map, nodes_for_groups

DEFAULT_ONNX_DIR = Path("artifacts/onnx")
DEFAULT_CACHE_DIR = Path("artifacts/quant")


@dataclass(frozen=True)
class ModelPaths:
    """The two source graphs Phase 2 produced for one model."""

    name: str
    deployable: Path
    quant_ready: Path

    @classmethod
    def resolve(cls, model: str, onnx_dir: Path = DEFAULT_ONNX_DIR) -> ModelPaths:
        deployable = onnx_dir / f"{model}.onnx"
        quant_ready = onnx_dir / f"{model}_quant_ready.onnx"
        for path in (deployable, quant_ready):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {path}. Run: python -m src.export_onnx --model {model} --device cpu"
                )
        return cls(name=model, deployable=deployable, quant_ready=quant_ready)


@dataclass(frozen=True)
class Artifact:
    """A built, gate-passing model file."""

    path: Path
    config: QuantConfig
    bytes: int
    sha256: str
    operators: dict[str, int]
    build_seconds: float
    from_cache: bool

    @property
    def int8_op_count(self) -> int:
        """QDQ node count -- zero on a config that claimed to quantize means the
        exclusions covered everything, which is worth catching in a report."""
        return self.operators.get("QuantizeLinear", 0) + self.operators.get("DequantizeLinear", 0)

    def as_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "bytes": self.bytes,
            "sha256": self.sha256,
            "operators": self.operators,
            "build_seconds": round(self.build_seconds, 3),
            "from_cache": self.from_cache,
            "config": self.config.as_dict(),
            "quant_hash": self.config.hash,
        }


class QuantizationFailure(Exception):
    """A candidate that cannot be built or loaded.

    Raised rather than returned so callers must decide explicitly; Phase 6
    catches it and records an infeasible trial instead of crashing the study.
    """


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _operator_inventory(path: Path) -> dict[str, int]:
    model = onnx.load(str(path))
    return dict(sorted(Counter(node.op_type for node in model.graph.node).items()))


def build_session(path: Path, run: RunConfig) -> ort.InferenceSession:
    """Create an ORT CPU session honouring the runtime config."""
    levels = {
        "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    options = ort.SessionOptions()
    options.intra_op_num_threads = run.intra_op_num_threads
    options.graph_optimization_level = levels[run.graph_optimization_level]
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _run_gates(path: Path) -> None:
    """The cheap staged-evaluation gates: structurally valid, and it loads.

    Both run before any accuracy work, which is the ordering PLAN.md calls the
    main performance trap in the search loop.
    """
    try:
        onnx.checker.check_model(str(path))
    except Exception as error:  # onnx raises several unrelated types here
        raise QuantizationFailure(f"onnx.checker rejected {path.name}: {error}") from error

    try:
        ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as error:
        raise QuantizationFailure(f"ONNX Runtime could not load {path.name}: {error}") from error


def _input_name(path: Path) -> str:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.get_inputs()[0].name


def build_artifact(
    model: str,
    config: QuantConfig,
    onnx_dir: Path = DEFAULT_ONNX_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    bundle_dir: Path | None = None,
    use_cache: bool = True,
    group_map: "OrderedDict[str, tuple[str, ...]] | None" = None,
) -> Artifact:
    """Materialize `config` for `model`, returning a validated artifact.

    `group_map` overrides the default node-to-block mapping, which is how a
    sub-block probe addresses names the standard grouping does not expose.

    Raises `QuantizationFailure` if the graph cannot be produced, fails
    `onnx.checker`, or will not load in ONNX Runtime.
    """
    paths = ModelPaths.resolve(model, onnx_dir)
    destination_dir = cache_dir / model
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{config.hash}.onnx"

    if use_cache and destination.exists():
        return Artifact(
            path=destination,
            config=config,
            bytes=destination.stat().st_size,
            sha256=_sha256_file(destination),
            operators=_operator_inventory(destination),
            build_seconds=0.0,
            from_cache=True,
        )

    started = time.perf_counter()
    try:
        if config.quant_type == "none":
            shutil.copyfile(paths.deployable, destination)
        else:
            excluded_nodes = []
            if config.excluded_groups:
                resolved_map = (
                    build_group_map(paths.quant_ready, model) if group_map is None else group_map
                )
                excluded_nodes = nodes_for_groups(resolved_map, config.excluded_groups)

            if config.quant_type == "dynamic":
                quantize_dynamic(
                    model_input=str(paths.quant_ready),
                    model_output=str(destination),
                    per_channel=config.per_channel,
                    reduce_range=config.reduce_range,
                    weight_type=config.ort_weight_type(),
                    nodes_to_exclude=excluded_nodes or None,
                )
            else:
                reader_kwargs = {} if bundle_dir is None else {"bundle_dir": bundle_dir}
                reader = build_calibration_reader(
                    input_name=_input_name(paths.quant_ready),
                    size=config.calibration_size,
                    **reader_kwargs,
                )
                quantize_static(
                    model_input=str(paths.quant_ready),
                    model_output=str(destination),
                    calibration_data_reader=reader,
                    per_channel=config.per_channel,
                    reduce_range=config.reduce_range,
                    activation_type=config.ort_activation_type(),
                    weight_type=config.ort_weight_type(),
                    calibrate_method=config.ort_calibration_method(),
                    nodes_to_exclude=excluded_nodes or None,
                )
    except QuantizationFailure:
        raise
    except Exception as error:
        destination.unlink(missing_ok=True)
        raise QuantizationFailure(
            f"Quantization failed for {model} [{config.describe()}]: {error}"
        ) from error

    elapsed = time.perf_counter() - started
    _run_gates(destination)

    artifact = Artifact(
        path=destination,
        config=config,
        bytes=destination.stat().st_size,
        sha256=_sha256_file(destination),
        operators=_operator_inventory(destination),
        build_seconds=elapsed,
        from_cache=False,
    )
    sidecar = destination.with_suffix(".json")
    sidecar.write_text(json.dumps(artifact.as_dict(), indent=2), encoding="utf-8")
    return artifact
