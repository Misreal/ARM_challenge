"""Probe two quantization knobs the campaign never searches: activation symmetry and QuantFormat.

Runs on the Pi and writes outside artifacts/quant/, so it cannot reach the
campaign's hash space or its measurement cache.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

from src.portable.bundle import DEFAULT_BUNDLE_DIR, load_bundle
from src.portable.onnx_eval import evaluate_accuracy
from src.quant.calibration import build_calibration_reader
from src.quant.config import RunConfig
from src.quant.quantize import ModelPaths, build_session

CALIBRATION_SIZE = 512
EVAL_SPLIT = "optval"
WARMUP_REPS = 20
TIMED_REPS = 300

# The campaign's deployed runtime settings, so latency here reads against it.
DEPLOYED_RUN = RunConfig()

# Overrides merged onto BASE_KWARGS. `baseline` reproduces QuantConfig()'s artifact,
# which is the control every other arm is read against.
ARMS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "u8_sym": {"extra_options": {"ActivationSymmetric": True}},
    "s8_asym": {"activation_type": QuantType.QInt8},
    "s8_sym": {
        "activation_type": QuantType.QInt8,
        "extra_options": {"ActivationSymmetric": True},
    },
    "qoperator": {"quant_format": QuantFormat.QOperator},
}

def base_kwargs(per_channel: bool) -> dict[str, Any]:
    """The control config. `per_channel` is a CLI knob because each model's Pareto
    leader sits on a different side of it, and an arm is only readable against its own leader."""
    return {
        "quant_format": QuantFormat.QDQ,
        "per_channel": per_channel,
        "reduce_range": False,
        "activation_type": QuantType.QUInt8,
        "weight_type": QuantType.QInt8,
        "calibrate_method": CalibrationMethod.MinMax,
    }


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _vcgencmd(argument: str) -> str | None:
    try:
        done = subprocess.run(
            ["vcgencmd", argument], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None


def device_state() -> dict[str, Any]:
    """Governor, clock, temperature and throttle flags: a latency delta means nothing without them."""
    return {
        "governor": _read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "cur_freq_khz": _read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"),
        "temp": _vcgencmd("measure_temp"),
        "throttled": _vcgencmd("get_throttled"),
    }


def operator_inventory(path: Path) -> dict[str, int]:
    model = onnx.load(str(path))
    return dict(sorted(Counter(node.op_type for node in model.graph.node).items()))


def zero_point_summary(path: Path) -> dict[str, list[int]]:
    """Distinct zero-point values per dtype: the evidence that ActivationSymmetric took effect."""
    model = onnx.load(str(path))
    found: dict[str, set[int]] = {}
    for initializer in model.graph.initializer:
        if "zero_point" not in initializer.name:
            continue
        values = np.asarray(numpy_helper.to_array(initializer)).reshape(-1)
        found.setdefault(str(values.dtype), set()).update(int(value) for value in values[:64])
    return {dtype: sorted(values)[:12] for dtype, values in sorted(found.items())}


def measure_latency(session: ort.InferenceSession, input_name: str, shape: list[int]) -> dict:
    batch = np.zeros(shape, dtype=np.float32)
    for _ in range(WARMUP_REPS):
        session.run(None, {input_name: batch})

    samples = []
    for _ in range(TIMED_REPS):
        started = time.perf_counter()
        session.run(None, {input_name: batch})
        samples.append((time.perf_counter() - started) * 1000.0)
    samples.sort()

    return {
        "reps": len(samples),
        "min_ms": round(samples[0], 4),
        "median_ms": round(samples[len(samples) // 2], 4),
        "mean_ms": round(sum(samples) / len(samples), 4),
        "p95_ms": round(samples[int(0.95 * (len(samples) - 1))], 4),
    }


def run_arm(
    name: str,
    overrides: dict[str, Any],
    paths: ModelPaths,
    input_name: str,
    out_dir: Path,
    bundle_dir: Path,
    limit: int | None,
    per_channel: bool,
) -> dict[str, Any]:
    record: dict[str, Any] = {"arm": name, "state_before": device_state()}

    overrides = dict(overrides)
    extra_options = dict(overrides.pop("extra_options", {}))
    kwargs: dict[str, Any] = {**base_kwargs(per_channel), **overrides}
    if extra_options:
        kwargs["extra_options"] = extra_options
    record["overrides"] = {
        **{key: str(value) for key, value in overrides.items()},
        **({"extra_options": extra_options} if extra_options else {}),
    }

    destination = out_dir / f"{paths.name}_{'pc' if per_channel else 'pt'}_{name}.onnx"
    # A fresh reader per arm: the calibrator's rewind contract varies by method.
    reader = build_calibration_reader(
        input_name=input_name, size=CALIBRATION_SIZE, bundle_dir=bundle_dir
    )

    started = time.perf_counter()
    try:
        quantize_static(
            model_input=str(paths.quant_ready),
            model_output=str(destination),
            calibration_data_reader=reader,
            **kwargs,
        )
    except Exception as error:
        return {**record, "status": "build_failed", "error": f"{type(error).__name__}: {error}"}
    record["build_seconds"] = round(time.perf_counter() - started, 2)

    record["bytes"] = destination.stat().st_size
    record["operators"] = operator_inventory(destination)
    record["zero_points"] = zero_point_summary(destination)

    try:
        session = build_session(destination, DEPLOYED_RUN)
    except Exception as error:
        return {**record, "status": "load_failed", "error": f"{type(error).__name__}: {error}"}

    spec = session.get_inputs()[0]
    shape = [dim if isinstance(dim, int) else 1 for dim in spec.shape]
    record["latency"] = measure_latency(session, spec.name, shape)

    bundle = load_bundle(EVAL_SPLIT, bundle_dir=bundle_dir)
    accuracy = evaluate_accuracy(session, bundle.images_uint8, bundle.labels, limit=limit)
    record["accuracy"] = accuracy.as_dict()
    record["split_fingerprint"] = bundle.split_fingerprint

    record["state_after"] = device_state()
    return {**record, "status": "ok"}


def summarize(report: dict[str, Any]) -> None:
    control = report["arms"].get("baseline", {})
    base_top1 = control.get("accuracy", {}).get("top1")
    base_ms = control.get("latency", {}).get("median_ms")

    print(f"\n{'arm':<12} {'status':<13} {'top1':>7} {'d_pt':>7} {'ms':>8} {'d_%':>7} {'MB':>6}")
    print("-" * 66)
    for name, data in report["arms"].items():
        if data["status"] != "ok":
            print(f"{name:<12} {data['status']:<13} {data.get('error', '')[:40]}")
            continue
        top1 = data["accuracy"]["top1"]
        ms = data["latency"]["median_ms"]
        delta_top1 = "" if base_top1 is None else f"{top1 - base_top1:+.2f}"
        delta_ms = "" if not base_ms else f"{100.0 * (ms - base_ms) / base_ms:+.1f}"
        print(
            f"{name:<12} {'ok':<13} {top1:>7.2f} {delta_top1:>7} "
            f"{ms:>8.3f} {delta_ms:>7} {data['bytes'] / 1e6:>6.2f}"
        )

    print("\noperator inventories (differences from baseline)")
    control_ops = control.get("operators", {})
    for name, data in report["arms"].items():
        if data["status"] == "build_failed":
            continue
        ops = data.get("operators", {})
        changed = {
            key: (control_ops.get(key, 0), ops.get(key, 0))
            for key in sorted(set(control_ops) | set(ops))
            if control_ops.get(key, 0) != ops.get(key, 0)
        }
        print(f"  {name:<12} {changed if changed else 'identical to baseline'}")

    print("\nzero points")
    for name, data in report["arms"].items():
        if "zero_points" in data:
            print(f"  {name:<12} {data['zero_points']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="custom_cnn")
    parser.add_argument("--per-channel", action="store_true", help="per-channel weights in every arm")
    parser.add_argument("--arms", nargs="*", default=None, help="subset of arms (default: all)")
    parser.add_argument("--limit", type=int, default=None, help="first N optval images")
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/quant_knobs/out"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = ModelPaths.resolve(args.model)

    probe = ort.InferenceSession(str(paths.quant_ready), providers=["CPUExecutionProvider"])
    input_name = probe.get_inputs()[0].name
    del probe

    selected = args.arms or list(ARMS)
    report: dict[str, Any] = {
        # These configs are deliberately outside QuantConfig's space; nothing here
        # may be merged into a Pareto front.
        "campaign_config": False,
        "model": args.model,
        "base": {key: str(value) for key, value in base_kwargs(args.per_channel).items()},
        "calibration_size": CALIBRATION_SIZE,
        "eval_split": EVAL_SPLIT,
        "limit": args.limit,
        "run_config": DEPLOYED_RUN.as_dict(),
        "onnxruntime": ort.__version__,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "arms": {},
    }

    for name in selected:
        print(f"\n=== {args.model} / {name} ===", flush=True)
        result = run_arm(
            name,
            ARMS[name],
            paths,
            input_name,
            args.out_dir,
            args.bundle_dir,
            args.limit,
            args.per_channel,
        )
        report["arms"][name] = result
        if result["status"] == "ok":
            print(
                f"  top1 {result['accuracy']['top1']:.2f}  "
                f"median {result['latency']['median_ms']:.3f} ms  "
                f"{result['bytes'] / 1e6:.2f} MB",
                flush=True,
            )
        else:
            print(f"  {result['status']}: {result.get('error', '')}", flush=True)

    suffix = "pc" if args.per_channel else "pt"
    destination = args.out_dir / f"{args.model}_{suffix}_activation_format.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summarize(report)
    print(f"\nWrote {destination}")


if __name__ == "__main__":
    main()
