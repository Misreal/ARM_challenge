"""Run the per-group sensitivity sweep and write the ranking report.

Belongs on the Pi: the ranking turns on a per-tensor vs per-channel contrast,
and this project's x86 host is a confirmed saturating outlier on exactly that axis.

Example:
    python -m src.sensitivity.analyze --model custom_cnn
"""

from __future__ import annotations

import argparse
import json
import platform
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from src.model_index import known_models
from src.portable.bundle import DEFAULT_BUNDLE_DIR, load_bundle
from src.quant.config import QuantConfig, RunConfig
from src.quant.groups import build_group_map, quantizable_groups, refine_group
from src.quant.quantize import DEFAULT_CACHE_DIR, DEFAULT_ONNX_DIR, ModelPaths
from src.sensitivity.metrics import accuracy_from_logits
from src.sensitivity.probes import (
    LEAVE_ONE_OUT,
    SCHEMES,
    Probe,
    anchor_configs,
    evaluate_config,
    logits_for_config,
    probe_config,
    probes_for,
)

EVAL_SPLIT = "optval"
PC_REPORT_DIR = Path("artifacts/reports")
DEVICE_REPORT_DIR = Path("artifacts/reports_pi")


def on_target() -> bool:
    return platform.machine().lower() in ("aarch64", "arm64")


def default_report_dir() -> Path:
    # Split by architecture so a PC dry run can never overwrite the device
    # report of record, which is the only one Phase 6 is allowed to consume.
    return DEVICE_REPORT_DIR if on_target() else PC_REPORT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=known_models())
    parser.add_argument("--schemes", nargs="+", choices=sorted(SCHEMES), default=sorted(SCHEMES))
    parser.add_argument(
        "--limit", type=int, default=None, help="evaluate only the first N optval images"
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument(
        "--refine",
        metavar="GROUP",
        help="split this group into dotted subgroups and probe those instead",
    )
    parser.add_argument("--refine-depth", type=int, default=3)
    parser.add_argument("--restart", action="store_true", help="ignore an existing report")
    parser.add_argument("--keep-artifacts", action="store_true")
    return parser.parse_args()


def report_path_for(model: str, report_dir: Path, refine: str | None) -> Path:
    suffix = "" if refine is None else f"_refine_{refine.replace('.', '-')}"
    return report_dir / f"{model}_sensitivity{suffix}.json"


def load_existing(path: Path, restart: bool) -> dict[str, Any]:
    if restart or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def rank_groups(
    groups: tuple[str, ...], anchors: dict[str, Any], probes: dict[str, Any], scheme: str
) -> list[dict[str, Any]]:
    """Order groups by how much keeping them FP32 heals the fully-quantized graph.

    The sign is the trap here. A leave-one-out probe quantizes everything *but*
    the group, so a sensitive group produces a *low* divergence -- the damage it
    would have caused is absent. Sensitivity is therefore the anchor's divergence
    minus the probe's, not the probe's own value.
    """
    anchor = anchors.get(f"static_{scheme}", {})
    anchor_kl = anchor.get("metrics", {}).get("kl_mean")

    rows: list[dict[str, Any]] = []
    for group in groups:
        loo = probes.get(Probe(group, LEAVE_ONE_OUT, scheme).key, {}).get("metrics")
        isolate = probes.get(Probe(group, "isolate", scheme).key, {}).get("metrics")
        if loo is None or isolate is None or anchor_kl is None:
            continue
        recovery = anchor_kl - loo["kl_mean"]
        rows.append(
            {
                "group": group,
                "recovery_kl": recovery,
                # Share of the fully-quantized graph's damage this group owns.
                # The two schemes sit at different absolute KL, so only the
                # normalized figure is comparable between them.
                "recovery_share": (recovery / anchor_kl) if anchor_kl else None,
                "isolate_kl": isolate["kl_mean"],
                "leave_one_out_kl": loo["kl_mean"],
                "recovery_top1": loo["top1_delta"] - anchor["metrics"]["top1_delta"],
                "isolate_flip_rate": isolate["flip_rate"],
            }
        )

    return sorted(rows, key=lambda row: row["recovery_kl"], reverse=True)


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir or default_report_dir()
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_path_for(args.model, report_dir, args.refine)

    run = RunConfig(intra_op_num_threads=args.threads)
    bundle = load_bundle(EVAL_SPLIT, bundle_dir=args.bundle_dir)
    images = bundle.images_uint8 if args.limit is None else bundle.images_uint8[: args.limit]
    labels = bundle.labels

    group_map = build_group_map(ModelPaths.resolve(args.model, args.onnx_dir).quant_ready, args.model)
    if args.refine:
        group_map = refine_group(group_map, args.refine, args.refine_depth)
    groups = quantizable_groups(group_map)
    all_groups = tuple(group_map)

    build_kwargs: dict[str, Any] = {
        "onnx_dir": args.onnx_dir,
        "cache_dir": args.cache_dir,
        "bundle_dir": args.bundle_dir,
        "group_map": group_map,
        "keep_artifact": args.keep_artifacts,
    }

    print(f"{args.model}: {len(groups)} groups, {len(images)} images, schemes {args.schemes}")

    # The reference is the FP32 graph's own logits; every metric is a paired
    # comparison against it, so it must be scored over the identical images.
    reference_logits, _, _ = logits_for_config(
        args.model, QuantConfig(quant_type="none"), images, run, **build_kwargs
    )
    reference_top1, reference_top5 = accuracy_from_logits(reference_logits, labels)
    print(f"  fp32 reference: top-1 {reference_top1:.2f}%  top-5 {reference_top5:.2f}%")

    existing = load_existing(path, args.restart)
    anchors: dict[str, Any] = existing.get("anchors", {})
    probe_results: dict[str, Any] = existing.get("probes", {})

    report: dict[str, Any] = {
        "model": args.model,
        "eval_split": EVAL_SPLIT,
        "eval_limit": args.limit,
        "samples": int(images.shape[0]),
        "split_fingerprint": bundle.split_fingerprint,
        "on_target": on_target(),
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "run_config": run.as_dict(),
        "refined_group": args.refine,
        "group_sizes": {name: len(nodes) for name, nodes in group_map.items()},
        "reference": {"top1": reference_top1, "top5": reference_top5},
        "anchors": anchors,
        "probes": probe_results,
        "onnxruntime": ort.__version__,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }

    def flush() -> None:
        report["ranking"] = {
            scheme: rank_groups(groups, anchors, probe_results, scheme) for scheme in args.schemes
        }
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def score(label: str, config: QuantConfig, sink: dict[str, Any]) -> None:
        if label in sink and sink[label].get("status") == "ok":
            print(f"  [skip] {label}")
            return
        outcome = evaluate_config(
            args.model,
            config,
            images,
            labels,
            reference_logits,
            reference_top1,
            run,
            **build_kwargs,
        )
        sink[label] = outcome.as_dict()
        flush()
        if outcome.metrics is None:
            print(f"  [fail] {label}: {outcome.error}")
        else:
            metrics = outcome.metrics
            print(
                f"  {label:44s} KL {metrics.kl_mean:.5f}  flip {metrics.flip_rate:5.2f}%  "
                f"top1 {metrics.top1:.2f}% ({metrics.top1_delta:+.2f})  {outcome.seconds:.0f}s"
            )

    for label, config in anchor_configs().items():
        if label == "fp32":
            continue
        if config.per_channel and "per_channel" not in args.schemes:
            continue
        if not config.per_channel and "per_tensor" not in args.schemes:
            continue
        score(label, config, anchors)

    for probe in probes_for(groups, tuple(args.schemes)):
        score(probe.key, probe_config(probe, all_groups), probe_results)

    flush()
    print(f"\nWrote {path}")
    for scheme in args.schemes:
        print(f"\nMost sensitive groups ({scheme}, by KL recovered):")
        for row in report["ranking"][scheme][:5]:
            print(f"  {row['group']:28s} recovery {row['recovery_kl']:+.5f}")


if __name__ == "__main__":
    main()
