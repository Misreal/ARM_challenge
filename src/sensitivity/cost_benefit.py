"""Join each group's accuracy benefit to its measured latency cost.

The Phase 5 ranking answers "what does sparing this group buy?" and the Pi
sweep answers "what does it cost?". Neither alone picks exclusions: the
top-ranked group is often the most expensive one to spare.

    python -m src.sensitivity.cost_benefit --model resnet18_cifar
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.model_index import known_models
from src.quant.candidates import PER_GROUP_PREFIX, per_group_configs
from src.quant.config import EXPORTED_MODELS, QuantConfig, RunConfig
from src.sensitivity.analyze import DEVICE_REPORT_DIR

DEFAULT_CACHE_DIR = Path("artifacts/bench_cache")
ANCHOR = QuantConfig(quant_type="static", per_channel=True)


# The runtime every cost is priced under. RunConfig's own defaults are the
# canonical recipe: 4 threads, full graph optimization, arena on.
REFERENCE_RUN = RunConfig()


def latencies_by_hash(
    model: str, cache_dir: Path, run: RunConfig = REFERENCE_RUN
) -> dict[str, float]:
    """Median latency of every admissible measurement taken under one runtime.

    Matching the whole run config is not optional. Threads alone was enough only
    while the runtime knobs were frozen; once the search could vary optimization
    level and the arena, the cache held several measurements of one artifact --
    the same bytes run with the optimizer off are several times slower -- under a
    single quant hash, and the last file read silently became the anchor.
    """
    wanted = run.as_dict()
    # Measurements taken before the runtime knobs were searchable do not record
    # them, and they ran at the defaults, so an absent knob reads as its default.
    defaults = RunConfig().as_dict()
    found: dict[str, float] = {}
    for path in sorted((cache_dir / model).glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "ok" or not result.get("admissible"):
            continue
        cached = result["config"]["run"]
        if any(cached.get(key, defaults[key]) != value for key, value in wanted.items()):
            continue
        found[result["quant_hash"]] = result["latency"]["median_ms"]
    return found


def build_rows(model: str, cache_dir: Path, report_dir: Path) -> list[dict[str, Any]]:
    report = json.loads(
        (report_dir / f"{model}_sensitivity.json").read_text(encoding="utf-8")
    )
    shares = {
        row["group"]: row["recovery_share"] for row in report["ranking"]["per_channel"]
    }
    latency = latencies_by_hash(model, cache_dir)
    anchor_ms = latency.get(ANCHOR.hash)
    if anchor_ms is None:
        raise SystemExit(f"No measured global per-channel INT8 baseline for {model}")

    rows: list[dict[str, Any]] = []
    for name, config in per_group_configs(model).items():
        group = name[len(PER_GROUP_PREFIX) :]
        spared_ms = latency.get(config.hash)
        if spared_ms is None or group not in shares:
            continue
        cost = spared_ms - anchor_ms
        share = shares[group]
        rows.append(
            {
                "group": group,
                "recovery_share": share,
                "latency_ms": spared_ms,
                "cost_ms": cost,
                # Share of distortion removed per millisecond paid. Groups whose
                # cost is at or below noise are reported as None rather than a
                # huge or negative ratio that would sort meaninglessly.
                "share_per_ms": (share / cost) if cost > 0.02 else None,
            }
        )
    return sorted(rows, key=lambda row: (row["share_per_ms"] is None, -(row["share_per_ms"] or 0)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", nargs="+", choices=known_models(), default=list(EXPORTED_MODELS))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEVICE_REPORT_DIR)
    parser.add_argument("--out", type=Path, default=DEVICE_REPORT_DIR / "cost_benefit.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    everything: dict[str, Any] = {}
    for model in args.model:
        rows = build_rows(model, args.cache_dir, args.report_dir)
        everything[model] = rows
        anchor_ms = latencies_by_hash(model, args.cache_dir)[ANCHOR.hash]
        print(f"\n{model}  (global per-channel INT8 = {anchor_ms:.3f} ms)")
        print(f"  {'group':16s} {'share':>7s} {'ms spared':>10s} {'cost ms':>9s} {'share/ms':>9s}")
        for row in rows:
            ratio = "free" if row["share_per_ms"] is None else f"{row['share_per_ms']:.2f}"
            print(
                f"  {row['group']:16s} {row['recovery_share']:7.3f} "
                f"{row['latency_ms']:10.3f} {row['cost_ms']:+9.3f} {ratio:>9s}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(everything, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
