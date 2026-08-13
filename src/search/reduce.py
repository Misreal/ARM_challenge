"""Turn measured cost/benefit into a reduced search space and a greedy reference.

Sparing a group raises latency, size and RAM together, so it never trades one
objective for another -- it only buys accuracy. The search is therefore a
knapsack over `recovery_share / cost_ms`, which both shrinks the space and gives
a zero-trial reference solution to judge the search against.

    python -m src.search.reduce --model resnet18_cifar
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.quant.config import EXPORTED_MODELS
from src.sensitivity.analyze import DEVICE_REPORT_DIR

DEFAULT_COST_BENEFIT = DEVICE_REPORT_DIR / "cost_benefit.json"
SPACE_SCHEMA = "search-space/2"

# Grouping bench_cache entries that differ only in calibration -- identical
# structure and shapes, artifact sizes equal to within 9 bytes -- gave 18 sets of
# byte-equivalent work whose median latencies still spread by 0.66% median and
# 1.91% at p90. The band is relative because the dispersion is: an absolute
# 0.06 ms is 3.3% of the custom CNN's reference but 1.8% of ResNet-18's.
DEFAULT_BAND_FRACTION = 0.02

# A group recovering less than this fraction of the best group's distortion per
# millisecond is never bought before the best group is, at any budget. Relative
# rather than absolute so the rule carries across models without retuning.
RATIO_FLOOR = 0.10

PIN_FP32 = "pin_fp32"
PIN_INT8 = "pin_int8"
SEARCH = "search"

BAND_FROM_SENTINEL = "sentinel"
BAND_FROM_DEFAULT = "default"


@dataclass(frozen=True)
class Band:
    """How far two latencies must differ before the difference means anything."""

    fraction: float
    source: str
    reference_ms: float

    @property
    def ms(self) -> float:
        return self.fraction * self.reference_ms

    def as_dict(self) -> dict[str, Any]:
        return {
            "fraction": round(self.fraction, 6),
            "ms": round(self.ms, 4),
            "source": self.source,
            "reference_ms": round(self.reference_ms, 4),
        }


def group_fingerprint(groups: tuple[str, ...]) -> str:
    """Identity of the group set a reduction was cut against.

    Sourced from the model index rather than by hashing the ONNX: the graphs are
    gitignored, so a fresh clone and every mock run have no graph to fingerprint.
    """
    payload = json.dumps(sorted(groups), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def reference_latency(rows: list[dict[str, Any]]) -> float:
    """The global per-channel INT8 anchor every cost was measured against."""
    anchors = [row["latency_ms"] - row["cost_ms"] for row in rows if row.get("latency_ms") is not None]
    if not anchors:
        raise ValueError("cost/benefit rows carry no latency, so there is no anchor to band against")
    return sorted(anchors)[len(anchors) // 2]


def band_for(rows: list[dict[str, Any]], sentinel: dict[str, Any] | None) -> Band:
    """The measured reproducibility band, or the documented default without one."""
    reference_ms = reference_latency(rows)
    fraction = None if sentinel is None else sentinel.get("spread_fraction")
    if fraction is None:
        return Band(DEFAULT_BAND_FRACTION, BAND_FROM_DEFAULT, reference_ms)
    return Band(float(fraction), BAND_FROM_SENTINEL, reference_ms)


@dataclass(frozen=True)
class ReducedSpace:
    """Which groups are decided up front and which are left to the sampler."""

    model: str
    pinned_fp32: tuple[str, ...]
    pinned_int8: tuple[str, ...]
    searchable: tuple[str, ...]
    fingerprint: str = ""
    band: Band | None = None

    @property
    def groups(self) -> tuple[str, ...]:
        return tuple(sorted(self.pinned_fp32 + self.pinned_int8 + self.searchable))

    @property
    def full_size(self) -> int:
        return 2 ** len(self.groups)

    @property
    def reduced_size(self) -> int:
        return 2 ** len(self.searchable)

    def as_dict(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "schema": SPACE_SCHEMA,
            "model": self.model,
            "fingerprint": self.fingerprint,
            "pinned_fp32": list(self.pinned_fp32),
            "pinned_int8": list(self.pinned_int8),
            "searchable": list(self.searchable),
            "precision_vectors_full": self.full_size,
            "precision_vectors_reduced": self.reduced_size,
        }
        if self.band is not None:
            document["band"] = self.band.as_dict()
        return document

    @classmethod
    def from_dict(cls, document: dict[str, Any]) -> ReducedSpace:
        band = document.get("band")
        return cls(
            model=document["model"],
            pinned_fp32=tuple(document.get("pinned_fp32", ())),
            pinned_int8=tuple(document.get("pinned_int8", ())),
            searchable=tuple(document.get("searchable", ())),
            fingerprint=document.get("fingerprint", ""),
            band=None if band is None else Band(band["fraction"], band["source"], band["reference_ms"]),
        )

    def validate(self, model: str, groups: tuple[str, ...]) -> None:
        """Refuse a partition that is not a partition, before any device work.

        An overlap silently makes one group both pinned and searched; a gap
        leaves a group with no decision at all and the sampler quietly inventing
        one. Both are the kind of error that only shows up as a front that
        cannot be reproduced.
        """
        if self.model != model:
            raise SystemExit(f"search space is for {self.model!r}, not {model!r}")

        expected = group_fingerprint(tuple(groups))
        if self.fingerprint and self.fingerprint != expected:
            raise SystemExit(
                f"search space was cut against a different group set ({self.fingerprint} != "
                f"{expected}). Re-run src.search.reduce against this graph."
            )

        buckets = {
            "pinned_fp32": self.pinned_fp32,
            "pinned_int8": self.pinned_int8,
            "searchable": self.searchable,
        }
        seen: dict[str, str] = {}
        for name, bucket in buckets.items():
            for group in bucket:
                if group in seen:
                    raise SystemExit(f"{group!r} is in both {seen[group]} and {name}")
                seen[group] = name

        unknown = sorted(set(seen) - set(groups))
        if unknown:
            raise SystemExit(f"search space names groups this graph does not expose: {', '.join(unknown)}")
        missing = sorted(set(groups) - set(seen))
        if missing:
            raise SystemExit(f"search space decides nothing for: {', '.join(missing)}")


def classify(row: dict[str, Any], best_ratio: float, band_ms: float) -> str:
    """Decide one group's fate from its measured cost and benefit.

    The asymmetry around the band is deliberate. A cost indistinguishable from
    zero is not evidence for pinning -- it is exactly the case enumeration should
    settle -- and an error that pins is silent and permanent, while an error that
    searches costs one more precision vector.
    """
    cost = row["cost_ms"]
    buys_something = row["recovery_share"] > 0

    if abs(cost) <= band_ms:
        return SEARCH if buys_something else PIN_INT8
    if cost < -band_ms:
        # Measurably faster when spared, so there is nothing to trade away.
        return PIN_FP32 if buys_something else PIN_INT8

    ratio = row["share_per_ms"]
    if ratio is None or ratio <= 0 or ratio < best_ratio * RATIO_FLOOR:
        return PIN_INT8
    return SEARCH


def reduce_space(
    model: str,
    rows: list[dict[str, Any]],
    band: Band | None = None,
    groups: tuple[str, ...] | None = None,
) -> ReducedSpace:
    priced = [row["share_per_ms"] for row in rows if row["share_per_ms"] is not None]
    best_ratio = max(priced, default=0.0)
    if band is None:
        band = band_for(rows, None)

    buckets: dict[str, list[str]] = {PIN_FP32: [], PIN_INT8: [], SEARCH: []}
    for row in rows:
        buckets[classify(row, best_ratio, band.ms)].append(row["group"])

    named = tuple(sorted(row["group"] for row in rows)) if groups is None else tuple(groups)
    return ReducedSpace(
        model=model,
        pinned_fp32=tuple(buckets[PIN_FP32]),
        pinned_int8=tuple(buckets[PIN_INT8]),
        searchable=tuple(buckets[SEARCH]),
        fingerprint=group_fingerprint(named),
        band=band,
    )


def greedy_order(rows: list[dict[str, Any]], band_ms: float) -> list[dict[str, Any]]:
    """The knapsack reference: spare groups cheapest-accuracy-first.

    Groups whose price is inside the band come first at any budget, then the
    priced ones by share per millisecond. Sparing in this order is what the
    search has to beat.
    """
    free = [row for row in rows if row["cost_ms"] <= band_ms and row["recovery_share"] > 0]
    priced = sorted(
        (row for row in rows if row["cost_ms"] > band_ms and (row["share_per_ms"] or 0) > 0),
        key=lambda row: -row["share_per_ms"],
    )
    return free + priced


def load_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_sentinel(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def known_groups(model: str) -> tuple[str, ...] | None:
    """The model index's group list, or None for a model not registered in it."""
    from src.model_index import entry_for

    try:
        return entry_for(model).groups
    except SystemExit:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", nargs="+", choices=EXPORTED_MODELS, default=list(EXPORTED_MODELS))
    parser.add_argument("--cost-benefit", type=Path, default=DEFAULT_COST_BENEFIT)
    parser.add_argument("--sentinel", type=Path, default=None, help="repeat-spread report for the band")
    parser.add_argument("--out", type=Path, default=DEVICE_REPORT_DIR / "search_space.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    everything = load_rows(args.cost_benefit)
    sentinel = load_sentinel(args.sentinel)
    spaces: dict[str, Any] = {}
    for model in args.model:
        rows = everything[model]
        band = band_for(rows, sentinel)
        groups = known_groups(model)
        space = reduce_space(model, rows, band, groups)
        # Validated here rather than at read time too, so a reduction that does
        # not cover the graph never reaches the artifact in the first place.
        space.validate(model, groups or space.groups)
        spaces[model] = {
            **space.as_dict(),
            "greedy_order": [row["group"] for row in greedy_order(rows, band.ms)],
        }
        print(f"\n{model}: {space.full_size:,} precision vectors -> {space.reduced_size:,}")
        print(f"  band       {band.fraction:.2%} of {band.reference_ms:.3f} ms "
              f"= {band.ms:.4f} ms (from {band.source})")
        print(f"  pin fp32   {', '.join(space.pinned_fp32) or '-'}")
        print(f"  pin int8   {', '.join(space.pinned_int8) or '-'}")
        print(f"  search     {', '.join(space.searchable) or '-'}")
        print(f"  greedy     {' > '.join(spaces[model]['greedy_order']) or '-'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(spaces, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
