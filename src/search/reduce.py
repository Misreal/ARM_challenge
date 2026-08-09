"""Turn measured cost/benefit into a reduced search space and a greedy reference.

Sparing a group raises latency, size and RAM together, so it never trades one
objective for another -- it only buys accuracy. The search is therefore a
knapsack over `recovery_share / cost_ms`, which both shrinks the space and gives
a zero-trial reference solution to judge the search against.

    python -m src.search.reduce --model resnet18_cifar
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.quant.config import EXPORTED_MODELS
from src.sensitivity.analyze import DEVICE_REPORT_DIR

DEFAULT_COST_BENEFIT = DEVICE_REPORT_DIR / "cost_benefit.json"

# Groups whose measured cost sits inside this band are free to spare: the Pi
# sweep produced negative costs down to -0.053 ms for graph-identical work, so
# anything under it is the harness's own noise, not a real price.
NOISE_MS = 0.06

# A group recovering less than this fraction of the best group's distortion per
# millisecond is never bought before the best group is, at any budget. Relative
# rather than absolute so the rule carries across models without retuning.
RATIO_FLOOR = 0.10

PIN_FP32 = "pin_fp32"
PIN_INT8 = "pin_int8"
SEARCH = "search"


@dataclass(frozen=True)
class ReducedSpace:
    """Which groups are decided up front and which are left to the sampler."""

    model: str
    pinned_fp32: tuple[str, ...]
    pinned_int8: tuple[str, ...]
    searchable: tuple[str, ...]

    @property
    def full_size(self) -> int:
        return 2 ** (len(self.pinned_fp32) + len(self.pinned_int8) + len(self.searchable))

    @property
    def reduced_size(self) -> int:
        return 2 ** len(self.searchable)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "pinned_fp32": list(self.pinned_fp32),
            "pinned_int8": list(self.pinned_int8),
            "searchable": list(self.searchable),
            "precision_vectors_full": self.full_size,
            "precision_vectors_reduced": self.reduced_size,
        }


def classify(row: dict[str, Any], best_ratio: float) -> str:
    """Decide one group's fate from its measured cost and benefit."""
    if row["cost_ms"] <= NOISE_MS:
        # Free either way, so the only question is whether it buys anything.
        return PIN_FP32 if row["recovery_share"] > 0 else PIN_INT8
    ratio = row["share_per_ms"]
    if ratio is None or ratio <= 0 or ratio < best_ratio * RATIO_FLOOR:
        return PIN_INT8
    return SEARCH


def reduce_space(model: str, rows: list[dict[str, Any]]) -> ReducedSpace:
    priced = [row["share_per_ms"] for row in rows if row["share_per_ms"] is not None]
    best_ratio = max(priced, default=0.0)

    buckets: dict[str, list[str]] = {PIN_FP32: [], PIN_INT8: [], SEARCH: []}
    for row in rows:
        buckets[classify(row, best_ratio)].append(row["group"])
    return ReducedSpace(
        model=model,
        pinned_fp32=tuple(buckets[PIN_FP32]),
        pinned_int8=tuple(buckets[PIN_INT8]),
        searchable=tuple(buckets[SEARCH]),
    )


def greedy_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The knapsack reference: spare groups cheapest-accuracy-first.

    Free groups come first at any budget, then the priced ones by share per
    millisecond. Sparing in this order is what the search has to beat.
    """
    free = [row for row in rows if row["cost_ms"] <= NOISE_MS and row["recovery_share"] > 0]
    priced = sorted(
        (row for row in rows if row["cost_ms"] > NOISE_MS and (row["share_per_ms"] or 0) > 0),
        key=lambda row: -row["share_per_ms"],
    )
    return free + priced


def load_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", nargs="+", choices=EXPORTED_MODELS, default=list(EXPORTED_MODELS))
    parser.add_argument("--cost-benefit", type=Path, default=DEFAULT_COST_BENEFIT)
    parser.add_argument("--out", type=Path, default=DEVICE_REPORT_DIR / "search_space.json")
    args = parser.parse_args()

    everything = load_rows(args.cost_benefit)
    spaces: dict[str, Any] = {}
    for model in args.model:
        rows = everything[model]
        space = reduce_space(model, rows)
        spaces[model] = {
            **space.as_dict(),
            "greedy_order": [row["group"] for row in greedy_order(rows)],
        }
        print(f"\n{model}: {space.full_size:,} precision vectors -> {space.reduced_size:,}")
        print(f"  pin fp32   {', '.join(space.pinned_fp32) or '-'}")
        print(f"  pin int8   {', '.join(space.pinned_int8) or '-'}")
        print(f"  search     {', '.join(space.searchable) or '-'}")
        print(f"  greedy     {' > '.join(spaces[model]['greedy_order']) or '-'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(spaces, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
