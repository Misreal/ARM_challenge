"""Decide what will be evaluated before anything is, and write it down.

Planning is separated from evaluation so a campaign is reproducible from its
report alone: the plan names every candidate, the strategy that chose them and
the reason, and it is fixed before the first device measurement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from itertools import combinations
from pathlib import Path
from typing import Any

from src.quant.config import CALIBRATION_METHODS, DeploymentConfig, QuantConfig, RunConfig
from src.search.reduce import Band, ReducedSpace
from src.search.space import (
    FIXED_SPINNING,
    FIXED_THREADS,
    calibration_sizes_for,
)
from src.search.version import SPACE_VERSION

PLAN_SCHEMA = "search-plan/1"

ENUMERATE = "enumerate"
NSGA2 = "nsga2"

# Static per-channel UINT8 with MinMax/512 at four threads and full graph
# optimization: the recipe every measured campaign so far was anchored on, and
# the point the one-factor-at-a-time rounds vary away from.
CANONICAL_QUANT = QuantConfig(
    quant_type="static",
    per_channel=True,
    activation_type="uint8",
    calibration_method="minmax",
    calibration_size=512,
)
CANONICAL_RUN = RunConfig(
    intra_op_num_threads=FIXED_THREADS,
    graph_optimization_level="all",
    enable_cpu_mem_arena=True,
    allow_intra_op_spinning=FIXED_SPINNING,
)

# How many structure rounds the default budget buys. Five rounds of 2^n is 40
# candidates for ResNet-18 and the custom CNN and 80 for MobileNetV2, which at
# the measured 15-47 s per candidate is 16-32 minutes of device time.
DEFAULT_ROUNDS = 5

# `rounds * 2^n` only sizes an enumeration. A space too large to enumerate would
# otherwise ask for tens of thousands of device measurements, so the default is
# capped at the largest enumerated budget any bundled model needs.
MAX_DEFAULT_BUDGET = 80


@dataclass(frozen=True)
class Round:
    """One factor moved off the canonical recipe, across every precision vector."""

    label: str
    reads_as: str
    quant: dict[str, Any] = field(default_factory=dict)
    run: dict[str, Any] = field(default_factory=dict)


# The schedule is justified by how the objectives factor. Calibration moves
# accuracy alone: across 18 byte-equivalent sets the artifact size varied by at
# most 9 bytes. Optimization level and the arena move latency and RAM but never
# size or accuracy. Only the precision vector, per_channel and activation_type
# move all four, so those come first and calibration comes last.
STRUCTURE_ROUNDS: tuple[Round, ...] = (
    Round("canonical", "the reduced space itself"),
    Round("no-arena", "the one knob that trades RAM against latency",
          run={"enable_cpu_mem_arena": False}),
    Round("per-tensor", "per-tensor against per-channel, the mixed-precision story",
          quant={"per_channel": False}),
    Round("act-int8", "the activation-type question", quant={"activation_type": "int8"}),
    Round("opt-extended", "the runtime half", run={"graph_optimization_level": "extended"}),
)


def calibration_rounds() -> tuple[Round, ...]:
    """Every calibration setting except the canonical one, at fixed structure."""
    rounds = []
    for method in CALIBRATION_METHODS:
        for size in calibration_sizes_for(method):
            if (method, size) == (CANONICAL_QUANT.calibration_method, CANONICAL_QUANT.calibration_size):
                continue
            rounds.append(
                Round(
                    f"calib-{method}-{size}",
                    "accuracy only; calibration does not move size or latency",
                    quant={"calibration_method": method, "calibration_size": size},
                )
            )
    return tuple(rounds)


def all_rounds() -> tuple[Round, ...]:
    return STRUCTURE_ROUNDS + calibration_rounds()


def precision_vectors(space: ReducedSpace) -> list[tuple[str, ...]]:
    """Every exclusion set the reduced space allows, fewest spared groups first.

    Ordering by how much is spared means a truncated budget keeps the candidates
    closest to full INT8, which is where the winners have been.
    """
    searchable = tuple(sorted(space.searchable))
    pinned = tuple(sorted(space.pinned_fp32))
    vectors = []
    for size in range(len(searchable) + 1):
        for spared in combinations(searchable, size):
            vectors.append(tuple(sorted(pinned + spared)))
    return vectors


def candidate_for(vector: tuple[str, ...], round_: Round) -> DeploymentConfig:
    """The canonical recipe with one round's factor moved and this vector spared."""
    quant = replace(CANONICAL_QUANT, excluded_groups=vector, **round_.quant)
    return DeploymentConfig(quant=quant, run=replace(CANONICAL_RUN, **round_.run))


def enumerate_candidates(
    space: ReducedSpace, budget: int
) -> tuple[list[tuple[str, DeploymentConfig]], list[str], str | None]:
    """Walk the rounds, covering every vector in one before starting the next.

    Returns the ordered candidates, the rounds that completed, and the round the
    budget cut short if there was one.
    """
    vectors = precision_vectors(space)
    ordered: list[tuple[str, DeploymentConfig]] = []
    seen: set[str] = set()
    completed: list[str] = []
    truncated: str | None = None

    for round_ in all_rounds():
        if len(ordered) >= budget:
            break
        for vector in vectors:
            if len(ordered) >= budget:
                truncated = round_.label
                break
            config = candidate_for(vector, round_)
            # Two rounds can land on the same bytes and the same runtime; paying
            # the device twice for one point is the duplicate cost this replaces.
            if config.hash in seen:
                continue
            seen.add(config.hash)
            ordered.append((round_.label, config))
        else:
            completed.append(round_.label)

    return ordered, completed, truncated


def default_budget(space: ReducedSpace) -> int:
    return min(DEFAULT_ROUNDS * space.reduced_size, MAX_DEFAULT_BUDGET)


def population_for(trials: int) -> int:
    """Small enough that a modest budget actually breeds rather than samples."""
    return min(10, max(4, trials // 4))


@dataclass(frozen=True)
class SearchPlan:
    """Everything decided before the first measurement."""

    model: str
    space: ReducedSpace
    strategy: str
    strategy_reason: str
    budget: int
    baseline_top1: float
    accuracy_budget_pt: float
    candidates: tuple[tuple[str, DeploymentConfig], ...] = ()
    rounds_completed: tuple[str, ...] = ()
    round_truncated: str | None = None
    population_size: int | None = None
    max_rss_mb: float | None = None

    @property
    def threshold_top1(self) -> float:
        return self.baseline_top1 - self.accuracy_budget_pt

    @property
    def band(self) -> Band | None:
        return self.space.band

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": PLAN_SCHEMA,
            "space_version": SPACE_VERSION,
            "model": self.model,
            "space": self.space.as_dict(),
            "strategy": self.strategy,
            "strategy_reason": self.strategy_reason,
            "budget": self.budget,
            "baseline_top1": self.baseline_top1,
            "accuracy_budget_pt": self.accuracy_budget_pt,
            "threshold_top1": round(self.threshold_top1, 4),
            "max_rss_mb": self.max_rss_mb,
            "population_size": self.population_size,
            "rounds_completed": list(self.rounds_completed),
            "round_truncated": self.round_truncated,
            "candidates": [
                {"round": label, "config_hash": config.hash, **config.as_dict()}
                for label, config in self.candidates
            ],
        }


def make_plan(
    model: str,
    space: ReducedSpace,
    baseline_top1: float,
    accuracy_budget_pt: float,
    trials: int | None = None,
    max_rss_mb: float | None = None,
) -> SearchPlan:
    """Choose the strategy from the space and the budget, then fix the candidates."""
    budget = default_budget(space) if trials is None else trials
    vectors = space.reduced_size

    if vectors <= budget:
        candidates, completed, truncated = enumerate_candidates(space, budget)
        return SearchPlan(
            model=model,
            space=space,
            strategy=ENUMERATE,
            strategy_reason=(
                f"{vectors} precision vectors fit a budget of {budget}, so every one is "
                f"measured under {len(completed)} complete one-factor round(s)"
            ),
            budget=budget,
            baseline_top1=baseline_top1,
            accuracy_budget_pt=accuracy_budget_pt,
            candidates=tuple(candidates),
            rounds_completed=tuple(completed),
            round_truncated=truncated,
            max_rss_mb=max_rss_mb,
        )

    return SearchPlan(
        model=model,
        space=space,
        strategy=NSGA2,
        strategy_reason=(
            f"{vectors} precision vectors exceed a budget of {budget}, so the space is "
            "sampled under the accuracy constraint rather than enumerated"
        ),
        budget=budget,
        baseline_top1=baseline_top1,
        accuracy_budget_pt=accuracy_budget_pt,
        population_size=population_for(budget),
        max_rss_mb=max_rss_mb,
    )


def load_space(path: Path, model: str, groups: tuple[str, ...]) -> ReducedSpace:
    """Read this run's reduction and refuse it before any device work is spent."""
    if not path.exists():
        raise SystemExit(
            f"No search space at {path}. Run the search_space stage first: "
            f"python -m src.app run --model {model} --only search_space"
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    # The stage file holds one model; the shared artifact holds every model.
    if model in document and "searchable" not in document:
        document = document[model]

    space = ReducedSpace.from_dict(document)
    space.validate(model, groups)
    return space
