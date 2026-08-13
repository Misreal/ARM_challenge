"""No candidate may spend device time on a decision the measurements already made.

The sampled fallback is unreachable for every bundled model, so the mock run
never covers it; these tests are the only thing that does.
"""

from __future__ import annotations

import optuna
import pytest

from src.search.plan import candidate_for, precision_vectors
from src.search.plan import STRUCTURE_ROUNDS
from src.search.reduce import Band, ReducedSpace, group_fingerprint
from src.search.space import suggest_within

GROUPS = ("conv1", "fc", "layer1", "layer2", "layer3")


def space(searchable=("conv1", "layer1"), pinned_fp32=("fc",)) -> ReducedSpace:
    return ReducedSpace(
        model="resnet18_cifar",
        pinned_fp32=tuple(pinned_fp32),
        pinned_int8=tuple(g for g in GROUPS if g not in searchable and g not in pinned_fp32),
        searchable=tuple(searchable),
        fingerprint=group_fingerprint(GROUPS),
        band=Band(0.02, "sentinel", 3.41),
    )


def sampled(reduced, trials: int = 60) -> list:
    """Draw from the constrained sampler the way the fallback actually would."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=optuna.samplers.NSGAIISampler(seed=0, population_size=4))
    configs = []

    def objective(trial):
        configs.append(suggest_within(trial, reduced))
        return 0.0

    study.optimize(objective, n_trials=trials)
    return configs


# ------------------------------------------------------------- the sampled path


def test_every_sampled_candidate_spares_every_pinned_fp32_group() -> None:
    reduced = space()
    for config in sampled(reduced):
        assert set(reduced.pinned_fp32) <= set(config.quant.excluded_groups)


def test_no_sampled_candidate_spares_a_pinned_int8_group() -> None:
    reduced = space()
    for config in sampled(reduced):
        assert not set(config.quant.excluded_groups) & set(reduced.pinned_int8)


def test_the_sampler_creates_no_parameter_for_a_pinned_group() -> None:
    reduced = space()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=optuna.samplers.NSGAIISampler(seed=0))
    study.optimize(lambda trial: (suggest_within(trial, reduced), 0.0)[1], n_trials=5)
    for trial in study.trials:
        precision = {key for key in trial.params if key.startswith("fp32__")}
        assert precision == {f"fp32__{group}" for group in reduced.searchable}


def test_the_sampled_candidates_stay_inside_the_reduced_space() -> None:
    reduced = space()
    allowed = {tuple(sorted(vector)) for vector in precision_vectors(reduced)}
    for config in sampled(reduced):
        assert tuple(sorted(config.quant.excluded_groups)) in allowed


def test_a_space_with_nothing_searchable_still_samples_the_pins() -> None:
    reduced = space(searchable=(), pinned_fp32=("fc",))
    for config in sampled(reduced, trials=5):
        assert config.quant.excluded_groups == ("fc",)


# --------------------------------------------------------- the enumerated path


@pytest.mark.parametrize("round_", STRUCTURE_ROUNDS, ids=lambda r: r.label)
def test_every_enumerated_candidate_obeys_the_pins(round_) -> None:
    reduced = space()
    for vector in precision_vectors(reduced):
        config = candidate_for(vector, round_)
        excluded = set(config.quant.excluded_groups)
        assert set(reduced.pinned_fp32) <= excluded
        assert not excluded & set(reduced.pinned_int8)
        assert excluded - set(reduced.pinned_fp32) <= set(reduced.searchable)
