"""What gets measured is decided here, so pin the schedule and the budget."""

from __future__ import annotations

import json

import pytest

from src.search.plan import (
    CANONICAL_QUANT,
    CANONICAL_RUN,
    DEFAULT_ROUNDS,
    ENUMERATE,
    MAX_DEFAULT_BUDGET,
    NSGA2,
    STRUCTURE_ROUNDS,
    all_rounds,
    default_budget,
    enumerate_candidates,
    load_space,
    make_plan,
    population_for,
    precision_vectors,
)
from src.search.reduce import Band, ReducedSpace, group_fingerprint

GROUPS = ("conv1", "fc", "layer1", "layer2", "layer3")


def space(searchable=("conv1", "fc", "layer1"), pinned_fp32=(), model="resnet18_cifar") -> ReducedSpace:
    pinned_int8 = tuple(g for g in GROUPS if g not in searchable and g not in pinned_fp32)
    return ReducedSpace(
        model=model,
        pinned_fp32=tuple(pinned_fp32),
        pinned_int8=pinned_int8,
        searchable=tuple(searchable),
        fingerprint=group_fingerprint(GROUPS),
        band=Band(0.02, "sentinel", 3.41),
    )


def test_every_precision_vector_appears_exactly_once() -> None:
    vectors = precision_vectors(space())
    assert len(vectors) == 8 == len(set(vectors))


def test_vectors_are_ordered_by_how_much_they_spare() -> None:
    # A truncated budget then keeps the candidates closest to full INT8.
    vectors = precision_vectors(space())
    assert vectors[0] == ()
    assert [len(v) for v in vectors] == sorted(len(v) for v in vectors)


def test_pinned_fp32_is_in_every_vector_and_pinned_int8_is_in_none() -> None:
    plan_space = space(searchable=("conv1", "layer1"), pinned_fp32=("fc",))
    for vector in precision_vectors(plan_space):
        assert "fc" in vector
        assert not set(vector) & set(plan_space.pinned_int8)


def test_a_space_with_nothing_searchable_is_one_vector() -> None:
    assert precision_vectors(space(searchable=())) == [()]


def test_a_round_covers_every_vector_before_the_next_round_starts() -> None:
    plan_space = space()
    candidates, completed, truncated = enumerate_candidates(plan_space, budget=24)
    labels = [label for label, _ in candidates]
    # Three full rounds of eight, in schedule order and not interleaved.
    assert labels[:8] == ["canonical"] * 8
    assert labels[8:16] == ["no-arena"] * 8
    assert completed == ["canonical", "no-arena", "per-tensor"]
    assert truncated is None


def test_enumeration_stops_at_exactly_the_budget_and_names_the_cut_round() -> None:
    candidates, completed, truncated = enumerate_candidates(space(), budget=10)
    assert len(candidates) == 10
    assert completed == ["canonical"]
    assert truncated == "no-arena"


def test_enumeration_is_deterministic() -> None:
    first = enumerate_candidates(space(), budget=40)[0]
    second = enumerate_candidates(space(), budget=40)[0]
    assert [c.hash for _, c in first] == [c.hash for _, c in second]


def test_no_candidate_is_measured_twice() -> None:
    candidates, _, _ = enumerate_candidates(space(), budget=200)
    hashes = [config.hash for _, config in candidates]
    assert len(hashes) == len(set(hashes))


def test_the_canonical_round_is_the_canonical_recipe() -> None:
    candidates, _, _ = enumerate_candidates(space(), budget=1)
    label, config = candidates[0]
    assert label == "canonical"
    assert config.quant == CANONICAL_QUANT
    assert config.run == CANONICAL_RUN


@pytest.mark.parametrize("round_", STRUCTURE_ROUNDS[1:], ids=lambda r: r.label)
def test_each_structure_round_moves_exactly_one_factor(round_) -> None:
    assert len(round_.quant) + len(round_.run) == 1


def test_calibration_rounds_come_after_every_structure_round() -> None:
    labels = [r.label for r in all_rounds()]
    first_calibration = next(i for i, label in enumerate(labels) if label.startswith("calib-"))
    assert first_calibration == len(STRUCTURE_ROUNDS)


def test_calibration_rounds_never_repeat_the_canonical_setting() -> None:
    for round_ in all_rounds():
        if round_.label.startswith("calib-"):
            assert (round_.quant["calibration_method"], round_.quant["calibration_size"]) != (
                CANONICAL_QUANT.calibration_method,
                CANONICAL_QUANT.calibration_size,
            )


def test_a_space_that_fits_the_budget_is_enumerated() -> None:
    plan = make_plan("resnet18_cifar", space(), 78.5, 0.2, trials=40)
    assert plan.strategy == ENUMERATE
    assert len(plan.candidates) == 40


def test_a_space_larger_than_the_budget_falls_back_to_sampling() -> None:
    big = space(searchable=GROUPS)
    plan = make_plan("resnet18_cifar", big, 78.5, 0.2, trials=16)
    assert plan.strategy == NSGA2
    assert plan.candidates == ()
    assert plan.population_size == population_for(16)


def test_the_boundary_enumerates_rather_than_samples() -> None:
    # 2^n == budget is still an exact answer, and exact beats sampled.
    plan = make_plan("resnet18_cifar", space(), 78.5, 0.2, trials=8)
    assert plan.strategy == ENUMERATE


def test_the_default_budget_buys_five_complete_rounds() -> None:
    plan_space = space()
    assert default_budget(plan_space) == DEFAULT_ROUNDS * 8
    plan = make_plan("resnet18_cifar", plan_space, 78.5, 0.2)
    assert len(plan.rounds_completed) == DEFAULT_ROUNDS
    assert plan.round_truncated is None


def test_the_default_budget_is_capped_for_a_space_it_cannot_enumerate() -> None:
    # 5 * 2^n on a large space would ask for tens of thousands of measurements.
    huge = ReducedSpace(
        model="imported",
        pinned_fp32=(),
        pinned_int8=(),
        searchable=tuple(f"block{index}" for index in range(13)),
        fingerprint=group_fingerprint(tuple(f"block{index}" for index in range(13))),
        band=Band(0.02, "sentinel", 3.41),
    )
    assert huge.reduced_size == 8192
    assert default_budget(huge) == MAX_DEFAULT_BUDGET
    assert make_plan("imported", huge, 78.5, 0.2).strategy == NSGA2


def test_the_population_stays_small_enough_to_breed() -> None:
    assert population_for(40) == 10
    assert population_for(8) == 4
    assert population_for(24) == 6


def test_the_threshold_is_the_baseline_less_the_budget() -> None:
    plan = make_plan("resnet18_cifar", space(), 78.5, 0.2, trials=8)
    assert plan.threshold_top1 == pytest.approx(78.3)


def test_the_plan_serializes_every_candidate_in_full() -> None:
    plan = make_plan("resnet18_cifar", space(), 78.5, 0.2, trials=8)
    document = plan.as_dict()
    assert len(document["candidates"]) == 8
    # Full configs, not labels: this is what retires the describe()-parsing path.
    assert document["candidates"][0]["quant"]["calibration_method"] == "minmax"
    assert document["space"]["band"]["source"] == "sentinel"
    json.dumps(document)  # must survive the round trip to the report


def test_a_missing_space_is_refused_before_any_device_work(tmp_path) -> None:
    with pytest.raises(SystemExit, match="No search space"):
        load_space(tmp_path / "absent.json", "resnet18_cifar", GROUPS)


def test_a_space_for_another_model_is_refused(tmp_path) -> None:
    path = tmp_path / "search_space.json"
    path.write_text(json.dumps(space(model="custom_cnn").as_dict()), encoding="utf-8")
    with pytest.raises(SystemExit, match="is for"):
        load_space(path, "resnet18_cifar", GROUPS)


def test_a_space_cut_against_a_different_graph_is_refused(tmp_path) -> None:
    path = tmp_path / "search_space.json"
    path.write_text(json.dumps(space().as_dict()), encoding="utf-8")
    with pytest.raises(SystemExit, match="different group set"):
        load_space(path, "resnet18_cifar", ("conv1", "fc"))


def test_the_shared_artifact_and_the_stage_file_both_load(tmp_path) -> None:
    single = space().as_dict()
    stage = tmp_path / "stage.json"
    stage.write_text(json.dumps(single), encoding="utf-8")
    shared = tmp_path / "shared.json"
    shared.write_text(json.dumps({"resnet18_cifar": single}), encoding="utf-8")

    assert load_space(stage, "resnet18_cifar", GROUPS) == load_space(shared, "resnet18_cifar", GROUPS)
