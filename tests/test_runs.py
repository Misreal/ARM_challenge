"""A run is the unit the whole app is organised around, so pin its contract."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from src.runs import (
    RunPaths,
    completed_stages,
    create_run,
    list_runs,
    load_run,
    read_manifest,
    record_stage,
    run_id_for,
)
from src.search.version import study_name


def make(tmp_path, model="resnet18_cifar", venue="pi", budget_pt=0.2, run_id=None):
    return create_run(
        model=model,
        budget_pt=budget_pt,
        trials=40,
        venue=venue,
        study=study_name(model, budget_pt, mock=venue == "mock"),
        run_id=run_id,
        runs_dir=tmp_path,
    )


def test_the_run_id_carries_model_budget_and_date() -> None:
    when = datetime(2026, 8, 8, tzinfo=UTC)
    assert run_id_for("resnet18_cifar", 0.2, when) == "resnet18_cifar_b0.2_20260808"


def test_every_path_in_a_run_derives_from_its_id(tmp_path) -> None:
    paths = RunPaths.under("demo", tmp_path)
    assert paths.config == tmp_path / "demo" / "run.json"
    assert paths.stage("study") == tmp_path / "demo" / "stages" / "study.json"
    assert paths.dashboard == tmp_path / "demo" / "dashboard" / "index.html"


def test_creating_a_run_writes_a_recipe_that_reloads_identically(tmp_path) -> None:
    created = make(tmp_path)
    loaded = load_run(created.run_id, tmp_path)
    assert loaded.as_dict() == created.as_dict()
    assert loaded.is_measured


def test_recreating_an_existing_run_reuses_it_rather_than_starting_over(tmp_path) -> None:
    first = make(tmp_path)
    record_stage(first, "search", "ok", 12.0)
    again = make(tmp_path, run_id=first.run_id)
    # Resuming must not wipe the stages already done, or an overnight campaign
    # restarts from zero every time the orchestrator is invoked.
    assert completed_stages(again) == {"search"}


def test_a_mock_run_cannot_resume_into_a_device_run(tmp_path) -> None:
    mock = make(tmp_path, venue="mock")
    # One directory holding both simulated and measured numbers is exactly the
    # confusion the venue field exists to prevent.
    with pytest.raises(SystemExit):
        make(tmp_path, venue="pi", run_id=mock.run_id)


def test_an_unknown_venue_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError):
        make(tmp_path, venue="laptop")


def test_only_successful_stages_count_as_complete(tmp_path) -> None:
    run = make(tmp_path)
    record_stage(run, "sensitivity", "ok", 3.0)
    record_stage(run, "search", "failed", 1.0, {"error": "device unreachable"})
    assert completed_stages(run) == {"sensitivity"}
    assert read_manifest(run)["stages"]["search"]["error"] == "device unreachable"


def test_a_failed_stage_can_be_rerun_and_recorded_as_done(tmp_path) -> None:
    run = make(tmp_path)
    record_stage(run, "search", "failed", 1.0)
    record_stage(run, "search", "ok", 90.0)
    assert completed_stages(run) == {"search"}


def test_runs_are_listed_grouped_by_model(tmp_path) -> None:
    make(tmp_path, model="resnet18_cifar")
    make(tmp_path, model="custom_cnn")
    assert [run.model for run in list_runs(tmp_path)] == ["custom_cnn", "resnet18_cifar"]


def test_listing_ignores_directories_that_are_not_runs(tmp_path) -> None:
    make(tmp_path)
    (tmp_path / "scratch").mkdir()
    assert len(list_runs(tmp_path)) == 1


def test_a_recipe_from_a_future_schema_is_refused_rather_than_guessed(tmp_path) -> None:
    run = make(tmp_path)
    document = json.loads(run.paths.config.read_text())
    document["schema"] = "run/99"
    run.paths.config.write_text(json.dumps(document))
    with pytest.raises(SystemExit):
        load_run(run.run_id, tmp_path)
