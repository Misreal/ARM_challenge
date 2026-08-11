"""The orchestrator decides what to re-run, so pin when it skips and when it stops."""

from __future__ import annotations

import json

import pytest

from src.pipeline import BY_NAME, STAGE_NAMES, resolve_order, run_pipeline, summarize
from src.runs import create_run


def make(tmp_path, venue="mock"):
    return create_run(
        model="resnet18_cifar",
        budget_pt=0.2,
        trials=4,
        venue=venue,
        study="resnet18_cifar_budget0.2_v2_mock",
        runs_dir=tmp_path,
    )


def recorder(run, written=(), fail=(), skip=()):
    """An executor that writes each stage's file, so `needs` can be satisfied."""
    seen: list[str] = []

    def executor(stage, run):
        seen.append(stage.name)
        if stage.name in fail:
            raise RuntimeError("boom")
        if stage.name in skip:
            return None
        if stage.name != "dashboard":
            run.paths.stage(stage.name).write_text(json.dumps({"stage": stage.name}))
        else:
            run.paths.dashboard.write_text("<p>page</p>", encoding="utf-8")
        return {"ok": True}

    return executor, seen


def test_every_stage_only_depends_on_stages_declared_before_it() -> None:
    # A cycle or a forward reference would make resume order meaningless.
    for index, name in enumerate(STAGE_NAMES):
        for need in BY_NAME[name].needs:
            assert STAGE_NAMES.index(need) < index, f"{name} needs {need}, declared later"


def test_the_dashboard_is_the_last_stage() -> None:
    assert STAGE_NAMES[-1] == "dashboard"


def test_selecting_stages_keeps_declaration_order_not_the_order_given() -> None:
    chosen = resolve_order(("dashboard", "baseline"))
    assert [stage.name for stage in chosen] == ["baseline", "dashboard"]


def test_an_unknown_stage_name_is_refused() -> None:
    with pytest.raises(SystemExit):
        resolve_order(("polish",))


def test_a_full_run_executes_every_stage_once(tmp_path) -> None:
    run = make(tmp_path)
    executor, seen = recorder(run)
    outcomes = run_pipeline(run, executor)
    assert seen == list(STAGE_NAMES)
    assert set(outcomes.values()) == {"ok"}


def test_a_second_run_repeats_nothing(tmp_path) -> None:
    run = make(tmp_path)
    run_pipeline(run, recorder(run)[0])
    executor, seen = recorder(run)
    outcomes = run_pipeline(run, executor)
    assert seen == []
    assert set(outcomes.values()) == {"skipped"}


def test_force_repeats_work_that_was_already_done(tmp_path) -> None:
    run = make(tmp_path)
    run_pipeline(run, recorder(run)[0])
    executor, seen = recorder(run)
    run_pipeline(run, executor, force=True)
    assert seen == list(STAGE_NAMES)


def test_a_failed_stage_stops_the_run_rather_than_carrying_on(tmp_path) -> None:
    run = make(tmp_path)
    executor, seen = recorder(run, fail=("sensitivity",))
    outcomes = run_pipeline(run, executor)
    # Everything after sensitivity depends on it directly or through the join,
    # so continuing would only produce a page built on a missing measurement.
    assert seen == ["baseline", "quant_baselines", "sensitivity"]
    assert outcomes["sensitivity"] == "failed"
    assert "search_space" not in outcomes


def test_a_stage_that_does_not_apply_is_not_a_failure(tmp_path) -> None:
    run = make(tmp_path)
    executor, seen = recorder(run, skip=("sentinel",))
    outcomes = run_pipeline(run, executor)
    assert outcomes["sentinel"] == "n/a"
    assert outcomes["dashboard"] == "ok"


def test_a_stage_that_did_not_apply_is_attempted_again_next_time(tmp_path) -> None:
    run = make(tmp_path)
    run_pipeline(run, recorder(run, skip=("sentinel",))[0])
    executor, seen = recorder(run, skip=("sentinel",))
    run_pipeline(run, executor)
    # A device may be attached next time, and then there is a noise floor to take.
    assert seen == ["sentinel"]


def test_running_one_stage_alone_is_blocked_when_its_input_is_missing(tmp_path) -> None:
    run = make(tmp_path)
    executor, seen = recorder(run)
    outcomes = run_pipeline(run, executor, only=("cost_benefit",))
    assert outcomes == {"cost_benefit": "blocked"}
    assert seen == []


def test_the_summary_counts_each_outcome(tmp_path) -> None:
    assert summarize({"a": "ok", "b": "ok", "c": "skipped"}) == "2 ok, 1 skipped"
