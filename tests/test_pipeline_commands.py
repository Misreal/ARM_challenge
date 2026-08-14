"""Every command the executors build must parse against the module it invokes.

This is the test the `--study` break needed: `pipeline_device` passed a flag
`final_test` did not define and omitted one it hard-refuses without, and nothing
caught it because the device path is not exercised by the mock run.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest

from src.pipeline import STAGES
from src.runs import create_run

# Stages whose work happens over SSH rather than through a subprocess: there is
# no argv to check, and the module runs on the Pi with its own entry point.
NO_SUBPROCESS = {"quant_baselines", "sensitivity"}


def make(tmp_path, **overrides: Any):
    recipe: dict[str, Any] = {
        "model": "resnet18_cifar",
        "budget_pt": 0.2,
        "trials": 4,
        "venue": "pi",
        "study": "resnet18_cifar_budget0.2_v4",
        "runs_dir": tmp_path,
    }
    recipe.update(overrides)
    return create_run(**recipe)


def captured_argv(monkeypatch, run, module: str = "src.pipeline_device") -> dict[str, list[str]]:
    """Drive an executor with everything stubbed out, keeping only the commands."""
    executors = importlib.import_module(module)
    calls: list[list[str]] = []

    monkeypatch.setattr(executors, "execute", lambda argv, echo=True: calls.append(argv))
    if module == "src.pipeline_device":
        monkeypatch.setattr(executors.PiConnection, "load", classmethod(lambda cls: object()))
        monkeypatch.setattr(executors, "RemoteBenchmarker", lambda connection: _NoDevice())
        monkeypatch.setattr(executors, "on_device", lambda *a, **k: None)
        monkeypatch.setattr(executors, "adopt_file", lambda *a, **k: {})
        # Force the export branch: this checkout has the graph already, so the
        # baseline stage would adopt it and build no command to check.
        monkeypatch.setattr(executors, "already_exported", lambda model: False)

    executor = executors.make_executor(run)
    by_stage: dict[str, list[str]] = {}
    for stage in STAGES:
        if stage.name in NO_SUBPROCESS:
            continue
        calls.clear()
        try:
            executor(stage, run)
        except Exception:  # a stage may still want files we did not fake
            pass
        if calls:
            by_stage[stage.name] = calls[-1]
    return by_stage


class _NoDevice:
    def push_code(self) -> None:
        return None


def parses(argv: list[str]) -> None:
    """Feed one `python -m module ...` command to that module's own parser."""
    assert argv[:2] == [sys.executable, "-m"], argv
    module = importlib.import_module(argv[2])
    if not hasattr(module, "parse_args"):
        pytest.fail(f"{argv[2]} has no parse_args(), so its command cannot be checked")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "argv", [argv[2], *argv[3:]])
        module.parse_args()


@pytest.mark.parametrize("stage", [s.name for s in STAGES if s.name not in NO_SUBPROCESS])
def test_the_device_executor_builds_a_command_its_module_accepts(tmp_path, monkeypatch, stage) -> None:
    argv = captured_argv(monkeypatch, make(tmp_path))
    assert stage in argv, f"{stage} issued no subprocess; move it to NO_SUBPROCESS deliberately"
    parses(argv[stage])


def test_the_mock_executor_builds_a_command_its_module_accepts(tmp_path, monkeypatch) -> None:
    run = make(tmp_path, venue="mock", study="resnet18_cifar_budget0.2_v4_mock")
    argv = captured_argv(monkeypatch, run, "src.pipeline_mock")
    assert "study" in argv, "the mock path stopped invoking the study"
    for command in argv.values():
        parses(command)


def test_the_final_test_stage_confirms_the_sealed_split(tmp_path, monkeypatch) -> None:
    # The module refuses to run without it, so a pipeline that forgets it fails
    # at the last stage of an overnight campaign.
    argv = captured_argv(monkeypatch, make(tmp_path))["final_test"]
    assert "--confirm" in argv
    assert "--study" in argv


def test_the_study_stage_passes_the_population_the_run_recorded(tmp_path, monkeypatch) -> None:
    run = make(tmp_path, population_size=20, study="resnet18_cifar_budget0.2_v4_pop20")
    for module in ("src.pipeline_device", "src.pipeline_mock"):
        argv = captured_argv(monkeypatch, run, module)["study"]
        assert argv[argv.index("--population-size") + 1] == "20", module
