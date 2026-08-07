"""Tests for the host-side Pi driver.

No SSH is involved: the transport is faked, so these tests cover the decisions
the driver makes -- what identifies a measurement, what may be served from
cache, what is worth retrying -- rather than whether OpenSSH works. Those
decisions are where a long campaign goes silently wrong: a cache key that
ignores a field replays the wrong number, and a retry policy that re-runs
deterministic failures turns a 3-hour campaign into a 9-hour one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.bench.agent import BenchSpec
from src.bench.protocol import RESULT_SCHEMA, SPEC_SCHEMA
from src.bench.remote import (
    CommandResult,
    MeasurementCache,
    PiConnection,
    RemoteBenchmarker,
    TransportError,
    spec_cache_key,
)
from src.quant.config import DeploymentConfig, QuantConfig, RunConfig

CONNECTION = PiConnection(
    host="172.20.10.2",
    user="tester",
    remote_root="/home/tester/arm_challenge",
    python="/home/tester/armopt/bin/python",
)


def _spec(**overrides) -> BenchSpec:
    defaults = {
        "model": "resnet18_cifar",
        "config": DeploymentConfig(
            quant=QuantConfig(quant_type="static", per_channel=True),
            run=RunConfig(intra_op_num_threads=4),
        ),
        "warmup": 20,
        "iterations": 100,
    }
    return BenchSpec(**{**defaults, **overrides})


def _ok_result(admissible: bool = True, status: str = "ok") -> dict:
    return {
        "schema": RESULT_SCHEMA,
        "status": status,
        "model": "resnet18_cifar",
        "admissible": admissible,
        "latency": {"median_ms": 3.34},
        "peak_rss_mb": 101.5,
    }


class FakeTransport:
    """Records commands and replays canned outcomes."""

    def __init__(self, result_payload: dict | None = None, run_results=None) -> None:
        self.commands: list[str] = []
        self.pushed: list[tuple[Path, str]] = []
        self.result_payload = result_payload
        self.run_results = list(run_results or [])
        self.fetch_calls = 0

    def _next(self) -> CommandResult:
        if self.run_results:
            outcome = self.run_results.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return CommandResult(0, "", "", 0.01)

    def run(self, remote_command: str, timeout: float | None = None) -> CommandResult:
        self.commands.append(remote_command)
        return self._next()

    def push(self, local: Path, remote: str, recursive: bool = False) -> CommandResult:
        self.pushed.append((local, remote))
        return CommandResult(0, "", "", 0.01)

    def fetch(self, remote: str, local: Path) -> CommandResult:
        self.fetch_calls += 1
        if self.result_payload is None:
            return CommandResult(1, "", "No such file", 0.01)
        local.write_text(json.dumps(self.result_payload), encoding="utf-8")
        return CommandResult(0, "", "", 0.01)


class TestSpecCacheKey:
    def test_the_same_measurement_maps_to_the_same_key(self):
        assert spec_cache_key(_spec()) == spec_cache_key(_spec())

    def test_thread_count_changes_the_key(self):
        # Threads do not change the artifact but do change the latency, so two
        # thread settings must never share a cached measurement.
        other = _spec(
            config=DeploymentConfig(
                quant=QuantConfig(quant_type="static", per_channel=True),
                run=RunConfig(intra_op_num_threads=1),
            )
        )
        assert spec_cache_key(_spec()) != spec_cache_key(other)

    def test_iteration_count_changes_the_key(self):
        assert spec_cache_key(_spec()) != spec_cache_key(_spec(iterations=500))

    def test_the_readiness_policy_does_not_change_the_key(self):
        # The policy decides whether a number may be taken, not what it is.
        from src.portable.device import ReadinessPolicy

        relaxed = _spec(policy=ReadinessPolicy(max_start_temperature_c=70.0))
        assert spec_cache_key(_spec()) == spec_cache_key(relaxed)


class TestMeasurementCache:
    def test_round_trips_an_admissible_result(self, tmp_path):
        cache = MeasurementCache(tmp_path)
        spec = _spec()

        assert cache.put(spec, _ok_result()) is True
        assert cache.get(spec)["latency"]["median_ms"] == pytest.approx(3.34)

    def test_refuses_to_cache_an_inadmissible_measurement(self, tmp_path):
        # Throttled or drifting: an environmental fact, not a property of the
        # candidate. Caching it would freeze a bad number into the campaign.
        cache = MeasurementCache(tmp_path)

        assert cache.put(_spec(), _ok_result(admissible=False)) is False
        assert cache.get(_spec()) is None

    def test_refuses_to_cache_a_device_not_ready_result(self, tmp_path):
        cache = MeasurementCache(tmp_path)

        stored = cache.put(_spec(), {**_ok_result(), "status": "device_not_ready"})

        assert stored is False

    def test_caches_deterministic_build_failures(self, tmp_path):
        # A config that cannot be quantized will not become quantizable on a
        # retry, and re-deriving that costs Pi time in every later campaign.
        cache = MeasurementCache(tmp_path)

        assert cache.put(_spec(), {**_ok_result(status="build_failed")}) is True

    def test_ignores_a_result_written_by_an_older_schema(self, tmp_path):
        cache = MeasurementCache(tmp_path)
        path = cache.path_for(_spec())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema": "bench_result/0", "status": "ok"}), encoding="utf-8")

        assert cache.get(_spec()) is None

    def test_survives_a_truncated_cache_file(self, tmp_path):
        # An interrupted write must cost one remeasurement, not a crashed campaign.
        cache = MeasurementCache(tmp_path)
        path = cache.path_for(_spec())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"schema": "bench_res', encoding="utf-8")

        assert cache.get(_spec()) is None


class TestRemoteBenchmarker:
    def _runner(self, tmp_path, transport) -> RemoteBenchmarker:
        return RemoteBenchmarker(
            CONNECTION,
            transport=transport,
            cache=MeasurementCache(tmp_path),
            sleep=lambda _: None,
        )

    def test_builds_before_timing_in_separate_invocations(self, tmp_path):
        # The build must not share a process with the timing, or peak RSS
        # reports the calibrator instead of the model.
        transport = FakeTransport(result_payload=_ok_result())
        runner = self._runner(tmp_path, transport)

        runner.measure(_spec())

        agent_calls = [command for command in transport.commands if "src.bench.agent" in command]
        assert len(agent_calls) == 2
        assert "--build-only" in agent_calls[0]
        assert "--build-only" not in agent_calls[1]

    def test_a_second_call_is_served_from_cache(self, tmp_path):
        transport = FakeTransport(result_payload=_ok_result())
        runner = self._runner(tmp_path, transport)

        first = runner.measure(_spec())
        commands_after_first = len(transport.commands)
        second = runner.measure(_spec())

        assert first["from_cache"] is False
        assert second["from_cache"] is True
        assert len(transport.commands) == commands_after_first

    def test_an_inadmissible_result_is_measured_again_next_time(self, tmp_path):
        transport = FakeTransport(result_payload=_ok_result(admissible=False))
        runner = self._runner(tmp_path, transport)

        runner.measure(_spec())
        commands_after_first = len(transport.commands)
        runner.measure(_spec())

        assert len(transport.commands) > commands_after_first

    def test_an_agent_failure_is_returned_not_retried(self, tmp_path):
        # Deterministic: retrying spends Pi time to reach the same answer.
        failure = {**_ok_result(status="build_failed"), "error": "unsupported op"}
        transport = FakeTransport(
            result_payload=failure,
            run_results=[
                CommandResult(0, "", "", 0.01),  # mkdir
                CommandResult(1, "", "quantization failed", 0.01),  # build
            ],
        )
        runner = self._runner(tmp_path, transport)

        result = runner.measure(_spec())

        assert result["status"] == "build_failed"
        build_calls = [c for c in transport.commands if "--build-only" in c]
        assert len(build_calls) == 1

    def test_transport_errors_are_retried_then_reported(self, tmp_path):
        transport = FakeTransport(
            run_results=[TransportError("connection reset")] * 3,
        )
        runner = self._runner(tmp_path, transport)

        with pytest.raises(TransportError, match="after 3 attempts"):
            runner.measure(_spec())

    def test_a_missing_result_file_becomes_a_transport_failure(self, tmp_path):
        # The agent never wrote anything: distinguishable from a candidate that
        # failed, because there is no result document to explain why.
        transport = FakeTransport(result_payload=None)
        runner = self._runner(tmp_path, transport)

        result = runner.measure(_spec())

        assert result["status"] == "transport_failed"
        assert result["config_hash"] == _spec().config.hash

    def test_device_state_uses_the_agent_probe(self, tmp_path):
        state = {"machine": "aarch64", "governor": "performance"}
        transport = FakeTransport(run_results=[CommandResult(0, json.dumps(state), "", 0.01)])
        runner = self._runner(tmp_path, transport)

        assert runner.device_state() == state
        assert "--print-device" in transport.commands[0]


class TestPiConnection:
    def test_builds_an_ssh_command_that_cannot_hang_on_a_password(self):
        argv = CONNECTION.ssh_argv("echo hi")

        assert "BatchMode=yes" in argv
        assert argv[-2] == "tester@172.20.10.2"

    def test_scp_uses_capital_p_for_the_port(self):
        connection = PiConnection(
            host="h", user="u", remote_root="/r", port=2222
        )
        argv = connection.scp_argv("a", "b")

        assert "-P" in argv and "2222" in argv

    def test_remote_paths_stay_posix_on_windows(self):
        assert CONNECTION.remote_path("src", "bench") == "/home/tester/arm_challenge/src/bench"

    def test_missing_configuration_names_what_is_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="host, user, remote_root"):
            PiConnection.load(tmp_path / "absent.json")

    def test_loads_from_file(self, tmp_path):
        target = tmp_path / "pi_target.json"
        target.write_text(
            json.dumps({"host": "h", "user": "u", "remote_root": "/r", "python": "py"}),
            encoding="utf-8",
        )

        connection = PiConnection.load(target)

        assert connection.host == "h"
        assert connection.python == "py"


class TestBenchSpecContract:
    def test_round_trips_through_json(self):
        spec = _spec()

        restored = BenchSpec.from_dict(json.loads(json.dumps(spec.as_dict())))

        assert restored == spec

    def test_rejects_an_unknown_schema(self):
        # Host and device are separately updated checkouts; a silently-parsed
        # older spec would return a number measured under the wrong settings.
        payload = _spec().as_dict()
        payload["schema"] = "bench_spec/99"

        with pytest.raises(ValueError, match="different versions"):
            BenchSpec.from_dict(payload)

    def test_declares_the_current_schema(self):
        assert _spec().as_dict()["schema"] == SPEC_SCHEMA
