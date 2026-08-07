"""Tests for the device-state guards.

sysfs reads, the firmware call and the clock are injected, so the assertions are
about which readings count as a violation, never about hardware behaviour.
"""

from __future__ import annotations

import pytest

from src.portable import device as dev


def _write_governors(root, values: list[str]) -> None:
    """Lay out a fake /sys/devices/system/cpu tree."""
    for index, value in enumerate(values):
        governor = root / f"cpu{index}" / "cpufreq" / "scaling_governor"
        governor.parent.mkdir(parents=True, exist_ok=True)
        governor.write_text(value)


class TestThrottleState:
    def test_decodes_live_and_sticky_bits(self, monkeypatch):
        # Arrange: bit 2 (throttled now) and bit 18 (throttled since boot).
        monkeypatch.setattr(dev, "read_throttle_mask", lambda: 0x40004)

        # Act
        state = dev.ThrottleState.read()

        # Assert
        assert state.live == ("currently_throttled",)
        assert state.sticky == ("throttled_since_boot",)
        assert state.available is True

    def test_reports_unavailable_where_vcgencmd_is_missing(self, monkeypatch):
        monkeypatch.setattr(dev, "read_throttle_mask", lambda: None)

        state = dev.ThrottleState.read()

        assert state.available is False
        assert state.as_dict()["hex"] is None

    def test_parses_the_firmware_output_format(self, monkeypatch):
        monkeypatch.setattr(dev, "_run_vcgencmd", lambda *args: "throttled=0x50005")

        assert dev.read_throttle_mask() == 0x50005

    def test_returns_none_on_unparseable_output(self, monkeypatch):
        monkeypatch.setattr(dev, "_run_vcgencmd", lambda *args: "VCHI initialization failed")

        assert dev.read_throttle_mask() is None


class TestThrottleEventsBetween:
    def test_a_clean_run_produces_no_events(self):
        before = dev.ThrottleState(raw=0x0, live=(), sticky=())
        after = dev.ThrottleState(raw=0x0, live=(), sticky=())

        assert dev.throttle_events_between(before, after) == ()

    def test_a_sticky_bit_set_before_the_run_is_not_attributed_to_it(self):
        # A brownout yesterday must not invalidate every measurement since.
        before = dev.ThrottleState(raw=0x10000, live=(), sticky=("under_voltage_since_boot",))
        after = dev.ThrottleState(raw=0x10000, live=(), sticky=("under_voltage_since_boot",))

        assert dev.throttle_events_between(before, after) == ()

    def test_a_sticky_bit_appearing_during_the_run_is_attributed(self):
        before = dev.ThrottleState(raw=0x0, live=(), sticky=())
        after = dev.ThrottleState(raw=0x40000, live=(), sticky=("throttled_since_boot",))

        assert dev.throttle_events_between(before, after) == ("throttled_since_boot",)

    def test_a_live_bit_at_the_end_always_counts(self):
        # Still throttling as the run finishes: the timing describes a hot SoC.
        before = dev.ThrottleState(raw=0x0, live=(), sticky=())
        after = dev.ThrottleState(raw=0x4, live=("currently_throttled",), sticky=())

        assert dev.throttle_events_between(before, after) == ("currently_throttled",)

    def test_unavailable_readings_report_nothing_rather_than_guessing(self):
        before = dev.ThrottleState(raw=None, live=(), sticky=())
        after = dev.ThrottleState(raw=0x4, live=("currently_throttled",), sticky=())

        assert dev.throttle_events_between(before, after) == ()


class TestGovernorReading:
    def test_reads_every_core_in_index_order(self, tmp_path, monkeypatch):
        # Arrange: 12 cores, so lexicographic sorting would put cpu10 before cpu2.
        _write_governors(tmp_path, ["performance"] * 12)
        monkeypatch.setattr(dev, "CPUFREQ_ROOT", tmp_path)

        # Act / Assert
        assert dev.read_governors() == ("performance",) * 12

    def test_absent_cpufreq_reads_as_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dev, "CPUFREQ_ROOT", tmp_path)

        assert dev.read_governors() == ()


class TestDeviceState:
    def _state(self, governors: tuple[str, ...], machine: str = "aarch64") -> dev.DeviceState:
        return dev.DeviceState(
            platform="Linux-test",
            machine=machine,
            is_target=machine == "aarch64",
            governors=governors,
            clock_mhz=2400.0,
            temperature_c=45.0,
            throttle=dev.ThrottleState(raw=0x0, live=(), sticky=()),
        )

    def test_uniform_governor_is_reported(self):
        assert self._state(("performance",) * 4).governor == "performance"

    def test_mixed_governors_report_none_rather_than_a_majority(self):
        # A machine where one core scales and three do not is misconfigured, and
        # picking a winner here would hide that from the guard.
        assert self._state(("performance", "performance", "ondemand", "performance")).governor is None

    def test_pinned_only_when_every_core_agrees(self):
        assert self._state(("performance",) * 4).governor_is_pinned is True
        assert self._state(("ondemand",) * 4).governor_is_pinned is False
        assert self._state(("performance", "ondemand")).governor_is_pinned is False


class TestWaitUntilCool:
    def test_returns_immediately_when_already_cool(self, monkeypatch):
        monkeypatch.setattr(dev, "read_cpu_temperature", lambda: 45.0)
        slept: list[float] = []

        report = dev.wait_until_cool(60.0, sleep=slept.append, clock=lambda: 0.0)

        assert report.reached is True
        assert slept == []

    def test_polls_until_the_threshold_is_met(self, monkeypatch):
        # Arrange: cools by 5 C per poll.
        temperatures = iter([72.0, 67.0, 62.0, 58.0])
        monkeypatch.setattr(dev, "read_cpu_temperature", lambda: next(temperatures))
        slept: list[float] = []

        # Act
        report = dev.wait_until_cool(60.0, poll_s=5.0, sleep=slept.append, clock=lambda: 0.0)

        # Assert
        assert report.reached is True
        assert report.temperature_c == pytest.approx(58.0)
        assert slept == [5.0, 5.0, 5.0]

    def test_gives_up_at_the_timeout(self, monkeypatch):
        # Arrange: never cools; the clock advances 60 s per reading.
        monkeypatch.setattr(dev, "read_cpu_temperature", lambda: 75.0)
        ticks = iter([0.0, 60.0, 120.0, 180.0])

        # Act
        report = dev.wait_until_cool(
            60.0, timeout_s=120.0, sleep=lambda _: None, clock=lambda: next(ticks)
        )

        # Assert
        assert report.reached is False
        assert "still 75.0 C" in report.reason

    def test_no_sensor_does_not_block_the_run(self, monkeypatch):
        # The dev box has no thermal_zone0. Blocking there would make the
        # harness untestable off-device for no measurement benefit.
        monkeypatch.setattr(dev, "read_cpu_temperature", lambda: None)

        report = dev.wait_until_cool(60.0, sleep=lambda _: None, clock=lambda: 0.0)

        assert report.reached is True
        assert report.temperature_c is None


class TestReadiness:
    def _patch_device(self, monkeypatch, *, governors, machine="aarch64", live=()):
        state = dev.DeviceState(
            platform="Linux-test",
            machine=machine,
            is_target=machine == "aarch64",
            governors=governors,
            clock_mhz=2400.0,
            temperature_c=45.0,
            throttle=dev.ThrottleState(raw=0x4 if live else 0x0, live=live, sticky=()),
        )
        monkeypatch.setattr(dev, "probe_device", lambda: state)
        monkeypatch.setattr(dev, "read_cpu_temperature", lambda: 45.0)
        monkeypatch.setattr(dev, "read_clock_mhz", lambda: 2400.0)
        return state

    def test_a_pinned_cool_device_passes(self, monkeypatch):
        self._patch_device(monkeypatch, governors=("performance",) * 4)

        report = dev.require_ready(sleep=lambda _: None, clock=lambda: 0.0)

        assert report.enforced is True
        assert report.violations == ()

    def test_a_scaling_governor_is_refused_with_the_fix_in_the_message(self, monkeypatch):
        self._patch_device(monkeypatch, governors=("ondemand",) * 4)

        with pytest.raises(dev.DeviceNotReady, match="pi_prepare.sh"):
            dev.require_ready(sleep=lambda _: None, clock=lambda: 0.0)

    def test_live_throttling_is_refused(self, monkeypatch):
        self._patch_device(
            monkeypatch, governors=("performance",) * 4, live=("under_voltage_now",)
        )

        with pytest.raises(dev.DeviceNotReady, match="under_voltage_now"):
            dev.require_ready(sleep=lambda _: None, clock=lambda: 0.0)

    def test_unreadable_governors_are_refused_rather_than_assumed_fine(self, monkeypatch):
        self._patch_device(monkeypatch, governors=())

        report = dev.check_readiness(sleep=lambda _: None, clock=lambda: 0.0)

        assert report.violations
        assert "cannot read any CPU governor" in report.violations[0]

    def test_guards_are_advisory_off_the_target_machine(self, monkeypatch):
        # The dev box has no governor to pin and no vcgencmd; failing there
        # would block every harness test for a number that is inadmissible
        # anyway.
        self._patch_device(monkeypatch, governors=("ondemand",) * 8, machine="AMD64")

        report = dev.require_ready(sleep=lambda _: None, clock=lambda: 0.0)

        assert report.enforced is False
        assert report.violations == ()
        assert "not the aarch64 target" in report.note
