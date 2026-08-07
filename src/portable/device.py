"""Device-state guards: governor, throttling and temperature checks that decide
whether a timing is worth recording. Probes return None off the Pi, and the
guards only enforce on aarch64, so this still imports on the dev box.
"""

from __future__ import annotations

import platform
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.portable.benchmark import read_cpu_temperature

try:  # POSIX-only, same rationale as in benchmark.py
    import resource
except ImportError:  # pragma: no cover - platform-dependent
    resource = None  # type: ignore[assignment]

CPUFREQ_ROOT = Path("/sys/devices/system/cpu")
VCGENCMD = "vcgencmd"
VCGENCMD_TIMEOUT_S = 5.0

TARGET_GOVERNOR = "performance"
DEFAULT_MAX_START_TEMPERATURE_C = 60.0
DEFAULT_COOLDOWN_TIMEOUT_S = 300.0
DEFAULT_COOLDOWN_POLL_S = 5.0

# Latency is only a result on the target ISA; see src/quant/measure.py.
TARGET_MACHINES = ("aarch64", "arm64")

# Pi throttle bitmask: bits 0-3 are live, bits 16-19 sticky since boot.
LIVE_THROTTLE_BITS = {
    0: "under_voltage_now",
    1: "arm_frequency_capped_now",
    2: "currently_throttled",
    3: "soft_temperature_limit_now",
}
STICKY_THROTTLE_BITS = {
    16: "under_voltage_since_boot",
    17: "arm_frequency_capped_since_boot",
    18: "throttled_since_boot",
    19: "soft_temperature_limit_since_boot",
}

_THROTTLED_PATTERN = re.compile(r"throttled=0x([0-9a-fA-F]+)")


# Raised, not warned: in a 300-run unattended campaign nobody reads warnings.
class DeviceNotReady(RuntimeError):
    """The host state would make any timing misleading."""


def _decode_bits(mask: int, names: dict[int, str]) -> tuple[str, ...]:
    return tuple(name for bit, name in sorted(names.items()) if mask & (1 << bit))


def _run_vcgencmd(*args: str) -> str | None:
    """Output of a vcgencmd call, or None where the tool is absent."""
    try:
        completed = subprocess.run(
            [VCGENCMD, *args],
            capture_output=True,
            text=True,
            timeout=VCGENCMD_TIMEOUT_S,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def read_throttle_mask() -> int | None:
    """The `vcgencmd get_throttled` bitmask, or None where unavailable."""
    output = _run_vcgencmd("get_throttled")
    if output is None:
        return None
    match = _THROTTLED_PATTERN.search(output)
    return int(match.group(1), 16) if match else None


@dataclass(frozen=True)
class ThrottleState:
    """One reading of the firmware's throttle bitmask."""

    raw: int | None
    live: tuple[str, ...]
    sticky: tuple[str, ...]

    @property
    def available(self) -> bool:
        return self.raw is not None

    @classmethod
    def read(cls) -> ThrottleState:
        mask = read_throttle_mask()
        if mask is None:
            return cls(raw=None, live=(), sticky=())
        return cls(
            raw=mask,
            live=_decode_bits(mask, LIVE_THROTTLE_BITS),
            sticky=_decode_bits(mask, STICKY_THROTTLE_BITS),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "hex": None if self.raw is None else f"0x{self.raw:x}",
            "live": list(self.live),
            "sticky": list(self.sticky),
            "available": self.available,
        }


def throttle_events_between(before: ThrottleState, after: ThrottleState) -> tuple[str, ...]:
    """Throttle events attributable to the window between two readings."""
    # Sticky bits are diffed, so a brownout yesterday does not invalidate today.
    if not (before.available and after.available):
        return ()
    newly_sticky = tuple(name for name in after.sticky if name not in before.sticky)
    return tuple(sorted(set(after.live) | set(newly_sticky)))


def read_governors() -> tuple[str, ...]:
    """Scaling governor of each CPU, in cpu-index order; empty where unavailable."""
    paths = sorted(
        CPUFREQ_ROOT.glob("cpu[0-9]*/cpufreq/scaling_governor"),
        key=lambda path: int(re.sub(r"\D", "", path.parts[-3])),
    )
    governors: list[str] = []
    for path in paths:
        try:
            governors.append(path.read_text().strip())
        except OSError:
            continue
    return tuple(governors)


def read_clock_mhz() -> float | None:
    """Current CPU0 clock in MHz, read from sysfs (in kHz) where exposed."""
    try:
        kilohertz = int((CPUFREQ_ROOT / "cpu0/cpufreq/scaling_cur_freq").read_text().strip())
    except (OSError, ValueError):
        return None
    return kilohertz / 1000.0


def read_process_cpu_seconds() -> float | None:
    """User+system CPU time for this process, or None off POSIX."""
    # Over wall time this gives cores-busy: proof that 4 threads used 4 cores.
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


@dataclass(frozen=True)
class DeviceState:
    """A snapshot of everything about the host that can invalidate a timing."""

    platform: str
    machine: str
    is_target: bool
    governors: tuple[str, ...]
    clock_mhz: float | None
    temperature_c: float | None
    throttle: ThrottleState

    @property
    def governor(self) -> str | None:
        """The single governor in force, or None if cores disagree (or unknown)."""
        unique = set(self.governors)
        return unique.pop() if len(unique) == 1 else None

    @property
    def governor_is_pinned(self) -> bool:
        return self.governor == TARGET_GOVERNOR

    def as_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "machine": self.machine,
            "is_target": self.is_target,
            "governors": list(self.governors),
            "governor": self.governor,
            "governor_is_pinned": self.governor_is_pinned,
            "clock_mhz": self.clock_mhz,
            "temperature_c": self.temperature_c,
            "throttle": self.throttle.as_dict(),
        }


def probe_device() -> DeviceState:
    machine = platform.machine()
    return DeviceState(
        platform=platform.platform(),
        machine=machine,
        is_target=machine.lower() in TARGET_MACHINES,
        governors=read_governors(),
        clock_mhz=read_clock_mhz(),
        temperature_c=read_cpu_temperature(),
        throttle=ThrottleState.read(),
    )


@dataclass(frozen=True)
class CooldownReport:
    """Outcome of waiting for the SoC to reach a starting temperature."""

    requested_c: float
    temperature_c: float | None
    waited_s: float
    reached: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_c": self.requested_c,
            "temperature_c": self.temperature_c,
            "waited_s": round(self.waited_s, 2),
            "reached": self.reached,
            "reason": self.reason,
        }


def wait_until_cool(
    threshold_c: float = DEFAULT_MAX_START_TEMPERATURE_C,
    timeout_s: float = DEFAULT_COOLDOWN_TIMEOUT_S,
    poll_s: float = DEFAULT_COOLDOWN_POLL_S,
    sleep=time.sleep,
    clock=time.monotonic,
) -> CooldownReport:
    """Block until the SoC is at or below `threshold_c`, or until timeout."""
    # sleep/clock are injected so the poll loop is testable without real seconds.
    started = clock()
    temperature = read_cpu_temperature()
    if temperature is None:
        return CooldownReport(threshold_c, None, 0.0, True, "no temperature sensor on this host")

    while temperature is not None and temperature > threshold_c:
        if clock() - started >= timeout_s:
            return CooldownReport(
                threshold_c,
                temperature,
                clock() - started,
                False,
                f"still {temperature:.1f} C after {timeout_s:.0f} s -- check cooling or ambient",
            )
        sleep(poll_s)
        temperature = read_cpu_temperature()

    return CooldownReport(
        threshold_c, temperature, clock() - started, True, "at or below threshold"
    )


@dataclass(frozen=True)
class ReadinessPolicy:
    """What the device must satisfy before a timed section may start."""

    require_governor: bool = True
    require_clean_throttle: bool = True
    max_start_temperature_c: float = DEFAULT_MAX_START_TEMPERATURE_C
    cooldown_timeout_s: float = DEFAULT_COOLDOWN_TIMEOUT_S
    cooldown_poll_s: float = DEFAULT_COOLDOWN_POLL_S

    def as_dict(self) -> dict[str, Any]:
        return {
            "require_governor": self.require_governor,
            "require_clean_throttle": self.require_clean_throttle,
            "max_start_temperature_c": self.max_start_temperature_c,
            "cooldown_timeout_s": self.cooldown_timeout_s,
            "cooldown_poll_s": self.cooldown_poll_s,
        }


@dataclass(frozen=True)
class ReadinessReport:
    """Whether the guards ran, and what they found."""

    enforced: bool
    state: DeviceState
    cooldown: CooldownReport | None
    violations: tuple[str, ...]
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "enforced": self.enforced,
            "violations": list(self.violations),
            "note": self.note,
            "cooldown": None if self.cooldown is None else self.cooldown.as_dict(),
            "device": self.state.as_dict(),
        }


def _governor_violation(state: DeviceState) -> str | None:
    if not state.governors:
        return (
            "cannot read any CPU governor from /sys -- the guard cannot confirm the "
            "clock is pinned, so the timing would be unverifiable"
        )
    if state.governor_is_pinned:
        return None
    observed = state.governor or f"mixed {sorted(set(state.governors))}"
    return (
        f"CPU governor is '{observed}', expected '{TARGET_GOVERNOR}'. Under a scaling "
        "governor the timing measures the frequency scheduler, not the model. "
        "Fix: sudo bash scripts/pi_prepare.sh"
    )


def check_readiness(
    policy: ReadinessPolicy = ReadinessPolicy(),
    *,
    wait: bool = True,
    sleep=time.sleep,
    clock=time.monotonic,
) -> ReadinessReport:
    """Probe the device, optionally cool it, and list any policy violations."""
    # Advisory off-target: an x86 latency is inadmissible anyway, and failing
    # there for a missing vcgencmd would make the harness untestable.
    state = probe_device()
    if not state.is_target:
        return ReadinessReport(
            enforced=False,
            state=state,
            cooldown=None,
            violations=(),
            note=(
                f"host machine '{state.machine}' is not the aarch64 target; device "
                "guards are advisory and the resulting latency is inadmissible"
            ),
        )

    cooldown = (
        wait_until_cool(
            policy.max_start_temperature_c,
            policy.cooldown_timeout_s,
            policy.cooldown_poll_s,
            sleep=sleep,
            clock=clock,
        )
        if wait
        else None
    )

    violations: list[str] = []
    if policy.require_governor:
        violation = _governor_violation(state)
        if violation:
            violations.append(violation)
    if policy.require_clean_throttle and state.throttle.live:
        violations.append(
            f"device is throttling right now ({', '.join(state.throttle.live)}); "
            "check the official 27 W PSU and the active cooler before measuring"
        )
    if cooldown is not None and not cooldown.reached:
        violations.append(f"cooldown gate not met: {cooldown.reason}")

    # Re-probe after cooling: record the state the run starts from, not the
    # state the device was found in.
    if cooldown is not None and cooldown.temperature_c is not None:
        state = DeviceState(
            platform=state.platform,
            machine=state.machine,
            is_target=state.is_target,
            governors=state.governors,
            clock_mhz=read_clock_mhz(),
            temperature_c=cooldown.temperature_c,
            throttle=state.throttle,
        )

    return ReadinessReport(
        enforced=True,
        state=state,
        cooldown=cooldown,
        violations=tuple(violations),
        note="device guards enforced" if not violations else "device guards failed",
    )


def require_ready(
    policy: ReadinessPolicy = ReadinessPolicy(),
    *,
    wait: bool = True,
    sleep=time.sleep,
    clock=time.monotonic,
) -> ReadinessReport:
    """`check_readiness`, but raise `DeviceNotReady` on any violation."""
    report = check_readiness(policy, wait=wait, sleep=sleep, clock=clock)
    if report.violations:
        listed = "\n  - ".join(report.violations)
        raise DeviceNotReady(f"Device is not fit to benchmark:\n  - {listed}")
    return report
