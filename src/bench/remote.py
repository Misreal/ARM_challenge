"""Drive the Pi from the PC over SSH, with retries and a measurement cache."""

# Key auth is mandatory: under password auth an unattended run does not fail, it
# hangs at a prompt nobody answers, so BatchMode is forced on.
#
#     python -m src.bench.remote --check
#     python -m src.bench.remote --model resnet18_cifar --configs fp32 --threads 1 2 4

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

from src.bench.agent import BenchSpec
from src.bench.protocol import RESULT_SCHEMA
from src.model_index import known_models
from src.portable.benchmark import DEFAULT_ITERATIONS, DEFAULT_WARMUP
from src.quant.baselines import BASELINE_CONFIGS
from src.quant.candidates import CANDIDATE_EXCLUSIONS, candidate_configs, per_group_configs
from src.quant.config import DeploymentConfig, QuantConfig, RunConfig

DEFAULT_TARGET_FILE = Path("pi_target.json")
DEFAULT_CACHE_DIR = Path("artifacts/bench_cache")
DEFAULT_REMOTE_WORK = "/tmp/armopt_bench"

# Retried: dropped link, sleeping device, moved lease. Never retried:
# anything the agent reported, which will report the same thing again.
DEFAULT_RETRIES = 3
RETRY_BACKOFF_S = (2.0, 8.0, 20.0)

# Excludes device_not_ready and transport_failed: facts about a moment,
# not about a candidate.
CACHEABLE_STATUSES = ("ok", "build_failed", "benchmark_failed")

SSH_OPTIONS = (
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=4",
)


class TransportError(RuntimeError):
    """SSH or SCP failed at the transport level -- the agent never ran."""


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    seconds: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True)
class PiConnection:
    """Where the device is and how to reach it."""

    host: str
    user: str
    remote_root: str
    python: str = "python3"
    port: int = 22
    identity_file: str | None = None
    connect_timeout_s: int = 15
    command_timeout_s: int = 1800
    remote_work: str = DEFAULT_REMOTE_WORK

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"

    def ssh_argv(self, remote_command: str) -> list[str]:
        argv = ["ssh", *SSH_OPTIONS, "-o", f"ConnectTimeout={self.connect_timeout_s}"]
        if self.port != 22:
            argv += ["-p", str(self.port)]
        if self.identity_file:
            argv += ["-i", self.identity_file]
        return [*argv, self.target, remote_command]

    def scp_argv(self, source: str, destination: str, recursive: bool = False) -> list[str]:
        argv = ["scp", *SSH_OPTIONS, "-o", f"ConnectTimeout={self.connect_timeout_s}"]
        if self.port != 22:
            # scp spells the port flag with a capital P, unlike ssh.
            argv += ["-P", str(self.port)]
        if self.identity_file:
            argv += ["-i", self.identity_file]
        if recursive:
            argv.append("-r")
        return [*argv, source, destination]

    def remote_path(self, *parts: str) -> str:
        return str(PurePosixPath(self.remote_root, *parts))

    def work_path(self, *parts: str) -> str:
        return str(PurePosixPath(self.remote_work, *parts))

    def as_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "user": self.user,
            "remote_root": self.remote_root,
            "python": self.python,
            "port": self.port,
            "identity_file": self.identity_file,
        }

    @classmethod
    def load(cls, path: Path = DEFAULT_TARGET_FILE) -> PiConnection:
        """Read connection settings from `pi_target.json`, overridable by env."""
        # Gitignored: committing one operator's host points a fresh clone at
        # the wrong machine.
        data: dict[str, Any] = {}
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))

        overrides = {
            "host": os.environ.get("ARMOPT_PI_HOST"),
            "user": os.environ.get("ARMOPT_PI_USER"),
            "remote_root": os.environ.get("ARMOPT_PI_ROOT"),
            "python": os.environ.get("ARMOPT_PI_PYTHON"),
            "identity_file": os.environ.get("ARMOPT_PI_KEY"),
        }
        data.update({key: value for key, value in overrides.items() if value})

        missing = [key for key in ("host", "user", "remote_root") if not data.get(key)]
        if missing:
            raise FileNotFoundError(
                f"Pi connection is not configured (missing {', '.join(missing)}). "
                f"Create {path} with host/user/remote_root/python, or set "
                "ARMOPT_PI_HOST / ARMOPT_PI_USER / ARMOPT_PI_ROOT."
            )
        return cls(**{key: value for key, value in data.items() if key in cls.__annotations__})


class SshTransport:
    """Subprocess-backed ssh/scp. Swapped for a fake in tests."""

    def __init__(self, connection: PiConnection) -> None:
        self.connection = connection

    def _execute(self, argv: Sequence[str], timeout: float) -> CommandResult:
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                list(argv), capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired as error:
            raise TransportError(f"timed out after {timeout:.0f}s: {' '.join(argv[:3])}") from error
        except OSError as error:
            raise TransportError(
                f"could not start {argv[0]!r} -- is the OpenSSH client installed and on PATH? ({error})"
            ) from error
        return CommandResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            seconds=time.perf_counter() - started,
        )

    def run(self, remote_command: str, timeout: float | None = None) -> CommandResult:
        return self._execute(
            self.connection.ssh_argv(remote_command),
            timeout or self.connection.command_timeout_s,
        )

    def push(self, local: Path, remote: str, recursive: bool = False) -> CommandResult:
        destination = f"{self.connection.target}:{remote}"
        return self._execute(
            self.connection.scp_argv(str(local), destination, recursive),
            self.connection.command_timeout_s,
        )

    def fetch(self, remote: str, local: Path) -> CommandResult:
        source = f"{self.connection.target}:{remote}"
        return self._execute(
            self.connection.scp_argv(source, str(local)),
            self.connection.command_timeout_s,
        )


def spec_cache_key(spec: BenchSpec) -> str:
    """Identity of a measurement: the config plus how it was timed."""
    # Excludes the readiness policy: it decides whether a number may be taken,
    # not what the number is.
    payload = {
        "model": spec.model,
        "config": spec.config.hash,
        "warmup": spec.warmup,
        "iterations": spec.iterations,
        "seed": spec.seed,
        "stability_tolerance": spec.stability_tolerance,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


class MeasurementCache:
    """One JSON per measurement, so a crashed campaign resumes where it stopped."""

    def __init__(self, root: Path = DEFAULT_CACHE_DIR) -> None:
        self.root = root

    def path_for(self, spec: BenchSpec) -> Path:
        return self.root / spec.model / f"{spec_cache_key(spec)}.json"

    def get(self, spec: BenchSpec) -> dict[str, Any] | None:
        path = self.path_for(spec)
        if not path.exists():
            return None
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Half-written by an interrupted run; costs one remeasurement.
            return None
        if cached.get("schema") != RESULT_SCHEMA:
            return None
        return cached

    def put(self, spec: BenchSpec, result: dict[str, Any]) -> bool:
        """Store a result if it is reproducible. Returns whether it was stored."""
        if not self.is_cacheable(result):
            return False
        path = self.path_for(spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return True

    @staticmethod
    def is_cacheable(result: dict[str, Any]) -> bool:
        status = result.get("status")
        if status not in CACHEABLE_STATUSES:
            return False
        # An inadmissible ok was throttled or drifting; caching it would
        # freeze a bad number into the campaign.
        if status == "ok" and not result.get("admissible", False):
            return False
        return True


class RemoteBenchmarker:
    """Measure candidates on the device, one spec at a time."""

    def __init__(
        self,
        connection: PiConnection,
        transport: SshTransport | None = None,
        cache: MeasurementCache | None = None,
        retries: int = DEFAULT_RETRIES,
        sleep=time.sleep,
    ) -> None:
        self.connection = connection
        self.transport = transport or SshTransport(connection)
        self.cache = cache or MeasurementCache()
        self.retries = retries
        self._sleep = sleep

    # -- low-level helpers -------------------------------------------------

    def _agent_command(self, arguments: str) -> str:
        return (
            f"cd {self.connection.remote_root} && "
            f"{self.connection.python} -m src.bench.agent {arguments}"
        )

    def _score_command(self, arguments: str) -> str:
        return (
            f"cd {self.connection.remote_root} && "
            f"{self.connection.python} -m src.bench.score {arguments}"
        )

    def _with_retries(self, operation, description: str) -> CommandResult:
        """Run a transport operation, retrying only genuine link failures."""
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                return operation()
            except TransportError as error:
                last = error
            if attempt + 1 < self.retries:
                self._sleep(RETRY_BACKOFF_S[min(attempt, len(RETRY_BACKOFF_S) - 1)])
        raise TransportError(f"{description} failed after {self.retries} attempts: {last}")

    def device_state(self) -> dict[str, Any]:
        """Ask the device to describe itself. The connectivity smoke test."""
        result = self._with_retries(
            lambda: self.transport.run(self._agent_command("--print-device"), timeout=60),
            "device probe",
        )
        if not result.ok:
            raise TransportError(
                f"agent could not run on {self.connection.target}:\n{result.stderr.strip()}"
            )
        return json.loads(result.stdout)

    def push_code(self) -> None:
        """Refresh `src/` on the device so both sides run the same harness."""
        self._with_retries(
            lambda: self.transport.push(Path("src"), self.connection.remote_path(""), recursive=True),
            "code push",
        )

    def push_bundle(self, local_dir: Path) -> str:
        """Copy an evaluation bundle to the device; returns its remote path."""
        remote_dir = self.connection.remote_path(local_dir.name)
        # Cleared first so a partial earlier copy cannot survive and fail the
        # manifest checksum as if the transfer had just corrupted.
        self._with_retries(
            lambda: self.transport.run(f"rm -rf {remote_dir}", timeout=60), "bundle cleanup"
        )
        self._with_retries(
            lambda: self.transport.push(local_dir, self.connection.remote_path(""), recursive=True),
            "bundle push",
        )
        return remote_dir

    # -- the measurement itself -------------------------------------------

    def measure(self, spec: BenchSpec, use_cache: bool = True) -> dict[str, Any]:
        """Return a result document for one candidate, measuring only if needed."""
        if use_cache:
            cached = self.cache.get(spec)
            if cached is not None:
                return {**cached, "from_cache": True}

        result = self._measure_uncached(spec)
        self.cache.put(spec, result)
        return {**result, "from_cache": False}

    def _measure_uncached(self, spec: BenchSpec) -> dict[str, Any]:
        key = spec_cache_key(spec)
        remote_spec = self.connection.work_path(f"{key}_spec.json")
        remote_result = self.connection.work_path(f"{key}_result.json")

        with tempfile.TemporaryDirectory() as staging_name:
            staging = Path(staging_name)
            local_spec = staging / "spec.json"
            local_spec.write_text(json.dumps(spec.as_dict(), indent=2), encoding="utf-8")

            self._with_retries(
                lambda: self.transport.run(f"mkdir -p {self.connection.remote_work}", timeout=60),
                "remote workdir",
            )
            self._with_retries(lambda: self.transport.push(local_spec, remote_spec), "spec upload")

            # Build in its own process so the timing never pays the
            # calibrator's memory cost. A build failure is a real answer.
            build = self._with_retries(
                lambda: self.transport.run(
                    self._agent_command(
                        f"--spec {remote_spec} --out {remote_result} --build-only"
                    )
                ),
                "artifact build",
            )
            if not build.ok:
                fetched = self._fetch_result(remote_result, staging)
                if fetched is not None:
                    return fetched
                return self._transport_failure(spec, "build", build)

            run = self._with_retries(
                lambda: self.transport.run(
                    self._agent_command(f"--spec {remote_spec} --out {remote_result}")
                ),
                "measurement",
            )
            fetched = self._fetch_result(remote_result, staging)
            if fetched is not None:
                return fetched
            return self._transport_failure(spec, "measure", run)

    def score(
        self,
        spec: BenchSpec,
        limit: int | None = None,
        split: str | None = None,
        bundle_dir: str | None = None,
    ) -> dict[str, Any]:
        """Measure one candidate's accuracy on the device.

        Uncached: the search calls this at two sample sizes for the same config,
        and a cache keyed on the config alone would return the screen's number
        when the full evaluation was asked for.
        """
        key = f"{spec_cache_key(spec)}_{split or 'optval'}_{limit or 'all'}"
        remote_spec = self.connection.work_path(f"{key}_spec.json")
        remote_result = self.connection.work_path(f"{key}_score.json")
        flags = "" if limit is None else f" --limit {limit}"
        if split:
            flags += f" --split {split}"
        if bundle_dir:
            flags += f" --bundle-dir {bundle_dir}"

        with tempfile.TemporaryDirectory() as staging_name:
            staging = Path(staging_name)
            local_spec = staging / "spec.json"
            local_spec.write_text(json.dumps(spec.as_dict(), indent=2), encoding="utf-8")

            self._with_retries(
                lambda: self.transport.run(f"mkdir -p {self.connection.remote_work}", timeout=60),
                "remote workdir",
            )
            self._with_retries(lambda: self.transport.push(local_spec, remote_spec), "spec upload")
            run = self._with_retries(
                lambda: self.transport.run(
                    self._score_command(f"--spec {remote_spec} --out {remote_result}{flags}")
                ),
                "accuracy scoring",
            )
            fetched = self._fetch_result(remote_result, staging)
            if fetched is not None:
                return fetched
            return self._transport_failure(spec, "score", run)

    def _fetch_result(self, remote_result: str, staging: Path) -> dict[str, Any] | None:
        """Bring the result document back, or None if the agent never wrote one."""
        local_result = staging / "result.json"
        try:
            fetch = self._with_retries(
                lambda: self.transport.fetch(remote_result, local_result), "result download"
            )
        except TransportError:
            return None
        if not fetch.ok or not local_result.exists():
            return None
        try:
            return json.loads(local_result.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _transport_failure(self, spec: BenchSpec, stage: str, result: CommandResult) -> dict[str, Any]:
        """Shape a transport failure like an agent result, so callers have one contract."""
        return {
            "schema": RESULT_SCHEMA,
            "status": "transport_failed",
            "model": spec.model,
            "config": spec.config.as_dict(),
            "quant_hash": spec.quant.hash,
            "config_hash": spec.config.hash,
            "error": (
                f"{stage} stage exited {result.returncode} and no result file came back.\n"
                f"stderr: {result.stderr.strip()[:2000]}"
            ),
        }


def summarize_sentinel(spec: BenchSpec, trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Run-to-run spread of repeated measurements of one config."""
    # Spread is (max-min)/median of the per-trial medians, matching how the
    # July 2026 baseline was computed so the two are comparable.
    medians = [t["latency"]["median_ms"] for t in trials if t.get("status") == "ok"]
    spread = None
    if len(medians) >= 2:
        spread = (max(medians) - min(medians)) / statistics.median(medians)

    return {
        "schema": "sentinel/1",
        "model": spec.model,
        "config": spec.config.as_dict(),
        "trials": len(trials),
        "admissible_trials": sum(1 for t in trials if t.get("admissible")),
        "medians_ms": [round(value, 4) for value in medians],
        "median_of_medians_ms": round(statistics.median(medians), 4) if medians else None,
        "min_ms": round(min(medians), 4) if medians else None,
        "max_ms": round(max(medians), 4) if medians else None,
        "spread_fraction": round(spread, 5) if spread is not None else None,
        "spread_percent": round(spread * 100, 2) if spread is not None else None,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "results": trials,
    }


def run_sentinel(
    runner: RemoteBenchmarker,
    spec: BenchSpec,
    repeats: int,
    spacing_s: float,
    sleep=time.sleep,
) -> dict[str, Any]:
    """Measure one config repeatedly, spaced out, to characterise drift."""
    # Cache is bypassed on purpose: identical configs are exactly what a
    # sentinel repeats, and a cache hit would return one run five times.
    trials: list[dict[str, Any]] = []
    for index in range(repeats):
        if index and spacing_s:
            sleep(spacing_s)
        result = runner.measure(spec, use_cache=False)
        trials.append(result)
        median = result.get("latency", {}).get("median_ms")
        print(
            f"  trial {index + 1}/{repeats}  "
            + (f"{median:.4f} ms" if median else f"{result['status']}")
        )
    return summarize_sentinel(spec, trials)


def configs_for(model: str) -> dict[str, QuantConfig]:
    """Every config name `--configs` accepts for this model."""
    return {**BASELINE_CONFIGS, **candidate_configs(model), **per_group_configs(model)}


def _specs_for(
    model: str, configs: Iterable[str], threads: Iterable[int], warmup: int, iterations: int
) -> list[BenchSpec]:
    available = configs_for(model)
    unknown = sorted(set(configs) - set(available))
    if unknown:
        raise SystemExit(
            f"Unknown config(s) for {model}: {', '.join(unknown)}.\n"
            f"Available: {', '.join(sorted(available))}"
        )
    return [
        BenchSpec(
            model=model,
            config=DeploymentConfig(
                quant=available[name],
                run=RunConfig(intra_op_num_threads=thread_count),
            ),
            warmup=warmup,
            iterations=iterations,
        )
        for thread_count in threads
        for name in configs
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="Probe the device and exit")
    parser.add_argument("--push-code", action="store_true", help="Copy src/ to the device first")
    parser.add_argument("--model", choices=known_models())
    # Not `choices`: the selective-FP32 candidates are model-specific, so the
    # valid set is only known once --model is parsed.
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["fp32"],
        help=f"{', '.join(sorted(BASELINE_CONFIGS))}, or {', '.join(sorted(CANDIDATE_EXCLUSIONS))}",
    )
    parser.add_argument(
        "--exclude-each",
        action="store_true",
        help="sweep every single-group exclusion, giving each group a latency cost",
    )
    parser.add_argument("--threads", nargs="+", type=int, default=[4])
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--no-cache", action="store_true", help="Re-measure even if cached")
    parser.add_argument(
        "--repeat", type=int, default=1, help="Repeat one config N times as a drift sentinel"
    )
    parser.add_argument(
        "--spacing-s", type=float, default=600.0, help="Seconds between sentinel trials"
    )
    parser.add_argument("--report", type=Path, help="Write the sentinel summary here")
    parser.add_argument("--target-file", type=Path, default=DEFAULT_TARGET_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    connection = PiConnection.load(args.target_file)
    runner = RemoteBenchmarker(connection)

    if args.push_code:
        print(f"Pushing src/ to {connection.target}:{connection.remote_root}")
        runner.push_code()

    if args.check or not args.model:
        state = runner.device_state()
        print(json.dumps(state, indent=2))
        if not state.get("governor_is_pinned"):
            print(
                "\nWARNING: governor is not 'performance'. Run on the Pi:\n"
                "  sudo bash scripts/pi_prepare.sh"
            )
        return

    configs = sorted(per_group_configs(args.model)) if args.exclude_each else args.configs
    specs = _specs_for(args.model, configs, args.threads, args.warmup, args.iterations)

    if args.repeat > 1:
        if len(specs) != 1:
            raise SystemExit("--repeat needs exactly one config and one thread count")
        print(
            f"Sentinel: {args.repeat} trials of {specs[0].quant.describe()} "
            f"@ {specs[0].run.intra_op_num_threads}t, {args.spacing_s:.0f}s apart\n"
        )
        summary = run_sentinel(runner, specs[0], args.repeat, args.spacing_s)
        print(
            f"\n  spread {summary['spread_percent']}%  "
            f"(min {summary['min_ms']} / max {summary['max_ms']} ms, "
            f"{summary['admissible_trials']}/{summary['trials']} admissible)"
        )
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"  wrote {args.report}")
        return

    print(f"{len(specs)} measurement(s) on {connection.target}\n")
    for spec in specs:
        label = f"{spec.quant.describe()} @ {spec.run.intra_op_num_threads}t"
        result = runner.measure(spec, use_cache=not args.no_cache)
        source = "cache" if result.get("from_cache") else "device"

        if result["status"] != "ok":
            print(f"  {label:<48} {result['status']}: {result.get('error', '')[:120]}")
            continue

        latency = result["latency"]
        flag = "" if result["admissible"] else f"  [INADMISSIBLE: {result['admissibility_reason']}]"
        print(
            f"  {label:<48} {latency['median_ms']:>8.3f} ms   "
            f"{result['peak_rss_mb']:>7.1f} MB RSS   ({source}){flag}"
        )


if __name__ == "__main__":
    main()
