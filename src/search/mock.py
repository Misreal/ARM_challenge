"""A fake device, so the study can be debugged without spending Pi hours.

Numbers are plausible but invented. Nothing this produces may reach a results
table; `--mock` marks the study name so a mock run cannot be mistaken for one.
"""

from __future__ import annotations

import hashlib
from typing import Any

from src.bench.agent import BenchSpec

MOCK_FP32_MS = 16.0
MOCK_INT8_MS = 3.4
MOCK_FP32_BYTES = 44_000_000
MOCK_BASELINE_TOP1 = 78.5

# Runtime knobs have to move the mock numbers or `--mock` cannot tell a working
# RunConfig plumbing from one that silently drops it. Ratios are shaped after the
# Phase 3 thread sweep; the rest are invented like everything else here.
MOCK_THREAD_SCALING = {1: 2.93, 2: 1.60, 4: 1.0}
MOCK_OPT_LEVEL_PENALTY = {"all": 1.0, "extended": 1.06, "basic": 1.35, "disabled": 4.0}
MOCK_NO_ARENA_PENALTY = 1.08
MOCK_NO_ARENA_RSS_SAVING_MB = 12.0
MOCK_NO_SPIN_PENALTY = 1.03

# Repeat-to-repeat spread. Without it the simulator is exact, so the finalist
# median, MAD and tie logic would never be exercised by the mock run that exists
# to cover them. Shaped after the measured dispersion on byte-equivalent work:
# 0.66% median, 1.91% at p90.
MOCK_JITTER_FRACTION = 0.012


def _int8_fraction(spec: BenchSpec, groups: tuple[str, ...]) -> float:
    """How much of the graph stayed quantized, by group count."""
    if not groups:
        return 1.0
    return max(0.0, 1.0 - len(spec.quant.excluded_groups) / len(groups))


class MockRunner:
    """Satisfies `search.evaluate.Runner` with a deterministic toy model."""

    def __init__(self, groups: tuple[str, ...]) -> None:
        self.groups = groups
        self.calls: list[tuple[str, str]] = []
        # How many uncached measurements this config has already had, so repeats
        # differ from each other while a cache hit repeats itself exactly.
        self.repeats: dict[str, int] = {}

    def _jitter(self, spec: BenchSpec, repeat: int) -> float:
        digest = hashlib.sha256(f"{spec.config.hash}:{repeat}".encode()).digest()
        return 1.0 + ((digest[0] / 255.0) - 0.5) * 2.0 * MOCK_JITTER_FRACTION

    def score(self, spec: BenchSpec, limit: int | None = None) -> dict[str, Any]:
        self.calls.append(("score", spec.quant.hash))
        if spec.quant.quant_type == "none":
            top1 = MOCK_BASELINE_TOP1
        else:
            # Quantizing costs a little accuracy; per-channel costs less.
            penalty = 0.4 if not spec.quant.per_channel else 0.1
            top1 = MOCK_BASELINE_TOP1 - penalty * _int8_fraction(spec, self.groups)
        return {
            "status": "ok",
            "accuracy": {"samples": limit or 3000, "top1": round(top1, 2), "top5": 94.0},
            "bytes": self._bytes(spec),
        }

    def measure(self, spec: BenchSpec, use_cache: bool = True) -> dict[str, Any]:
        self.calls.append(("measure", spec.quant.hash))
        fraction = _int8_fraction(spec, self.groups)
        latency = MOCK_FP32_MS - (MOCK_FP32_MS - MOCK_INT8_MS) * fraction
        if spec.quant.quant_type == "dynamic":
            latency *= 4.0  # dynamic is slower than fp32 on ARM CNNs, as measured

        run = spec.run
        latency *= MOCK_THREAD_SCALING.get(run.intra_op_num_threads, 1.0)
        latency *= MOCK_OPT_LEVEL_PENALTY.get(run.graph_optimization_level, 1.0)
        rss = 100.0 + 60.0 * (1.0 - fraction)
        if not run.enable_cpu_mem_arena:
            latency *= MOCK_NO_ARENA_PENALTY
            rss -= MOCK_NO_ARENA_RSS_SAVING_MB
        if not run.allow_intra_op_spinning:
            latency *= MOCK_NO_SPIN_PENALTY

        repeat = 0 if use_cache else self.repeats.get(spec.config.hash, 0) + 1
        if not use_cache:
            self.repeats[spec.config.hash] = repeat
        latency *= self._jitter(spec, repeat)

        return {
            "status": "ok",
            "admissible": True,
            "latency": {"median_ms": round(latency, 4)},
            "peak_rss_mb": round(rss, 1),
            "bytes": self._bytes(spec),
            "from_cache": use_cache,
            # A simulator has no die to read, so the page can say "simulated"
            # rather than print an invented temperature as if it were measured.
            "readiness": {"device": {"temperature_c": None, "throttled": False}},
        }

    def _bytes(self, spec: BenchSpec) -> int:
        if spec.quant.quant_type == "none":
            return MOCK_FP32_BYTES
        fraction = _int8_fraction(spec, self.groups)
        return int(MOCK_FP32_BYTES * (0.25 + 0.75 * (1.0 - fraction)))
