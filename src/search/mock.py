"""A fake device, so the study can be debugged without spending Pi hours.

Numbers are plausible but invented. Nothing this produces may reach a results
table; `--mock` marks the study name so a mock run cannot be mistaken for one.
"""

from __future__ import annotations

from typing import Any

from src.bench.agent import BenchSpec

MOCK_FP32_MS = 16.0
MOCK_INT8_MS = 3.4
MOCK_FP32_BYTES = 44_000_000
MOCK_BASELINE_TOP1 = 78.5


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
        return {
            "status": "ok",
            "admissible": True,
            "latency": {"median_ms": round(latency, 4)},
            "peak_rss_mb": round(100.0 + 60.0 * (1.0 - fraction), 1),
            "bytes": self._bytes(spec),
        }

    def _bytes(self, spec: BenchSpec) -> int:
        if spec.quant.quant_type == "none":
            return MOCK_FP32_BYTES
        fraction = _int8_fraction(spec, self.groups)
        return int(MOCK_FP32_BYTES * (0.25 + 0.75 * (1.0 - fraction)))
