"""Versioned host<->device contract. Bump when a field's meaning changes.

Without the schema check an older agent silently defaults fields it does not
know, and returns a plausible number measured under the wrong settings.
"""

from __future__ import annotations

SPEC_SCHEMA = "bench_spec/1"
RESULT_SCHEMA = "bench_result/1"

# Only `ok` may hold a latency. The rest let Phase 6 log an infeasible trial
# instead of crashing a study.
RESULT_STATUSES = (
    "ok",
    "build_failed",
    "benchmark_failed",
    "device_not_ready",
    "transport_failed",
)
