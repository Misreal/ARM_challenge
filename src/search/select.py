"""The Pareto front and the named deployment choices drawn from it.

Comparisons happen inside the measured band: two latencies closer than the
harness can reproduce are a tie, and a tie is broken by the stated rules rather
than by whichever printed digit happened to fall first.
"""

from __future__ import annotations

from typing import Any, Callable

# Minimize the first three, maximize the last. Mirrors the study's objectives.
OBJECTIVES: tuple[tuple[str, str], ...] = (
    ("latency_ms", "min"),
    ("size_bytes", "min"),
    ("peak_rss_mb", "min"),
    ("top1", "max"),
)

FASTEST = "fastest_valid"
SMALLEST = "smallest_valid"
LOWEST_RAM = "lowest_ram"
MOST_ACCURATE = "most_accurate"
RECOMMENDED = "recommended"

CHOICES = (FASTEST, SMALLEST, LOWEST_RAM, MOST_ACCURATE, RECOMMENDED)


def values(row: dict[str, Any]) -> tuple[float, ...]:
    result = row["result"]
    return tuple(float(result[key]) for key, _ in OBJECTIVES)


def dominates(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """First is at least as good everywhere and strictly better somewhere."""
    left, right = values(first), values(second)
    better_anywhere = False
    for index, (_, sense) in enumerate(OBJECTIVES):
        a, b = (left[index], right[index]) if sense == "min" else (-left[index], -right[index])
        if a > b:
            return False
        if a < b:
            better_anywhere = True
    return better_anywhere


def feasible(rows: list[dict[str, Any]], max_rss_mb: float | None = None) -> list[dict[str, Any]]:
    """Candidates that passed every gate, and the RAM ceiling when one was set.

    The accuracy budget is applied upstream as a hard filter, so anything still
    marked ok already satisfies it.
    """
    kept = [row for row in rows if row["result"].get("status") == "ok"]
    if max_rss_mb is not None:
        kept = [row for row in kept if row["result"]["peak_rss_mb"] <= max_rss_mb]
    return kept


def non_dominated(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if not any(dominates(other, row) for other in rows if other is not row)]


def _tie_break(row: dict[str, Any]) -> tuple[float, float, str]:
    """Smaller artifact, then higher accuracy, then a stable hash."""
    result = row["result"]
    return (float(result["size_bytes"]), -float(result["top1"]), row["config_hash"])


def best_by(
    rows: list[dict[str, Any]],
    key: Callable[[dict[str, Any]], float],
    band_ms: float | None = None,
    banded: bool = False,
) -> dict[str, Any] | None:
    """The best row on one objective, with anything inside the band counted a tie.

    `banded` applies only to latency: the other three objectives are exact
    properties of the artifact or of one measurement, not of a repeated one.
    """
    if not rows:
        return None
    best = min(key(row) for row in rows)
    if banded and band_ms:
        tied = [row for row in rows if key(row) - best <= band_ms]
    else:
        tied = [row for row in rows if key(row) == best]
    return sorted(tied, key=_tie_break)[0]


def latency_of(row: dict[str, Any]) -> float:
    """The repeated median when a finalist round measured one, else the single pass."""
    repeated = row.get("repeated", {}).get("median_ms")
    return float(repeated if repeated is not None else row["result"]["latency_ms"])


def selections(
    rows: list[dict[str, Any]], band_ms: float | None = None, max_rss_mb: float | None = None
) -> dict[str, dict[str, Any] | None]:
    """The five named choices. `recommended` is the fastest that clears every gate."""
    valid = feasible(rows, max_rss_mb)
    return {
        FASTEST: best_by(valid, latency_of, band_ms, banded=True),
        SMALLEST: best_by(valid, lambda row: float(row["result"]["size_bytes"])),
        LOWEST_RAM: best_by(valid, lambda row: float(row["result"]["peak_rss_mb"])),
        MOST_ACCURATE: best_by(valid, lambda row: -float(row["result"]["top1"])),
        RECOMMENDED: best_by(
            [row for row in valid if row["result"].get("admissible", True)],
            latency_of,
            band_ms,
            banded=True,
        ),
    }


def finalists(rows: list[dict[str, Any]], band_ms: float | None = None, limit: int = 5) -> list[dict[str, Any]]:
    """The candidates worth re-measuring: the fastest three, the smallest, the lowest-RAM.

    Deduplicated by configuration, so a candidate that wins on two objectives
    does not spend two finalist slots.
    """
    valid = feasible(rows)
    if not valid:
        return []

    ordered = sorted(valid, key=lambda row: (latency_of(row), _tie_break(row)))
    chosen: list[dict[str, Any]] = list(ordered[:3])
    for pick in (
        best_by(valid, lambda row: float(row["result"]["size_bytes"])),
        best_by(valid, lambda row: float(row["result"]["peak_rss_mb"])),
    ):
        if pick is not None:
            chosen.append(pick)

    seen: set[str] = set()
    unique = []
    for row in chosen:
        if row["config_hash"] in seen:
            continue
        seen.add(row["config_hash"])
        unique.append(row)
    return unique[:limit]
