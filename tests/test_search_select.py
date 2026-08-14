"""Ranking rules and the named choices, including what counts as a tie."""

from __future__ import annotations

import pytest

from src.search import select
from src.search.finalists import aggregate, median_absolute_deviation, run_finalists, ties_within_band

BAND_MS = 0.068


def candidate(
    name: str, latency: float, size: int = 11_000_000, rss: float = 100.0,
    top1: float = 78.4, status: str = "ok", admissible: bool = True,
) -> dict:
    return {
        "config_hash": name,
        "describe": name,
        "describe_run": "4t opt:all",
        "config": {"quant": {"quant_type": "static"}, "run": {}},
        "result": {
            "status": status,
            "latency_ms": latency,
            "size_bytes": size,
            "peak_rss_mb": rss,
            "top1": top1,
            "admissible": admissible,
        },
    }


def test_a_candidate_better_everywhere_dominates() -> None:
    assert select.dominates(candidate("a", 3.0, 10, 90.0, 79.0), candidate("b", 4.0, 20, 100.0, 78.0))


def test_a_candidate_better_on_one_objective_only_does_not_dominate() -> None:
    faster = candidate("a", 3.0, top1=78.0)
    accurate = candidate("b", 4.0, top1=79.0)
    assert not select.dominates(faster, accurate)
    assert not select.dominates(accurate, faster)


def test_an_identical_candidate_does_not_dominate() -> None:
    assert not select.dominates(candidate("a", 3.0), candidate("b", 3.0))


def test_the_front_keeps_only_the_undominated() -> None:
    rows = [
        candidate("best", 3.0, 10, 90.0, 79.0),
        candidate("dominated", 4.0, 20, 100.0, 78.0),
        candidate("accurate", 5.0, 30, 120.0, 80.0),
    ]
    assert {row["config_hash"] for row in select.non_dominated(rows)} == {"best", "accurate"}


def test_a_candidate_that_failed_a_gate_is_never_selected() -> None:
    rows = [candidate("broken", 1.0, status="build_failed"), candidate("ok", 5.0)]
    assert select.selections(rows)[select.FASTEST]["config_hash"] == "ok"


def test_the_ram_ceiling_rejects_rather_than_ranks() -> None:
    rows = [candidate("hungry", 1.0, rss=200.0), candidate("lean", 5.0, rss=90.0)]
    assert select.selections(rows, max_rss_mb=100.0)[select.FASTEST]["config_hash"] == "lean"
    assert select.selections(rows)[select.FASTEST]["config_hash"] == "hungry"


def test_recommended_skips_a_candidate_the_device_called_inadmissible() -> None:
    rows = [candidate("throttled", 1.0, admissible=False), candidate("clean", 5.0)]
    chosen = select.selections(rows, BAND_MS)
    assert chosen[select.RECOMMENDED]["config_hash"] == "clean"
    # It is still the fastest thing measured, and the front should say so.
    assert chosen[select.FASTEST]["config_hash"] == "throttled"


def test_two_latencies_inside_the_band_are_a_tie_broken_by_size() -> None:
    rows = [candidate("big", 3.400, size=30_000_000), candidate("small", 3.440, size=10_000_000)]
    # 0.040 ms apart on a 0.068 ms band: the harness cannot tell them apart, so
    # the smaller artifact wins rather than the smaller printed number.
    assert select.selections(rows, BAND_MS)[select.FASTEST]["config_hash"] == "small"


def test_a_gap_wider_than_the_band_is_not_a_tie() -> None:
    rows = [candidate("fast", 3.400, size=30_000_000), candidate("slow", 3.600, size=10_000_000)]
    assert select.selections(rows, BAND_MS)[select.FASTEST]["config_hash"] == "fast"


def test_without_a_band_only_an_exact_match_ties() -> None:
    rows = [candidate("a", 3.400, size=30_000_000), candidate("b", 3.440, size=10_000_000)]
    assert select.selections(rows, None)[select.FASTEST]["config_hash"] == "a"


def test_ties_are_reported_as_groups() -> None:
    rows = [candidate("a", 3.40), candidate("b", 3.42), candidate("c", 3.90)]
    assert ties_within_band(rows, BAND_MS) == [["a", "b"]]


def test_a_field_with_no_ties_reports_none() -> None:
    rows = [candidate("a", 3.40), candidate("b", 3.90)]
    assert ties_within_band(rows, BAND_MS) == []


def test_tie_breaks_are_deterministic_all_the_way_down() -> None:
    # Same latency, same size, same accuracy: the config hash is the last resort.
    rows = [candidate("zzz", 3.40), candidate("aaa", 3.40)]
    assert select.selections(rows, BAND_MS)[select.FASTEST]["config_hash"] == "aaa"


def test_every_named_choice_is_produced() -> None:
    rows = [candidate("a", 3.0, 30_000_000, 120.0, 78.0), candidate("b", 5.0, 10_000_000, 90.0, 79.0)]
    chosen = select.selections(rows, BAND_MS)
    assert set(chosen) == set(select.CHOICES)
    assert chosen[select.FASTEST]["config_hash"] == "a"
    assert chosen[select.SMALLEST]["config_hash"] == "b"
    assert chosen[select.LOWEST_RAM]["config_hash"] == "b"
    assert chosen[select.MOST_ACCURATE]["config_hash"] == "b"


def test_nothing_feasible_names_nothing_rather_than_guessing() -> None:
    chosen = select.selections([candidate("broken", 1.0, status="build_failed")], BAND_MS)
    assert all(value is None for value in chosen.values())


def test_the_shortlist_is_the_fastest_three_plus_smallest_and_lowest_ram() -> None:
    rows = [
        candidate("f1", 1.0), candidate("f2", 2.0), candidate("f3", 3.0), candidate("f4", 4.0),
        candidate("tiny", 9.0, size=1),
        candidate("lean", 9.5, rss=1.0),
    ]
    picked = [row["config_hash"] for row in select.finalists(rows)]
    assert picked == ["f1", "f2", "f3", "tiny", "lean"]


def test_a_candidate_winning_twice_does_not_spend_two_slots() -> None:
    rows = [candidate("both", 1.0, size=1, rss=1.0), candidate("f2", 2.0), candidate("f3", 3.0)]
    picked = [row["config_hash"] for row in select.finalists(rows)]
    assert picked == ["both", "f2", "f3"]


def test_the_shortlist_is_empty_when_nothing_is_feasible() -> None:
    assert select.finalists([candidate("broken", 1.0, status="build_failed")]) == []


def test_the_repeated_median_replaces_the_single_measurement_in_ranking() -> None:
    row = candidate("a", 9.99)
    measured = aggregate(row, [{"status": "ok", "latency_ms": v, "peak_rss_mb": 100.0,
                                "admissible": True} for v in (3.0, 3.1, 3.2)])
    assert measured["repeated"]["median_ms"] == 3.1
    assert select.latency_of(measured) == 3.1


def test_a_partial_finalist_failure_is_recorded_not_hidden() -> None:
    row = candidate("a", 3.0)
    measured = aggregate(row, [
        {"status": "ok", "latency_ms": 3.0, "peak_rss_mb": 100.0, "admissible": True},
        {"status": "measurement_failed", "latency_ms": None},
    ])
    assert measured["repeated"]["succeeded"] == 1
    assert measured["repeated"]["failed"] == 1
    assert measured["repeated"]["median_ms"] == 3.0


def test_the_deviation_is_zero_for_a_single_measurement_and_positive_for_spread() -> None:
    assert median_absolute_deviation([3.0]) == 0.0
    assert median_absolute_deviation([3.0, 3.1, 3.2]) == pytest.approx(0.1)


class Repeater:
    """A device that returns a slightly different number every uncached call."""

    def __init__(self) -> None:
        self.calls = 0

    def measure(self, spec, use_cache: bool = True) -> dict:
        self.calls += 1
        return {
            "status": "ok",
            "admissible": True,
            "latency": {"median_ms": 3.4 + (self.calls % 3) * 0.01},
            "peak_rss_mb": 100.0,
            "readiness": {"device": {"temperature_c": 50.0, "throttled": False}},
        }


def study_summary(rows: list[dict]) -> dict:
    return {
        "study": "test",
        "band": {"fraction": 0.02, "ms": BAND_MS, "source": "sentinel", "reference_ms": 3.4},
        "evaluated_configs": rows,
        "search_plan": {"max_rss_mb": None},
    }


def test_the_stage_measures_every_finalist_the_stated_number_of_times() -> None:
    rows = [candidate("a", 3.0), candidate("b", 3.5), candidate("c", 4.0)]
    runner = Repeater()
    report = run_finalists(runner, "resnet18_cifar", study_summary(rows), repeats=5)
    assert len(report["finalists"]) == 3
    assert runner.calls == 15
    assert all(row["repeated"]["repeats"] == 5 for row in report["finalists"])


def test_the_stage_names_choices_from_every_candidate_not_just_the_shortlist() -> None:
    # The most accurate candidate is the slowest, so it is not a finalist. It is
    # still the most accurate, and accuracy needed no re-measurement.
    rows = [
        candidate("fast", 3.0, top1=78.0),
        candidate("mid", 3.5, top1=78.2),
        candidate("slow", 9.0, top1=79.9),
        candidate("slower", 9.5, top1=78.1),
    ]
    report = run_finalists(Repeater(), "resnet18_cifar", study_summary(rows), repeats=3)
    assert {row["config_hash"] for row in report["finalists"]} == {"fast", "mid", "slow"}
    assert report["selections"][select.MOST_ACCURATE] == "slow"


def test_repeat_rounds_alternate_direction() -> None:
    seen: list[str] = []

    class Recorder(Repeater):
        def measure(self, spec, use_cache: bool = True) -> dict:
            seen.append(spec.config.quant.describe())
            return super().measure(spec, use_cache)

    rows = [candidate("a", 3.0), candidate("b", 3.5)]
    run_finalists(Recorder(), "resnet18_cifar", study_summary(rows), repeats=2)
    # Two candidates over two rounds: forward then reversed.
    assert seen[:2] == list(reversed(seen[2:]))
