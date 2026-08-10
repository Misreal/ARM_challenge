"""The dashboard restates measured results, so pin the transformations it applies."""

from __future__ import annotations

import pytest

from scripts.build_dashboard import (
    OBJECTIVES,
    front_rows,
    normalize,
    parse_quant_describe,
    parse_run_describe,
    rank,
    unreachable_rows,
)
from src.quant.config import QuantConfig, RunConfig

DESCRIBES = (
    "fp32",
    "dynamic",
    "dynamic per-channel",
    "dynamic fp32:avgpool",
    "dynamic per-channel fp32:avgpool+conv1+relu",
    "static per-tensor act:int8 entropy/128 fp32:avgpool+relu",
    "static per-channel act:uint8 minmax/512",
    "static per-channel act:int8 percentile/128 fp32:avgpool+fc+relu",
)

RUN_DESCRIBES = ("4t opt:all", "4t opt:extended no-arena", "1t opt:basic no-arena no-spin")


@pytest.mark.parametrize("text", DESCRIBES)
def test_quant_describe_round_trips(text: str) -> None:
    assert parse_quant_describe(text).describe() == text


@pytest.mark.parametrize("text", RUN_DESCRIBES)
def test_run_describe_round_trips(text: str) -> None:
    assert parse_run_describe(text).describe() == text


def test_parsed_config_hash_matches_the_config_it_names() -> None:
    # The hash is what joins a page row back to its measurement, so a parse that
    # merely prints the same label is not good enough.
    text = "static per-channel act:int8 percentile/128 fp32:avgpool+fc+relu"
    expected = QuantConfig(
        quant_type="static",
        per_channel=True,
        activation_type="int8",
        calibration_method="percentile",
        calibration_size=128,
        excluded_groups=("avgpool", "fc", "relu"),
    )
    assert parse_quant_describe(text).hash == expected.hash


def test_run_describe_recovers_the_non_default_knobs() -> None:
    assert parse_run_describe("4t opt:extended no-arena") == RunConfig(
        graph_optimization_level="extended", enable_cpu_mem_arena=False
    )


def member(latency: float, size_mb: float, rss: float, top1: float, describe: str) -> dict:
    return {
        "describe": describe,
        "describe_run": "4t opt:all",
        "quant_hash": f"{hash(describe) & 0xFFFFFFFF:08x}",
        "result": {
            "status": "ok",
            "latency_ms": latency,
            "size_bytes": int(size_mb * 1e6),
            "peak_rss_mb": rss,
            "top1": top1,
            "screen_top1": top1 + 2.0,
        },
    }


def fixture_rows() -> list[dict]:
    # Three corners plus one point deliberately inside their hull.
    summary = {
        "pareto": [
            member(3.0, 12.0, 100.0, 78.0, "static per-channel act:uint8 minmax/512"),
            member(9.0, 11.0, 99.0, 78.4, "static per-tensor act:uint8 minmax/512"),
            member(6.0, 11.6, 99.6, 78.15, "dynamic"),
            member(5.0, 13.0, 104.0, 79.0, "static per-channel act:uint8 minmax/512 fp32:fc"),
        ]
    }
    return normalize(front_rows(summary, baseline_top1=78.5, samples=3000))


def test_rows_are_ordered_by_latency_and_labelled_in_that_order() -> None:
    rows = fixture_rows()
    assert [row["id"] for row in rows] == ["C1", "C2", "C3", "C4"]
    assert [row["latency_ms"] for row in rows] == [3.0, 5.0, 6.0, 9.0]


def test_normalization_puts_the_best_value_at_one_whichever_way_it_points() -> None:
    rows = fixture_rows()
    # Latency is minimized, so the fastest row normalizes to 1.0; top-1 is
    # maximized, so the most accurate one does.
    assert rows[0]["norm"]["latency_ms"] == pytest.approx(1.0)
    assert max(row["norm"]["top1"] for row in rows) == pytest.approx(1.0)
    assert min(row["norm"]["top1"] for row in rows) == pytest.approx(0.0)


def test_an_objective_with_one_distinct_value_cannot_move_the_ranking() -> None:
    rows = normalize(
        front_rows(
            {"pareto": [member(3.0, 12.0, 100.0, 78.0, "dynamic"), member(5.0, 12.0, 100.0, 79.0, "dynamic per-channel")]},
            baseline_top1=78.5,
            samples=3000,
        )
    )
    assert [row["norm"]["size_mb"] for row in rows] == [1.0, 1.0]
    assert rank(rows, {"size_mb": 100}) == [0, 1]  # a tie, resolved by original order


@pytest.mark.parametrize(
    ("key", "expected_id"),
    # C4 is the extreme on both size and RAM in this fixture; that is what makes
    # C3 interior, so the two expectations are meant to coincide.
    [("latency_ms", "C1"), ("size_mb", "C4"), ("peak_rss_mb", "C4"), ("top1", "C2")],
)
def test_an_extreme_weighting_selects_that_objectives_extreme(key: str, expected_id: str) -> None:
    rows = fixture_rows()
    assert rows[rank(rows, {key: 100})[0]]["id"] == expected_id


def test_images_gap_is_reported_against_the_split_size() -> None:
    rows = fixture_rows()
    # 79.0 against a 78.5 baseline is 0.5 pt, which on 3,000 images is 15 pictures.
    assert rows[1]["images_vs_baseline"] == 15


def test_a_point_inside_the_hull_is_never_rank_one() -> None:
    rows = fixture_rows()
    # C3 is worse than C2 on latency and RAM and worse than C4 on size, while
    # sitting between them on every axis -- no weighting can lift it.
    assert unreachable_rows(rows, step=10) == ["C3"]


def test_every_hull_member_is_reachable_by_the_weighting_that_favours_it() -> None:
    rows = fixture_rows()
    unreachable = set(unreachable_rows(rows, step=10))
    for objective in OBJECTIVES:
        winner = rows[rank(rows, {objective["key"]: 100})[0]]["id"]
        assert winner not in unreachable
