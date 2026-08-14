"""The dashboard restates measured results, so pin the transformations it applies."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.build_dashboard import (
    OBJECTIVES,
    configs_for,
    find_sentinel,
    front_rows,
    group_index,
    lexicographic,
    noise_floor,
    orderings,
    parse_quant_describe,
    parse_run_describe,
    tie_groups,
    trial_rows,
    unmeasured_noise,
    unreachable_rows,
)
from src.quant.config import DeploymentConfig, QuantConfig, RunConfig

# Perfect resolution, so a grouping test isolates whichever effect it is about.
NO_NOISE = {"latency_ms": 0.0, "size_mb": 0.0, "peak_rss_mb": 0.0, "top1": 0.0}
LATENCY_NOISE = {**NO_NOISE, "latency_ms": 2.55}

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


def front_member(**overrides) -> dict:
    row = {
        "describe": "static per-channel act:uint8 minmax/512",
        "describe_run": "4t opt:all",
        "quant_hash": "x",
        "result": {"status": "ok", "latency_ms": 3.4, "size_bytes": 1, "peak_rss_mb": 90.0,
                   "top1": 78.4, "screen_top1": 78.0},
    }
    row.update(overrides)
    return row


def test_a_stored_config_is_read_back_rather_than_parsed() -> None:
    stored = DeploymentConfig(
        quant=QuantConfig(quant_type="static", per_channel=True, calibration_size=128),
        run=RunConfig(graph_optimization_level="extended"),
    )
    # The label deliberately disagrees with the stored config: whichever wins is
    # visible, and the stored bytes are the ones that were measured.
    quant, run = configs_for(front_member(config=stored.as_dict()))
    assert quant == stored.quant
    assert run == stored.run


def test_a_campaign_without_stored_configs_still_parses_its_labels() -> None:
    quant, run = configs_for(front_member())
    assert quant.describe() == "static per-channel act:uint8 minmax/512"
    assert run.describe() == "4t opt:all"


def test_the_trial_panel_prefers_the_committed_summary_over_sqlite(tmp_path) -> None:
    # An enumerated campaign writes no sqlite at all, so reading the summary is
    # what keeps the panel on the page.
    rows = [{**front_member(), "config_hash": "a", "round": "canonical"}]
    summary = {"evaluated_configs": rows}
    panel = trial_rows(summary, tmp_path / "absent.db", "study", [])
    assert [row["number"] for row in panel] == [0]
    assert panel[0]["round"] == "canonical"
    assert panel[0]["on_front"] is False


def test_a_summary_without_evaluations_and_without_sqlite_drops_the_panel(tmp_path) -> None:
    assert trial_rows({"trials": 40}, tmp_path / "absent.db", "study", []) == []


def test_the_panel_marks_which_evaluations_reached_the_front() -> None:
    rows = [{**front_member(), "config_hash": "a"}]
    panel = trial_rows({"evaluated_configs": rows}, Path("absent.db"), "study",
                       [{"quant_hash": "x", "describe_run": "4t opt:all"}])
    assert panel[0]["on_front"] is True


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


def rows_from(*members: dict) -> list[dict]:
    return front_rows({"pareto": list(members)}, baseline_top1=78.5, samples=3000)


def fixture_rows() -> list[dict]:
    # Three configs that each lead an objective, plus one that leads none.
    return rows_from(
        member(3.0, 12.0, 100.0, 78.0, "static per-channel act:uint8 minmax/512"),
        member(9.0, 11.0, 99.0, 78.4, "static per-tensor act:uint8 minmax/512"),
        member(6.0, 11.6, 99.6, 78.15, "dynamic"),
        member(5.0, 13.0, 104.0, 79.0, "static per-channel act:uint8 minmax/512 fp32:fc"),
    )


def test_rows_are_ordered_by_latency_and_labelled_in_that_order() -> None:
    rows = fixture_rows()
    assert [row["id"] for row in rows] == ["C1", "C2", "C3", "C4"]
    assert [row["latency_ms"] for row in rows] == [3.0, 5.0, 6.0, 9.0]


def test_images_gap_is_reported_against_the_split_size() -> None:
    rows = fixture_rows()
    # 79.0 against a 78.5 baseline is 0.5 pt, which on 3,000 images is 15 pictures.
    assert rows[1]["images_vs_baseline"] == 15


def test_only_the_timed_objectives_carry_repeat_noise() -> None:
    noise = noise_floor(
        {
            "admissible_trials": 3,
            "spread_percent": 2.55,
            "results": [{"peak_rss_mb": 100.0}, {"peak_rss_mb": 100.1}, {"peak_rss_mb": 99.9}],
        }
    )
    assert noise["percent"]["latency_ms"] == 2.55
    assert noise["percent"]["peak_rss_mb"] == pytest.approx(0.2)
    # Rerunning does not move either of these, so their only limit is the printout.
    assert noise["percent"]["size_mb"] == 0.0
    assert noise["percent"]["top1"] == 0.0


def test_a_run_without_a_sentinel_claims_no_noise_rather_than_borrowing_one() -> None:
    floor = unmeasured_noise()
    assert floor["measured"] is False
    assert set(floor["percent"]) == {objective["key"] for objective in OBJECTIVES}
    assert all(value == 0.0 for value in floor["percent"].values())


def test_the_sentinel_is_found_by_model_and_never_from_another_model(tmp_path) -> None:
    (tmp_path / "sentinel_resnet18_cifar_20260808.json").write_text("{}")
    (tmp_path / "sentinel_resnet18_cifar_20260901.json").write_text("{}")
    # Dated names sort chronologically, so the newest repeat run is the one used.
    assert find_sentinel(tmp_path, "resnet18_cifar").name == "sentinel_resnet18_cifar_20260901.json"
    assert find_sentinel(tmp_path, "mobilenetv2_cifar") is None


def test_values_that_print_the_same_share_a_tie_group() -> None:
    # Both round to 11.27 MB, so the page has shown nothing to rank them on.
    rows = rows_from(
        member(3.0, 11.271, 100.0, 78.0, "dynamic"),
        member(4.0, 11.274, 101.0, 78.1, "dynamic per-channel"),
        member(5.0, 11.400, 102.0, 78.2, "dynamic fp32:avgpool"),
    )
    assert tie_groups(rows, NO_NOISE)["size_mb"] == [["C1", "C2"], ["C3"]]


def test_repeat_noise_groups_values_that_print_differently() -> None:
    rows = rows_from(
        member(3.500, 12.0, 100.0, 78.0, "dynamic"),
        member(3.560, 12.1, 101.0, 78.1, "dynamic per-channel"),
        member(3.700, 12.2, 102.0, 78.2, "dynamic fp32:avgpool"),
    )
    # 3.500 and 3.560 print differently but are 1.7% apart, inside the 2.55% floor.
    assert tie_groups(rows, LATENCY_NOISE)["latency_ms"] == [["C1", "C2"], ["C3"]]


def test_a_group_is_measured_from_its_leader_not_its_last_member() -> None:
    rows = rows_from(
        member(3.500, 12.0, 100.0, 78.0, "dynamic"),
        member(3.570, 12.1, 101.0, 78.1, "dynamic per-channel"),
        member(3.640, 12.2, 102.0, 78.2, "dynamic fp32:avgpool"),
    )
    # Each neighbouring pair is inside the noise floor but the ends are not, so
    # chaining off the last member would swallow all three into one group.
    assert tie_groups(rows, LATENCY_NOISE)["latency_ms"] == [["C1", "C2"], ["C3"]]


def tied_on_latency() -> list[dict]:
    return rows_from(
        member(3.500, 12.0, 100.0, 78.0, "dynamic"),
        member(3.560, 12.1, 101.0, 78.1, "dynamic per-channel"),
        member(3.700, 12.2, 102.0, 78.2, "dynamic fp32:avgpool"),
    )


def test_a_tie_on_the_first_priority_is_broken_by_the_second() -> None:
    rows = tied_on_latency()
    index = group_index(tie_groups(rows, LATENCY_NOISE))
    ranked = lexicographic(rows, ("latency_ms", "top1", "size_mb", "peak_rss_mb"), index)
    # C1 is faster on paper, but not by more than the device's repeat spread, so
    # the reader's second priority is what actually separates them.
    assert [row["id"] for row in ranked] == ["C2", "C1", "C3"]


def test_each_row_records_the_priority_that_separated_it_from_the_row_above() -> None:
    rows = tied_on_latency()
    entries = orderings(rows, tie_groups(rows, LATENCY_NOISE))["latency_ms,top1,size_mb,peak_rss_mb"]
    assert [entry["id"] for entry in entries] == ["C2", "C1", "C3"]
    assert [entry["decided_by"] for entry in entries] == [None, "top1", "latency_ms"]


def test_every_priority_order_the_reader_can_build_is_precomputed() -> None:
    rows = fixture_rows()
    assert len(orderings(rows, tie_groups(rows, NO_NOISE))) == 24


@pytest.mark.parametrize(
    ("key", "expected_id"),
    # C4 leads on both size and RAM in this fixture; that is what leaves C3
    # leading nothing, so the two expectations are meant to coincide.
    [("latency_ms", "C1"), ("size_mb", "C4"), ("peak_rss_mb", "C4"), ("top1", "C2")],
)
def test_the_first_priority_selects_that_objectives_best(key: str, expected_id: str) -> None:
    rows = fixture_rows()
    resolved = orderings(rows, tie_groups(rows, NO_NOISE))
    order = [key] + [o["key"] for o in OBJECTIVES if o["key"] != key]
    assert resolved[",".join(order)][0]["id"] == expected_id


def test_a_row_that_leads_no_objective_never_ranks_first() -> None:
    rows = fixture_rows()
    # C3 is slower than C1, larger than C4 and less accurate than C2, while
    # sitting between them on every axis -- no ordering can lift it.
    resolved = orderings(rows, tie_groups(rows, NO_NOISE))
    assert unreachable_rows(resolved, rows) == ["C3"]


def test_a_tie_break_win_counts_as_reachable() -> None:
    rows = tied_on_latency()
    # C2 leads no objective outright, but it ties C1 on latency and takes the
    # next round, so a latency-first reader really can see it at rank 1.
    resolved = orderings(rows, tie_groups(rows, LATENCY_NOISE))
    assert unreachable_rows(resolved, rows) == []
