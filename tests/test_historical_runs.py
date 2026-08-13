"""The committed campaigns are measured evidence, so pin what they report.

These runs were adopted from before the run layout existed and cost real Pi
hours. A refactor is allowed to change what the *next* campaign searches; it is
not allowed to change what these three already found, or to stop reading them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.runs import RUNS_DIR, list_runs, load_run

# Headline numbers as measured, read off the committed stage files. A change
# here is either a bug or history being rewritten; neither should be quiet.
MEASURED = {
    "custom_cnn_b0.2_20260811": ("custom_cnn_budget0.2_v2", 40, 13),
    "mobilenetv2_cifar_b0.2_20260811": ("mobilenetv2_cifar_budget0.2_v2", 36, 5),
    "resnet18_cifar_b0.2_20260809": ("resnet18_cifar_budget0.2_v2", 40, 8),
}

# What the absolute-noise-floor rule produced, before the banded rule replaced
# it. Kept so the two partitions can be compared rather than confused: these are
# the spaces the v2 and v3 campaigns actually searched.
ABSOLUTE_FLOOR_PARTITIONS = {
    "custom_cnn": {
        "pinned_fp32": ["classifier"],
        "pinned_int8": ["pool", "stage1", "stage3"],
        "searchable": ["stage2", "stem"],
    },
    "mobilenetv2_cifar": {
        # The 16 pinned INT8 groups are `features.2` through `features.17`, and
        # they are where the 2^20 to 2^4 collapse actually comes from.
        "pinned_fp32": ["classifier.1", "features.18"],
        "pinned_int8": sorted(f"features.{index}" for index in range(2, 18)),
        "searchable": ["features.0", "features.1"],
    },
    "resnet18_cifar": {
        "pinned_fp32": ["fc"],
        "pinned_int8": ["avgpool", "layer2", "layer3", "layer4", "relu"],
        "searchable": ["conv1", "layer1"],
    },
}


def skip_without(path: Path):
    if not path.exists():
        pytest.skip(f"{path} is not in this checkout")


@pytest.mark.parametrize("run_id", sorted(MEASURED))
def test_a_committed_campaign_still_loads_and_reports_what_it_measured(run_id: str) -> None:
    skip_without(RUNS_DIR / run_id / "run.json")
    run = load_run(run_id)
    study = json.loads(run.paths.stage("study").read_text(encoding="utf-8"))

    name, trials, front_size = MEASURED[run_id]
    assert run.study == name
    assert study["trials"] == trials
    assert len(study["pareto"]) == front_size


def test_every_run_in_the_tree_loads_under_the_current_schema() -> None:
    skip_without(RUNS_DIR)
    # run/1 recipes predate the population and RAM fields, and must keep loading.
    assert list_runs(), "no runs found; the reader stopped seeing the committed tree"


@pytest.mark.parametrize("model", sorted(ABSOLUTE_FLOOR_PARTITIONS))
def test_the_historical_partition_is_recorded_beside_the_campaign_it_cut(model: str) -> None:
    run_id = next(key for key in MEASURED if key.startswith(model))
    skip_without(RUNS_DIR / run_id / "stages" / "search_space.json")
    recorded = json.loads((RUNS_DIR / run_id / "stages" / "search_space.json").read_text(encoding="utf-8"))
    for field, expected in ABSOLUTE_FLOOR_PARTITIONS[model].items():
        assert sorted(recorded[field]) == expected, field
