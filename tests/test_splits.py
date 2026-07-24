"""Tests for the data split -- the single most correctness-critical module.

Every accuracy number in the project rests on one property: the four subsets
never overlap, and the same partition is used on every machine. A leak here
does not crash anything; it just makes results quietly better than they are.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.splits import (
    IMAGES_PER_CLASS,
    NUM_CLASSES,
    SPLIT_NAMES,
    SPLIT_SIZES,
    create_split,
    load_split,
    save_split,
    split_fingerprint,
)


@pytest.fixture(scope="module")
def labels() -> list[int]:
    """A synthetic stand-in for CIFAR-100's target list: 500 images per class."""
    return [class_id for class_id in range(NUM_CLASSES) for _ in range(IMAGES_PER_CLASS)]


def test_counts_match_the_declared_sizes(labels: list[int]) -> None:
    split = create_split(labels)
    for name in SPLIT_NAMES:
        assert len(split.get(name)) == SPLIT_SIZES[name] * NUM_CLASSES


def test_subsets_are_disjoint_and_cover_everything(labels: list[int]) -> None:
    split = create_split(labels)
    index_sets = [set(split.get(name)) for name in SPLIT_NAMES]

    for i, first in enumerate(index_sets):
        for second in index_sets[i + 1 :]:
            assert not (first & second)
    assert len(set().union(*index_sets)) == NUM_CLASSES * IMAGES_PER_CLASS


def test_every_subset_is_class_balanced(labels: list[int]) -> None:
    """Matters most for calib: an unstratified 1k sample could miss classes
    entirely and skew the INT8 activation ranges."""
    split = create_split(labels)
    for name in SPLIT_NAMES:
        per_class: dict[int, int] = {}
        for index in split.get(name):
            per_class[labels[index]] = per_class.get(labels[index], 0) + 1
        assert set(per_class.values()) == {SPLIT_SIZES[name]}


def test_same_seed_reproduces_the_split(labels: list[int]) -> None:
    assert create_split(labels, seed=42) == create_split(labels, seed=42)


def test_different_seed_changes_the_split(labels: list[int]) -> None:
    assert create_split(labels, seed=42) != create_split(labels, seed=7)


def test_wrong_dataset_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="training labels"):
        create_split([0, 1, 2])


def test_save_load_round_trip(labels: list[int], tmp_path: Path) -> None:
    split = create_split(labels)
    path = tmp_path / "split.json"
    save_split(split, path)

    assert load_split(path) == split


def test_edited_split_file_is_rejected(labels: list[int], tmp_path: Path) -> None:
    """The guard that catches a hand-edited or truncated split file -- the
    failure mode that would otherwise leak training images into optval."""
    path = tmp_path / "split.json"
    save_split(create_split(labels), path)

    document = json.loads(path.read_text(encoding="utf-8"))
    # Move one index from calib into optval: still disjoint, still plausible.
    moved = document["indices"]["calib"].pop()
    document["indices"]["optval"].append(moved)
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="integrity check"):
        load_split(path)


def test_fingerprint_is_stable_and_seed_sensitive(labels: list[int]) -> None:
    split = create_split(labels, seed=42)
    assert split_fingerprint(split) == split_fingerprint(create_split(labels, seed=42))
    assert split_fingerprint(split) != split_fingerprint(create_split(labels, seed=7))
