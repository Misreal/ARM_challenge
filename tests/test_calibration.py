"""Calibration subset selection shapes the INT8 activation ranges, so an
unbalanced subset skews every static-quantized candidate in the same direction
-- a bias that looks like a property of quantization rather than a sampling
artifact."""

from __future__ import annotations

import numpy as np
import pytest

from src.quant.calibration import stratified_prefix


def _calib_like_labels(per_class: int = 10, classes: int = 100) -> np.ndarray:
    """Labels shaped like the real calib split: 10 per class, index-ordered so
    class membership is scattered rather than grouped."""
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(classes), per_class)
    return labels[rng.permutation(labels.size)]


@pytest.mark.parametrize("size", [128, 512, 1000])
def test_subsets_are_class_balanced(size: int) -> None:
    labels = _calib_like_labels()
    chosen = labels[stratified_prefix(labels, size)]
    _, counts = np.unique(chosen, return_counts=True)
    # Round-robin selection can never differ by more than one per class.
    assert counts.max() - counts.min() <= 1


def test_small_subset_still_covers_most_classes() -> None:
    # The failure this guards against: a raw index prefix of 128 covers only
    # ~79 of 100 classes, leaving 21 classes contributing no activation range.
    labels = _calib_like_labels()
    chosen = labels[stratified_prefix(labels, 128)]
    assert len(np.unique(chosen)) == 100


def test_full_size_returns_every_image_once() -> None:
    labels = _calib_like_labels()
    indices = stratified_prefix(labels, labels.size)
    assert sorted(indices.tolist()) == list(range(labels.size))


def test_selection_is_deterministic() -> None:
    labels = _calib_like_labels()
    assert np.array_equal(stratified_prefix(labels, 256), stratified_prefix(labels, 256))


def test_prefixes_are_nested() -> None:
    # A smaller calibration size should be a subset of a larger one, so the
    # calibration_size search dimension varies sample size and nothing else.
    labels = _calib_like_labels()
    small = stratified_prefix(labels, 128).tolist()
    large = stratified_prefix(labels, 512).tolist()
    assert small == large[:128]


def test_oversized_request_is_rejected() -> None:
    labels = _calib_like_labels()
    with pytest.raises(ValueError, match="only"):
        stratified_prefix(labels, labels.size + 1)


def test_nonpositive_size_is_rejected() -> None:
    with pytest.raises(ValueError):
        stratified_prefix(_calib_like_labels(), 0)
