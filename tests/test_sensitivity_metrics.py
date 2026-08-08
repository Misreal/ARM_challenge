"""These metrics rank the layers, so a sign error or a NaN leak would produce a
plausible-looking ranking built on nothing. Hand-computed values throughout."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.sensitivity.metrics import (
    accuracy_from_logits,
    compare,
    flip_rate,
    kl_divergence,
    log_softmax,
    logit_mse,
)


def test_log_softmax_rows_are_normalized() -> None:
    logits = np.array([[1.0, 2.0, 3.0], [-5.0, 0.0, 5.0]], dtype=np.float32)
    probabilities = np.exp(log_softmax(logits))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6)


def test_log_softmax_survives_large_logits() -> None:
    # Without the max-subtraction the exponential overflows to inf and every
    # downstream metric becomes NaN.
    logits = np.array([[1000.0, 999.0]], dtype=np.float32)
    assert np.isfinite(log_softmax(logits)).all()


def test_kl_of_a_distribution_against_itself_is_zero() -> None:
    logits = np.array([[0.5, 1.5, -2.0], [3.0, 0.0, 1.0]], dtype=np.float32)
    assert kl_divergence(logits, logits) == pytest.approx(0.0, abs=1e-7)


def test_kl_is_invariant_to_a_constant_logit_shift() -> None:
    # Softmax is shift-invariant, so a candidate that added a constant to every
    # logit is the same distribution and must score zero divergence.
    logits = np.array([[0.5, 1.5, -2.0]], dtype=np.float32)
    assert kl_divergence(logits, logits + 4.0) == pytest.approx(0.0, abs=1e-6)


def test_kl_matches_a_hand_computed_two_class_case() -> None:
    # p = softmax([0, 0]) = [0.5, 0.5]; q = softmax([0, log 3]) = [0.25, 0.75].
    reference = np.array([[0.0, 0.0]], dtype=np.float64)
    candidate = np.array([[0.0, math.log(3.0)]], dtype=np.float64)
    expected = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    assert kl_divergence(reference, candidate) == pytest.approx(expected, rel=1e-9)


def test_kl_stays_finite_when_the_candidate_probability_underflows() -> None:
    # A confident candidate drives one class to ~0 probability; the ratio form of
    # KL would divide by it and return NaN.
    reference = np.array([[0.0, 0.0]], dtype=np.float32)
    candidate = np.array([[0.0, 300.0]], dtype=np.float32)
    assert np.isfinite(kl_divergence(reference, candidate))


def test_logit_mse_is_hand_checkable() -> None:
    reference = np.array([[1.0, 2.0]], dtype=np.float32)
    candidate = np.array([[3.0, 2.0]], dtype=np.float32)
    assert logit_mse(reference, candidate) == pytest.approx(2.0)


def test_flip_rate_counts_argmax_disagreements_in_both_directions() -> None:
    reference = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    candidate = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    assert flip_rate(reference, candidate) == pytest.approx(50.0)


def test_flip_rate_sees_damage_that_accuracy_cancels_out() -> None:
    # One image flips right->wrong and another wrong->right, so top-1 is
    # unchanged. This is the failure mode that makes accuracy useless for ranking.
    labels = np.array([0, 1], dtype=np.int64)
    reference = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    candidate = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)

    result = compare(reference, candidate, labels)
    assert result.top1_delta == pytest.approx(0.0)
    assert result.flip_rate == pytest.approx(100.0)


def test_accuracy_from_logits() -> None:
    logits = np.array(
        [
            [9.0, 1.0, 2.0, 3.0, 4.0, 5.0],  # top-1 = 0, label hits
            [1.0, 2.0, 3.0, 4.0, 5.0, 9.0],  # top-5 = {5,4,3,2,1}, so label 0 misses
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0], dtype=np.int64)
    top1, top5 = accuracy_from_logits(logits, labels)
    assert top1 == pytest.approx(50.0)
    assert top5 == pytest.approx(50.0)


def test_compare_populates_every_field() -> None:
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(32, 100)).astype(np.float32)
    candidate = reference + rng.normal(scale=0.1, size=(32, 100)).astype(np.float32)
    labels = rng.integers(0, 100, size=32).astype(np.int64)

    result = compare(reference, candidate, labels)
    assert result.samples == 32
    assert result.kl_mean > 0.0
    assert result.logit_mse > 0.0
    assert set(result.as_dict()) == {
        "samples",
        "kl_mean",
        "logit_mse",
        "flip_rate",
        "top1",
        "top5",
        "top1_delta",
    }


def test_compare_accepts_a_precomputed_reference_accuracy() -> None:
    # The sweep scores one reference against many candidates; recomputing its
    # accuracy per candidate would be wasted work on the Pi.
    labels = np.array([0, 1], dtype=np.int64)
    reference = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    candidate = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    assert compare(reference, candidate, labels).top1_delta == pytest.approx(-50.0)
    assert compare(reference, candidate, labels, reference_top1=100.0).top1_delta == pytest.approx(
        -50.0
    )


def test_compare_refuses_mismatched_image_counts() -> None:
    # A candidate scored over a different number of images is not comparable
    # pairwise, and silently truncating would fabricate a divergence.
    reference = np.zeros((4, 10), dtype=np.float32)
    candidate = np.zeros((3, 10), dtype=np.float32)
    labels = np.zeros(4, dtype=np.int64)

    with pytest.raises(ValueError, match="disagree"):
        compare(reference, candidate, labels)
