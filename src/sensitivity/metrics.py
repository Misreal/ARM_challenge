# Compare a quantized candidate's logits against the FP32 reference on identical images.
#
# Top-1 delta cannot rank layers here: full INT8 costs 3-11 images out of 3000, so
# per-group deltas sit at the sampling floor. These metrics are continuous or
# high-count instead, which is what gives the ranking any resolution at all.

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

TOP_K = 5


@dataclass(frozen=True)
class ComparisonResult:
    """One candidate scored against the FP32 reference."""

    samples: int
    kl_mean: float
    logit_mse: float
    flip_rate: float
    top1: float
    top5: float
    top1_delta: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise log-softmax over the class axis."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def kl_divergence(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Mean KL(reference || candidate) in nats, over images.

    Computed from log-probabilities rather than probabilities: a ratio form
    divides by a candidate probability that underflows to zero on confident
    predictions, and 0*inf would poison the mean with NaN.
    """
    log_p = log_softmax(reference)
    log_q = log_softmax(candidate)
    per_image = (np.exp(log_p) * (log_p - log_q)).sum(axis=1)
    return float(per_image.mean())


def logit_mse(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Mean squared difference of raw logits."""
    difference = reference.astype(np.float64) - candidate.astype(np.float64)
    return float(np.square(difference).mean())


def flip_rate(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Percent of images whose predicted class differs from the reference.

    Counts disturbance in both directions, unlike accuracy, where a wrong->right
    flip cancels a right->wrong one and hides the damage.
    """
    changed = np.argmax(reference, axis=1) != np.argmax(candidate, axis=1)
    return float(100.0 * changed.mean())


def accuracy_from_logits(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Top-1 and top-5 accuracy, as percentages."""
    targets = labels[: logits.shape[0]].astype(np.int64)
    top1 = float(100.0 * (np.argmax(logits, axis=1) == targets).mean())
    # Clamped so a narrow test fixture cannot blow up on an out-of-bounds kth.
    k = min(TOP_K, logits.shape[1])
    top_k = np.argpartition(logits, -k, axis=1)[:, -k:]
    top5 = float(100.0 * (top_k == targets[:, None]).any(axis=1).mean())
    return top1, top5


def _check_shapes(reference: np.ndarray, candidate: np.ndarray, labels: np.ndarray) -> None:
    if reference.shape != candidate.shape:
        raise ValueError(
            f"Reference logits {reference.shape} and candidate {candidate.shape} disagree; "
            "the two were scored over different images and cannot be compared pairwise."
        )
    if reference.ndim != 2:
        raise ValueError(f"Expected (images, classes) logits, got shape {reference.shape}")
    if labels.shape[0] < reference.shape[0]:
        raise ValueError(f"Have {labels.shape[0]} labels for {reference.shape[0]} images")


def compare(
    reference: np.ndarray,
    candidate: np.ndarray,
    labels: np.ndarray,
    reference_top1: float | None = None,
) -> ComparisonResult:
    """Score `candidate` against `reference`; both must cover the same images in order."""
    _check_shapes(reference, candidate, labels)

    top1, top5 = accuracy_from_logits(candidate, labels)
    baseline_top1 = (
        accuracy_from_logits(reference, labels)[0] if reference_top1 is None else reference_top1
    )

    return ComparisonResult(
        samples=int(reference.shape[0]),
        kl_mean=kl_divergence(reference, candidate),
        logit_mse=logit_mse(reference, candidate),
        flip_rate=flip_rate(reference, candidate),
        top1=top1,
        top5=top5,
        top1_delta=top1 - baseline_top1,
    )
