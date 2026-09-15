# ONNX Runtime accuracy evaluation, shared verbatim between PC and Pi.
#
# The graphs are exported static at batch size 1 (a project invariant), so this
# feeds one image per call on both machines. That is slower than batching but it
# is what the deployment target actually does, and identical code on both sides
# means a PC-vs-Pi accuracy gap can only come from the kernels -- which is
# precisely the quantity Phase 4 warns is worth about +/-0.1 pt.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

from src.portable.preprocess import normalize_uint8_nchw


@dataclass(frozen=True)
class AccuracyResult:
    samples: int
    top1: float
    top5: float

    def as_dict(self) -> dict[str, float | int]:
        return {"samples": self.samples, "top1": self.top1, "top5": self.top5}


def evaluate_accuracy(
    session: ort.InferenceSession,
    images_uint8: np.ndarray,
    labels: np.ndarray,
    limit: int | None = None,
) -> AccuracyResult:
    """Top-1/top-5 accuracy of `session` over uint8 NCHW images.

    `limit` truncates to the first N images, which is how the search's cheap
    accuracy screen works. Callers relying on that must ensure the bundle order
    is class-balanced (see `src.quant.calibration.stratified_prefix`), or the
    screen measures the wrong subset.
    """
    total = int(images_uint8.shape[0]) if limit is None else min(limit, int(images_uint8.shape[0]))
    if total <= 0:
        raise ValueError("No images to evaluate")
    if labels.shape[0] < total:
        raise ValueError(f"Have {labels.shape[0]} labels for {total} images")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    top1_hits = 0
    top5_hits = 0
    for index in range(total):
        # Slice rather than index so the leading batch axis survives.
        batch = normalize_uint8_nchw(images_uint8[index : index + 1])
        logits = session.run([output_name], {input_name: batch})[0][0]

        target = int(labels[index])
        if int(np.argmax(logits)) == target:
            top1_hits += 1
        # argpartition is O(n) and enough for a membership test.
        if target in np.argpartition(logits, -5)[-5:]:
            top5_hits += 1

    return AccuracyResult(
        samples=total,
        top1=100.0 * top1_hits / total,
        top5=100.0 * top5_hits / total,
    )


def collect_logits(
    session: ort.InferenceSession,
    images_uint8: np.ndarray,
    limit: int | None = None,
) -> np.ndarray:
    """Raw logits for every image, shape (N, classes) float32.

    Kept separate from `evaluate_accuracy` rather than shared with it: that
    function runs inside the bench agent's measured process, and the buffer this
    one allocates would land in `ru_maxrss` and misattribute per-candidate RAM.
    """
    total = int(images_uint8.shape[0]) if limit is None else min(limit, int(images_uint8.shape[0]))
    if total <= 0:
        raise ValueError("No images to evaluate")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    rows: list[np.ndarray] = []
    for index in range(total):
        batch = normalize_uint8_nchw(images_uint8[index : index + 1])
        rows.append(session.run([output_name], {input_name: batch})[0][0])

    return np.asarray(rows, dtype=np.float32)
