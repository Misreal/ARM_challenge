# Calibration data for static INT8 quantization.
#
# Calibration images come exclusively from the 1,000-image `calib` split, loaded
# through the Pi bundle -- a structural guarantee rather than a rule someone has
# to remember, since there is no code path here that can reach optval or test.
# `splits.py` stratifies calib 10-per-class then sorts by dataset index, so a
# raw prefix of 128 images covers only ~79 of 100 classes and skews the
# activation ranges calibration measures. `stratified_prefix` takes round-robin
# across classes instead so every subset size stays balanced, while the arrays
# on disk keep their faithful index order.

from __future__ import annotations

from pathlib import Path

import numpy as np
from onnxruntime.quantization import CalibrationDataReader

from src.portable.bundle import DEFAULT_BUNDLE_DIR, load_bundle
from src.portable.preprocess import normalize_uint8_nchw

CALIBRATION_SPLIT = "calib"


def stratified_prefix(labels: np.ndarray, size: int) -> np.ndarray:
    """Indices of a class-balanced subset of `size` items, deterministically.

    Walks the classes round-robin, taking one unused member of each in turn, so
    any prefix of the result is as balanced as that size allows. Ties are broken
    by original position, which keeps the output a pure function of the input.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    total = int(labels.shape[0])
    if size > total:
        raise ValueError(f"Requested {size} calibration images but only {total} are available")

    by_class: dict[int, list[int]] = {}
    for position, label in enumerate(labels.tolist()):
        by_class.setdefault(int(label), []).append(position)

    selected: list[int] = []
    ordered_classes = sorted(by_class)
    round_index = 0
    while len(selected) < size:
        progressed = False
        for class_id in ordered_classes:
            members = by_class[class_id]
            if round_index < len(members):
                selected.append(members[round_index])
                progressed = True
                if len(selected) == size:
                    break
        if not progressed:  # pragma: no cover -- guarded by the size check above
            raise RuntimeError("Ran out of calibration images before reaching the requested size")
        round_index += 1

    return np.asarray(selected, dtype=np.int64)


class NumpyCalibrationReader(CalibrationDataReader):
    """Feeds normalized batch-1 images to ONNX Runtime's calibrator.

    Batch 1 because the graphs are exported with a static batch axis; feeding
    anything else would fail shape inference during calibration.
    """

    def __init__(self, images_uint8: np.ndarray, input_name: str) -> None:
        self._batches = [
            normalize_uint8_nchw(images_uint8[index : index + 1])
            for index in range(images_uint8.shape[0])
        ]
        self._input_name = input_name
        self._cursor = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._cursor >= len(self._batches):
            return None
        batch = self._batches[self._cursor]
        self._cursor += 1
        return {self._input_name: batch}

    def rewind(self) -> None:
        """Reset for calibrators that make more than one pass (Entropy, Percentile)."""
        self._cursor = 0

    def __len__(self) -> int:
        return len(self._batches)


def build_calibration_reader(
    input_name: str,
    size: int,
    bundle_dir: Path = DEFAULT_BUNDLE_DIR,
) -> NumpyCalibrationReader:
    """Load `size` class-balanced calibration images as an ORT data reader."""
    bundle = load_bundle(CALIBRATION_SPLIT, bundle_dir=bundle_dir)
    indices = stratified_prefix(bundle.labels, size)
    return NumpyCalibrationReader(bundle.images_uint8[indices], input_name)
