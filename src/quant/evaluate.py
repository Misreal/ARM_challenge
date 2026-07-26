"""PC-side accuracy scoring for quantized candidates.

Screening happens here, on x86, because it is fast and needs no device. The
honesty caveat from PLAN.md applies and is worth repeating in any report built
on these numbers: x86 and ARM INT8 kernels can differ by roughly +/-0.1 pt from
requantization rounding, so a PC score is a filter, not a finding. Finalists get
their accuracy re-verified on the Pi.

The evaluation loop itself lives in `src.portable.onnx_eval`, shared verbatim
with the device, so any PC-vs-Pi gap is attributable to kernels rather than to
two implementations drifting apart.
"""

from __future__ import annotations

from pathlib import Path

from src.portable.bundle import DEFAULT_BUNDLE_DIR, load_bundle
from src.portable.onnx_eval import AccuracyResult, evaluate_accuracy
from src.quant.config import RunConfig
from src.quant.quantize import Artifact, build_session

EVAL_SPLIT = "optval"


def score_artifact(
    artifact: Artifact,
    run: RunConfig | None = None,
    limit: int | None = None,
    bundle_dir: Path = DEFAULT_BUNDLE_DIR,
) -> AccuracyResult:
    """Top-1/top-5 of one artifact on the optimization-validation split.

    `limit` drives the search's cheap accuracy screen. The bundle is stored in
    dataset-index order, which is effectively random with respect to class, so a
    prefix is an unbiased sample -- but note it is not *stratified*, so at small
    N the estimate carries the usual binomial spread (about 2 pts at N=500,
    which is why PLAN.md screens against a threshold minus a noise margin).
    """
    bundle = load_bundle(EVAL_SPLIT, bundle_dir=bundle_dir)
    session = build_session(artifact.path, run or RunConfig())
    return evaluate_accuracy(session, bundle.images_uint8, bundle.labels, limit=limit)
