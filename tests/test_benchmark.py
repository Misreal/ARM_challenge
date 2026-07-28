"""Tests for the shared latency harness.

The statistics and the drift verdict are tested against synthetic timing series
rather than real runs: a real benchmark's numbers are machine-dependent and
noisy, so asserting on them would produce a test that fails for reasons having
nothing to do with the code. The end-to-end tests use a trivial graph and assert
only on structure and invariants, never on how fast anything was.
"""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from src.portable.benchmark import (
    BenchmarkError,
    benchmark_session,
    resolve_input_shape,
    summarize,
)

SHAPE = (1, 3, 4, 4)


def _tiny_session(tmp_path, dynamic_batch: bool = False) -> ort.InferenceSession:
    """A one-node graph, so the tests exercise the harness and not a model."""
    dims: list = list(SHAPE)
    if dynamic_batch:
        dims[0] = "batch"

    graph = helper.make_graph(
        [helper.make_node("Relu", ["images"], ["logits"])],
        "tiny",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, dims)],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, dims)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10

    path = tmp_path / ("dynamic.onnx" if dynamic_batch else "static.onnx")
    onnx.save(model, str(path))
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


class TestSummarize:
    def test_reports_the_expected_distribution(self):
        # Arrange: 1..100 has arithmetic that is easy to verify by hand.
        samples = [float(value) for value in range(1, 101)]

        # Act
        stats = summarize(samples, warmup=5, batch_size=1)

        # Assert
        assert stats.iterations == 100
        assert stats.warmup == 5
        assert stats.median_ms == pytest.approx(50.5)
        assert stats.min_ms == pytest.approx(1.0)
        assert stats.max_ms == pytest.approx(100.0)
        assert stats.p90_ms == pytest.approx(90.1)
        assert stats.p95_ms == pytest.approx(95.05)

    def test_throughput_scales_with_batch_size(self):
        # Arrange: a flat 10 ms per run.
        samples = [10.0] * 20

        # Act
        single = summarize(samples, warmup=0, batch_size=1)
        batched = summarize(samples, warmup=0, batch_size=8)

        # Assert: 10 ms/run is 100 runs/s, so 100 or 800 images/s.
        assert single.throughput_ips == pytest.approx(100.0)
        assert batched.throughput_ips == pytest.approx(800.0)

    def test_steady_timings_are_stable(self):
        # Arrange
        samples = [10.0, 10.1, 9.9, 10.0, 10.2, 9.8, 10.1, 9.9]

        # Act
        stats = summarize(samples, warmup=0, batch_size=1)

        # Assert
        assert stats.stable is True
        assert stats.drift_ratio == pytest.approx(1.0, abs=0.05)

    def test_thermal_drift_is_flagged_unstable(self):
        # Arrange: second half 2x slower, the signature of a device still heating.
        samples = [10.0] * 50 + [20.0] * 50

        # Act
        stats = summarize(samples, warmup=0, batch_size=1)

        # Assert
        assert stats.stable is False
        assert stats.drift_ratio == pytest.approx(2.0)

    def test_drift_uses_arrival_order_not_sorted_order(self):
        # Arrange: the same values, but slow runs come *first* -- a machine
        # settling down, not heating up. Sorting before splitting would report
        # this identically to the heating case, hiding the difference.
        samples = [20.0] * 50 + [10.0] * 50

        # Act
        stats = summarize(samples, warmup=0, batch_size=1)

        # Assert
        assert stats.drift_ratio == pytest.approx(0.5)
        assert stats.stable is True

    def test_tolerance_is_respected(self):
        # Arrange: 4% drift, either side of a 5% and a 2% tolerance.
        samples = [10.0] * 50 + [10.4] * 50

        # Act / Assert
        assert summarize(samples, 0, 1, stability_tolerance=0.05).stable is True
        assert summarize(samples, 0, 1, stability_tolerance=0.02).stable is False

    def test_rejects_too_few_samples(self):
        with pytest.raises(BenchmarkError, match="at least 2"):
            summarize([1.0], warmup=0, batch_size=1)


class TestResolveInputShape:
    def test_returns_static_shape(self, tmp_path):
        session = _tiny_session(tmp_path)
        assert resolve_input_shape(session) == SHAPE

    def test_rejects_dynamic_axis(self, tmp_path):
        # A symbolic batch axis means the export invariant was violated
        # upstream; the harness must refuse rather than assume a batch size.
        session = _tiny_session(tmp_path, dynamic_batch=True)
        with pytest.raises(BenchmarkError, match="non-static axis 0"):
            resolve_input_shape(session)


class TestBenchmarkSession:
    def test_produces_stats_and_host_probe(self, tmp_path):
        # Arrange
        session = _tiny_session(tmp_path)

        # Act
        stats, host = benchmark_session(session, warmup=2, iterations=10)

        # Assert
        assert stats.iterations == 10
        assert stats.warmup == 2
        assert stats.batch_size == 1
        assert stats.min_ms <= stats.median_ms <= stats.max_ms
        assert stats.throughput_ips > 0
        assert host.machine  # always populated, unlike the optional probes

    def test_is_reproducible_in_its_input(self, tmp_path):
        # The seed must fix the payload, so two runs feed identical tensors and
        # any timing difference is the machine rather than the data.
        session = _tiny_session(tmp_path)
        first = np.random.default_rng(7).standard_normal(SHAPE, dtype=np.float32)
        second = np.random.default_rng(7).standard_normal(SHAPE, dtype=np.float32)
        assert np.array_equal(first, second)

        stats, _ = benchmark_session(session, warmup=1, iterations=4, seed=7)
        assert stats.iterations == 4

    def test_rejects_degenerate_iteration_counts(self, tmp_path):
        session = _tiny_session(tmp_path)
        with pytest.raises(BenchmarkError, match="iterations must be >= 2"):
            benchmark_session(session, warmup=1, iterations=1)

    def test_rejects_negative_warmup(self, tmp_path):
        session = _tiny_session(tmp_path)
        with pytest.raises(BenchmarkError, match="warmup must be >= 0"):
            benchmark_session(session, warmup=-1, iterations=4)
