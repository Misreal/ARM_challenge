"""The importer is the only gate between a stranger's file and the whole pipeline."""

from __future__ import annotations

import onnx
import pytest
from onnx import TensorProto, helper

from src.import_onnx import ImportRejected, check_graph, tensor_shape

CLASSES = 100


def graph_with(shape=(1, 3, 32, 32), classes=CLASSES, opset=17) -> onnx.ModelProto:
    """A one-node classifier, which is all the shape and opset guards look at."""
    weight = helper.make_tensor(
        "w", TensorProto.FLOAT, [classes, shape[1] * shape[2] * shape[3]],
        [0.0] * (classes * shape[1] * shape[2] * shape[3]),
    )
    nodes = [
        helper.make_node("Flatten", ["images"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "w"], ["logits"], transB=1),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, list(shape))],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [shape[0], classes])],
        initializer=[weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 9  # what onnx 1.22 checks against for opset 17
    return model


def test_a_conforming_graph_reports_what_the_pipeline_needs(tmp_path) -> None:
    facts = check_graph(graph_with(), tmp_path / "m.onnx")
    assert facts["input_shape"] == [1, 3, 32, 32]
    assert facts["opset"] == 17
    assert facts["operators"]["Gemm"] == 1


def test_a_224_pixel_model_is_refused_with_the_reason(tmp_path) -> None:
    # The eval bundle is 32x32 uint8; accepting this would mean resizing, and
    # resizing is where preprocessing divergence quietly costs accuracy.
    with pytest.raises(ImportRejected, match="32, 32"):
        check_graph(graph_with(shape=(1, 3, 224, 224)), tmp_path / "vit.onnx")


def test_a_batched_export_is_refused(tmp_path) -> None:
    with pytest.raises(ImportRejected):
        check_graph(graph_with(shape=(8, 3, 32, 32)), tmp_path / "batched.onnx")


def test_a_dynamic_batch_axis_is_refused(tmp_path) -> None:
    model = graph_with()
    # Dynamic axes break static calibration and the latency methodology both.
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch"
    with pytest.raises(ImportRejected):
        check_graph(model, tmp_path / "dynamic.onnx")


def test_a_model_with_the_wrong_number_of_classes_is_refused(tmp_path) -> None:
    with pytest.raises(ImportRejected, match="10 classes"):
        check_graph(graph_with(classes=10), tmp_path / "cifar10.onnx")


def test_an_opset_too_old_for_per_channel_quantization_is_refused(tmp_path) -> None:
    with pytest.raises(ImportRejected, match="opset 11"):
        check_graph(graph_with(opset=11), tmp_path / "old.onnx")


def test_a_dynamic_axis_reads_back_by_name_not_as_a_number() -> None:
    model = graph_with()
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch"
    assert tensor_shape(model.graph.input[0])[0] == "batch"
