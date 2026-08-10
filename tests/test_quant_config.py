"""Config hashing is correctness-critical: it is the identity the artifact cache
and every Pi measurement are keyed by. A hash that collides wastes a
measurement; a hash that splits when it should not causes the same file to be
benchmarked twice under different names."""

from __future__ import annotations

import pytest

from src.quant.config import DeploymentConfig, QuantConfig, RunConfig


def test_hash_is_stable_across_equal_configs() -> None:
    assert QuantConfig(quant_type="static").hash == QuantConfig(quant_type="static").hash


def test_excluded_group_order_does_not_change_hash() -> None:
    # Optuna may emit groups in any order; the artifact is identical either way.
    a = QuantConfig(excluded_groups=("stem", "stage3"))
    b = QuantConfig(excluded_groups=("stage3", "stem"))
    assert a.hash == b.hash


def test_irrelevant_fields_are_canonicalized_away() -> None:
    # Dynamic quantization never reads calibration settings, so configs that
    # differ only there produce byte-identical artifacts and must share a hash.
    a = QuantConfig(quant_type="dynamic", calibration_method="minmax", calibration_size=128)
    b = QuantConfig(quant_type="dynamic", calibration_method="entropy", calibration_size=1000)
    assert a.hash == b.hash

    # Same for activation type, which only exists for static quantization.
    c = QuantConfig(quant_type="dynamic", activation_type="int8")
    assert a.hash == c.hash


def test_fp32_ignores_every_quantization_field() -> None:
    a = QuantConfig(quant_type="none")
    b = QuantConfig(quant_type="none", per_channel=True, calibration_size=1000)
    assert a.hash == b.hash


def test_relevant_fields_do_change_hash() -> None:
    base = QuantConfig(quant_type="static")
    assert base.hash != QuantConfig(quant_type="static", per_channel=True).hash
    assert base.hash != QuantConfig(quant_type="static", activation_type="int8").hash
    assert base.hash != QuantConfig(quant_type="static", calibration_size=128).hash
    assert base.hash != QuantConfig(quant_type="static", calibration_method="entropy").hash
    assert base.hash != QuantConfig(quant_type="static", excluded_groups=("stem",)).hash
    assert base.hash != QuantConfig(quant_type="dynamic").hash


def test_runtime_settings_do_not_affect_the_artifact_hash() -> None:
    # The whole point of the two-object split: one quantization serves every
    # thread count, so the artifact cache must not see runtime settings.
    quant = QuantConfig()
    one = DeploymentConfig(quant=quant, run=RunConfig(intra_op_num_threads=1))
    four = DeploymentConfig(quant=quant, run=RunConfig(intra_op_num_threads=4))

    assert one.quant.hash == four.quant.hash
    assert one.hash != four.hash  # but they are distinct *measurements*


def test_round_trips_through_dict() -> None:
    config = DeploymentConfig(
        quant=QuantConfig(quant_type="static", per_channel=True, excluded_groups=("stem", "fc")),
        run=RunConfig(intra_op_num_threads=2, graph_optimization_level="basic"),
    )
    restored = DeploymentConfig.from_dict(config.as_dict())
    assert restored == config
    assert restored.hash == config.hash


@pytest.mark.parametrize(
    "kwargs",
    [
        {"quant_type": "int4"},
        {"activation_type": "float16"},
        {"weight_type": "bogus"},
        {"calibration_method": "kl"},
        {"calibration_size": 0},
        {"excluded_groups": ("stem", "stem")},
    ],
)
def test_invalid_configs_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises((ValueError, TypeError)):
        QuantConfig(**kwargs)  # type: ignore[arg-type]


def test_excluded_groups_must_be_hashable() -> None:
    with pytest.raises(TypeError):
        QuantConfig(excluded_groups=["stem"])  # type: ignore[arg-type]


def test_invalid_run_configs_are_rejected() -> None:
    with pytest.raises(ValueError):
        RunConfig(intra_op_num_threads=0)
    with pytest.raises(ValueError):
        RunConfig(graph_optimization_level="maximum")


def test_new_runtime_knobs_change_the_measurement_identity() -> None:
    base = RunConfig()
    assert base.hash != RunConfig(enable_cpu_mem_arena=False).hash
    assert base.hash != RunConfig(allow_intra_op_spinning=False).hash


def test_default_run_hash_survived_adding_knobs() -> None:
    # Pinned literal, not recomputed: every Pi measurement cached and every
    # sensitivity result joined during Phases 3-6 keys on this digest. If adding
    # a knob changes it, those results silently stop resolving.
    assert RunConfig(intra_op_num_threads=4, graph_optimization_level="all").hash == (
        "facf1876de07230d"
    )


def test_run_config_ignores_fields_it_does_not_know() -> None:
    # A cached result written by a newer PC must stay readable by an older Pi.
    restored = RunConfig.from_dict({"intra_op_num_threads": 2, "future_knob": "on"})
    assert restored.intra_op_num_threads == 2
