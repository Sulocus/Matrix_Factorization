import pytest
import torch

from matrix_factorization.core.contracts import (
    get_algorithm_specs,
    get_normalization_specs,
    get_precision_policy_specs,
)
from matrix_factorization.core.experiment.config import (
    AlgorithmParams,
    ExperimentConfig,
    MatrixParams,
    ScanConfig,
    TrainingParams,
)
from matrix_factorization.core.experiment.data_factory import DataFactory
from matrix_factorization.core.memory_calibration import estimation_params_from_config
from matrix_factorization.core.planning import build_experiment_plan
from matrix_factorization.modules.algorithms.agd import AGDAlgorithm
from matrix_factorization.modules.algorithms.bigamp.spreading import BiGAMPSpreading
from matrix_factorization.modules.algorithms.bigamp.standard import BiGAMPAlgorithm
from matrix_factorization.modules.algorithms.bigamp.tensor_spreading import BiGAMPTensorSpreading
from matrix_factorization.modules.algorithms.bigamp.tensor_spreading_parallel import BiGAMPTensorSpreadingParallel


def _active_algorithm_keys():
    return [
        key
        for key, spec in get_algorithm_specs().items()
        if spec.status == "active"
    ]


def _small_config(algorithm_key: str) -> ExperimentConfig:
    return ExperimentConfig(
        matrix=MatrixParams(N1=4, N2=4, M=2),
        training=TrainingParams(samples_per_alpha=1, max_steps=1, max_epochs=1),
        algorithm_key=algorithm_key,
        scan=ScanConfig(dimension="alpha", values=[0.1]),
        algorithm_params=AlgorithmParams(
            use_compile=False,
            precision_profile="safe",
            use_bf16=False,
            normalization_profile="paper_sparse_sampling",
        ),
    )


def test_active_algorithms_have_normalization_and_precision_specs():
    normalization_specs = get_normalization_specs()
    precision_specs = get_precision_policy_specs()

    for key in _active_algorithm_keys():
        normalization = normalization_specs[key]
        precision = precision_specs[key]

        assert normalization.status == "active"
        assert normalization.default_profile == "paper_sparse_sampling"
        assert "internal_normalized_legacy" in normalization.profiles
        assert normalization.latent_scale == "O(1)"
        assert normalization.teacher_init_variance == "1"
        assert normalization.student_init_variance == "1"
        assert normalization.prior_precision_base == "1"
        assert normalization.schema_version >= 5

        assert precision.status == "active"
        assert set(precision.profiles) >= {"safe", "fast", "aggressive"}
        role_map = precision.role_dtype_map("fast")
        if key != "agd":
            assert role_map["student_factors"]["storage"] == "bfloat16"
            assert role_map["factor_variances"]["storage"] == "bfloat16"
            assert role_map["metric_reductions"]["storage"] == "float32"


def test_teacher_initialization_uses_paper_unit_scale_by_default():
    factory = DataFactory(device=torch.device("cpu"))
    M = 16

    W, X, _ = factory.create_teacher(
        N1=512,
        N2=512,
        M=M,
        teacher_key="standard",
        seed=123,
        init_distribution="gaussian",
    )
    assert float(W.pow(2).mean()) == pytest.approx(1.0, rel=0.20)
    assert float(X.pow(2).mean()) == pytest.approx(1.0, rel=0.20)

    W_orth, X_orth, _ = factory.create_teacher(
        N1=64,
        N2=64,
        M=8,
        teacher_key="orthogonal",
        seed=123,
    )
    assert float(W_orth.pow(2).mean()) == pytest.approx(1.0, rel=1e-3)
    assert float(X_orth.pow(2).mean()) == pytest.approx(1.0, rel=1e-3)


def test_teacher_initialization_legacy_profile_keeps_inverse_sqrt_m_scale():
    factory = DataFactory(device=torch.device("cpu"))
    M = 16

    W, X, _ = factory.create_teacher(
        N1=512,
        N2=512,
        M=M,
        teacher_key="standard",
        seed=123,
        init_distribution="gaussian",
        normalization_profile="internal_normalized_legacy",
    )
    assert float(W.pow(2).mean()) == pytest.approx(1.0 / M, rel=0.20)
    assert float(X.pow(2).mean()) == pytest.approx(1.0 / M, rel=0.20)


def test_tensor_direct_teacher_default_uses_paper_unit_scale():
    algorithm = BiGAMPTensorSpreading(
        tensor_order=3,
        dims=(64, 64, 64),
        M=16,
        max_steps=1,
        S=1,
        device=torch.device("cpu"),
    )
    factors = algorithm.create_teacher(torch.device("cpu"), seed=123)

    for factor in factors:
        assert float(factor.pow(2).mean()) == pytest.approx(1.0, rel=0.25)


def test_tensor_direct_teacher_legacy_profile_keeps_inverse_sqrt_m_scale():
    algorithm = BiGAMPTensorSpreading(
        tensor_order=3,
        dims=(64, 64, 64),
        M=16,
        max_steps=1,
        S=1,
        device=torch.device("cpu"),
        normalization_profile="internal_normalized_legacy",
    )
    factors = algorithm.create_teacher(torch.device("cpu"), seed=123)

    for factor in factors:
        assert float(factor.pow(2).mean()) == pytest.approx(1.0 / 16, rel=0.25)


@pytest.mark.parametrize(
    ("algorithm_key", "algorithm_cls"),
    [
        ("agd", AGDAlgorithm),
        ("bigamp", BiGAMPAlgorithm),
        ("bigamp_spreading", BiGAMPSpreading),
        ("bigamp_tensor", BiGAMPTensorSpreading),
        ("bigamp_tensor_parallel", BiGAMPTensorSpreadingParallel),
    ],
)
def test_active_algorithms_resolve_paper_normalization_profile(algorithm_key, algorithm_cls):
    config = _small_config(algorithm_key)
    algorithm = algorithm_cls(config, torch.device("cpu"))

    assert algorithm.normalization_profile == "paper_sparse_sampling"
    assert algorithm._norm.student_init_std == pytest.approx(1.0)
    assert algorithm._norm.prior_variance == pytest.approx(1.0)
    assert algorithm._norm.prior_precision_base == pytest.approx(1.0)


def test_precision_profile_controls_estimation_role_dtypes():
    config = ExperimentConfig(
        matrix=MatrixParams(N1=8, N2=8, M=2),
        training=TrainingParams(samples_per_alpha=1, max_steps=1),
        algorithm_key="bigamp_spreading",
        scan=ScanConfig(dimension="alpha", values=[0.1]),
        algorithm_params=AlgorithmParams(
            precision_profile="aggressive",
            use_bf16=False,
            use_compile=False,
        ),
    )
    params = estimation_params_from_config(config)

    assert params.precision_profile == "aggressive"
    assert params.use_bf16 is True
    assert params.role_dtype_map["student_factors"]["storage"] == "bfloat16"
    assert params.role_dtype_map["observations_Y"]["storage"] == "bfloat16"


def test_invalid_precision_profile_is_rejected():
    with pytest.raises(ValueError, match="precision_profile"):
        AlgorithmParams(precision_profile="turbo")


def test_invalid_normalization_profile_is_rejected():
    with pytest.raises(ValueError, match="normalization_profile"):
        AlgorithmParams(normalization_profile="paperish")


def test_strict_plan_rejects_precision_legacy_conflict():
    config = ExperimentConfig(
        matrix=MatrixParams(N1=4, N2=4, M=2),
        training=TrainingParams(samples_per_alpha=1, max_steps=1),
        algorithm_key="bigamp",
        scan=ScanConfig(dimension="alpha", values=[0.0]),
        algorithm_params=AlgorithmParams(
            precision_profile="safe",
            use_bf16=True,
            use_compile=False,
        ),
    )
    raw_yaml = """
algorithm_params:
  precision_profile: safe
  use_bf16: true
"""

    plan = build_experiment_plan(config, {}, raw_yaml, strict=True)

    assert any("precision_profile" in error and "use_bf16" in error for error in plan.errors)
