from pathlib import Path

import pytest
import torch

from matrix_factorization.cli import load_yaml_config
from matrix_factorization.core.experiment.config import (
    AlgorithmParams,
    ExperimentConfig,
    MatrixParams,
    ScanConfig,
    SpreadingConfig,
    TeacherConfig,
    TrainingParams,
)
from matrix_factorization.core.experiment.result import ExperimentResult, SingleRunResult
from matrix_factorization.core.experiment.runner import ExperimentRunner
from matrix_factorization.modules.algorithms.bigamp.tensor_spreading_parallel import (
    BiGAMPTensorSpreadingParallel,
)
from matrix_factorization.modules.algorithms.bigamp.spreading import BiGAMPSpreading
from matrix_factorization.modules.algorithms.bigamp.standard import BiGAMPAlgorithm
from matrix_factorization.modules.algorithms.agd import AGDAlgorithm


def _write_config(path: Path, tensor_order: int, algorithm):
    path.write_text(
        f"""
tensor_order: {tensor_order}
algorithm: {algorithm}
teacher: 2
teacher_config:
  init_distribution: 2
matrix:
  N1: 4
  N2: 4
  M: 2
scan:
  axes:
    alpha:
      path: alpha
      values:
        start: 0.0
        stop: 0.2
        step: 0.2
training:
  samples_per_alpha: 1
  max_steps: 2
  num_workers: 3
  seed: 7
algorithm_params:
  damping: 0.5
  noise_var: 1.0e-5
  use_compile: false
  use_bf16: false
spreading:
  f_distribution: 1
  seed: 123
  onsager_correction: false
  chunk_size: 0
output:
  save_tensors: false
  enable_heatmap: false
  storage_mode: lightweight
  heatmap_metric: Q_Y
""",
        encoding="utf-8",
    )


def test_tensor_order_routes_to_parallel_tensor_algorithm(tmp_path):
    config_path = tmp_path / "tensor.yaml"
    _write_config(config_path, tensor_order=3, algorithm=4)

    config, output_options, raw_yaml = load_yaml_config(config_path)

    assert "tensor_order: 3" in raw_yaml
    assert config.algorithm_key == "bigamp_tensor_parallel"
    assert config.spreading.tensor_order == 3
    assert config.spreading.allow_intra_connection is False
    assert config.spreading.seed == 123
    assert config.training.num_workers == 3
    assert config.seeds.base_seed == 7
    assert config.teacher.init_distribution == "rademacher"
    assert output_options["save_tensors"] is False
    assert output_options["enable_heatmap"] is False


def test_tensor_order_one_enables_general_spreading(tmp_path):
    config_path = tmp_path / "general.yaml"
    _write_config(config_path, tensor_order=1, algorithm=2)

    config, output_options, _ = load_yaml_config(config_path)

    assert config.algorithm_key == "bigamp_spreading"
    assert config.spreading.tensor_order == 1
    assert config.spreading.allow_intra_connection is True
    assert output_options["storage_mode"] == "lightweight"


def test_spreading_respects_use_bf16_false(monkeypatch, tmp_path):
    config_path = tmp_path / "general.yaml"
    _write_config(config_path, tensor_order=1, algorithm=2)
    config, _, _ = load_yaml_config(config_path)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    algorithm = BiGAMPSpreading(config, device=torch.device("cpu"))

    assert algorithm.requested_use_bf16 is False
    assert algorithm.use_bf16 is False
    assert algorithm.storage_dtype == torch.float32
    assert algorithm.dtype_status == "bf16_disabled_by_config"
    assert algorithm._contract_execution_metadata["requested_use_bf16"] is False
    assert algorithm._contract_execution_metadata["dtype_status"] == "bf16_disabled_by_config"


def test_agd_respects_use_bf16_false_on_cuda_device():
    config = ExperimentConfig(
        matrix=MatrixParams(N1=2, N2=2, M=1),
        training=TrainingParams(samples_per_alpha=1, max_steps=1, max_epochs=1),
        algorithm_key="agd",
        scan=ScanConfig(dimension="alpha", values=[0.0]),
        algorithm_params=AlgorithmParams(
            use_bf16=False,
            dtype_fallback_policy="allow",
        ),
        teacher_key="standard",
    )
    algo_config = ExperimentRunner(device=torch.device("cpu"), verbose=False)._build_algorithm_config(config)
    algorithm = AGDAlgorithm(algo_config, device=torch.device("cuda"))

    assert algorithm.requested_use_bf16 is False
    assert algorithm.use_bf16 is False
    assert algorithm.compute_dtype == torch.float32
    assert algorithm._contract_execution_metadata["dtype_status"] == "bf16_disabled_by_config"


def test_tensor_parallel_respects_use_compile_config(tmp_path):
    config_path = tmp_path / "tensor.yaml"
    _write_config(config_path, tensor_order=3, algorithm=4)

    config, _, _ = load_yaml_config(config_path)
    algorithm = BiGAMPTensorSpreadingParallel(config, device=None)

    assert algorithm.use_compile is False


def test_tensor_parallel_records_compile_fallback_policy(tmp_path):
    config_path = tmp_path / "tensor.yaml"
    _write_config(config_path, tensor_order=3, algorithm=4)
    text = config_path.read_text(encoding="utf-8").replace(
        "  use_compile: false",
        "  use_compile: false\n  compile_fallback_policy: error",
    )
    config_path.write_text(text, encoding="utf-8")

    config, output_options, raw_yaml = load_yaml_config(config_path)
    from matrix_factorization.core.planning import build_experiment_plan

    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)
    chain = {item["path"]: item for item in plan.parameter_chain()}
    algorithm = BiGAMPTensorSpreadingParallel(config, device=None)

    assert chain["algorithm_params.compile_fallback_policy"]["effective_value"] == "error"
    assert plan.resource_plan["config_effective"]["compile_fallback_policy"] == "error"
    assert algorithm.compile_fallback_policy == "error"


def test_spreading_records_compile_fallback_policy(tmp_path):
    config_path = tmp_path / "general.yaml"
    _write_config(config_path, tensor_order=1, algorithm=2)
    text = config_path.read_text(encoding="utf-8").replace(
        "  use_compile: false",
        "  use_compile: false\n  compile_fallback_policy: error",
    )
    config_path.write_text(text, encoding="utf-8")

    config, output_options, raw_yaml = load_yaml_config(config_path)
    from matrix_factorization.core.planning import build_experiment_plan

    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)
    chain = {item["path"]: item for item in plan.parameter_chain()}
    algorithm = BiGAMPSpreading(config, device=torch.device("cpu"))

    assert chain["algorithm_params.compile_fallback_policy"]["effective_value"] == "error"
    assert plan.resource_plan["config_effective"]["compile_fallback_policy"] == "error"
    assert algorithm.compile_fallback_policy == "error"
    assert algorithm._contract_execution_metadata["compile_status"] == "disabled_by_config"


def test_bigamp_records_compile_fallback_policy(tmp_path):
    config_path = tmp_path / "matrix.yaml"
    _write_config(config_path, tensor_order=2, algorithm=1)
    text = config_path.read_text(encoding="utf-8").replace(
        "  use_compile: false",
        "  use_compile: false\n  compile_fallback_policy: error",
    )
    config_path.write_text(text, encoding="utf-8")

    config, output_options, raw_yaml = load_yaml_config(config_path)
    from matrix_factorization.core.planning import build_experiment_plan

    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)
    chain = {item["path"]: item for item in plan.parameter_chain()}
    algo_config = ExperimentRunner(device=torch.device("cpu"), verbose=False)._build_algorithm_config(config)
    algorithm = BiGAMPAlgorithm(algo_config, device=torch.device("cpu"))

    assert chain["algorithm_params.compile_fallback_policy"]["effective_value"] == "error"
    assert plan.resource_plan["config_effective"]["compile_fallback_policy"] == "error"
    assert algorithm.compile_fallback_policy == "error"
    assert algorithm._contract_execution_metadata["compile_status"] == "disabled_by_config"


def test_agd_partition_invariant_seed_is_alpha_batch_independent():
    config = ExperimentConfig(
        matrix=MatrixParams(N1=2, N2=2, M=1),
        training=TrainingParams(samples_per_alpha=2, max_steps=1, max_epochs=1),
        algorithm_key="agd",
        scan=ScanConfig(dimension="alpha", values=[0.5]),
        algorithm_params=AlgorithmParams(
            use_bf16=False,
            seed_partition_policy="partition_invariant",
        ),
        teacher_key="standard",
    )
    algo_config = ExperimentRunner(device=torch.device("cpu"), verbose=False)._build_algorithm_config(config)
    algorithm = AGDAlgorithm(algo_config, device=torch.device("cpu"))

    single = algorithm._randn_partitioned_matrix(
        alpha_values=[0.5],
        sample_count=2,
        shape=(2, 1),
        seed=17,
        device=torch.device("cpu"),
        scale=1.0,
        role="W_student",
    )
    combined = algorithm._randn_partitioned_matrix(
        alpha_values=[0.5, 0.8],
        sample_count=2,
        shape=(2, 1),
        seed=17,
        device=torch.device("cpu"),
        scale=1.0,
        role="W_student",
    )

    assert torch.equal(single[0], combined[0])
    assert algorithm._contract_execution_metadata["seed_partition_policy"] == "partition_invariant"


def test_bigamp_partition_invariant_seed_is_alpha_batch_independent():
    config = ExperimentConfig(
        matrix=MatrixParams(N1=2, N2=2, M=1),
        training=TrainingParams(samples_per_alpha=2, max_steps=1),
        algorithm_key="bigamp",
        scan=ScanConfig(dimension="alpha", values=[0.5]),
        algorithm_params=AlgorithmParams(
            use_compile=False,
            seed_partition_policy="partition_invariant",
        ),
        teacher_key="standard",
    )
    algo_config = ExperimentRunner(device=torch.device("cpu"), verbose=False)._build_algorithm_config(config)
    algorithm = BiGAMPAlgorithm(algo_config, device=torch.device("cpu"))

    single = algorithm._randn_partitioned_matrix(
        alpha_values=[0.5],
        sample_count=2,
        shape=(2, 1),
        seed=17,
        device=torch.device("cpu"),
        scale=1.0,
        role="W_student",
    )
    combined = algorithm._randn_partitioned_matrix(
        alpha_values=[0.5, 0.8],
        sample_count=2,
        shape=(2, 1),
        seed=17,
        device=torch.device("cpu"),
        scale=1.0,
        role="W_student",
    )

    assert torch.equal(single[0], combined[0])
    assert algorithm._contract_execution_metadata["seed_partition_policy"] == "partition_invariant"


def test_spreading_partition_invariant_seed_is_alpha_batch_independent():
    config = ExperimentConfig(
        matrix=MatrixParams(N1=2, N2=2, M=1),
        training=TrainingParams(samples_per_alpha=2, max_steps=1),
        algorithm_key="bigamp_spreading",
        scan=ScanConfig(dimension="alpha", values=[0.5]),
        algorithm_params=AlgorithmParams(
            use_compile=False,
            use_bf16=False,
            seed_partition_policy="partition_invariant",
        ),
        spreading=SpreadingConfig(tensor_order=2),
        teacher_key="standard",
    )
    algorithm = BiGAMPSpreading(config, device=torch.device("cpu"))

    single = algorithm._randn_partitioned_spreading_flat(
        alpha_values=[0.5],
        sample_count=2,
        node_count=2,
        latent_dim=1,
        seed=17,
        role="W_student",
        scale=0.1,
    )
    combined = algorithm._randn_partitioned_spreading_flat(
        alpha_values=[0.5, 0.8],
        sample_count=2,
        node_count=2,
        latent_dim=1,
        seed=17,
        role="W_student",
        scale=0.1,
    )

    assert torch.equal(single[0], combined[0])
    assert algorithm._spreading_batch_seed(17, 3) == 17
    assert algorithm._contract_execution_metadata["seed_partition_policy"] == "partition_invariant"


def test_spreading_partition_invariant_restart_noise_is_alpha_batch_independent():
    config = ExperimentConfig(
        matrix=MatrixParams(N1=2, N2=2, M=1),
        training=TrainingParams(samples_per_alpha=2, max_steps=1),
        algorithm_key="bigamp_spreading",
        scan=ScanConfig(dimension="alpha", values=[0.5]),
        algorithm_params=AlgorithmParams(
            use_compile=False,
            use_bf16=False,
            seed_partition_policy="partition_invariant",
            adaptive_restart=True,
        ),
        spreading=SpreadingConfig(tensor_order=2),
        teacher_key="standard",
    )
    algorithm = BiGAMPSpreading(config, device=torch.device("cpu"))

    single = algorithm._randn_partitioned_restart_noise(
        alpha_values=[0.5],
        sample_count=2,
        node_count=2,
        latent_dim=1,
        seed=17,
        role="W_student",
        step=3,
        scale=0.1,
    )
    combined = algorithm._randn_partitioned_restart_noise(
        alpha_values=[0.5, 0.8],
        sample_count=2,
        node_count=2,
        latent_dim=1,
        seed=17,
        role="W_student",
        step=3,
        scale=0.1,
    )
    different_step = algorithm._randn_partitioned_restart_noise(
        alpha_values=[0.5],
        sample_count=2,
        node_count=2,
        latent_dim=1,
        seed=17,
        role="W_student",
        step=4,
        scale=0.1,
    )
    different_role = algorithm._randn_partitioned_restart_noise(
        alpha_values=[0.5],
        sample_count=2,
        node_count=2,
        latent_dim=1,
        seed=17,
        role="X_student",
        step=3,
        scale=0.1,
    )

    assert torch.equal(single[0], combined[0])
    assert not torch.equal(single, different_step)
    assert not torch.equal(single, different_role)


def test_bigamp_honors_use_tf32_false(tmp_path):
    original_matmul = torch.backends.cuda.matmul.allow_tf32
    original_cudnn = torch.backends.cudnn.allow_tf32
    try:
        config_path = tmp_path / "matrix.yaml"
        _write_config(config_path, tensor_order=2, algorithm=1)
        text = config_path.read_text(encoding="utf-8").replace(
            "  use_bf16: false",
            "  use_bf16: false\n  use_tf32: false",
        )
        config_path.write_text(text, encoding="utf-8")

        config, output_options, raw_yaml = load_yaml_config(config_path)
        from matrix_factorization.core.planning import build_experiment_plan

        plan = build_experiment_plan(config, output_options, raw_yaml, config_path)
        chain = {item["path"]: item for item in plan.parameter_chain()}
        algo_config = ExperimentRunner(device=torch.device("cpu"), verbose=False)._build_algorithm_config(config)
        algorithm = BiGAMPAlgorithm(algo_config, device=torch.device("cpu"))

        assert chain["algorithm_params.use_tf32"]["effective_value"] is False
        assert chain["algorithm_params.use_tf32"]["consumption_status"] == "effective"
        assert plan.resource_plan["config_effective"]["use_tf32"] is False
        assert algorithm._contract_execution_metadata["requested_use_tf32"] is False
        assert algorithm._contract_execution_metadata["tf32_matmul_enabled"] is False
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul
        torch.backends.cudnn.allow_tf32 = original_cudnn


def test_tensor_parallel_records_dtype_fallback_policy(tmp_path):
    config_path = tmp_path / "tensor.yaml"
    _write_config(config_path, tensor_order=3, algorithm=4)
    text = config_path.read_text(encoding="utf-8").replace(
        "  use_bf16: false",
        "  use_bf16: false\n  dtype_fallback_policy: error",
    )
    config_path.write_text(text, encoding="utf-8")

    config, output_options, raw_yaml = load_yaml_config(config_path)
    from matrix_factorization.core.planning import build_experiment_plan

    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)
    chain = {item["path"]: item for item in plan.parameter_chain()}
    algorithm = BiGAMPTensorSpreadingParallel(config, device=None)

    assert chain["algorithm_params.dtype_fallback_policy"]["effective_value"] == "error"
    assert plan.resource_plan["config_effective"]["dtype_fallback_policy"] == "error"
    assert algorithm.dtype_fallback_policy == "error"


def test_tensor_parallel_honors_use_tf32_false(tmp_path):
    original_matmul = torch.backends.cuda.matmul.allow_tf32
    original_cudnn = torch.backends.cudnn.allow_tf32
    try:
        config_path = tmp_path / "tensor.yaml"
        _write_config(config_path, tensor_order=3, algorithm=4)
        text = config_path.read_text(encoding="utf-8").replace(
            "  use_bf16: false",
            "  use_bf16: false\n  use_tf32: false",
        )
        config_path.write_text(text, encoding="utf-8")

        config, output_options, raw_yaml = load_yaml_config(config_path)
        from matrix_factorization.core.planning import build_experiment_plan

        plan = build_experiment_plan(config, output_options, raw_yaml, config_path)
        algorithm = BiGAMPTensorSpreadingParallel(config, device=torch.device("cpu"))
        algorithm._batch_metrics = {0.0: {"Q_Y_mean": 0.1}}
        result = algorithm._metrics_only_algorithm_result("bigamp_tensor_parallel")
        execution = result.metadata["tensor_execution"]

        assert plan.resource_plan["config_effective"]["use_tf32"] is False
        assert execution["requested_use_tf32"] is False
        assert execution["tf32_matmul_enabled"] is False
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul
        torch.backends.cudnn.allow_tf32 = original_cudnn


def test_tensor_parallel_compile_fallback_policy_error_raises(monkeypatch):
    monkeypatch.setattr(BiGAMPTensorSpreadingParallel, "_compiled_step", None)
    monkeypatch.setattr(BiGAMPTensorSpreadingParallel, "_compiled_step_super", None)

    def fail_compile(*args, **kwargs):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(torch, "compile", fail_compile)
    config = ExperimentConfig(
        matrix=MatrixParams(N1=2, N2=2, M=1),
        training=TrainingParams(samples_per_alpha=1, max_steps=1),
        algorithm_key="bigamp_tensor_parallel",
        scan=ScanConfig(dimension="alpha", values=[0.0]),
        algorithm_params=AlgorithmParams(
            use_compile=True,
            use_bf16=False,
            compile_fallback_policy="error",
        ),
        spreading=SpreadingConfig(tensor_order=3),
        teacher_key="standard",
    )

    with pytest.raises(RuntimeError, match="compile_fallback_policy='error'"):
        BiGAMPTensorSpreadingParallel(config, device=torch.device("cpu"))


def test_tensor_parallel_compile_fallback_policy_allow_preserves_legacy_fallback(monkeypatch):
    monkeypatch.setattr(BiGAMPTensorSpreadingParallel, "_compiled_step", None)
    monkeypatch.setattr(BiGAMPTensorSpreadingParallel, "_compiled_step_super", None)

    def fail_compile(*args, **kwargs):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(torch, "compile", fail_compile)
    config = ExperimentConfig(
        matrix=MatrixParams(N1=2, N2=2, M=1),
        training=TrainingParams(samples_per_alpha=1, max_steps=1),
        algorithm_key="bigamp_tensor_parallel",
        scan=ScanConfig(dimension="alpha", values=[0.0]),
        algorithm_params=AlgorithmParams(
            use_compile=True,
            use_bf16=False,
            compile_fallback_policy="allow",
        ),
        spreading=SpreadingConfig(tensor_order=3),
        teacher_key="standard",
    )
    algorithm = BiGAMPTensorSpreadingParallel(config, device=torch.device("cpu"))

    assert algorithm.use_compile is False
    assert algorithm.compile_attempts[0]["target"] == "tensor_step_batch"
    assert algorithm.compile_attempts[0]["success"] is False


def test_spreading_compile_fallback_policy_error_raises(monkeypatch):
    monkeypatch.setattr(BiGAMPSpreading, "_compiled_step", None)
    monkeypatch.setattr(BiGAMPSpreading, "_compiled_step_adaptive", None)

    def fail_compile(*args, **kwargs):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(torch, "compile", fail_compile)
    config = ExperimentConfig(
        matrix=MatrixParams(N1=2, N2=2, M=1),
        training=TrainingParams(samples_per_alpha=1, max_steps=1),
        algorithm_key="bigamp_spreading",
        scan=ScanConfig(dimension="alpha", values=[0.0]),
        algorithm_params=AlgorithmParams(
            use_compile=True,
            use_bf16=False,
            compile_fallback_policy="error",
        ),
        spreading=SpreadingConfig(tensor_order=1),
        teacher_key="standard",
    )

    with pytest.raises(RuntimeError, match="compile_fallback_policy='error'"):
        BiGAMPSpreading(config, device=torch.device("cpu"))


def test_spreading_compile_fallback_policy_allow_preserves_legacy_fallback(monkeypatch):
    monkeypatch.setattr(BiGAMPSpreading, "_compiled_step", None)
    monkeypatch.setattr(BiGAMPSpreading, "_compiled_step_adaptive", None)

    def fail_compile(*args, **kwargs):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(torch, "compile", fail_compile)
    config = ExperimentConfig(
        matrix=MatrixParams(N1=2, N2=2, M=1),
        training=TrainingParams(samples_per_alpha=1, max_steps=1),
        algorithm_key="bigamp_spreading",
        scan=ScanConfig(dimension="alpha", values=[0.0]),
        algorithm_params=AlgorithmParams(
            use_compile=True,
            use_bf16=False,
            compile_fallback_policy="allow",
        ),
        spreading=SpreadingConfig(tensor_order=1),
        teacher_key="standard",
    )
    algorithm = BiGAMPSpreading(config, device=torch.device("cpu"))

    assert algorithm.use_compile is False
    assert algorithm.compile_attempts[0]["target"] == "bigamp_step_disjoint_union_flat"
    assert algorithm.compile_attempts[0]["success"] is False
    assert algorithm._contract_execution_metadata["compile_status"] == "fallback_to_eager_spreading_step"


def test_bigamp_compile_fallback_policy_error_raises(monkeypatch):
    monkeypatch.setattr(BiGAMPAlgorithm, "_compiled_step", None)

    def fail_compile(*args, **kwargs):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(torch, "compile", fail_compile)
    config = ExperimentConfig(
        matrix=MatrixParams(N1=2, N2=2, M=1),
        training=TrainingParams(samples_per_alpha=1, max_steps=1),
        algorithm_key="bigamp",
        scan=ScanConfig(dimension="alpha", values=[0.0]),
        algorithm_params=AlgorithmParams(
            use_compile=True,
            compile_fallback_policy="error",
        ),
        teacher_key="standard",
    )
    algo_config = ExperimentRunner(device=torch.device("cpu"), verbose=False)._build_algorithm_config(config)

    with pytest.raises(RuntimeError, match="compile_fallback_policy='error'"):
        BiGAMPAlgorithm(algo_config, device=torch.device("cuda"))


def test_bigamp_compile_fallback_policy_allow_preserves_legacy_fallback(monkeypatch):
    monkeypatch.setattr(BiGAMPAlgorithm, "_compiled_step", None)

    def fail_compile(*args, **kwargs):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(torch, "compile", fail_compile)
    config = ExperimentConfig(
        matrix=MatrixParams(N1=2, N2=2, M=1),
        training=TrainingParams(samples_per_alpha=1, max_steps=1),
        algorithm_key="bigamp",
        scan=ScanConfig(dimension="alpha", values=[0.0]),
        algorithm_params=AlgorithmParams(
            use_compile=True,
            compile_fallback_policy="allow",
        ),
        teacher_key="standard",
    )
    algo_config = ExperimentRunner(device=torch.device("cpu"), verbose=False)._build_algorithm_config(config)
    algorithm = BiGAMPAlgorithm(algo_config, device=torch.device("cuda"))

    assert algorithm.use_compile is False
    assert algorithm.compile_attempts[0]["target"] == "bigamp_step"
    assert algorithm.compile_attempts[0]["success"] is False
    assert algorithm._contract_execution_metadata["compile_status"] == "fallback_to_eager_bigamp_step"


def test_tensor_parallel_dtype_fallback_policy_error_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = ExperimentConfig(
        matrix=MatrixParams(N1=2, N2=2, M=1),
        training=TrainingParams(samples_per_alpha=1, max_steps=1),
        algorithm_key="bigamp_tensor_parallel",
        scan=ScanConfig(dimension="alpha", values=[0.0]),
        algorithm_params=AlgorithmParams(
            use_compile=False,
            use_bf16=True,
            dtype_fallback_policy="error",
        ),
        spreading=SpreadingConfig(tensor_order=3),
        teacher_key="standard",
    )

    with pytest.raises(RuntimeError, match="dtype_fallback_policy='error'"):
        BiGAMPTensorSpreadingParallel(config, device=torch.device("cpu"))


def test_tensor_parallel_dtype_fallback_policy_allow_preserves_float32_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = ExperimentConfig(
        matrix=MatrixParams(N1=2, N2=2, M=1),
        training=TrainingParams(samples_per_alpha=1, max_steps=1),
        algorithm_key="bigamp_tensor_parallel",
        scan=ScanConfig(dimension="alpha", values=[0.0]),
        algorithm_params=AlgorithmParams(
            use_compile=False,
            use_bf16=True,
            dtype_fallback_policy="allow",
        ),
        spreading=SpreadingConfig(tensor_order=3),
        teacher_key="standard",
    )
    algorithm = BiGAMPTensorSpreadingParallel(config, device=torch.device("cpu"))

    assert algorithm.use_bf16 is False
    assert algorithm.storage_dtype == torch.float32
    assert algorithm.dtype_status == "fallback_to_float32_bf16_unavailable"


def test_spreading_dtype_fallback_policy_error_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = ExperimentConfig(
        matrix=MatrixParams(N1=2, N2=2, M=1),
        training=TrainingParams(samples_per_alpha=1, max_steps=1),
        algorithm_key="bigamp_spreading",
        scan=ScanConfig(dimension="alpha", values=[0.0]),
        algorithm_params=AlgorithmParams(
            use_compile=False,
            use_bf16=True,
            dtype_fallback_policy="error",
        ),
        spreading=SpreadingConfig(tensor_order=1),
        teacher_key="standard",
    )

    with pytest.raises(RuntimeError, match="dtype_fallback_policy='error'"):
        BiGAMPSpreading(config, device=torch.device("cpu"))


def test_agd_dtype_fallback_policy_error_raises_on_cpu():
    config = ExperimentConfig(
        matrix=MatrixParams(N1=2, N2=2, M=1),
        training=TrainingParams(samples_per_alpha=1, max_steps=1, max_epochs=1),
        algorithm_key="agd",
        scan=ScanConfig(dimension="alpha", values=[0.0]),
        algorithm_params=AlgorithmParams(
            use_bf16=True,
            dtype_fallback_policy="error",
        ),
        teacher_key="standard",
    )
    algo_config = ExperimentRunner(device=torch.device("cpu"), verbose=False)._build_algorithm_config(config)

    with pytest.raises(RuntimeError, match="dtype_fallback_policy='error'"):
        AGDAlgorithm(algo_config, device=torch.device("cpu"))


def test_scaling_sweep_propagates_teacher_and_contract_metadata(monkeypatch):
    base_config = ExperimentConfig(
        matrix=MatrixParams(N1=2, N2=2, M=1),
        training=TrainingParams(samples_per_alpha=1, max_steps=2),
        algorithm_key="bigamp",
        scan=ScanConfig(dimension="alpha", values=[0.0]),
        algorithm_params=AlgorithmParams(use_compile=False),
        teacher_key="standard",
        teacher=TeacherConfig(init_distribution="rademacher"),
        experiment_name="nested_contract",
    )
    output_options = {"experiment_plan": {"algorithm": "bigamp", "is_valid": True}}
    captured = {}
    runner = ExperimentRunner(device=None, verbose=False)

    def fake_run(config, observer=None, output_options=None):
        captured["config"] = config
        captured["output_options"] = output_options
        return ExperimentResult(
            experiment_id=config.experiment_name,
            config=config,
            scan_dimension=config.scan.dimension,
            scan_values=config.scan.values,
        )

    monkeypatch.setattr(runner, "run", fake_run)
    runner.run_scaling_sweep(
        base_config=base_config,
        matrix_sizes=[(4, 4, 2)],
        output_options=output_options,
    )

    assert captured["config"].teacher.init_distribution == "rademacher"
    assert captured["config"].matrix.N1 == 4
    assert captured["output_options"]["experiment_plan"]["algorithm"] == "bigamp"


def test_algorithm_cache_is_keyed_by_effective_config():
    runner = ExperimentRunner(device=None, verbose=False)
    config = ExperimentConfig(
        matrix=MatrixParams(N1=2, N2=2, M=1),
        training=TrainingParams(samples_per_alpha=1, max_steps=2),
        algorithm_key="bigamp",
        scan=ScanConfig(dimension="alpha", values=[0.0]),
        algorithm_params=AlgorithmParams(damping=0.5, use_compile=False),
        teacher_key="standard",
    )

    first = runner._get_algorithm(config)
    second = runner._get_algorithm(config)

    config_changed = ExperimentConfig(
        matrix=MatrixParams(N1=2, N2=2, M=1),
        training=TrainingParams(samples_per_alpha=1, max_steps=2),
        algorithm_key="bigamp",
        scan=ScanConfig(dimension="alpha", values=[0.0]),
        algorithm_params=AlgorithmParams(damping=0.8, use_compile=False),
        teacher_key="standard",
    )
    third = runner._get_algorithm(config_changed)

    assert first is second
    assert third is not first
    assert first._contract_config_trace["algorithm_params"]["damping"] == 0.5
    assert third._contract_config_trace["algorithm_params"]["damping"] == 0.8
    assert first._contract_config_trace["cache_signature"] != third._contract_config_trace["cache_signature"]


def test_runner_writes_algorithm_config_trace_to_metadata(monkeypatch):
    config = ExperimentConfig(
        matrix=MatrixParams(N1=2, N2=2, M=1),
        training=TrainingParams(samples_per_alpha=1, max_steps=2),
        algorithm_key="bigamp",
        scan=ScanConfig(dimension="alpha", values=[0.0]),
        algorithm_params=AlgorithmParams(damping=0.7, use_compile=False),
        teacher_key="standard",
    )
    runner = ExperimentRunner(device=None, verbose=False)

    def fake_standard_scan(config, algorithm, result, observer, resume_results=None, output_options=None, raw_yaml=""):
        result.add_result(
            0.0,
            SingleRunResult(scan_value=0.0, metrics={"Q_Y_mean": 0.1}),
        )

    monkeypatch.setattr(runner, "_run_standard_scan", fake_standard_scan)
    result = runner.run(config, output_options={"enable_heatmap": False})
    trace = result.metadata.contract["algorithm_config_trace"]

    assert trace["algorithm_key"] == "bigamp"
    assert trace["algorithm_params"]["damping"] == 0.7
    assert trace["mock_config_shape"]["has_algorithm_params_alias"] is True
    assert trace["metadata_only"] is True
