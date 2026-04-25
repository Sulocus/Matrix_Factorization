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
scan_mode: 1
alpha_scan:
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
