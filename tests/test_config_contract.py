from pathlib import Path

from matrix_factorization.cli import load_yaml_config
from matrix_factorization.core.experiment.config import (
    AlgorithmParams,
    ExperimentConfig,
    MatrixParams,
    ScanConfig,
    TeacherConfig,
    TrainingParams,
)
from matrix_factorization.core.experiment.result import ExperimentResult
from matrix_factorization.core.experiment.runner import ExperimentRunner
from matrix_factorization.modules.algorithms.bigamp.tensor_spreading_parallel import (
    BiGAMPTensorSpreadingParallel,
)


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


def test_tensor_parallel_respects_use_compile_config(tmp_path):
    config_path = tmp_path / "tensor.yaml"
    _write_config(config_path, tensor_order=3, algorithm=4)

    config, _, _ = load_yaml_config(config_path)
    algorithm = BiGAMPTensorSpreadingParallel(config, device=None)

    assert algorithm.use_compile is False


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
