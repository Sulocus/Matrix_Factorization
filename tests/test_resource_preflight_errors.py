from pathlib import Path

from matrix_factorization.cli import load_yaml_config
from matrix_factorization.core.planning import build_experiment_plan


def _write(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def test_sample_folding_request_is_allowed_after_sample_sharding_contract(tmp_path):
    config_path = tmp_path / "sample_fold.yaml"
    _write(
        config_path,
        """
tensor_order: 2
algorithm: 1
teacher: 2
matrix: {N1: 8, N2: 8, M: 2}
scan:
  axes:
    alpha: {path: alpha, values: [0.0, 0.1]}
  execution:
    allowed_fold_axes: [alpha, sample]
training: {samples_per_alpha: 2, max_steps: 1, max_epochs: 1}
algorithm_params: {use_compile: false, use_bf16: false}
output: {save_tensors: false, enable_heatmap: false}
""",
    )
    config, _, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, raw_yaml=raw_yaml, config_path=config_path)

    assert not any("sample folding requested" in error for error in plan.errors)
    assert not any("unsupported axis 'sample'" in error for error in plan.errors)


def test_single_alpha_full_sample_oversize_is_preflight_error(tmp_path):
    config_path = tmp_path / "oversize.yaml"
    _write(
        config_path,
        """
tensor_order: 2
algorithm: 1
teacher: 2
matrix: {N1: 512, N2: 512, M: 8}
scan:
  axes:
    alpha: {path: alpha, values: [0.2]}
  execution:
    max_allocated_gb: 0.001
    target_utilization: 0.75
training: {samples_per_alpha: 4, max_steps: 1, max_epochs: 1}
algorithm_params: {use_compile: false, use_bf16: false}
output: {save_tensors: false, enable_heatmap: false}
""",
    )
    config, _, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, raw_yaml=raw_yaml, config_path=config_path)

    assert any("ResourceExecutionPlan preflight" in error for error in plan.errors)


def test_unknown_scan_execution_field_is_config_error(tmp_path):
    config_path = tmp_path / "unknown_execution.yaml"
    _write(
        config_path,
        """
tensor_order: 2
algorithm: 1
teacher: 2
matrix: {N1: 8, N2: 8, M: 2}
scan:
  axes:
    alpha: {path: alpha, values: [0.0]}
  execution:
    made_up_budget: 1
training: {samples_per_alpha: 1, max_steps: 1, max_epochs: 1}
algorithm_params: {use_compile: false, use_bf16: false}
output: {save_tensors: false, enable_heatmap: false}
""",
    )
    config, _, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, raw_yaml=raw_yaml, config_path=config_path)

    assert any("未注册 scan.execution 字段" in error for error in plan.errors)
