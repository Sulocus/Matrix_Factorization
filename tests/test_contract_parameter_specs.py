from dataclasses import fields

from matrix_factorization.cli import load_yaml_config
from matrix_factorization.core.experiment.config import (
    AlgorithmParams,
    MatrixParams,
    SeedConfig,
    SpreadingConfig,
    TeacherConfig,
    TrainingParams,
)
from matrix_factorization.core.contracts import get_parameter_specs
from matrix_factorization.core.planning import build_experiment_plan


def test_parameter_specs_cover_core_yaml_fields():
    specs = get_parameter_specs()

    for path in [
        "tensor_order",
        "algorithm",
        "teacher_config.init_distribution",
        "scan.axes",
        "training.num_workers",
        "spreading.seed",
        "algorithm_params.use_compile",
        "algorithm_params.use_metric_plateau_stop",
        "algorithm_params.plateau_signal",
        "output.enable_heatmap",
    ]:
        assert path in specs


def test_config_dataclass_fields_have_parameter_specs():
    specs = set(get_parameter_specs())
    dataclass_prefixes = [
        (MatrixParams, "matrix"),
        (TrainingParams, "training"),
        (SeedConfig, "seeds"),
        (SpreadingConfig, "spreading"),
        (TeacherConfig, "teacher_config"),
        (AlgorithmParams, "algorithm_params"),
    ]

    missing = []
    for cls, prefix in dataclass_prefixes:
        missing.extend(
            f"{prefix}.{field.name}"
            for field in fields(cls)
            if f"{prefix}.{field.name}" not in specs
        )

    assert missing == []


def test_unknown_yaml_field_is_validation_error(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """
tensor_order: 2
algorithm: 1
matrix:
  N1: 4
  N2: 4
  M: 2
training:
  samples_per_alpha: 1
  max_steps: 2
scan:
  axes:
    alpha:
      path: alpha
      values:
        start: 0.0
        stop: 0.0
        step: 1.0
unknown_block:
  value: 1
""",
        encoding="utf-8",
    )
    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)

    assert any("未知 YAML 字段: unknown_block.value" in error for error in plan.errors)


def test_unknown_algorithm_param_is_validation_error(tmp_path):
    config_path = tmp_path / "bad_algorithm_param.yaml"
    config_path.write_text(
        """
tensor_order: 2
algorithm: 1
matrix:
  N1: 4
  N2: 4
  M: 2
training:
  samples_per_alpha: 1
  max_steps: 2
scan:
  axes:
    alpha:
      path: alpha
      values:
        start: 0.0
        stop: 0.0
        step: 1.0
algorithm_params:
  damping: 0.5
  made_up_parameter: 10
""",
        encoding="utf-8",
    )
    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)

    assert any("未注册参数字段: algorithm_params.made_up_parameter" in error for error in plan.errors)


def test_invalid_enum_parameter_is_validation_error(tmp_path):
    config_path = tmp_path / "bad_enum_param.yaml"
    config_path.write_text(
        """
tensor_order: 3
algorithm: 4
matrix:
  N1: 4
  N2: 4
  M: 2
training:
  samples_per_alpha: 1
  max_steps: 2
scan:
  axes:
    alpha:
      path: alpha
      values:
        start: 0.0
        stop: 0.0
        step: 1.0
algorithm_params:
  use_compile: false
  seed_partition_policy: maybe
output:
  enable_heatmap: false
""",
        encoding="utf-8",
    )
    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)

    assert any("algorithm_params.seed_partition_policy" in error and "允许值" in error for error in plan.errors)
    assert "INVALID_PARAMETER_VALUE" in plan.to_dict()["error_codes"]


def test_invalid_bool_parameter_is_validation_error(tmp_path):
    config_path = tmp_path / "bad_bool_param.yaml"
    config_path.write_text(
        """
tensor_order: 2
algorithm: 1
matrix:
  N1: 4
  N2: 4
  M: 2
training:
  samples_per_alpha: 1
  max_steps: 2
scan:
  axes:
    alpha:
      path: alpha
      values:
        start: 0.0
        stop: 0.0
        step: 1.0
algorithm_params:
  use_compile: "no"
output:
  enable_heatmap: false
""",
        encoding="utf-8",
    )
    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)

    assert any("algorithm_params.use_compile 类型无效" in error for error in plan.errors)


def test_parsed_only_fields_are_warnings(tmp_path):
    config_path = tmp_path / "parsed_only.yaml"
    config_path.write_text(
        """
tensor_order: 2
algorithm: 1
matrix:
  N1: 4
  N2: 4
  M: 2
training:
  samples_per_alpha: 1
  max_steps: 2
scan:
  axes:
    alpha:
      path: alpha
      values:
        start: 0.0
        stop: 0.0
        step: 1.0
output:
  storage_mode: lightweight
""",
        encoding="utf-8",
    )
    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)

    assert not plan.errors
    assert any("output.storage_mode 当前是 parsed_only" in warning for warning in plan.warnings)


def test_strict_mode_turns_parsed_only_fields_into_errors(tmp_path):
    config_path = tmp_path / "strict_parsed_only.yaml"
    config_path.write_text(
        """
tensor_order: 2
algorithm: 1
matrix:
  N1: 4
  N2: 4
  M: 2
training:
  samples_per_alpha: 1
  max_steps: 2
scan:
  axes:
    alpha:
      path: alpha
      values:
        start: 0.0
        stop: 0.0
        step: 1.0
output:
  storage_mode: lightweight
""",
        encoding="utf-8",
    )
    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path, strict=True)

    assert any("strict mode: output.storage_mode 当前是 parsed_only" in error for error in plan.errors)


def test_parameter_chain_reports_effective_values_and_scan_effects(tmp_path):
    config_path = tmp_path / "trace.yaml"
    config_path.write_text(
        """
tensor_order: 2
algorithm: 1
teacher_config:
  init_distribution: 2
matrix:
  N1: 4
  N2: 5
  M: 2
training:
  samples_per_alpha: 1
  max_steps: 2
  num_workers: 3
scan:
  axes:
    alpha:
      path: alpha
      values:
        start: 0.0
        stop: 0.2
        step: 0.1
algorithm_params:
  use_compile: false
output:
  enable_heatmap: false
""",
        encoding="utf-8",
    )
    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)
    chain = {item["path"]: item for item in plan.parameter_chain()}

    assert chain["training.num_workers"]["effective_value"] == 3
    assert chain["training.num_workers"]["effective_source"] == "effective_parameters"
    assert chain["training.num_workers"]["consumption_status"] == "effective"
    assert chain["teacher_config.init_distribution"]["effective_value"] == "rademacher"
    assert chain["scan.axes"]["derived_effect"]["scan_num_points"] == 3
    assert chain["scan.axes"]["consumption_status"] == "effective"


def test_legacy_scan_fields_are_rejected_before_planning(tmp_path):
    config_path = tmp_path / "legacy_scan.yaml"
    config_path.write_text(
        """
tensor_order: 2
algorithm: 1
matrix:
  N1: 4
  N2: 4
  M: 2
training:
  samples_per_alpha: 1
  max_steps: 2
scan_mode: 1
alpha_scan:
  start: 0.0
  stop: 0.2
  step: 0.1
output:
  enable_heatmap: false
""",
        encoding="utf-8",
    )

    try:
        load_yaml_config(config_path)
    except ValueError as exc:
        assert "旧 scan 配置已从主链路移除" in str(exc)
    else:
        raise AssertionError("legacy scan fields should be rejected")


def test_parameter_chain_reports_seed_aliases_and_specific_seed_override(tmp_path):
    config_path = tmp_path / "seed_trace.yaml"
    config_path.write_text(
        """
tensor_order: 2
algorithm: 2
matrix:
  N1: 4
  N2: 4
  M: 2
training:
  samples_per_alpha: 1
  max_steps: 2
scan:
  axes:
    alpha:
      path: alpha
      values:
        start: 0.0
        stop: 0.0
        step: 1.0
seeds:
  model: 11
  data: 22
  spreading_seed: 33
spreading:
  seed: 44
output:
  enable_heatmap: false
""",
        encoding="utf-8",
    )
    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)
    chain = {item["path"]: item for item in plan.parameter_chain()}

    assert chain["seeds.model"]["effective_source"] == "seeds.base_seed"
    assert chain["seeds.model"]["effective_value"] == 11
    assert chain["seeds.data"]["effective_source"] == "seeds.teacher_seed"
    assert chain["seeds.data"]["effective_value"] == 22
    assert chain["seeds.spreading_seed"]["effective_value"] == 33
    assert chain["seeds.spreading_seed"]["derived_effect"]["actual_spreading_seed"] == 44
    assert chain["seeds.spreading_seed"]["derived_effect"]["overridden_by"] == "spreading.seed"
    assert chain["seeds.spreading_seed"]["consumption_status"] == "overridden_current_route"
    assert chain["spreading.seed"]["effective_value"] == 44
    assert chain["spreading.seed"]["consumption_status"] == "effective"


def test_parameter_chain_marks_algorithm_specific_params_inactive_for_current_route(tmp_path):
    config_path = tmp_path / "inactive_algorithm_params.yaml"
    config_path.write_text(
        """
tensor_order: 2
algorithm: 3
matrix:
  N1: 4
  N2: 4
  M: 2
training:
  samples_per_alpha: 1
  max_steps: 2
  max_epochs: 3
scan:
  axes:
    alpha:
      path: alpha
      values:
        start: 0.0
        stop: 0.0
        step: 1.0
algorithm_params:
  damping: 0.2
  learning_rate: 0.01
spreading:
  seed: 77
output:
  enable_heatmap: false
""",
        encoding="utf-8",
    )
    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)
    chain = {item["path"]: item for item in plan.parameter_chain()}

    assert config.algorithm_key == "agd"
    assert chain["algorithm_params.learning_rate"]["consumption_status"] == "effective"
    assert chain["algorithm_params.damping"]["active_in_current_plan"] is False
    assert chain["algorithm_params.damping"]["consumption_status"] == "inactive_current_route"
    assert chain["spreading.seed"]["active_in_current_plan"] is False
    assert chain["spreading.seed"]["consumption_status"] == "inactive_current_route"
    assert any("algorithm_params.damping 在当前 algorithm/scan 路由下不会生效" in warning for warning in plan.warnings)


def test_parameter_chain_marks_resource_params_inactive_when_algorithm_does_not_consume_them(tmp_path):
    config_path = tmp_path / "inactive_resource_params.yaml"
    config_path.write_text(
        """
tensor_order: 2
algorithm: 1
matrix:
  N1: 4
  N2: 4
  M: 2
training:
  samples_per_alpha: 1
  max_steps: 2
scan:
  axes:
    alpha:
      path: alpha
      values:
        start: 0.0
        stop: 0.0
        step: 1.0
algorithm_params:
  damping: 0.5
  use_compile: false
  use_bf16: false
output:
  enable_heatmap: false
""",
        encoding="utf-8",
    )
    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)
    chain = {item["path"]: item for item in plan.parameter_chain()}

    assert config.algorithm_key == "bigamp"
    assert chain["algorithm_params.use_compile"]["consumption_status"] == "effective"
    assert chain["algorithm_params.use_bf16"]["consumption_status"] == "legacy"
    assert chain["algorithm_params.precision_profile"]["consumption_status"] == "effective"
    assert any("algorithm_params.use_bf16 是 legacy 字段" in warning for warning in plan.warnings)


def test_parameter_chain_marks_compile_inactive_for_agd(tmp_path):
    config_path = tmp_path / "inactive_agd_compile.yaml"
    config_path.write_text(
        """
tensor_order: 2
algorithm: 3
matrix:
  N1: 4
  N2: 4
  M: 2
training:
  samples_per_alpha: 1
  max_steps: 2
  max_epochs: 3
scan:
  axes:
    alpha:
      path: alpha
      values:
        start: 0.0
        stop: 0.0
        step: 1.0
algorithm_params:
  learning_rate: 0.01
  use_compile: false
  use_bf16: false
output:
  enable_heatmap: false
""",
        encoding="utf-8",
    )
    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)
    chain = {item["path"]: item for item in plan.parameter_chain()}

    assert config.algorithm_key == "agd"
    assert chain["algorithm_params.use_bf16"]["consumption_status"] == "legacy"
    assert chain["algorithm_params.precision_profile"]["consumption_status"] == "effective"
    assert chain["algorithm_params.use_compile"]["active_in_current_plan"] is False
    assert chain["algorithm_params.use_compile"]["consumption_status"] == "inactive_current_route"


def test_metric_plateau_params_are_effective_only_for_flat_spreading(tmp_path):
    config_path = tmp_path / "plateau_flat_spreading.yaml"
    config_path.write_text(
        """
tensor_order: 2
algorithm: 2
matrix:
  N1: 4
  N2: 4
  M: 2
training:
  samples_per_alpha: 1
  max_steps: 2
scan:
  axes:
    alpha:
      path: alpha
      values:
        start: 0.0
        stop: 0.0
        step: 1.0
algorithm_params:
  use_metric_plateau_stop: true
  plateau_check_interval: 1
  plateau_window_steps: 1
  plateau_patience: 1
  plateau_abs_tol: 0.003
  plateau_rel_tol: 0.01
  plateau_min_steps: 0
  plateau_signal: teacher_latent_overlap
  plateau_monitor: teacher_latent_overlap_qw_qx
spreading:
  f_distribution: 1
output:
  enable_heatmap: false
""",
        encoding="utf-8",
    )
    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)
    chain = {item["path"]: item for item in plan.parameter_chain()}

    assert config.algorithm_key == "bigamp_spreading"
    assert config.spreading.allow_intra_connection is False
    assert chain["algorithm_params.use_metric_plateau_stop"]["consumption_status"] == "effective"
    assert chain["algorithm_params.plateau_rel_tol"]["consumption_status"] == "effective"
    assert chain["algorithm_params.plateau_monitor"]["consumption_status"] == "effective"
    assert chain["algorithm_params.plateau_window_steps"]["consumption_status"] == "effective"
    assert plan.resource_plan["config_effective"]["metric_plateau_stop"]["enabled"] is True
    assert plan.resource_plan["config_effective"]["metric_plateau_stop"]["teacher_assisted"] is True
    assert plan.resource_plan["config_effective"]["metric_plateau_stop"]["strategy"] == "self_convergence_window_trend_decay"
    assert plan.resource_plan["config_effective"]["metric_plateau_stop"]["effective_window_steps"] == 1


def test_metric_plateau_params_are_inactive_for_general_spreading(tmp_path):
    config_path = tmp_path / "plateau_general_spreading.yaml"
    config_path.write_text(
        """
tensor_order: 1
algorithm: 2
matrix:
  N1: 4
  N2: 4
  M: 2
training:
  samples_per_alpha: 1
  max_steps: 2
scan:
  axes:
    alpha:
      path: alpha
      values:
        start: 0.0
        stop: 0.0
        step: 1.0
algorithm_params:
  use_metric_plateau_stop: true
  plateau_check_interval: 1
  plateau_window_steps: 1
  plateau_patience: 1
  plateau_abs_tol: 0.003
  plateau_rel_tol: 0.01
  plateau_min_steps: 0
  plateau_signal: teacher_latent_overlap
  plateau_monitor: teacher_latent_overlap_qw_qx
spreading:
  f_distribution: 1
output:
  enable_heatmap: false
""",
        encoding="utf-8",
    )
    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)
    chain = {item["path"]: item for item in plan.parameter_chain()}

    assert config.algorithm_key == "bigamp_spreading"
    assert config.spreading.allow_intra_connection is True
    assert chain["algorithm_params.use_metric_plateau_stop"]["active_in_current_plan"] is False
    assert chain["algorithm_params.use_metric_plateau_stop"]["consumption_status"] == "inactive_current_route"
    assert any(
        "algorithm_params.use_metric_plateau_stop 在当前 algorithm/scan 路由下不会生效" in warning
        for warning in plan.warnings
    )


def test_strict_mode_rejects_metric_plateau_on_general_spreading(tmp_path):
    config_path = tmp_path / "strict_plateau_general_spreading.yaml"
    config_path.write_text(
        """
tensor_order: 1
algorithm: 2
matrix:
  N1: 4
  N2: 4
  M: 2
training:
  samples_per_alpha: 1
  max_steps: 2
scan:
  axes:
    alpha:
      path: alpha
      values:
        start: 0.0
        stop: 0.0
        step: 1.0
algorithm_params:
  use_metric_plateau_stop: true
  plateau_check_interval: 1
  plateau_window_steps: 1
  plateau_patience: 1
  plateau_abs_tol: 0.003
  plateau_rel_tol: 0.01
  plateau_min_steps: 0
  plateau_signal: teacher_latent_overlap
  plateau_monitor: teacher_latent_overlap_qw_qx
spreading:
  f_distribution: 1
output:
  enable_heatmap: false
""",
        encoding="utf-8",
    )
    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path, strict=True)

    assert any(
        "strict mode: algorithm_params.use_metric_plateau_stop 在当前 algorithm/scan 路由下不会生效" in error
        for error in plan.errors
    )


def test_strict_mode_rejects_inactive_current_route_parameters(tmp_path):
    config_path = tmp_path / "strict_inactive_algorithm_param.yaml"
    config_path.write_text(
        """
tensor_order: 2
algorithm: 3
matrix:
  N1: 4
  N2: 4
  M: 2
training:
  samples_per_alpha: 1
  max_steps: 2
  max_epochs: 3
scan:
  axes:
    alpha:
      path: alpha
      values:
        start: 0.0
        stop: 0.0
        step: 1.0
algorithm_params:
  damping: 0.2
  learning_rate: 0.01
output:
  enable_heatmap: false
""",
        encoding="utf-8",
    )
    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path, strict=True)

    assert any("strict mode: algorithm_params.damping 在当前 algorithm/scan 路由下不会生效" in error for error in plan.errors)
    assert "STRICT_INACTIVE_PARAMETER_CURRENT_ROUTE" in plan.to_dict()["error_codes"]
