import pytest

from matrix_factorization.cli import load_yaml_config
from matrix_factorization.core.planning import build_experiment_plan


def _write_config(path, scan_block: str, extra: str = ""):
    path.write_text(
        f"""
tensor_order: 2
algorithm: 1
teacher: 2
matrix:
  N1: 4
  N2: 4
  M: 2
training:
  samples_per_alpha: 1
  max_steps: 2
  max_epochs: 2
{scan_block}
algorithm_params:
  use_compile: false
  use_bf16: false
output:
  save_tensors: false
  enable_heatmap: false
{extra}
""",
        encoding="utf-8",
    )


def test_canonical_scan_schema_loads(tmp_path):
    config_path = tmp_path / "canonical.yaml"
    _write_config(
        config_path,
        """
scan:
  axes:
    alpha:
      path: alpha
      values: {start: 0.0, stop: 0.2, step: 0.1}
""",
    )

    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)

    assert plan.errors == []
    assert config.scan_spec["axes"]["alpha"]["path"] == "alpha"
    assert plan.scan_plan.num_points == 3


def test_legacy_scan_fields_are_not_accepted_by_loader(tmp_path):
    config_path = tmp_path / "legacy.yaml"
    _write_config(
        config_path,
        """
scan_mode: 1
alpha_scan:
  start: 0.0
  stop: 0.2
  step: 0.1
""",
    )

    with pytest.raises(ValueError, match="旧 scan 配置已从主链路移除"):
        load_yaml_config(config_path)


def test_scan_axis_unknown_parameter_path_is_preflight_error(tmp_path):
    config_path = tmp_path / "bad_axis.yaml"
    _write_config(
        config_path,
        """
scan:
  axes:
    mystery:
      path: algorithm_params.not_real
      values: [1, 2]
""",
    )

    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)

    assert any("scan axis 'mystery' 引用了未注册参数路径" in error for error in plan.errors)


def test_composite_axis_override_paths_are_validated(tmp_path):
    config_path = tmp_path / "bad_composite.yaml"
    _write_config(
        config_path,
        """
scan:
  axes:
    init:
      kind: composite
      values:
        bad:
          algorithm_params.not_real: 1
    alpha:
      path: alpha
      values: [0.0]
""",
    )

    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)

    assert any("scan axis 'init' 引用了未注册参数路径" in error for error in plan.errors)
