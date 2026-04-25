import subprocess
import sys
import json


def _write_min_config(path):
    path.write_text(
        """
tensor_order: 3
algorithm: 4
teacher: 2
teacher_config:
  init_distribution: 1
matrix:
  N1: 4
  N2: 4
  M: 2
scan_mode: 1
alpha_scan:
  start: 0.0
  stop: 0.0
  step: 1.0
training:
  samples_per_alpha: 1
  max_steps: 2
algorithm_params:
  damping: 0.5
  noise_var: 1.0e-5
  use_compile: false
  use_bf16: false
spreading:
  f_distribution: 1
  seed: 321
output:
  enable_heatmap: true
  save_tensors: false
""",
        encoding="utf-8",
    )


def test_validate_command_reports_contract_status(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_min_config(config_path)

    result = subprocess.run(
        [sys.executable, "-m", "matrix_factorization.cli", "validate", str(config_path)],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "配置校验结果" in result.stdout
    assert "algorithm: bigamp_tensor_parallel" in result.stdout
    assert "errors: none" in result.stdout


def test_explain_config_command_reports_effective_route(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_min_config(config_path)

    result = subprocess.run(
        [sys.executable, "-m", "matrix_factorization.cli", "explain-config", str(config_path)],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "配置解释" in result.stdout
    assert "实际 algorithm: bigamp_tensor_parallel" in result.stdout
    assert "YAML algorithm: 4" in result.stdout
    assert "spreading.seed: 321" in result.stdout
    assert "参数链路:" in result.stdout


def test_validate_command_supports_json_output(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_min_config(config_path)

    result = subprocess.run(
        [sys.executable, "-m", "matrix_factorization.cli", "validate", "--json", str(config_path)],
        check=False,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert payload["algorithm"] == "bigamp_tensor_parallel"
    assert payload["algorithm_spec"]["result_contract"] == "legacy_tensor_metrics_only"
    assert "tensor_heatmap" in payload["outputs"]
    assert "overlap_matrix" in payload["output_plan"]["required_artifacts"]
    assert "plots/animation_Y.gif" in payload["output_plan"]["output_files"]
    assert "ALGORITHM_ROUTE_OVERRIDE" in payload["warning_codes"]
    assert all("code" in issue and "message" in issue for issue in payload["issues"])
    parameter_chain = {item["path"]: item for item in payload["parameter_chain"]}
    assert parameter_chain["spreading.seed"]["consumers"] == ["SpreadingConfig"]
    assert parameter_chain["spreading.seed"]["effective_value"] == 321
    assert parameter_chain["spreading.seed"]["consumption_status"] == "effective"
    assert payload["parameter_consumption"] == payload["parameter_chain"]
    assert "algorithm_params.damping" in payload["physical_sensitive_parameters"]


def test_validate_command_strict_mode_fails_on_parsed_only_field(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_min_config(config_path)
    text = config_path.read_text(encoding="utf-8")
    text = text.replace(
        "output:\n  enable_heatmap: true",
        "output:\n  storage_mode: lightweight\n  enable_heatmap: true",
    )
    config_path.write_text(text, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "matrix_factorization.cli", "validate", "--strict", str(config_path)],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "strict mode: output.storage_mode 当前是 parsed_only" in result.stdout


def test_validate_json_reports_machine_readable_error_codes(tmp_path):
    config_path = tmp_path / "bad_intervention.yaml"
    _write_min_config(config_path)
    text = config_path.read_text(encoding="utf-8")
    text = text.replace(
        "algorithm_params:\n  damping: 0.5",
        "algorithm_params:\n  adaptive_restart: true\n  damping: 0.5",
    )
    config_path.write_text(text, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "matrix_factorization.cli", "validate", "--json", str(config_path)],
        check=False,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 1
    assert "INTERVENTION_UNSUPPORTED" in payload["error_codes"]
    assert any(issue["severity"] == "error" for issue in payload["issues"])
