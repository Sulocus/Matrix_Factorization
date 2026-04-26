import subprocess
import sys

from matrix_factorization.core.memory_calibration import (
    build_calibration_config,
    get_memory_calibration_profiles,
)
from matrix_factorization.core.parallel.execution_modes import EstimationParams
from matrix_factorization.core.parallel.memory_estimator import MemoryEstimator


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "matrix_factorization.cli", *args],
        check=False,
        text=True,
        capture_output=True,
    )


def test_memory_calibration_profiles_build_small_configs():
    profiles = get_memory_calibration_profiles()

    assert {"matrix_bigamp_small", "matrix_agd_small", "spreading_bigamp_small", "tensor_parallel_small"} <= set(profiles)
    assert {"matrix_bigamp_target_10gb", "matrix_bigamp_target_16gb"} <= set(profiles)
    tensor_config = build_calibration_config(profiles["tensor_parallel_small"])
    assert tensor_config.algorithm_key == "bigamp_tensor_parallel"
    assert tensor_config.spreading.tensor_order == 3
    assert tensor_config.training.max_steps <= 2
    target_config = build_calibration_config(profiles["matrix_bigamp_target_10gb"])
    assert target_config.matrix.N1 >= 3000
    assert target_config.training.max_steps == 2


def test_memory_calibration_cli_list_and_explain():
    listed = _run_cli("calibrate", "memory", "list")
    explained = _run_cli("calibrate", "memory", "explain", "matrix_bigamp_target_10gb")

    assert listed.returncode == 0
    assert "memory calibration profiles" in listed.stdout
    assert "matrix_bigamp_small" in listed.stdout
    assert "matrix_bigamp_target_10gb" in listed.stdout
    assert explained.returncode == 0
    assert "memory calibration profile" in explained.stdout
    assert "target_tensor_gb: 10.000" in explained.stdout
    assert "theoretical_tensor_estimate_gb" in explained.stdout
    assert "estimated_total_with_runtime_gb" in explained.stdout
    assert "output_root: runs/calibration/memory/matrix_bigamp_target_10gb" in explained.stdout


def test_memory_estimator_applies_local_calibration_coefficients(tmp_path, monkeypatch):
    gpu_model = MemoryEstimator().gpu_model
    coeff_path = tmp_path / "runs" / "calibration" / "memory" / "latest_coefficients.json"
    coeff_path.parent.mkdir(parents=True)
    coeff_path.write_text(
        """
{
  "format": "mf_memory_calibration_v1",
  "gpu_model": "%s",
  "algorithms": {
    "bigamp": {
      "factor": 2.0,
      "min_raw_gb_for_apply": 0.0,
      "latest_status": "active"
    }
  }
}
""" % gpu_model,
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    estimator = MemoryEstimator()
    params = EstimationParams(
        N1=32,
        N2=32,
        M=4,
        S=1,
        alpha_values=[0.1],
        algorithm_key="bigamp",
        use_compile=False,
    )
    estimate = estimator.estimate(params)

    assert estimate.breakdown["calibration.applied"] == 1.0
    assert estimate.breakdown["calibration.applied_factor"] == 2.0
