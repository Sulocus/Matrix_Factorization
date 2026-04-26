import subprocess
import sys

from matrix_factorization.core.memory_calibration import (
    build_calibration_config,
    get_memory_calibration_profiles,
)


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
    tensor_config = build_calibration_config(profiles["tensor_parallel_small"])
    assert tensor_config.algorithm_key == "bigamp_tensor_parallel"
    assert tensor_config.spreading.tensor_order == 3
    assert tensor_config.training.max_steps <= 2


def test_memory_calibration_cli_list_and_explain():
    listed = _run_cli("calibrate", "memory", "list")
    explained = _run_cli("calibrate", "memory", "explain", "matrix_bigamp_small")

    assert listed.returncode == 0
    assert "memory calibration profiles" in listed.stdout
    assert "matrix_bigamp_small" in listed.stdout
    assert explained.returncode == 0
    assert "memory calibration profile" in explained.stdout
    assert "theoretical_tensor_estimate_gb" in explained.stdout
    assert "estimated_total_with_runtime_gb" in explained.stdout
    assert "output_root: runs/calibration/memory/matrix_bigamp_small" in explained.stdout
