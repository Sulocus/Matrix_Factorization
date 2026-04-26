from matrix_factorization.cli import _build_run_directory_name
from matrix_factorization.core.experiment import (
    AlgorithmParams,
    ExperimentConfig,
    MatrixParams,
    ScanConfig,
    TrainingParams,
)


def _config(scan_spec):
    alpha_values = scan_spec["axes"].get("alpha", {"values": [1.0]}).get("values", [1.0])
    if isinstance(alpha_values, dict):
        alpha_values = [0.0, 0.5, 1.0]
    return ExperimentConfig(
        matrix=MatrixParams(N1=200, N2=200, M=50),
        training=TrainingParams(samples_per_alpha=2, max_steps=3),
        algorithm_key="bigamp_spreading",
        scan=ScanConfig(dimension="alpha", values=alpha_values),
        algorithm_params=AlgorithmParams(damping=0.2),
        experiment_name="verbose_legacy_name_that_should_not_be_used_for_folder",
        teacher_key="standard",
        scan_spec=scan_spec,
    )


def test_run_directory_name_is_compact_and_uses_output_name():
    scan_spec = {
        "axes": {
            "alpha": {
                "path": "alpha",
                "values": {"start": 0.0, "stop": 1.0, "step": 0.5},
            }
        }
    }
    name = _build_run_directory_name(
        "20260427_073012",
        _config(scan_spec),
        {"name": "my phase test"},
        raw_yaml="stable payload",
    )

    assert name.startswith("20260427_073012_my-phase-test_bgs_N200_M50_a3_")
    assert "verbose_legacy_name" not in name
    assert len(name.rsplit("_", 1)[-1]) == 6


def test_run_directory_name_summarizes_mixed_scan_axes():
    scan_spec = {
        "axes": {
            "alpha": {"path": "alpha", "values": [0.5, 1.0]},
            "damping": {"path": "algorithm_params.damping", "values": [0.2, 0.5]},
            "init": {
                "kind": "composite",
                "values": {
                    "cold": {"algorithm_params.init_mode": "random"},
                    "warm": {"algorithm_params.init_mode": "teacher"},
                },
            },
        }
    }
    name = _build_run_directory_name(
        "20260427_073012",
        _config(scan_spec),
        {"name": ""},
        raw_yaml="mixed payload",
    )

    assert "_bgs_N200_M50_a2-dmp2-init2_" in name
