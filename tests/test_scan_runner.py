from matrix_factorization.core.experiment.config import (
    AlgorithmParams,
    ExperimentConfig,
    MatrixParams,
    ScanConfig,
    TrainingParams,
)
from matrix_factorization.core.experiment.runner import ExperimentRunner


def _base_config(scan_spec):
    alpha_axis = scan_spec["axes"].get("alpha", {"values": [0.0]})
    alpha_values = list(alpha_axis.get("values", [0.0]))
    return ExperimentConfig(
        matrix=MatrixParams(N1=3, N2=3, M=1),
        training=TrainingParams(samples_per_alpha=1, max_steps=1, max_epochs=1),
        algorithm_key="bigamp",
        scan=ScanConfig(dimension="alpha", values=alpha_values),
        algorithm_params=AlgorithmParams(
            damping=0.5,
            noise_var=1.0e-5,
            use_compile=False,
            use_bf16=False,
        ),
        teacher_key="standard",
        experiment_name="scan_runner_test",
        scan_spec=scan_spec,
    )


def test_alpha_only_scan_uses_result_cube_without_group_executor():
    config = _base_config({"axes": {"alpha": {"path": "alpha", "values": [0.0, 0.1]}}})

    result = ExperimentRunner(device=None, verbose=False).run(
        config,
        output_options={"save_tensors": False, "enable_heatmap": False},
    )

    assert result.scan_dimension == "alpha"
    assert set(result.result_cube.axes) == {"alpha"}
    assert len(result.result_cube.points) == 2


def test_steps_axis_runs_through_canonical_executor():
    config = _base_config({
        "axes": {
            "alpha": {"path": "alpha", "values": [0.0]},
            "max_steps": {"path": "max_steps", "values": [1, 2]},
        }
    })

    result = ExperimentRunner(device=None, verbose=False).run(
        config,
        output_options={"save_tensors": False, "enable_heatmap": False},
    )

    assert result.scan_dimension == "scan"
    assert len(result.result_cube.points) == 2
    assert {point.coordinates["max_steps"] for point in result.result_cube.points.values()} == {1, 2}


def test_size_axis_runs_as_isolated_groups():
    config = _base_config({
        "axes": {
            "size": {
                "kind": "composite",
                "values": {
                    "N3_M1": {"matrix.N1": 3, "matrix.N2": 3, "matrix.M": 1},
                    "N4_M1": {"matrix.N1": 4, "matrix.N2": 4, "matrix.M": 1},
                },
            },
            "alpha": {"path": "alpha", "values": [0.0]},
        }
    })

    result = ExperimentRunner(device=None, verbose=False).run(
        config,
        output_options={"save_tensors": False, "enable_heatmap": False},
    )

    assert len(result.result_cube.points) == 2
    assert {point.coordinates["size"] for point in result.result_cube.points.values()} == {"N3_M1", "N4_M1"}


def test_init_axis_runs_cold_and_warm_groups():
    config = _base_config({
        "axes": {
            "init": {
                "kind": "composite",
                "values": {
                    "cold": {"algorithm_params.init_mode": "random"},
                    "warm_095": {
                        "algorithm_params.init_mode": "teacher",
                        "algorithm_params.init_overlap": 0.95,
                    },
                },
            },
            "alpha": {"path": "alpha", "values": [0.0]},
        }
    })

    result = ExperimentRunner(device=None, verbose=False).run(
        config,
        output_options={"save_tensors": False, "enable_heatmap": False},
    )

    assert len(result.result_cube.points) == 2
    assert {point.coordinates["init"] for point in result.result_cube.points.values()} == {"cold", "warm_095"}
