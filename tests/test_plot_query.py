import pytest

from matrix_factorization.core.experiment.config import (
    AlgorithmParams,
    ExperimentConfig,
    MatrixParams,
    ScanConfig,
    TrainingParams,
)
from matrix_factorization.core.experiment.result import ExperimentResult, ResultCube, SingleRunResult


def _result_with_cube():
    scan_spec = {
        "axes": {
            "damping": {"path": "algorithm_params.damping", "values": [0.2, 0.5]},
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
            "alpha": {"path": "alpha", "values": [0.0, 0.1]},
        }
    }
    config = ExperimentConfig(
        matrix=MatrixParams(N1=2, N2=2, M=1),
        training=TrainingParams(samples_per_alpha=1, max_steps=1, max_epochs=1),
        algorithm_key="bigamp",
        scan=ScanConfig(dimension="alpha", values=[0.0, 0.1]),
        algorithm_params=AlgorithmParams(use_compile=False, use_bf16=False),
        teacher_key="standard",
        scan_spec=scan_spec,
    )
    result = ExperimentResult("plot_query", config, scan_dimension="scan", scan_values=[])
    result.result_cube = ResultCube(axes={
        "damping": {"key": "damping", "path": "algorithm_params.damping", "values": [0.2, 0.5]},
        "init": {"key": "init", "kind": "composite", "values": ["cold", "warm_095"]},
        "alpha": {"key": "alpha", "path": "alpha", "values": [0.0, 0.1]},
    })
    idx = 0
    for damping in [0.2, 0.5]:
        for init in ["cold", "warm_095"]:
            for alpha in [0.0, 0.1]:
                point_id = f"p{idx:04d}"
                metric = {"Q_Y_mean": damping + alpha + (0.1 if init == "warm_095" else 0.0)}
                single = SingleRunResult(scan_value=point_id, metrics=metric)
                result.scan_values.append(point_id)
                result.add_result(point_id, single)
                result.result_cube.add_point(
                    point_id,
                    {"damping": damping, "init": init, "alpha": alpha},
                    metric,
                    group_id=f"damping={damping}|init={init}",
                )
                idx += 1
    return result


def test_plot_query_can_compare_selected_parameter_groups(tmp_path):
    result = _result_with_cube()

    result.save(
        tmp_path,
        save_tensors=False,
        output_options={
            "enable_heatmap": False,
            "plots": [
                {
                    "x": "alpha",
                    "y": "Q_Y_mean",
                    "series_by": ["damping", "init"],
                    "where": {"init": "warm_095"},
                    "filename": "qy_compare.png",
                }
            ],
        },
    )

    assert (tmp_path / "plots" / "qy_compare.png").exists()


def test_plot_query_missing_metric_is_error(tmp_path):
    result = _result_with_cube()

    with pytest.raises(ValueError, match="does_not_exist"):
        result.save(
            tmp_path,
            save_tensors=False,
            output_options={
                "enable_heatmap": False,
                "plots": [{"x": "alpha", "y": "does_not_exist", "filename": "bad.png"}],
            },
        )
