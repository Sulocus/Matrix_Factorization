import json

from matrix_factorization.core.experiment.config import (
    AlgorithmParams,
    ExperimentConfig,
    MatrixParams,
    ScanConfig,
    TrainingParams,
)
from matrix_factorization.core.experiment.result import ExperimentResult, ResultCube, SingleRunResult


def _config(scan_spec):
    return ExperimentConfig(
        matrix=MatrixParams(N1=2, N2=2, M=1),
        training=TrainingParams(samples_per_alpha=1, max_steps=1, max_epochs=1),
        algorithm_key="bigamp",
        scan=ScanConfig(dimension="alpha", values=[0.0]),
        algorithm_params=AlgorithmParams(use_compile=False, use_bf16=False),
        teacher_key="standard",
        scan_spec=scan_spec,
    )


def test_result_cube_is_saved_in_metrics_json(tmp_path):
    config = _config({
        "axes": {
            "damping": {"path": "algorithm_params.damping", "values": [0.5]},
            "alpha": {"path": "alpha", "values": [0.0]},
        }
    })
    result = ExperimentResult("cube", config, scan_dimension="scan", scan_values=["p0000"])
    result.result_cube = ResultCube(
        axes={
            "damping": {"key": "damping", "path": "algorithm_params.damping", "values": [0.5]},
            "alpha": {"key": "alpha", "path": "alpha", "values": [0.0]},
        }
    )
    single = SingleRunResult(scan_value="p0000", metrics={"Q_Y_mean": 0.4, "Q_W_mean": 0.6})
    result.add_result("p0000", single)
    result.result_cube.add_point(
        "p0000",
        {"damping": 0.5, "alpha": 0.0},
        single.metrics,
        overrides={"algorithm_params.damping": 0.5},
        group_id="damping=0.5",
    )

    result.save(tmp_path, save_tensors=False, output_options={"enable_heatmap": False})

    payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert payload["result_cube"]["points"]["p0000"]["coordinates"]["damping"] == 0.5
    assert payload["result_cube"]["metrics"]["p0000"]["Q_Y_mean"] == 0.4
    loaded = ExperimentResult.load(tmp_path)
    assert loaded.result_cube.points["p0000"].coordinates["damping"] == 0.5


def test_alpha_only_result_cube_keeps_legacy_metrics_payload(tmp_path):
    config = _config({"axes": {"alpha": {"path": "alpha", "values": [0.0]}}})
    result = ExperimentResult("alpha", config, scan_dimension="alpha", scan_values=[0.0])
    single = SingleRunResult(scan_value=0.0, metrics={"Q_Y_mean": 0.7, "Q_W_mean": 0.3})
    result.add_result(0.0, single)
    result.result_cube = ResultCube(axes={"alpha": {"key": "alpha", "path": "alpha", "values": [0.0]}})
    result.result_cube.add_point("p0000", {"alpha": 0.0}, single.metrics)

    result.save(tmp_path, save_tensors=False, output_options={"enable_heatmap": False})

    payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert payload["metrics"]["0.0"]["Q_Y_mean"] == 0.7
    assert payload["result_cube"]["metrics"]["p0000"]["Q_Y_mean"] == 0.7
