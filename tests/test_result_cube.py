import json
import torch

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


def test_precision_scan_writes_comparison_report(tmp_path):
    config = _config({
        "axes": {
            "precision": {"path": "algorithm_params.precision_profile", "values": ["fast", "aggressive"]},
            "alpha": {"path": "alpha", "values": [0.5]},
        }
    })
    result = ExperimentResult("precision_cube", config, scan_dimension="scan", scan_values=["fast", "aggressive"])
    result.result_cube = ResultCube(
        axes={
            "precision": {"key": "precision", "path": "algorithm_params.precision_profile", "values": ["fast", "aggressive"]},
            "alpha": {"key": "alpha", "path": "alpha", "values": [0.5]},
        }
    )
    fast = SingleRunResult(scan_value="fast", metrics={"Q_Y_mean": 0.8, "Q_W_mean": 0.7})
    aggressive = SingleRunResult(scan_value="aggressive", metrics={"Q_Y_mean": 0.78, "Q_W_mean": 0.71})
    result.add_result("fast", fast)
    result.add_result("aggressive", aggressive)
    result.result_cube.add_point("p_fast", {"precision": "fast", "alpha": 0.5}, fast.metrics)
    result.result_cube.add_point("p_aggressive", {"precision": "aggressive", "alpha": 0.5}, aggressive.metrics)

    result.save(tmp_path, save_tensors=False, output_options={"enable_heatmap": False})

    report = (tmp_path / "precision_comparison.md").read_text(encoding="utf-8")
    assert "Precision comparison" in report
    assert "Q_Y_mean" in report
    assert "No NaN/Inf" in report


def test_canonical_heatmap_uses_alpha_coordinate_for_point_ids(tmp_path, monkeypatch):
    import numpy as np
    from matrix_factorization.modules.outputs import plotting

    config = _config({
        "axes": {
            "mean_scale": {"path": "teacher_config.mean_scale", "values": [0.4]},
            "alpha": {"path": "alpha", "values": [0.2]},
        }
    })
    result = ExperimentResult("cube_heatmap", config, scan_dimension="scan", scan_values=["p0000"])
    result.result_cube = ResultCube(
        axes={
            "mean_scale": {"key": "mean_scale", "path": "teacher_config.mean_scale", "values": [0.4]},
            "alpha": {"key": "alpha", "path": "alpha", "values": [0.2]},
        }
    )
    single = SingleRunResult(
        scan_value="p0000",
        metrics={"Q_Y_mean": 1.0, "Q_W_mean": 1.0},
        W_students=torch.ones(2, 2, 1),
    )
    result.W_teacher = torch.ones(2, 1)
    result.add_result("p0000", single)
    result.result_cube.add_point(
        "p0000",
        {"mean_scale": 0.4, "alpha": 0.2},
        single.metrics,
        group_id="mean_scale=0.4",
    )
    captured = {}

    def fake_heatmap(matrix, alpha, output_dir, metric_name="Q_W", filename_prefix="heatmap", **kwargs):
        captured["alpha"] = alpha
        captured["prefix"] = filename_prefix
        assert np.asarray(matrix).shape == (3, 3)
        path = output_dir / f"{filename_prefix}_alpha_{alpha:.6f}.png"
        path.write_text("fake", encoding="utf-8")
        return path

    monkeypatch.setattr(plotting, "plot_replica_heatmap", fake_heatmap)
    monkeypatch.setattr(plotting, "create_gif", lambda paths, output_path, duration=0.2: output_path)

    result.save(tmp_path, save_tensors=False, output_options={"enable_heatmap": True, "heatmap_metric": "Q_W"})

    assert captured["alpha"] == 0.2
    assert captured["prefix"].startswith("heatmap_W_mean_scale_0.4_p0000")


def test_heatmap_metric_can_use_w_sign_aligned(tmp_path, monkeypatch):
    import numpy as np
    from matrix_factorization.modules.outputs import plotting

    config = _config({"axes": {"alpha": {"path": "alpha", "values": [0.0]}}})
    result = ExperimentResult("sign_heatmap", config, scan_dimension="alpha", scan_values=[0.0])
    single = SingleRunResult(
        scan_value=0.0,
        metrics={"Q_Y_mean": 1.0, "Q_W_SIGN_ALIGNED_mean": 1.0},
        W_students=torch.tensor([
            [[1.0, -2.0], [3.0, -4.0]],
            [[-1.0, 2.0], [-3.0, 4.0]],
        ]),
    )
    result.W_teacher = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result.add_result(0.0, single)

    captured = {}

    def fake_heatmap(matrix, alpha, output_dir, metric_name="Q_W", filename_prefix="heatmap", **kwargs):
        captured["matrix"] = np.asarray(matrix)
        captured["metric_name"] = metric_name
        captured["prefix"] = filename_prefix
        path = output_dir / f"{filename_prefix}_alpha_{alpha:.6f}.png"
        path.write_text("fake", encoding="utf-8")
        return path

    monkeypatch.setattr(plotting, "plot_replica_heatmap", fake_heatmap)
    monkeypatch.setattr(plotting, "create_gif", lambda paths, output_path, duration=0.2: output_path)

    result.save(
        tmp_path,
        save_tensors=False,
        output_options={"enable_heatmap": True, "heatmap_metric": "Q_W_SIGN_ALIGNED"},
    )

    assert captured["prefix"].startswith("heatmap_W_sign")
    assert "Sign-Aligned" in captured["metric_name"]
    assert captured["matrix"].shape == (3, 3)


def test_heatmap_rsb_ordering_accepts_projection_values_above_one(tmp_path):
    import numpy as np
    from matrix_factorization.modules.outputs.plotting import plot_replica_heatmap

    matrix = np.array([
        [1.0, 1.2, 0.4, 0.2],
        [1.2, 1.0, 1.1, 0.3],
        [0.4, 1.1, 1.0, 1.4],
        [0.2, 0.3, 1.4, 1.0],
    ])

    path = plot_replica_heatmap(
        matrix,
        alpha=3.4,
        output_dir=tmp_path,
        metric_name="W Sign-Aligned Projection",
        filename_prefix="heatmap_W_sign",
        rsb_ordering=True,
        enhance_high_values=False,
    )

    assert path.exists()
