import json

from matrix_factorization.core.experiment.config import (
    AlgorithmParams,
    ExperimentConfig,
    MatrixParams,
    ScanConfig,
    TrainingParams,
)
from matrix_factorization.core.experiment.runner import ExperimentRunner
from matrix_factorization.core.parallel.parallel_coordinator import ParallelCoordinator


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


def test_canonical_child_runner_uses_forced_resource_batch(monkeypatch):
    config = _base_config({
        "axes": {
            "damping": {"path": "algorithm_params.damping", "values": [0.5]},
            "alpha": {"path": "alpha", "values": [0.0, 0.1]},
        }
    })

    def fail_child_replan(*args, **kwargs):
        raise AssertionError("child runner must not re-plan forced canonical batches")

    monkeypatch.setattr(ParallelCoordinator, "plan_resource_execution", fail_child_replan)

    result = ExperimentRunner(device=None, verbose=False).run(
        config,
        output_options={
            "save_tensors": False,
            "storage_mode": "lightweight",
            "enable_heatmap": False,
        },
    )

    assert len(result.result_cube.points) == 2
    assert set(result.results) == set(result.result_cube.points)


def test_canonical_scan_progress_wraps_child_events_without_nested_lifecycle():
    config = _base_config({
        "axes": {
            "damping": {"path": "algorithm_params.damping", "values": [0.5, 0.6]},
            "alpha": {"path": "alpha", "values": [0.0, 0.1]},
        }
    })
    events = []

    ExperimentRunner(device=None, verbose=False).run(
        config,
        observer=events.append,
        output_options={
            "save_tensors": False,
            "storage_mode": "lightweight",
            "enable_heatmap": False,
        },
    )

    names = [event.type.name for event in events]
    assert names.count("EXPERIMENT_START") == 1
    assert names.count("EXPERIMENT_END") == 1
    assert names.count("EXECUTION_PLAN") == 1
    batch_starts = [event.payload for event in events if event.type.name == "BATCH_START"]
    assert batch_starts
    assert all(payload.get("scan_context", {}).get("canonical_scan") is True for payload in batch_starts)
    assert {payload["scan_context"]["coordinates"]["damping"] for payload in batch_starts} == {0.5, 0.6}
    assert all(payload.get("total_batches") == len(batch_starts) for payload in batch_starts)


def test_canonical_scan_error_mentions_current_coordinates(monkeypatch):
    config = _base_config({
        "axes": {
            "damping": {"path": "algorithm_params.damping", "values": [0.5]},
            "alpha": {"path": "alpha", "values": [0.0]},
        }
    })
    events = []

    def fail_algorithm_result(*args, **kwargs):
        raise RuntimeError("forced algorithm failure")

    monkeypatch.setattr(ExperimentRunner, "_run_algorithm_result", fail_algorithm_result)

    try:
        ExperimentRunner(device=None, verbose=False).run(
            config,
            observer=events.append,
            output_options={
                "save_tensors": False,
                "storage_mode": "lightweight",
                "enable_heatmap": False,
            },
        )
    except RuntimeError as exc:
        assert "damping=0.5" in str(exc)
        assert "forced algorithm failure" in str(exc)
    else:
        raise AssertionError("canonical scan failure was expected")

    error_events = [event for event in events if event.type.name == "ERROR"]
    assert error_events
    assert "damping=0.5" in error_events[-1].payload["error"]


def test_canonical_lightweight_aggregate_does_not_retain_factor_tensors():
    config = _base_config({
        "axes": {
            "damping": {"path": "algorithm_params.damping", "values": [0.5]},
            "alpha": {"path": "alpha", "values": [0.0, 0.1]},
        }
    })

    result = ExperimentRunner(device=None, verbose=False).run(
        config,
        output_options={
            "save_tensors": False,
            "storage_mode": "lightweight",
            "enable_heatmap": False,
        },
    )

    assert result.W_teacher is None
    assert all(single.W_students is None for single in result.results.values())
    assert all(single.X_students is None for single in result.results.values())


def test_canonical_scan_writes_partial_snapshot_after_batch(tmp_path):
    config = _base_config({
        "axes": {
            "damping": {"path": "algorithm_params.damping", "values": [0.5]},
            "alpha": {"path": "alpha", "values": [0.0, 0.1]},
        }
    })
    run_dir = tmp_path / "run"

    result = ExperimentRunner(device=None, verbose=False).run(
        config,
        output_options={
            "save_tensors": False,
            "storage_mode": "lightweight",
            "enable_heatmap": False,
            "checkpoint_path": str(run_dir / "checkpoints" / "latest.pt"),
        },
    )

    partial_path = run_dir / "partial" / "metrics_partial.json"
    assert partial_path.exists()
    payload = json.loads(partial_path.read_text())
    assert payload["partial_snapshot"] is True
    assert payload["snapshot_mode"] == "compact_progress"
    assert payload["completed"] == result.num_completed == 2
    assert payload["total"] == 2
    assert set(payload["completed_scan_values"]) == {str(value) for value in result.results}
    assert (run_dir / "metrics.partial.json").exists()
    group_dirs = sorted((run_dir / "groups").glob("*"))
    assert len(group_dirs) == 1
    assert (group_dirs[0] / "GROUP.md").exists()
    assert (group_dirs[0] / "metrics.json").exists()


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
