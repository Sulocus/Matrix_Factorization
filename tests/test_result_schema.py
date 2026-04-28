import json
from pathlib import Path

import pytest
import torch

from matrix_factorization.core.contracts import AlgorithmResult, AlgorithmStateView
from matrix_factorization.core.experiment.config import (
    AlgorithmParams,
    ExperimentConfig,
    MatrixParams,
    ScanConfig,
    SpreadingConfig,
    TrainingParams,
)
from matrix_factorization.core.experiment.data_factory import ExperimentData
from matrix_factorization.core.experiment.result import ExperimentResult, SingleRunResult
from matrix_factorization.core.experiment.runner import ExperimentRunner, ProgressEventType
from matrix_factorization.modules.algorithms.base import AlgorithmBase
from matrix_factorization.modules.algorithms.agd import AGDAlgorithm
from matrix_factorization.modules.algorithms.bigamp.standard import BiGAMPAlgorithm
from matrix_factorization.modules.algorithms.bigamp.spreading import BiGAMPSpreading
from matrix_factorization.modules.algorithms.bigamp.tensor_spreading import BiGAMPTensorSpreading
from matrix_factorization.modules.algorithms.bigamp.tensor_spreading_parallel import BiGAMPTensorSpreadingParallel
from matrix_factorization.modules.registry import get_algorithm
from matrix_factorization.modules.outputs.latest import refresh_latest_results


def _tiny_config():
    return ExperimentConfig(
        matrix=MatrixParams(N1=2, N2=2, M=1),
        training=TrainingParams(samples_per_alpha=1, max_steps=2),
        algorithm_key="bigamp",
        scan=ScanConfig(dimension="alpha", values=[0.0, 0.5]),
        algorithm_params=AlgorithmParams(use_compile=False, use_bf16=False),
        experiment_name="schema_contract",
        teacher_key="standard",
    )


def test_experiment_result_directory_schema(tmp_path):
    config = _tiny_config()
    result = ExperimentResult(
        experiment_id="schema_contract",
        config=config,
        scan_dimension="alpha",
        scan_values=[0.0, 0.5],
    )
    for alpha, q_y in [(0.0, 0.1), (0.5, 0.8)]:
        result.add_result(
            alpha,
            SingleRunResult(
                scan_value=alpha,
                metrics={
                    "Q_W_mean": q_y / 2,
                    "Q_W_COS_ROOT_mean": q_y / 2,
                    "Q_Y_mean": q_y,
                },
                metric_contract={
                    "algorithm_key": "bigamp",
                    "source": "runner_matrix_metrics",
                    "metric_specs": ["matrix.full.Q_Y", "matrix.factor.Q_W"],
                },
            ),
        )
    result.metadata.contract = {
        "algorithm": "bigamp",
        "is_valid": True,
        "output_plan": {
            "specs": ["scalar_curves", "latest_display"],
            "required_metrics": [],
            "required_artifacts": ["metrics_by_alpha", "config.json", "metadata.json"],
            "output_files": ["plots/qy_evolution.png"],
        },
    }

    run_dir = tmp_path / "20260101_0000_schema_contract"
    result.save(
        run_dir,
        save_tensors=False,
        output_options={"enable_heatmap": False},
    )

    assert (run_dir / "config.json").exists()
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "output_contract.json").exists()
    assert (run_dir / "events.jsonl").exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "plots" / "qy_evolution.png").exists()
    assert not (run_dir / "artifacts" / "results.pt").exists()

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["schema_version"] == 4
    assert metrics["result_cube"]["schema_version"] == 1
    assert metrics["contract"]["algorithm"] == "bigamp"
    assert metrics["scan_dimension"] == "alpha"
    assert metrics["scan_values"] == ["0.0", "0.5"]
    assert metrics["available_metric_keys"] == ["Q_W_COS_ROOT_mean", "Q_W_mean", "Q_Y_mean"]
    assert metrics["metric_schema"]["schema_version"] == 4
    assert metrics["metric_schema"]["compatibility"]["legacy_flat_keys_preserved"] is True
    assert metrics["metric_schema"]["compatibility"]["physical_overlap_metric_migration"] is True
    assert metrics["metric_schema"]["metric_policy"] == {
        "Q_Y_formula": "1 - normalized_mse",
        "Q_W_Q_X_normalization": "fixed_coordinate_count",
        "legacy_projection_suffix": "_PROJ_ABS",
        "teacher_norm_epsilon": 1e-12,
        "degenerate_teacher_norm": "return_zero",
        "clipped": False,
    }
    assert metrics["metric_schema"]["flat_key_index"]["Q_Y_mean"][0]["canonical_key"] == "measurement.full.teacher_student.Q_Y_fit"
    assert metrics["metric_schema"]["flat_key_index"]["Q_W_COS_ROOT_mean"][0]["canonical_key"] == "latent.W.teacher_student.Q_W_COS_ROOT"
    assert "measurement.full.teacher_student.Q_Y_fit" in metrics["metric_schema"]["semantic_classes"]
    assert metrics["metric_semantics"]["Q_Y_mean"][0]["space"] == "measurement"
    assert metrics["metric_contracts"]["0.5"]["source"] == "runner_matrix_metrics"
    assert "matrix.full.Q_Y" in metrics["metric_contracts"]["0.5"]["metric_specs"]
    assert metrics["metrics"]["0.5"]["Q_Y_mean"] == 0.8
    assert "MSE" not in metrics["results"]["0.5"]["metrics"]
    assert metrics["results"]["0.5"]["metric_contract"]["algorithm_key"] == "bigamp"
    assert metrics["results"]["0.5"]["metric_contract"]["metric_definition_policy"]["degenerate_teacher_norm"] == "return_zero"
    assert metrics["factor_payload_contract"]["matrix_factors"] == {
        "W_students": False,
        "X_students": False,
        "W_teacher": False,
        "X_teacher": False,
    }

    output_contract = json.loads((run_dir / "output_contract.json").read_text(encoding="utf-8"))
    assert output_contract["is_valid"] is True
    assert "config.json" in output_contract["available_artifacts"]
    assert "metrics_by_alpha" in output_contract["available_artifacts"]


def test_result_schema_contract_doc_covers_canonical_files():
    doc = (Path("docs/result_schema_contract.md")).read_text(encoding="utf-8")

    for required in [
        "config.json",
        "metadata.json",
        "metrics.json",
        "output_contract.json",
        "events.jsonl",
        "manifest.json",
        "artifacts/results.pt",
        "artifacts/teacher_factors.pt",
        "results/latest",
        "index.json",
        "summary.json",
        "selected_overview.png",
    ]:
        assert required in doc


def test_latest_export_is_lightweight_display_schema(tmp_path):
    source_dir = tmp_path / "runs"
    run_dir = source_dir / "20260101_0000_schema_contract"
    result = ExperimentResult(
        experiment_id="schema_contract",
        config=_tiny_config(),
        scan_dimension="alpha",
        scan_values=[0.0],
    )
    result.add_result(
        0.0,
        SingleRunResult(
            scan_value=0.0,
            metrics={"Q_W_mean": 0.0, "Q_Y_mean": 0.1},
        ),
    )
    result.save(run_dir, save_tensors=False, output_options={"enable_heatmap": False})

    latest_dir = tmp_path / "latest"
    summaries = refresh_latest_results(source_dir, latest_dir=latest_dir, count=1)

    assert len(summaries) == 1
    assert (latest_dir / "index.json").exists()
    assert (latest_dir / run_dir.name / "summary.json").exists()
    assert (latest_dir / run_dir.name / "plots" / "selected_overview.png").exists()

    index = json.loads((latest_dir / "index.json").read_text(encoding="utf-8"))
    assert index["schema_version"] == 1
    assert index["count"] == 1
    assert index["runs"][0]["artifacts"]["results_pt_omitted"] is False


def test_latest_export_requires_display_contract_files(tmp_path):
    source_dir = tmp_path / "runs"
    only_config = source_dir / "20260101_0000_only_config"
    only_metadata = source_dir / "20260101_0001_only_metadata"
    only_config.mkdir(parents=True)
    only_metadata.mkdir(parents=True)
    (only_config / "config.json").write_text("{}", encoding="utf-8")
    (only_metadata / "metadata.json").write_text("{}", encoding="utf-8")

    latest_dir = tmp_path / "latest"
    summaries = refresh_latest_results(source_dir, latest_dir=latest_dir, count=3)

    assert summaries == []
    index = json.loads((latest_dir / "index.json").read_text(encoding="utf-8"))
    assert index["count"] == 0


def test_algorithm_result_matrix_adapter_tracks_real_factor_outputs():
    result = AlgorithmResult.from_matrix_factors(
        alpha=0.5,
        metrics={"Q_Y_mean": 0.8},
        W_students="W",
        X_students="X",
        metadata={"algorithm_key": "bigamp"},
    )

    assert result.metrics_by_alpha[0.5]["Q_Y_mean"] == 0.8
    assert result.matrix_factors == {"W_students": "W", "X_students": "X"}
    assert result.tensor_factors is None
    assert result.metadata["result_kind"] == "matrix_factors"
    assert "matrix_factors" in result.available_outputs()


def test_algorithm_result_metrics_only_does_not_create_dummy_matrix_factors():
    result = AlgorithmResult.from_metrics_only(
        metrics_by_alpha={0.5: {"Q_Y_mean": 0.8}},
        artifacts={"overlap_matrix": "matrix"},
        metadata={"algorithm_key": "bigamp_tensor_parallel"},
    )

    assert result.matrix_factors is None
    assert result.tensor_factors is None
    assert "overlap_matrix" in result.available_outputs()
    assert result.metadata["result_kind"] == "metrics_only"


def test_metrics_only_result_save_marks_factor_payload_unavailable(tmp_path):
    config = _tiny_config()
    config.algorithm_key = "bigamp_tensor_parallel"
    result = ExperimentResult(
        experiment_id="tensor_metrics_only",
        config=config,
        scan_dimension="alpha",
        scan_values=[0.5],
    )
    result.add_result(
        0.5,
        SingleRunResult(
            scan_value=0.5,
            metrics={"Q_Y_mean": 0.8, "overlap_matrix": [[1.0]]},
        ),
    )

    run_dir = tmp_path / "tensor_metrics_only"
    result.save(run_dir, save_tensors=True, output_options={"enable_heatmap": False})

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["factor_payload_contract"]["unavailable_fields"] == [
        "W_students",
        "X_students",
        "W_teacher",
        "X_teacher",
    ]
    tensor_payload = torch.load(run_dir / "artifacts" / "results.pt", map_location="cpu")
    assert "W_students" not in tensor_payload
    assert "X_students" not in tensor_payload
    assert tensor_payload["factor_payload_contract"]["matrix_factors"]["W_students"] is False


def test_save_tensors_writes_shared_teacher_factor_artifact_for_points(tmp_path):
    config = _tiny_config()
    result = ExperimentResult(
        experiment_id="teacher_payload",
        config=config,
        scan_dimension="canonical",
        scan_values=["p0000"],
        W_teacher=torch.tensor([[1.0], [2.0]]),
        X_teacher=torch.tensor([[3.0, 4.0]]),
    )
    single = SingleRunResult(
        scan_value="p0000",
        metrics={"Q_Y_mean": 1.0},
        W_students=torch.ones(1, 2, 1),
        X_students=torch.ones(1, 1, 2),
    )
    result.add_result("p0000", single)
    result.result_cube.axes = {
        "alpha": {"values": [0.5]},
        "onsager_policy": {"values": ["test"]},
    }
    result.result_cube.add_point(
        "p0000",
        {"alpha": 0.5, "onsager_policy": "test"},
        single.metrics,
    )

    run_dir = tmp_path / "teacher_payload"
    result.save(run_dir, save_tensors=True, output_options={"enable_heatmap": False})

    teacher_payload = torch.load(run_dir / "artifacts" / "teacher_factors.pt", map_location="cpu")
    assert torch.equal(teacher_payload["W_teacher"], result.W_teacher)
    assert torch.equal(teacher_payload["X_teacher"], result.X_teacher)
    assert teacher_payload["W_teacher"].dtype == torch.float32

    point_payload = torch.load(run_dir / "artifacts" / "points" / "p0000" / "results.pt", map_location="cpu")
    assert point_payload["teacher_factors_path"] == "artifacts/teacher_factors.pt"
    assert point_payload["factor_payload_contract"]["matrix_factors"]["W_teacher"] is True

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["result_cube"]["artifacts"]["_shared"]["teacher_factors"] == "artifacts/teacher_factors.pt"

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert any(item["path"] == "artifacts/teacher_factors.pt" for item in manifest["artifacts"])

    loaded = ExperimentResult.load(run_dir)
    assert torch.equal(loaded.W_teacher, result.W_teacher)
    assert torch.equal(loaded.X_teacher, result.X_teacher)


def test_algorithm_result_does_not_advertise_empty_factor_payloads():
    result = AlgorithmResult(matrix_factors={"W_students": None, "X_students": None})

    assert "matrix_factors" not in result.available_outputs()


def test_algorithm_state_view_exposes_only_declared_capability_names():
    state = AlgorithmStateView(
        student_factors={"W": "W"},
        step_index=10,
        metadata={"custom_probe_state": True},
    )

    assert state.available_capabilities() == [
        "student_factors",
        "step_index",
        "custom_probe_state",
    ]


def test_runner_resource_plan_report_is_metadata_only():
    config = _tiny_config()
    runner = ExperimentRunner(device=torch.device("cpu"), verbose=False)
    params = runner._estimation_params_for_config(config, config.scan.values)
    plan = runner.parallel_coordinator.plan_execution(params)
    report = runner._runtime_resource_plan_report(config, plan)

    assert report["algorithm_key"] == "bigamp"
    assert report["metadata_only"] is True
    assert report["mode"]
    assert report["num_batches"] >= 1
    assert report["batches"][0]["sample_range_honored_by_runner"] is False
    assert report["batches"][0]["memory_breakdown"]
    assert "allocation_ratio" in report["allocation"]
    assert report["seed_policy"]["policy_key"] == "legacy_vectorized_batch_manual_seed"
    assert report["seed_policy"]["automatic_rebatch_allowed"] is False


def test_runtime_resource_plan_uses_effective_tensor_seed_policy():
    config = ExperimentConfig(
        matrix=MatrixParams(N1=2, N2=2, M=1),
        training=TrainingParams(samples_per_alpha=1, max_steps=2),
        algorithm_key="bigamp_tensor_parallel",
        scan=ScanConfig(dimension="alpha", values=[0.0, 0.5]),
        algorithm_params=AlgorithmParams(
            use_compile=False,
            use_bf16=False,
            seed_partition_policy="partition_invariant",
        ),
        spreading=SpreadingConfig(tensor_order=3),
        teacher_key="standard",
    )
    runner = ExperimentRunner(device=torch.device("cpu"), verbose=False)
    params = runner._estimation_params_for_config(config, config.scan.values)
    plan = runner.parallel_coordinator.plan_execution(params)
    report = runner._runtime_resource_plan_report(config, plan)

    assert report["config_effective"]["seed_partition_policy"] == "partition_invariant"
    assert report["seed_policy"]["policy_key"] == "tensor_parallel_partition_invariant_v1"
    assert report["seed_policy"]["partition_invariant"] is True
    assert report["seed_policy"]["batch_partition_sensitive"] is False
    assert report["seed_policy"]["automatic_rebatch_allowed"] is True
    assert report["replan_safety"]["replan_policy_key"] == "tensor_parallel_partition_invariant_v1"
    assert report["replan_safety"]["automatic_rebatch_allowed"] is True
    assert report["replan_safety"]["replan_implemented"] is False


@pytest.mark.parametrize("algorithm_key", ["agd", "bigamp"])
def test_runtime_resource_plan_uses_effective_matrix_seed_policy(algorithm_key):
    config = ExperimentConfig(
        matrix=MatrixParams(N1=2, N2=2, M=1),
        training=TrainingParams(samples_per_alpha=1, max_steps=2, max_epochs=2),
        algorithm_key=algorithm_key,
        scan=ScanConfig(dimension="alpha", values=[0.0, 0.5]),
        algorithm_params=AlgorithmParams(
            use_compile=False,
            use_bf16=False,
            seed_partition_policy="partition_invariant",
        ),
        teacher_key="standard",
    )
    runner = ExperimentRunner(device=torch.device("cpu"), verbose=False)
    params = runner._estimation_params_for_config(config, config.scan.values)
    plan = runner.parallel_coordinator.plan_execution(params)
    report = runner._runtime_resource_plan_report(config, plan)

    assert report["config_effective"]["seed_partition_policy"] == "partition_invariant"
    assert report["seed_policy"]["policy_key"] == "matrix_student_init_partition_invariant_v1"
    assert report["seed_policy"]["random_streams"] == ["student_initialization"]
    assert report["seed_policy"]["partition_invariant"] is True
    assert report["seed_policy"]["batch_partition_sensitive"] is False
    assert report["seed_policy"]["automatic_rebatch_allowed"] is True
    assert report["replan_safety"]["replan_policy_key"] == "matrix_student_init_partition_invariant_v1"
    assert report["replan_safety"]["automatic_rebatch_allowed"] is True
    assert report["replan_safety"]["replan_implemented"] is False


def test_runtime_resource_plan_uses_effective_spreading_seed_policy():
    config = ExperimentConfig(
        matrix=MatrixParams(N1=2, N2=2, M=1),
        training=TrainingParams(samples_per_alpha=1, max_steps=2),
        algorithm_key="bigamp_spreading",
        scan=ScanConfig(dimension="alpha", values=[0.0, 0.5]),
        algorithm_params=AlgorithmParams(
            use_compile=False,
            use_bf16=False,
            seed_partition_policy="partition_invariant",
        ),
        spreading=SpreadingConfig(tensor_order=2),
        teacher_key="standard",
    )
    runner = ExperimentRunner(device=torch.device("cpu"), verbose=False)
    params = runner._estimation_params_for_config(config, config.scan.values)
    plan = runner.parallel_coordinator.plan_execution(params)
    report = runner._runtime_resource_plan_report(config, plan)

    assert report["config_effective"]["seed_partition_policy"] == "partition_invariant"
    assert report["seed_policy"]["policy_key"] == "spreading_partition_invariant_v2"
    assert report["seed_policy"]["random_streams"] == [
        "student_initialization",
        "spreading_graph",
        "F_super",
        "restart_noise",
    ]
    assert "step" in report["seed_policy"]["seed_inputs"]
    assert report["seed_policy"]["partition_invariant"] is True
    assert report["seed_policy"]["batch_partition_sensitive"] is False
    assert report["seed_policy"]["automatic_rebatch_allowed"] is True


def test_spreading_chunk_policy_is_preserved_in_algorithm_result_metadata():
    algorithm = object.__new__(BiGAMPSpreading)
    algorithm.chunk_size = 1024
    algorithm.use_compile = True
    algorithm.use_bf16 = False
    algorithm.storage_dtype = torch.float32
    algorithm._contract_execution_metadata = algorithm._build_spreading_execution_metadata(
        [0.1],
        [(0, 1, 0.1)],
    )

    result = algorithm.coerce_legacy_batch_result(
        algorithm_key="bigamp_spreading",
        W_students=torch.zeros(1, 1, 2, 1),
        X_students=torch.zeros(1, 1, 1, 2),
    )
    metadata = result.metadata["execution_metadata"]

    assert metadata["chunk_size"] == 1024
    assert metadata["chunking_enabled"] is True
    assert metadata["chunk_policy"] == "manual_config"
    assert metadata["dynamic_batches"][0]["alpha_max"] == 0.1
    assert metadata["metadata_only"] is True


def test_runner_batch_end_event_records_elapsed_duration():
    config = _tiny_config()
    config.scan.values = [0.1]
    config.training.max_steps = 1
    runner = ExperimentRunner(device=torch.device("cpu"), verbose=False)
    events = []

    runner.run(
        config,
        observer=events.append,
        output_options={"enable_heatmap": False},
    )

    batch_end_events = [
        event for event in events if event.type is ProgressEventType.BATCH_END
    ]
    assert batch_end_events
    assert batch_end_events[-1].payload["duration"] >= 0.0


def test_runner_persists_native_matrix_algorithm_result_summary():
    config = _tiny_config()
    config.scan.values = [0.1]
    config.training.max_steps = 1
    result = ExperimentRunner(device=torch.device("cpu"), verbose=False).run(
        config,
        output_options={"enable_heatmap": False},
    )

    summaries = result.metadata.contract["algorithm_result_batches"]
    assert summaries[0]["result_source"] == "native_bigamp_algorithm_result"
    assert summaries[0]["result_kind"] == "matrix_factors"
    assert summaries[0]["matrix_factors_available"] is True
    assert summaries[0]["available_outputs"] == ["matrix_factors"]


def test_runner_wraps_tensor_legacy_metrics_without_dummy_matrix_factors():
    class DummyTensorAlgorithm:
        _batch_metrics = {
            0.5: {
                "Q_Y_mean": 0.8,
                "overlap_matrix": [[1.0]],
            }
        }

    config = _tiny_config()
    config.algorithm_key = "bigamp_tensor_parallel"
    runner = ExperimentRunner(device=None, verbose=False)
    result = runner._coerce_algorithm_result(
        config=config,
        algorithm=DummyTensorAlgorithm(),
        W_students=None,
        X_students=None,
    )

    assert result.matrix_factors is None
    assert result.metrics_by_alpha[0.5]["Q_Y_mean"] == 0.8
    assert result.artifacts["overlap_matrix"] == "per_alpha_metric_payload"


def test_runner_wraps_matrix_outputs_as_matrix_factors():
    class DummyMatrixAlgorithm:
        _batch_metrics = {}

    config = _tiny_config()
    runner = ExperimentRunner(device=None, verbose=False)
    result = runner._coerce_algorithm_result(
        config=config,
        algorithm=DummyMatrixAlgorithm(),
        W_students="W",
        X_students="X",
    )

    assert result.matrix_factors == {"W_students": "W", "X_students": "X"}
    assert result.metadata["result_kind"] == "matrix_factors"


def test_runner_rejects_missing_matrix_factors_for_matrix_contract():
    class DummyMatrixAlgorithm:
        _batch_metrics = {}

    config = _tiny_config()
    runner = ExperimentRunner(device=None, verbose=False)

    with pytest.raises(ValueError, match="must return real W_students/X_students"):
        runner._coerce_algorithm_result(
            config=config,
            algorithm=DummyMatrixAlgorithm(),
            W_students=None,
            X_students=None,
        )


def test_steps_checkpoint_path_returns_algorithm_result_contract():
    class DummyCheckpointAlgorithm:
        _batch_metrics = {}

        def __init__(self):
            self.max_steps_seen = []

        def train_batch_alphas(
            self,
            W_teacher,
            X_teacher,
            Y_teacher,
            masks,
            alpha_values,
            seed,
            max_steps=None,
            progress_callback=None,
        ):
            self.max_steps_seen.append(max_steps)
            W = torch.zeros(len(alpha_values), 1, 2, 1)
            X = torch.zeros(len(alpha_values), 1, 1, 2)
            return W, X

    runner = ExperimentRunner(device=torch.device("cpu"), verbose=False)
    algorithm = DummyCheckpointAlgorithm()
    data = ExperimentData(
        W_teacher=torch.zeros(2, 1),
        X_teacher=torch.zeros(1, 2),
        Y_teacher=torch.zeros(2, 2),
        masks=torch.ones(1, 2, 2),
        alpha_values=[0.5],
    )

    result, checkpoint = runner._run_algorithm_with_checkpoint_result(
        algorithm=algorithm,
        config=_tiny_config(),
        data=data,
        additional_steps=3,
        checkpoint=None,
        step_callback=None,
    )

    assert isinstance(result, AlgorithmResult)
    assert result.matrix_factors["W_students"].shape == (1, 1, 2, 1)
    assert checkpoint.step == 3
    assert algorithm.max_steps_seen == [3]

    _, next_checkpoint = runner._run_algorithm_with_checkpoint_result(
        algorithm=algorithm,
        config=_tiny_config(),
        data=data,
        additional_steps=2,
        checkpoint=checkpoint,
        step_callback=None,
    )

    assert next_checkpoint.step == 5
    assert algorithm.max_steps_seen == [3, 5]


def test_steps_checkpoint_tensor_path_keeps_metrics_only_contract():
    class DummyTensorCheckpointAlgorithm:
        _batch_metrics = {
            0.5: {
                "Q_Y_mean": 0.8,
                "overlap_matrix": [[1.0]],
            }
        }

        def train_batch_alphas(
            self,
            W_teacher,
            X_teacher,
            Y_teacher,
            masks,
            alpha_values,
            seed,
            max_steps=None,
            step_callback=None,
        ):
            return None, None

    config = _tiny_config()
    config.algorithm_key = "bigamp_tensor_parallel"
    runner = ExperimentRunner(device=torch.device("cpu"), verbose=False)
    result, checkpoint = runner._run_algorithm_with_checkpoint_result(
        algorithm=DummyTensorCheckpointAlgorithm(),
        config=config,
        data=ExperimentData(
            W_teacher=torch.zeros(2, 1),
            X_teacher=torch.zeros(1, 2),
            Y_teacher=torch.zeros(2, 2),
            alpha_values=[0.5],
        ),
        additional_steps=4,
        checkpoint=None,
        step_callback=None,
    )

    assert result.matrix_factors is None
    assert result.metrics_by_alpha[0.5]["Q_Y_mean"] == 0.8
    assert result.artifacts["overlap_matrix"] == "per_alpha_metric_payload"
    assert checkpoint.W_state is None
    assert checkpoint.X_state is None
    assert checkpoint.step == 4


def test_steps_checkpoint_prefers_formal_train_batch_result_contract():
    class FormalCheckpointAlgorithm:
        def __init__(self):
            self.max_steps_seen = []

        def train_batch_result(
            self,
            *,
            algorithm_key,
            W_teacher,
            X_teacher,
            Y_teacher,
            masks,
            alpha_values,
            seed,
            max_steps=None,
            progress_callback=None,
            step_callback=None,
        ):
            self.max_steps_seen.append(max_steps)
            return AlgorithmResult(
                metrics_by_alpha={0.5: {"Q_Y_mean": 0.8}},
                matrix_factors={
                    "W_students": torch.zeros(1, 1, 2, 1),
                    "X_students": torch.zeros(1, 1, 1, 2),
                },
                metadata={"algorithm_key": algorithm_key, "result_kind": "matrix_factors"},
            )

        def train_batch_alphas(self, *args, **kwargs):
            raise AssertionError("runner should use train_batch_result for checkpoint scans")

    config = _tiny_config()
    runner = ExperimentRunner(device=torch.device("cpu"), verbose=False)
    algorithm = FormalCheckpointAlgorithm()
    data = ExperimentData(
        W_teacher=torch.zeros(2, 1),
        X_teacher=torch.zeros(1, 2),
        Y_teacher=torch.zeros(2, 2),
        masks=torch.ones(1, 2, 2),
        alpha_values=[0.5],
    )

    result, checkpoint = runner._run_algorithm_with_checkpoint_result(
        algorithm=algorithm,
        config=config,
        data=data,
        additional_steps=3,
        checkpoint=None,
        step_callback=None,
    )

    assert result.metrics_by_alpha[0.5]["Q_Y_mean"] == 0.8
    assert checkpoint.step == 3
    assert checkpoint.W_state.shape == (1, 1, 2, 1)
    assert algorithm.max_steps_seen == [3]


def test_algorithm_base_train_batch_result_filters_unsupported_legacy_kwargs():
    class MinimalAlgorithm(AlgorithmBase):
        def train_single_alpha(self, W_teacher, X_teacher, Y_teacher, mask, alpha, seed):
            raise NotImplementedError

        def train_batch_alphas(
            self,
            W_teacher,
            X_teacher,
            Y_teacher,
            masks,
            alpha_values,
            seed,
            progress_callback=None,
        ):
            W = torch.zeros(len(alpha_values), 1, 2, 1)
            X = torch.zeros(len(alpha_values), 1, 1, 2)
            return W, X

    class Config:
        pass

    algorithm = MinimalAlgorithm(Config(), device=torch.device("cpu"))
    result = algorithm.train_batch_result(
        algorithm_key="bigamp",
        W_teacher=torch.zeros(2, 1),
        X_teacher=torch.zeros(1, 2),
        Y_teacher=torch.zeros(2, 2),
        masks=torch.ones(1, 2, 2),
        alpha_values=[0.1],
        seed=1,
        step_callback=lambda *_: None,
    )

    assert result.matrix_factors["W_students"].shape == (1, 1, 2, 1)
    assert result.metadata["result_source"] == "legacy_matrix_adapter"
    assert result.metadata["matrix_factors_available"] is True


@pytest.mark.parametrize(
    ("algorithm_cls", "algorithm_key", "result_source"),
    [
        (AGDAlgorithm, "agd", "native_agd_algorithm_result"),
        (BiGAMPAlgorithm, "bigamp", "native_bigamp_algorithm_result"),
        (BiGAMPSpreading, "bigamp_spreading", "native_bigamp_spreading_algorithm_result"),
    ],
)
def test_matrix_algorithms_return_native_algorithm_result(algorithm_cls, algorithm_key, result_source):
    algorithm = object.__new__(algorithm_cls)
    algorithm._contract_execution_metadata = {
        "path": "unit_fixture",
        "metadata_only": True,
    }

    def fake_train_batch_alphas(**kwargs):
        return (
            torch.zeros(len(kwargs["alpha_values"]), 1, 2, 1),
            torch.zeros(len(kwargs["alpha_values"]), 1, 1, 2),
        )

    algorithm.train_batch_alphas = fake_train_batch_alphas

    result = algorithm_cls.train_batch_result(
        algorithm,
        algorithm_key=algorithm_key,
        W_teacher=torch.zeros(2, 1),
        X_teacher=torch.zeros(1, 2),
        Y_teacher=torch.zeros(2, 2),
        masks=torch.ones(1, 2, 2),
        alpha_values=[0.1],
        seed=1,
        step_callback=lambda *_: None,
        runtime_step_callback=lambda *_: None,
    )

    assert isinstance(result, AlgorithmResult)
    assert result.metrics_by_alpha == {}
    assert result.matrix_factors["W_students"].shape == (1, 1, 2, 1)
    assert result.matrix_factors["X_students"].shape == (1, 1, 1, 2)
    assert result.metadata["result_source"] == result_source
    assert result.metadata["result_kind"] == "matrix_factors"
    assert result.metadata["matrix_factors_available"] is True
    assert result.metadata["execution_metadata"]["path"] == "unit_fixture"


@pytest.mark.parametrize("algorithm_key", ["agd", "bigamp", "bigamp_spreading"])
def test_active_matrix_algorithm_registry_uses_native_train_batch_result(algorithm_key):
    algorithm_cls = get_algorithm(algorithm_key).cls

    assert algorithm_cls.train_batch_result is not AlgorithmBase.train_batch_result


def test_runner_alpha_path_uses_formal_train_batch_result_contract():
    class FormalAlphaAlgorithm:
        def train_batch_result(
            self,
            *,
            algorithm_key,
            W_teacher,
            X_teacher,
            Y_teacher,
            masks,
            alpha_values,
            seed,
            progress_callback=None,
            step_callback=None,
            **kwargs,
        ):
            A = len(alpha_values)
            return AlgorithmResult(
                metrics_by_alpha={float(alpha): {"Q_Y_mean": 0.5, "Q_W_mean": 0.4} for alpha in alpha_values},
                matrix_factors={
                    "W_students": torch.zeros(A, 1, 2, 1),
                    "X_students": torch.zeros(A, 1, 1, 2),
                },
                metadata={"algorithm_key": algorithm_key, "result_kind": "matrix_factors"},
            )

        def train_batch_alphas(self, *args, **kwargs):
            raise AssertionError("runner should use train_batch_result for alpha scans")

    config = _tiny_config()
    runner = ExperimentRunner(device=torch.device("cpu"), verbose=False)
    data = ExperimentData(
        W_teacher=torch.zeros(2, 1),
        X_teacher=torch.zeros(1, 2),
        Y_teacher=torch.zeros(2, 2),
        masks=torch.ones(1, 2, 2),
        alpha_values=[0.5],
    )

    result = runner._run_algorithm_result(
        algorithm=FormalAlphaAlgorithm(),
        config=config,
        data=data,
        step_callback=None,
    )

    assert result.metrics_by_alpha[0.5]["Q_Y_mean"] == 0.5
    assert result.matrix_factors["W_students"].shape == (1, 1, 2, 1)


@pytest.mark.parametrize(
    ("algorithm_cls", "algorithm_key", "metric_payload"),
    [
        (
            BiGAMPTensorSpreading,
            "bigamp_tensor",
            {0.5: {"Q_Y_mean": 0.7, "Q_Y_std": 0.0, "Q_N_mean": 0.6, "Q_N_std": 0.0}},
        ),
        (
            BiGAMPTensorSpreadingParallel,
            "bigamp_tensor_parallel",
            {
                0.5: {
                    "Q_Y_mean": 0.8,
                    "Q_Y_std": 0.0,
                    "Q_Y_observed_mean": 0.75,
                    "Q_Y_observed_std": 0.0,
                    "Q_Y_unobserved_mean": 0.7,
                    "Q_Y_unobserved_std": 0.0,
                    "Q_N_mean": 0.6,
                    "Q_N_std": 0.0,
                    "overlap_matrix": [[1.0]],
                    "overlap_matrix_metric": "Q_Y",
                }
            },
        ),
    ],
)
def test_tensor_algorithms_return_formal_metrics_only_result(
    algorithm_cls,
    algorithm_key,
    metric_payload,
):
    algorithm = object.__new__(algorithm_cls)
    algorithm.order = 3
    algorithm.dims = (2, 2, 2)
    algorithm.device = torch.device("cpu")
    algorithm.requested_use_bf16 = False
    algorithm.use_bf16 = False
    algorithm.dtype_fallback_policy = "allow"
    algorithm.dtype_status = "bf16_disabled_by_config"
    algorithm.storage_dtype = torch.float32
    algorithm.requested_use_compile = False
    algorithm.use_compile = False

    def fake_train_batch_alphas(**kwargs):
        algorithm._batch_metrics = metric_payload
        return "legacy_placeholder_W", "legacy_placeholder_X"

    if hasattr(algorithm_cls, "_run_batch_metrics_only"):
        algorithm._run_batch_metrics_only = lambda **kwargs: setattr(algorithm, "_batch_metrics", metric_payload)
    else:
        algorithm.train_batch_alphas = fake_train_batch_alphas
    result = algorithm.train_batch_result(
        algorithm_key=algorithm_key,
        W_teacher=torch.zeros(2, 1),
        X_teacher=torch.zeros(1, 2),
        Y_teacher=torch.zeros(2, 2),
        masks=None,
        alpha_values=[0.5],
        seed=1,
    )

    assert isinstance(result, AlgorithmResult)
    assert result.matrix_factors is None
    assert result.metrics_by_alpha[0.5]["Q_Y_mean"] == metric_payload[0.5]["Q_Y_mean"]
    assert result.metadata["result_source"] == "tensor_algorithm_train_batch_result"
    assert result.metadata["matrix_factors_available"] is False
    assert result.metadata["tensor_order"] == 3
    assert result.metadata["dims"] == [2, 2, 2]
    assert result.metadata["graph_kind"] in {"tensor_hypergraph", "tensor_supergraph"}
    assert result.metadata["tensor_execution"]["metadata_only"] is True
    assert result.metadata["tensor_execution"]["effective_use_bf16"] is False
    assert result.metadata["tensor_execution"]["compile_status"]
    assert result.metadata["internal_alpha_batch_plan"]["metadata_only"] is True

    check = ExperimentRunner._validate_metric_payload(
        algorithm_key,
        result.metrics_by_alpha[0.5],
        source="algorithm_result",
    )
    assert check.is_valid


def test_tensor_parallel_legacy_tuple_path_is_separate_from_metrics_only_path():
    algorithm = object.__new__(BiGAMPTensorSpreadingParallel)
    algorithm.order = 3
    algorithm.S = 1
    algorithm.dims = (2, 2, 2)
    algorithm.M = 1
    algorithm.device = torch.device("cpu")
    algorithm._batch_metrics = {}
    algorithm._run_batch_metrics_only = lambda **kwargs: setattr(
        algorithm,
        "_batch_metrics",
        {0.5: {"Q_Y_mean": 0.8}},
    )

    W_all, X_all = algorithm.train_batch_alphas(
        W_teacher=torch.zeros(2, 1),
        X_teacher=torch.zeros(1, 2),
        Y_teacher=torch.zeros(2, 2),
        masks=None,
        alpha_values=[0.5],
        seed=1,
    )

    assert W_all.shape == (1, 1, 2, 1)
    assert X_all.shape == (1, 1, 1, 2)
    assert algorithm._batch_metrics[0.5]["Q_Y_mean"] == 0.8


def test_tensor_parallel_respects_requested_bf16_false(monkeypatch):
    config = _tiny_config()
    config.algorithm_key = "bigamp_tensor_parallel"
    config.spreading = SpreadingConfig(tensor_order=3)
    config.algorithm_params.use_bf16 = False
    config.algorithm_params.use_compile = False

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    algorithm = BiGAMPTensorSpreadingParallel(config, device=torch.device("cpu"))

    assert algorithm.requested_use_bf16 is False
    assert algorithm.use_bf16 is False
    assert algorithm.storage_dtype == torch.float32


def test_tensor_parallel_execution_metadata_reports_super_compile_fallback(monkeypatch):
    monkeypatch.setattr(BiGAMPTensorSpreadingParallel, "_compiled_step", object())
    monkeypatch.setattr(BiGAMPTensorSpreadingParallel, "_compiled_step_super", None)

    algorithm = object.__new__(BiGAMPTensorSpreadingParallel)
    algorithm.order = 3
    algorithm.dims = (2, 2, 2)
    algorithm.device = torch.device("cpu")
    algorithm.requested_use_bf16 = False
    algorithm.use_bf16 = False
    algorithm.dtype_fallback_policy = "allow"
    algorithm.dtype_status = "bf16_disabled_by_config"
    algorithm.storage_dtype = torch.float32
    algorithm.requested_use_compile = True
    algorithm.use_compile = True
    algorithm.compile_attempts = [
        {"target": "tensor_step_super", "success": False, "error": "compile_exception"}
    ]
    algorithm._batch_metrics = {0.5: {"Q_Y_mean": 0.8}}

    result = algorithm._metrics_only_algorithm_result("bigamp_tensor_parallel")
    execution = result.metadata["tensor_execution"]

    assert execution["requested_use_compile"] is True
    assert execution["dtype_fallback_policy"] == "allow"
    assert execution["dtype_status"] == "bf16_disabled_by_config"
    assert execution["compile_fallback_policy"] == "allow"
    assert execution["effective_use_compile"] is False
    assert execution["compiled_step_available"] is True
    assert execution["compiled_super_step_available"] is False
    assert execution["compile_status"] == "fallback_to_eager_tensor_step_super"
    assert execution["compile_attempts"][0]["target"] == "tensor_step_super"


def test_tensor_parallel_cpu_alpha_batch_plan_is_recorded_without_probe(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    algorithm = object.__new__(BiGAMPTensorSpreadingParallel)
    algorithm.device = torch.device("cpu")

    batches = algorithm._compute_alpha_batches([0.2, 0.1])

    assert batches == [[0.2, 0.1]]
    plan = algorithm._last_internal_alpha_batch_plan
    assert plan["planner"] == "cpu_no_internal_alpha_batching"
    assert plan["alpha_values_execution_order"] == [0.2, 0.1]
    assert plan["probe_enabled"] is False
    assert plan["seed_partition_sensitive"] is True
    assert plan["metadata_only"] is True


def test_runner_rejects_undeclared_metric_payload_keys():
    with pytest.raises(ValueError, match="undeclared metric keys"):
        ExperimentRunner._validate_metric_payload(
            "bigamp",
            {"Q_Y_mean": 0.1, "new_unregistered_metric": 1.0},
        )


def test_experiment_result_save_rejects_undeclared_metric_keys(tmp_path):
    result = ExperimentResult(
        experiment_id="bad_metric_payload",
        config=_tiny_config(),
        scan_dimension="alpha",
        scan_values=[0.0],
    )
    result.add_result(
        0.0,
        SingleRunResult(
            scan_value=0.0,
            metrics={"Q_Y_mean": 0.2, "new_unregistered_metric": 1.0},
        ),
    )

    with pytest.raises(ValueError, match="failed MetricSpec validation"):
        result.save(
            tmp_path / "bad_metric_payload",
            save_tensors=False,
            output_options={"enable_heatmap": False},
        )


def test_active_metrics_only_path_does_not_read_private_batch_metrics():
    class DummyTensorAlgorithm:
        _batch_metrics = {0.5: {"Q_Y_mean": 0.8}}

    runner = ExperimentRunner(device=torch.device("cpu"), verbose=False)
    data = ExperimentData(
        W_teacher=torch.zeros(2, 1),
        X_teacher=torch.zeros(1, 2),
        Y_teacher=torch.zeros(2, 2),
        alpha_values=[0.5],
    )
    algorithm_result = AlgorithmResult.from_metrics_only(
        metrics_by_alpha={},
        metadata={"algorithm_key": "bigamp_tensor_parallel"},
    )

    with pytest.raises(RuntimeError, match="must not recover missing metrics from private _batch_metrics"):
        runner._compute_metrics(
            W_students=None,
            X_students=None,
            data=data,
            algorithm=DummyTensorAlgorithm(),
            algorithm_result=algorithm_result,
        )


def test_runner_accepts_declared_tensor_artifact_metric_keys():
    check = ExperimentRunner._validate_metric_payload(
        "bigamp_tensor_parallel",
        {
            "Q_Y_mean": 0.1,
            "Q_Y_std": 0.0,
            "Q_Y_observed_mean": 0.1,
            "Q_Y_unobserved_mean": 0.1,
            "Q_N_mean": 0.1,
            "overlap_matrix": [[1.0]],
            "overlap_matrix_metric": "Q_Y",
        },
        source="algorithm_result",
    )

    assert check.source == "algorithm_result"
    assert "tensor.full.Q_Y" in check.metric_specs
    assert "tensor.factor.Q_N" in check.metric_specs


def test_custom_plot_missing_metric_fails_before_silent_zero(tmp_path):
    result = ExperimentResult(
        experiment_id="missing_metric_plot",
        config=_tiny_config(),
        scan_dimension="alpha",
        scan_values=[0.0],
    )
    result.add_result(
        0.0,
        SingleRunResult(scan_value=0.0, metrics={"Q_Y_mean": 0.2}),
    )

    with pytest.raises(ValueError, match="Q_W_mean"):
        result.save(
            tmp_path / "missing_metric_plot",
            save_tensors=False,
            output_options={
                "enable_heatmap": False,
                "plots": [{"curves": ["A.w"]}],
            },
        )


def test_heatmap_without_matrix_or_overlap_artifact_fails(tmp_path):
    result = ExperimentResult(
        experiment_id="missing_heatmap_payload",
        config=_tiny_config(),
        scan_dimension="alpha",
        scan_values=[0.0],
    )
    result.add_result(
        0.0,
        SingleRunResult(scan_value=0.0, metrics={"Q_Y_mean": 0.2}),
    )

    with pytest.raises(ValueError, match="enable_heatmap=true"):
        result.save(
            tmp_path / "missing_heatmap_payload",
            save_tensors=False,
            output_options={"enable_heatmap": True},
        )


def test_heatmap_plot_failure_is_hard_error(tmp_path, monkeypatch):
    from matrix_factorization.modules.outputs import plotting

    def fail_plot(*args, **kwargs):
        raise RuntimeError("plot backend failed")

    monkeypatch.setattr(plotting, "plot_replica_heatmap", fail_plot)
    config = _tiny_config()
    config.algorithm_key = "bigamp_tensor_parallel"
    result = ExperimentResult(
        experiment_id="hard_heatmap_failure",
        config=config,
        scan_dimension="alpha",
        scan_values=[0.0],
    )
    result.add_result(
        0.0,
        SingleRunResult(
            scan_value=0.0,
            metrics={
                "Q_Y_mean": 0.2,
                "overlap_matrix": [[1.0, 0.2], [0.2, 1.0]],
                "overlap_matrix_metric": "Q_Y",
            },
        ),
    )

    with pytest.raises(RuntimeError, match="Heatmap/GIF output failed"):
        result.save(
            tmp_path / "hard_heatmap_failure",
            save_tensors=False,
            output_options={"enable_heatmap": True},
        )


def test_output_plan_missing_artifact_fails_before_plotting(tmp_path):
    result = ExperimentResult(
        experiment_id="missing_output_plan_artifact",
        config=_tiny_config(),
        scan_dimension="alpha",
        scan_values=[0.0],
    )
    result.metadata.contract = {
        "output_plan": {
            "specs": ["tensor_heatmap"],
            "required_metrics": [],
            "required_artifacts": ["overlap_matrix"],
            "output_files": ["plots/heatmap_Y_alpha_*.png"],
        },
    }
    result.add_result(
        0.0,
        SingleRunResult(scan_value=0.0, metrics={"Q_Y_mean": 0.2}),
    )

    with pytest.raises(ValueError, match="overlap_matrix"):
        result.save(
            tmp_path / "missing_output_plan_artifact",
            save_tensors=False,
            output_options={"enable_heatmap": False},
        )


def test_default_scalar_plot_does_not_synthesize_missing_qw_curve(tmp_path):
    result = ExperimentResult(
        experiment_id="qy_only_plot",
        config=_tiny_config(),
        scan_dimension="alpha",
        scan_values=[0.0],
    )
    result.add_result(
        0.0,
        SingleRunResult(scan_value=0.0, metrics={"Q_Y_mean": 0.2}),
    )

    run_dir = tmp_path / "qy_only_plot"
    result.save(run_dir, save_tensors=False, output_options={"enable_heatmap": False})

    assert (run_dir / "plots" / "qy_evolution.png").exists()
    assert not (run_dir / "plots" / "overlap_evolution.png").exists()


def test_default_scalar_plot_requires_real_qy_metric(tmp_path):
    result = ExperimentResult(
        experiment_id="missing_qy_plot",
        config=_tiny_config(),
        scan_dimension="alpha",
        scan_values=[0.0],
    )
    result.add_result(
        0.0,
        SingleRunResult(scan_value=0.0, metrics={"Q_W_mean": 1.0}),
    )

    with pytest.raises(ValueError, match="scalar_curves output requires"):
        result.save(
            tmp_path / "missing_qy_plot",
            save_tensors=False,
            output_options={"enable_heatmap": False},
        )
