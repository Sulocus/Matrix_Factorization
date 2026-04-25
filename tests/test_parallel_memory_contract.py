from pathlib import Path

import pytest
import torch

from matrix_factorization.cli import load_yaml_config
from matrix_factorization.core.contracts import (
    get_effective_seed_policy_summary,
    get_batching_specs,
    get_memory_model_specs,
    get_resource_specs,
    get_seed_policy_specs,
    get_tensor_parity_report,
)
from matrix_factorization.core.planning import build_experiment_plan
from matrix_factorization.core.experiment.result import ExperimentResult
from matrix_factorization.core.experiment.runner import ExperimentRunner
from matrix_factorization.core.parallel import (
    AllocationPresets,
    BatchConfig,
    EstimationParams,
    ExecutionPlan,
    ParallelMode,
    get_parallel_coordinator,
)
from matrix_factorization.core.parallel.batch_checkpoint import CheckpointManager
from matrix_factorization.core.parallel.memory_guard import MemoryAbortException
from matrix_factorization.core.parallel.memory_estimator import MemoryEstimator
from matrix_factorization.modules.algorithms.bigamp.tensor_contract import (
    build_tensor_alpha_batch_metadata,
    build_tensor_execution_metadata,
)
from matrix_factorization.modules.algorithms.bigamp.tensor_supergraph import (
    create_tensor_superdata,
    create_tensor_supergraph,
)
from matrix_factorization.modules.algorithms.bigamp.spreading import BiGAMPSpreading


def _write_tensor_config(path: Path) -> None:
    path.write_text(
        """
tensor_order: 3
algorithm: 4
teacher: 2
matrix:
  N1: 4
  N2: 4
  M: 2
scan_mode: 1
alpha_scan:
  start: 0.0
  stop: 0.2
  step: 0.1
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
  enable_heatmap: false
  save_tensors: false
""",
        encoding="utf-8",
    )


def _write_matrix_config(path: Path, algorithm: int) -> None:
    path.write_text(
        f"""
tensor_order: 2
algorithm: {algorithm}
teacher: 2
matrix:
  N1: 4
  N2: 4
  M: 2
scan_mode: 1
alpha_scan:
  start: 0.0
  stop: 0.2
  step: 0.1
training:
  samples_per_alpha: 1
  max_steps: 2
  max_epochs: 2
algorithm_params:
  damping: 0.5
  noise_var: 1.0e-5
  learning_rate: 0.01
  use_compile: false
  use_bf16: false
  seed_partition_policy: partition_invariant
output:
  enable_heatmap: false
  save_tensors: false
""",
        encoding="utf-8",
    )


def _write_spreading_config(path: Path, adaptive_restart: bool = False) -> None:
    path.write_text(
        f"""
tensor_order: 2
algorithm: 2
teacher: 2
matrix:
  N1: 4
  N2: 4
  M: 2
scan_mode: 1
alpha_scan:
  start: 0.0
  stop: 0.2
  step: 0.1
training:
  samples_per_alpha: 1
  max_steps: 2
algorithm_params:
  damping: 0.5
  noise_var: 1.0e-5
  use_compile: false
  use_bf16: false
  seed_partition_policy: partition_invariant
  adaptive_restart: {str(adaptive_restart).lower()}
spreading:
  f_distribution: 1
  seed: 321
output:
  enable_heatmap: false
  save_tensors: false
""",
        encoding="utf-8",
    )


def test_parallel_memory_contract_docs_cover_runtime_metadata():
    contract = Path("docs/parallel_memory_contract.md")
    review = Path("docs/parallel_memory_review_queue.md")

    assert contract.exists()
    assert review.exists()

    contract_text = contract.read_text(encoding="utf-8")
    assert "tensor_execution" in contract_text
    assert "internal_alpha_batch_plan" in contract_text
    assert "metadata_only" in contract_text

    review_text = review.read_text(encoding="utf-8")
    assert "Seed Partition Invariant" in review_text
    assert "OOM Retry / Auto Shrink Batch" in review_text


def test_checkpoint_flush_waits_for_async_save(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    manager = CheckpointManager(checkpoint_path)

    manager.save(
        config_dict={"algorithm_key": "bigamp"},
        completed_alphas=[0.1],
        results={0.1: {"metrics": {"Q_Y_mean": 0.5}}},
        output_options={"save_tensors": False},
        raw_yaml="algorithm: 1",
    )
    manager.flush()

    loaded = manager.load()
    assert loaded is not None
    assert loaded.completed_alphas == [0.1]
    assert loaded.results[0.1]["metrics"]["Q_Y_mean"] == 0.5

    manager.delete()
    assert not checkpoint_path.exists()


def test_checkpoint_flush_surfaces_async_save_failure(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "checkpoint.pt"
    manager = CheckpointManager(checkpoint_path)

    def fail_save(payload, path):
        raise RuntimeError("forced checkpoint failure")

    monkeypatch.setattr(manager, "_atomic_save", fail_save)
    manager.save(config_dict={}, completed_alphas=[], results={})

    with pytest.raises(RuntimeError, match="forced checkpoint failure"):
        manager.flush()


def test_memory_abort_flushes_checkpoint_before_exit(tmp_path):
    config_path = tmp_path / "matrix_config.yaml"
    checkpoint_path = tmp_path / "oom_checkpoint.pt"
    _write_matrix_config(config_path, algorithm=1)
    config, _, raw_yaml = load_yaml_config(config_path)
    result = ExperimentResult(
        experiment_id="oom_checkpoint_contract",
        config=config,
        scan_dimension=config.scan.dimension,
        scan_values=config.scan.values,
    )

    class AbortGuard:
        is_running = True

        def check_abort(self):
            raise MemoryAbortException("forced abort")

    runner = ExperimentRunner(device=torch.device("cpu"), verbose=False)
    runner._memory_guard = AbortGuard()

    with pytest.raises(SystemExit) as exc:
        runner._run_standard_scan(
            config,
            algorithm=object(),
            result=result,
            observer=None,
            output_options={"checkpoint_path": str(checkpoint_path), "save_tensors": False},
            raw_yaml=raw_yaml,
        )

    assert exc.value.code == 0
    loaded = CheckpointManager(checkpoint_path).load()
    assert loaded is not None
    assert loaded.completed_alphas == []
    assert loaded.raw_yaml == raw_yaml


def test_tensor_parallel_resource_and_batching_specs_remain_metadata_only():
    resource = get_resource_specs()["bigamp_tensor_parallel"]
    batching = get_batching_specs()["bigamp_tensor_parallel"]
    memory_model = get_memory_model_specs()["bigamp_tensor_parallel"]
    seed_policy = get_seed_policy_specs()["bigamp_tensor_parallel"]

    assert resource.probe_support == "A=1 tensor supergraph probe"
    assert "tf32 matmul" in resource.dtype_modes
    assert "algorithm internal probe batches" in batching.planner_layers
    assert batching.seed_partition_sensitive is True
    assert batching.metadata_only is True
    assert batching.sample_range_honored is False
    assert memory_model.probe_required is True
    assert memory_model.drives_execution is False
    assert "tensorsupergraph" in memory_model.formula_basis.lower()
    assert seed_policy.partition_invariant is False
    assert seed_policy.batch_partition_sensitive is True
    assert seed_policy.automatic_rebatch_allowed is False
    assert "batch_idx" in seed_policy.notes


def test_effective_seed_policy_summary_is_shared_contract_source():
    matrix_policy = get_effective_seed_policy_summary("bigamp", "partition_invariant")
    spreading_policy = get_effective_seed_policy_summary("bigamp_spreading", "partition_invariant")
    tensor_policy = get_effective_seed_policy_summary("bigamp_tensor_parallel", "partition_invariant")

    assert matrix_policy["policy_key"] == "matrix_student_init_partition_invariant_v1"
    assert spreading_policy["policy_key"] == "spreading_partition_invariant_v2"
    assert tensor_policy["policy_key"] == "tensor_parallel_partition_invariant_v1"
    assert matrix_policy["automatic_rebatch_allowed"] is True
    assert spreading_policy["random_streams"] == [
        "student_initialization",
        "spreading_graph",
        "F_super",
        "restart_noise",
    ]
    assert "step" in spreading_policy["seed_inputs"]
    assert "dimension" in tensor_policy["seed_inputs"]


def test_tensor_supergraph_partition_invariant_indices_do_not_depend_on_batch_cmax():
    dims = (4, 4, 4)
    device = torch.device("cpu")

    single = create_tensor_supergraph(
        dims,
        [0.5],
        M=2,
        S=2,
        seed=123,
        device=device,
        partition_invariant=True,
    )
    combined = create_tensor_supergraph(
        dims,
        [0.5, 1.0],
        M=2,
        S=2,
        seed=123,
        device=device,
        partition_invariant=True,
    )

    c_alpha = single.C_per_alpha[0]
    assert c_alpha == combined.C_per_alpha[0]
    for dim_idx in range(len(dims)):
        assert torch.equal(
            single.indices[dim_idx][:, :c_alpha],
            combined.indices[dim_idx][:, :c_alpha],
        )


def test_tensor_superdata_partition_invariant_F_and_Y_do_not_depend_on_batch_cmax():
    dims = (4, 4, 4)
    device = torch.device("cpu")
    teacher_factors = [
        torch.arange(dim * 2, dtype=torch.float32, device=device).reshape(dim, 2) / 10.0
        for dim in dims
    ]
    single_graph = create_tensor_supergraph(
        dims,
        [0.5],
        M=2,
        S=2,
        seed=123,
        device=device,
        partition_invariant=True,
    )
    combined_graph = create_tensor_supergraph(
        dims,
        [0.5, 1.0],
        M=2,
        S=2,
        seed=123,
        device=device,
        partition_invariant=True,
    )
    single_data = create_tensor_superdata(
        single_graph,
        teacher_factors,
        f_distribution="rademacher",
        seed=1123,
        partition_invariant=True,
    )
    combined_data = create_tensor_superdata(
        combined_graph,
        teacher_factors,
        f_distribution="rademacher",
        seed=1123,
        partition_invariant=True,
    )

    c_alpha = single_graph.C_per_alpha[0]
    assert torch.equal(single_data.F_super[:, :c_alpha, :], combined_data.F_super[:, :c_alpha, :])
    assert torch.allclose(single_data.Y_super[:, :c_alpha], combined_data.Y_super[:, :c_alpha])


def test_active_algorithms_have_memory_model_specs():
    resource_specs = get_resource_specs()
    memory_models = get_memory_model_specs()
    seed_policies = get_seed_policy_specs()

    for algorithm_key in resource_specs:
        assert algorithm_key in memory_models
        assert algorithm_key in seed_policies
        assert memory_models[algorithm_key].algorithm_key == algorithm_key
        assert memory_models[algorithm_key].calibration_status
        assert seed_policies[algorithm_key].policy_key


def test_tensor_parity_report_keeps_seed_and_batching_as_review_items():
    report = get_tensor_parity_report()

    assert report["parity_item_details"]["seed_partition"]["review_required"] is True
    assert report["parity_item_details"]["seed_partition"]["risk"] == "high"
    assert report["parity_item_details"]["batching_semantics"]["review_required"] is True
    assert report["parity_item_details"]["batching_semantics"]["area"] == "resource"


def test_tensor_execution_metadata_is_json_friendly_and_non_driving():
    metadata = build_tensor_execution_metadata(
        path="parallel_tensor_supergraph",
        device="cuda:0",
        requested_use_bf16=False,
        effective_use_bf16=False,
        storage_dtype="float32",
        requested_use_compile=False,
        effective_use_compile=False,
        compiled_step_available=False,
        compiled_super_step_available=False,
        compile_status="disabled_by_config",
        compile_attempts=[],
        tf32_matmul_enabled=True,
        tf32_cudnn_enabled=True,
    )

    assert metadata["path"] == "parallel_tensor_supergraph"
    assert metadata["storage_dtype"] == "float32"
    assert metadata["requested_use_bf16"] is False
    assert metadata["effective_use_compile"] is False
    assert metadata["compile_status"] == "disabled_by_config"
    assert metadata["compile_attempts"] == []
    assert metadata["metadata_only"] is True


def test_tensor_alpha_batch_metadata_schema_records_seed_sensitivity():
    metadata = build_tensor_alpha_batch_metadata(
        planner="tensor_parallel_probe_alpha_batching",
        device="cuda:0",
        alpha_values_input=[0.2, 0.1],
        alpha_batches=[[0.1], [0.2]],
        sort_policy="ascending_alpha_before_batching",
        probe_enabled=True,
        probe_method="probe_tensor_super_memory(A=1)",
        alpha_max=0.2,
        probe_result_gb=0.5,
        total_memory_gb=8.0,
        target_memory_gb=6.4,
        target_memory_fraction=0.8,
        max_alphas_per_batch=1,
        seed_partition_sensitive=True,
        empty_cache_between_batches=True,
    )

    assert metadata["alpha_values_input"] == [0.2, 0.1]
    assert metadata["alpha_values_execution_order"] == [0.1, 0.2]
    assert metadata["seed_partition_sensitive"] is True
    assert metadata["metadata_only"] is True
    assert metadata["empty_cache_between_batches"] is True


def test_tensor_parallel_experiment_plan_resource_summary_is_metadata_only(tmp_path):
    config_path = tmp_path / "tensor_config.yaml"
    _write_tensor_config(config_path)

    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)

    assert plan.algorithm_spec.key == "bigamp_tensor_parallel"
    assert plan.resource_plan["probe_support"] == "A=1 tensor supergraph probe"
    assert plan.resource_plan["metadata_only"] is True
    assert plan.resource_plan["seed_partition_sensitive"] is True
    assert plan.resource_plan["memory_model"]["drives_execution"] is False
    assert plan.resource_plan["memory_model"]["probe_required"] is True
    assert plan.resource_plan["seed_policy"]["batch_partition_sensitive"] is True
    assert plan.resource_plan["seed_policy"]["automatic_rebatch_allowed"] is False
    assert plan.seed_policy_spec.policy_key == "legacy_tensor_parallel_batch_idx_seed"
    assert plan.resource_plan["config_effective"]["use_bf16"] is False
    assert plan.resource_plan["config_effective"]["use_compile"] is False


def test_tensor_parallel_partition_invariant_seed_policy_updates_resource_plan(tmp_path):
    config_path = tmp_path / "tensor_config.yaml"
    _write_tensor_config(config_path)
    text = config_path.read_text(encoding="utf-8").replace(
        "  use_bf16: false",
        "  use_bf16: false\n  seed_partition_policy: partition_invariant",
    )
    config_path.write_text(text, encoding="utf-8")

    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)

    assert plan.resource_plan["config_effective"]["seed_partition_policy"] == "partition_invariant"
    assert plan.resource_plan["seed_policy"]["policy_key"] == "tensor_parallel_partition_invariant_v1"
    assert plan.resource_plan["seed_policy"]["partition_invariant"] is True
    assert plan.resource_plan["seed_policy"]["batch_partition_sensitive"] is False
    assert plan.resource_plan["seed_policy"]["automatic_rebatch_allowed"] is True
    assert plan.resource_plan["replan_safety"]["automatic_rebatch_allowed"] is True
    assert plan.resource_plan["replan_safety"]["replan_implemented"] is False


@pytest.mark.parametrize(
    ("algorithm_id", "algorithm_key"),
    [
        (1, "bigamp"),
        (3, "agd"),
    ],
)
def test_matrix_partition_invariant_seed_policy_updates_resource_plan(
    tmp_path, algorithm_id, algorithm_key
):
    config_path = tmp_path / "matrix_config.yaml"
    _write_matrix_config(config_path, algorithm_id)

    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)

    assert plan.algorithm_spec.key == algorithm_key
    assert plan.resource_plan["config_effective"]["seed_partition_policy"] == "partition_invariant"
    assert plan.resource_plan["seed_policy"]["policy_key"] == "matrix_student_init_partition_invariant_v1"
    assert plan.resource_plan["seed_policy"]["random_streams"] == ["student_initialization"]
    assert plan.resource_plan["seed_policy"]["partition_invariant"] is True
    assert plan.resource_plan["seed_policy"]["batch_partition_sensitive"] is False
    assert plan.resource_plan["seed_policy"]["automatic_rebatch_allowed"] is True


def test_spreading_partition_invariant_seed_policy_updates_resource_plan(tmp_path):
    config_path = tmp_path / "spreading_config.yaml"
    _write_spreading_config(config_path)

    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)

    assert plan.algorithm_spec.key == "bigamp_spreading"
    assert plan.resource_plan["config_effective"]["seed_partition_policy"] == "partition_invariant"
    assert plan.resource_plan["seed_policy"]["policy_key"] == "spreading_partition_invariant_v2"
    assert plan.resource_plan["seed_policy"]["random_streams"] == [
        "student_initialization",
        "spreading_graph",
        "F_super",
        "restart_noise",
    ]
    assert "step" in plan.resource_plan["seed_policy"]["seed_inputs"]
    assert plan.resource_plan["seed_policy"]["partition_invariant"] is True
    assert plan.resource_plan["seed_policy"]["batch_partition_sensitive"] is False
    assert plan.resource_plan["seed_policy"]["automatic_rebatch_allowed"] is True


def test_spreading_partition_invariant_accepts_adaptive_restart(tmp_path):
    config_path = tmp_path / "spreading_config.yaml"
    _write_spreading_config(config_path, adaptive_restart=True)

    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)

    assert not any("adaptive_restart" in error for error in plan.errors)


def test_spreading_partition_invariant_data_prefix_is_batch_independent(tmp_path):
    config_path = tmp_path / "spreading_config.yaml"
    _write_spreading_config(config_path)
    config, _, _ = load_yaml_config(config_path)
    algorithm = BiGAMPSpreading(config, device=torch.device("cpu"))
    W_teacher = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 10.0
    X_teacher = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 10.0

    single = algorithm.create_spreading_data(W_teacher, X_teacher, [0.5], S=2, base_seed=17)
    combined = algorithm.create_spreading_data(W_teacher, X_teacher, [0.5, 0.8], S=2, base_seed=17)
    c_small = int(single.supergraph.C_per_alpha[0].item())

    assert torch.equal(single.supergraph.i_idx[:, :c_small], combined.supergraph.i_idx[:, :c_small])
    assert torch.equal(single.supergraph.j_idx[:, :c_small], combined.supergraph.j_idx[:, :c_small])
    assert torch.equal(single.F_super[:, :c_small], combined.F_super[:, :c_small])
    assert torch.equal(single.Y_super[:, :c_small], combined.Y_super[:, :c_small])


def test_tensor_memory_estimator_consumes_tensor_order_and_dims():
    estimator = MemoryEstimator()
    order3 = EstimationParams(
        N1=4,
        N2=5,
        M=2,
        S=1,
        alpha_values=[0.2],
        algorithm_key="bigamp_tensor_parallel",
        use_bf16=False,
        tensor_order=3,
        tensor_dims=(4, 5, 4),
    )
    order4 = EstimationParams(
        N1=4,
        N2=5,
        M=2,
        S=1,
        alpha_values=[0.2],
        algorithm_key="bigamp_tensor_parallel",
        use_bf16=False,
        tensor_order=4,
        tensor_dims=(4, 5, 4, 4),
    )

    assert estimator.estimate(order4).total_gb > estimator.estimate(order3).total_gb


@pytest.mark.parametrize(
    ("algorithm_key", "expected_component"),
    [
        ("agd", "Parameters & Gradients"),
        ("bigamp", "Dense Intermediate (N1×N2)"),
        ("bigamp_spreading", "SuperGraph Data"),
        ("bigamp_tensor_parallel", "Tensor Scatter/Gather Temporaries"),
    ],
)
def test_memory_estimates_expose_component_breakdown(algorithm_key, expected_component):
    estimator = MemoryEstimator()
    params = EstimationParams(
        N1=4,
        N2=5,
        M=2,
        S=1,
        alpha_values=[0.2],
        algorithm_key=algorithm_key,
        use_bf16=False,
        tensor_order=3,
        tensor_dims=(4, 5, 4),
    )

    estimate = estimator.estimate(params)

    assert estimate.breakdown
    assert expected_component in estimate.breakdown
    assert estimate.breakdown[expected_component] > 0.0


def test_execution_plan_batches_preserve_memory_breakdown():
    ParallelCoordinator = get_parallel_coordinator()
    coordinator = ParallelCoordinator(
        estimator=MemoryEstimator(),
        config=AllocationPresets.CONSERVATIVE,
    )
    plan = coordinator.plan_execution(
        EstimationParams(
            N1=4,
            N2=5,
            M=2,
            S=1,
            alpha_values=[0.1],
            algorithm_key="bigamp",
            use_bf16=False,
        )
    )

    assert plan.batches[0].memory_breakdown
    assert "Dense Intermediate (N1×N2)" in plan.batches[0].memory_breakdown


def test_parallel_coordinator_replan_is_hard_gated_by_seed_policy():
    ParallelCoordinator = get_parallel_coordinator()
    coordinator = ParallelCoordinator(
        estimator=MemoryEstimator(),
        config=AllocationPresets.CONSERVATIVE,
    )
    coordinator.current_plan = ExecutionPlan(
        mode=ParallelMode.FULL_PARALLEL,
        batches=[
            BatchConfig(
                sample_range=(0, 1),
                alpha_range=(0, 1),
                estimated_memory_gb=1.0,
                alpha_values=[0.2],
            )
        ],
        total_estimated_memory_gb=1.0,
        allocation_config=AllocationPresets.CONSERVATIVE,
        algorithm_key="bigamp_tensor_parallel",
        gpu_model="test",
        available_memory_gb=8.0,
    )

    with pytest.raises(RuntimeError, match="Automatic OOM replan is disabled"):
        coordinator.replan_with_safety()


def test_parallel_coordinator_records_effective_replan_policy():
    ParallelCoordinator = get_parallel_coordinator()
    coordinator = ParallelCoordinator(
        estimator=MemoryEstimator(),
        config=AllocationPresets.CONSERVATIVE,
    )
    params = EstimationParams(
        N1=2,
        N2=2,
        M=1,
        S=1,
        alpha_values=[0.1],
        algorithm_key="bigamp",
        seed_partition_policy="partition_invariant",
    )
    plan = coordinator.plan_execution(params)

    assert plan.seed_partition_policy == "partition_invariant"
    assert plan.replan_policy_key == "matrix_student_init_partition_invariant_v1"
    assert plan.automatic_rebatch_allowed is True
    assert plan.replan_implemented is False
    assert plan.plan_id.startswith("plan_")
    assert plan.estimation_params["algorithm_key"] == "bigamp"
    assert plan.estimation_params["alpha_values"] == [0.1]
    assert plan.estimation_params["seed_partition_policy"] == "partition_invariant"
    assert plan.replan_provenance["source"] == "initial_plan"
    assert plan.replan_provenance["num_batches"] == plan.num_batches
    assert plan.replan_provenance["batch_summary"][0]["alpha_values"] == [0.1]
    assert plan.replan_provenance["metadata_only"] is True

    coordinator.current_plan = plan
    with pytest.raises(NotImplementedError, match="Automatic replan is not implemented"):
        coordinator.replan_with_safety()


def test_runtime_resource_plan_reports_replan_provenance():
    config_path = Path("trials/active/matrix_bigamp_quick/config.yaml")
    config, _, _ = load_yaml_config(config_path)
    config.algorithm_params.seed_partition_policy = "partition_invariant"
    runner = ExperimentRunner(device=torch.device("cpu"), verbose=False)
    params = runner._estimation_params_for_config(config, config.scan.values)
    plan = runner.parallel_coordinator.plan_execution(params)

    report = runner._runtime_resource_plan_report(config, plan)

    assert report["replan_safety"]["plan_id"] == plan.plan_id
    assert report["replan_safety"]["estimation_params"]["algorithm_key"] == config.algorithm_key
    assert report["replan_safety"]["estimation_params"]["seed_partition_policy"] == "partition_invariant"
    assert report["replan_safety"]["replan_provenance"]["batch_summary"]
    assert report["replan_safety"]["replan_provenance"]["metadata_only"] is True
