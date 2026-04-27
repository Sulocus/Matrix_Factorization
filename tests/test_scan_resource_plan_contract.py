from pathlib import Path

from matrix_factorization.cli import load_yaml_config
from matrix_factorization.core.planning import build_experiment_plan
from matrix_factorization.core.scan_planning import build_scan_plan
from matrix_factorization.core.parallel import EstimationParams, get_parallel_coordinator
from matrix_factorization.core.parallel.memory_estimator import MemoryEstimator
from matrix_factorization.core.parallel.resource_execution import build_scan_resource_execution_plan
from matrix_factorization.core.contracts import get_batching_specs


def _write_config(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def test_scan_plan_alpha_steps_nested_and_hysteresis(tmp_path):
    alpha_path = tmp_path / "alpha.yaml"
    _write_config(
        alpha_path,
        """
tensor_order: 2
algorithm: 1
teacher: 2
matrix: {N1: 4, N2: 4, M: 2}
scan:
  axes:
    alpha: {path: alpha, values: {start: 0.0, stop: 0.2, step: 0.1}}
training: {samples_per_alpha: 1, max_steps: 2, max_epochs: 2}
algorithm_params: {use_compile: false, use_bf16: false}
output: {save_tensors: false, enable_heatmap: false}
""",
    )
    alpha_config, _, alpha_raw = load_yaml_config(alpha_path)
    alpha_plan = build_scan_plan(alpha_config)
    experiment_plan = build_experiment_plan(alpha_config, raw_yaml=alpha_raw, config_path=alpha_path)
    assert alpha_plan.scan_kind == "parameter_space"
    assert alpha_plan.num_points == 3
    assert alpha_plan.execution_constraints.sample_folding is False
    assert experiment_plan.scan_plan.scan_kind == "parameter_space"

    steps_path = tmp_path / "steps.yaml"
    _write_config(
        steps_path,
        """
tensor_order: 2
algorithm: 1
teacher: 2
matrix: {N1: 4, N2: 4, M: 2}
scan:
  axes:
    alpha: {path: alpha, values: [0.3]}
    max_steps: {path: max_steps, values: [1, 2, 4]}
training: {samples_per_alpha: 1, max_steps: 4, max_epochs: 4}
algorithm_params: {use_compile: false, use_bf16: false}
output: {save_tensors: false, enable_heatmap: false}
""",
    )
    steps_config, _, steps_raw = load_yaml_config(steps_path)
    steps_plan = build_experiment_plan(steps_config, raw_yaml=steps_raw, config_path=steps_path).scan_plan
    assert steps_plan.scan_kind == "parameter_space"
    assert steps_plan.execution_constraints.steps_reuse is True
    assert steps_plan.execution_constraints.sample_folding is False
    assert [point.max_steps for point in steps_plan.points] == [1, 2, 4]

    nested_path = tmp_path / "nested.yaml"
    _write_config(
        nested_path,
        """
tensor_order: 2
algorithm: 1
teacher: 2
matrix: {N1: 4, N2: 4, M: 2}
scan:
  axes:
    size:
      kind: composite
      values:
        N4_M2: {matrix.N1: 4, matrix.N2: 4, matrix.M: 2}
        N6_M3: {matrix.N1: 6, matrix.N2: 6, matrix.M: 3}
    alpha: {path: alpha, values: {start: 0.0, stop: 0.1, step: 0.1}}
training: {samples_per_alpha: 1, max_steps: 2, max_epochs: 2}
algorithm_params: {use_compile: false, use_bf16: false}
output: {save_tensors: false, enable_heatmap: false}
""",
    )
    nested_config, _, nested_raw = load_yaml_config(nested_path)
    nested_plan = build_experiment_plan(nested_config, raw_yaml=nested_raw, config_path=nested_path).scan_plan
    assert nested_plan.scan_kind == "parameter_space"
    assert nested_plan.num_points == 4
    assert nested_plan.execution_constraints.sample_folding is False
    assert {group.group_id for group in nested_plan.grouping} == {"size=N4_M2", "size=N6_M3"}

    hyst_path = tmp_path / "hysteresis.yaml"
    _write_config(
        hyst_path,
        """
tensor_order: 2
algorithm: 1
teacher: 2
matrix: {N1: 4, N2: 4, M: 2}
scan:
  axes:
    init:
      kind: composite
      values:
        cold: {algorithm_params.init_mode: random}
        warm_095: {algorithm_params.init_mode: teacher, algorithm_params.init_overlap: 0.95}
    alpha: {path: alpha, values: {start: 0.0, stop: 0.1, step: 0.1}}
training: {samples_per_alpha: 1, max_steps: 2, max_epochs: 2}
algorithm_params: {use_compile: false, use_bf16: false}
output: {save_tensors: false, enable_heatmap: false}
""",
    )
    hyst_config, _, hyst_raw = load_yaml_config(hyst_path)
    hyst_plan = build_experiment_plan(hyst_config, raw_yaml=hyst_raw, config_path=hyst_path).scan_plan
    assert hyst_plan.scan_kind == "parameter_space"
    assert hyst_plan.num_points == 4
    assert hyst_plan.execution_constraints.sample_folding is False
    assert {point.init_overlap for point in hyst_plan.points} == {0.0, 0.95}


def test_resource_execution_plan_exposes_work_items_for_alpha_batches():
    ParallelCoordinator = get_parallel_coordinator()
    params = EstimationParams(
        N1=4,
        N2=4,
        M=2,
        S=2,
        alpha_values=[0.0, 0.1, 0.2],
        algorithm_key="bigamp",
        use_compile=False,
        use_bf16=False,
        seed_partition_policy="partition_invariant",
    )
    scan_plan = build_scan_plan({"scan": {"axes": {"alpha": {"path": "alpha", "values": params.alpha_values}}}})
    coordinator = ParallelCoordinator(estimator=MemoryEstimator())
    execution_plan, resource_plan = coordinator.plan_resource_execution(
        scan_plan,
        params,
        batching_spec=get_batching_specs()["bigamp"],
    )

    assert execution_plan.num_batches == resource_plan.num_batches
    assert resource_plan.num_work_items == len(params.alpha_values)
    assert sorted(
        item.scan_point_id
        for batch in resource_plan.batches
        for item in batch.work_items
    ) == ["p0000", "p0001", "p0002"]
    assert resource_plan.batches[0].batch_axes
    assert "sample" not in resource_plan.batches[0].batch_axes
    assert resource_plan.sample_range_honored is False


def test_resource_execution_plan_does_not_fold_nested_or_hysteresis_groups():
    ParallelCoordinator = get_parallel_coordinator()
    coordinator = ParallelCoordinator(estimator=MemoryEstimator(apply_calibration=False))
    params = EstimationParams(
        N1=4,
        N2=4,
        M=2,
        S=2,
        alpha_values=[0.0, 0.1],
        algorithm_key="bigamp",
        use_compile=False,
        use_bf16=False,
        seed_partition_policy="partition_invariant",
    )
    nested_plan = build_scan_plan({"scan": {"axes": {
        "size": {
            "kind": "composite",
            "values": {
                "N4_M2": {"matrix.N1": 4, "matrix.N2": 4, "matrix.M": 2},
                "N6_M3": {"matrix.N1": 6, "matrix.N2": 6, "matrix.M": 3},
            },
        },
        "alpha": {"path": "alpha", "values": [0.0, 0.1]},
    }}})
    _, resource_plan = coordinator.plan_resource_execution(
        nested_plan,
        params,
        batching_spec=get_batching_specs()["bigamp"],
    )

    assert resource_plan.num_work_items == 4
    for batch in resource_plan.batches:
        assert len({item.output_group_id for item in batch.work_items}) == 1
        assert "scan_axis" not in batch.batch_axes

    hysteresis_plan = build_scan_plan({"scan": {"axes": {
        "init": {
            "kind": "composite",
            "values": {
                "cold": {"algorithm_params.init_mode": "random"},
                "warm_095": {"algorithm_params.init_mode": "teacher", "algorithm_params.init_overlap": 0.95},
            },
        },
        "alpha": {"path": "alpha", "values": [0.0, 0.1]},
    }}})
    _, hyst_resource_plan = coordinator.plan_resource_execution(
        hysteresis_plan,
        params,
        batching_spec=get_batching_specs()["bigamp"],
    )
    assert hyst_resource_plan.num_work_items == 4
    for batch in hyst_resource_plan.batches:
        assert len({item.output_group_id for item in batch.work_items}) == 1
        assert "scan_axis" not in batch.batch_axes


def test_steps_resource_plan_isolates_step_budget_points():
    ParallelCoordinator = get_parallel_coordinator()
    coordinator = ParallelCoordinator(estimator=MemoryEstimator(apply_calibration=False))
    params = EstimationParams(
        N1=4,
        N2=4,
        M=2,
        S=2,
        alpha_values=[0.3],
        algorithm_key="bigamp",
        use_compile=False,
        use_bf16=False,
        seed_partition_policy="partition_invariant",
    )
    steps_plan = build_scan_plan({"scan": {"axes": {
        "alpha": {"path": "alpha", "values": [0.3]},
        "max_steps": {"path": "max_steps", "values": [1, 2, 4]},
    }}})
    _, resource_plan = coordinator.plan_resource_execution(
        steps_plan,
        params,
        batching_spec=get_batching_specs()["bigamp"],
    )

    assert resource_plan.num_batches == 3
    assert [batch.work_items[0].axis_values["max_steps"] for batch in resource_plan.batches] == [1, 2, 4]
    assert all(batch.batch_axes == ["point"] for batch in resource_plan.batches)


def test_scan_resource_plan_applies_teacher_config_overrides(tmp_path):
    config_path = tmp_path / "mean_scale_scan.yaml"
    _write_config(
        config_path,
        """
tensor_order: 2
algorithm: 2
teacher: 2
teacher_config:
  init_distribution: 3
  mean_scale: 0.0
matrix: {N1: 4, N2: 4, M: 2}
scan:
  axes:
    mean_scale: {path: teacher_config.mean_scale, values: [0.0, 0.4]}
    alpha: {path: alpha, values: [0.0, 0.1]}
training: {samples_per_alpha: 1, max_steps: 2}
algorithm_params: {use_compile: false, use_bf16: false}
spreading: {f_distribution: 1, seed: 123, onsager_correction: false, chunk_size: 0}
output: {save_tensors: false, enable_heatmap: false}
""",
    )
    config, _, _ = load_yaml_config(config_path)
    scan_plan = build_scan_plan(config)
    ParallelCoordinator = get_parallel_coordinator()
    plan = build_scan_resource_execution_plan(
        scan_plan=scan_plan,
        base_config=config,
        coordinator=ParallelCoordinator(estimator=MemoryEstimator(apply_calibration=False)),
        batching_spec=get_batching_specs()["bigamp_spreading"],
    )

    assert plan.preflight_errors == []
    assert [group.coordinates["mean_scale"] for group in plan.groups] == [0.0, 0.4]
    assert [group.effective_overrides["teacher_config.mean_scale"] for group in plan.groups] == [0.0, 0.4]
