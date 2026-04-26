from pathlib import Path

from matrix_factorization.cli import load_yaml_config
from matrix_factorization.core.planning import build_experiment_plan
from matrix_factorization.core.scan_planning import build_scan_plan
from matrix_factorization.core.parallel import EstimationParams, get_parallel_coordinator
from matrix_factorization.core.parallel.memory_estimator import MemoryEstimator
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
scan_mode: 1
alpha_scan: {start: 0.0, stop: 0.2, step: 0.1}
training: {samples_per_alpha: 1, max_steps: 2, max_epochs: 2}
algorithm_params: {use_compile: false, use_bf16: false}
output: {save_tensors: false, enable_heatmap: false}
""",
    )
    alpha_config, _, alpha_raw = load_yaml_config(alpha_path)
    alpha_plan = build_scan_plan(alpha_config)
    experiment_plan = build_experiment_plan(alpha_config, raw_yaml=alpha_raw, config_path=alpha_path)
    assert alpha_plan.scan_kind == "alpha"
    assert alpha_plan.num_points == 3
    assert experiment_plan.scan_plan.scan_kind == "alpha"

    steps_path = tmp_path / "steps.yaml"
    _write_config(
        steps_path,
        """
tensor_order: 2
algorithm: 1
teacher: 2
matrix: {N1: 4, N2: 4, M: 2}
scan_mode: 2
steps_scan:
  alpha: 0.3
  step_values: [1, 2, 4]
training: {samples_per_alpha: 1, max_steps: 4, max_epochs: 4}
algorithm_params: {use_compile: false, use_bf16: false}
output: {save_tensors: false, enable_heatmap: false}
""",
    )
    steps_config, _, steps_raw = load_yaml_config(steps_path)
    steps_plan = build_experiment_plan(steps_config, raw_yaml=steps_raw, config_path=steps_path).scan_plan
    assert steps_plan.scan_kind == "steps"
    assert steps_plan.execution_constraints.steps_reuse is True
    assert [point.max_steps for point in steps_plan.points] == [1, 2, 4]

    nested_path = tmp_path / "nested.yaml"
    _write_config(
        nested_path,
        """
tensor_order: 2
algorithm: 1
teacher: 2
matrix: {N1: 4, N2: 4, M: 2}
scan_mode: 3
nested_scan:
  sizes: [[4, 2], [6, 3]]
  alpha: {start: 0.0, stop: 0.1, step: 0.1}
training: {samples_per_alpha: 1, max_steps: 2, max_epochs: 2}
algorithm_params: {use_compile: false, use_bf16: false}
output: {save_tensors: false, enable_heatmap: false}
""",
    )
    nested_config, _, nested_raw = load_yaml_config(nested_path)
    nested_plan = build_experiment_plan(nested_config["base_config"] if "base_config" in nested_config else nested_config, raw_yaml=nested_raw, config_path=nested_path).scan_plan
    assert nested_plan.scan_kind == "nested"
    assert nested_plan.num_points == 4
    assert {group.group_id for group in nested_plan.grouping} == {"size:4x4_M2", "size:6x6_M3"}

    hyst_path = tmp_path / "hysteresis.yaml"
    _write_config(
        hyst_path,
        """
tensor_order: 2
algorithm: 1
teacher: 2
matrix: {N1: 4, N2: 4, M: 2}
scan_mode: 4
hysteresis_scan:
  init_overlaps: [0.0, 0.95]
  alpha: {start: 0.0, stop: 0.1, step: 0.1}
training: {samples_per_alpha: 1, max_steps: 2, max_epochs: 2}
algorithm_params: {use_compile: false, use_bf16: false}
output: {save_tensors: false, enable_heatmap: false}
""",
    )
    hyst_config, _, hyst_raw = load_yaml_config(hyst_path)
    hyst_plan = build_experiment_plan(hyst_config["base_config"], raw_yaml=hyst_raw, config_path=hyst_path).scan_plan
    assert hyst_plan.scan_kind == "hysteresis"
    assert hyst_plan.num_points == 4
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
    scan_plan = build_scan_plan(type("Config", (), {
        "scan": type("Scan", (), {"dimension": "alpha", "values": params.alpha_values})()
    })())
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
    ) == ["alpha:0", "alpha:1", "alpha:2"]
    assert resource_plan.batches[0].batch_axes
    assert resource_plan.sample_range_honored is False
