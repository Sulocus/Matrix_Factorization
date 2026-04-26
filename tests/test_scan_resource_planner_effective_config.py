import pytest

from matrix_factorization.core.experiment.config import (
    AlgorithmParams,
    ExperimentConfig,
    MatrixParams,
    ScanConfig,
    SpreadingConfig,
    TrainingParams,
)
from matrix_factorization.core.contracts import get_batching_specs
from matrix_factorization.core.parallel.memory_estimator import MemoryEstimator
from matrix_factorization.core.parallel.parallel_coordinator import ParallelCoordinator
from matrix_factorization.core.parallel.resource_execution import build_scan_resource_execution_plan
from matrix_factorization.core.scan_planning import build_scan_plan


def _mixed_config():
    scan_spec = {
        "axes": {
            "size": {
                "kind": "composite",
                "values": {
                    "N4_M1": {"matrix.N1": 4, "matrix.N2": 4, "matrix.M": 1},
                    "N8_M2": {"matrix.N1": 8, "matrix.N2": 8, "matrix.M": 2},
                },
            },
            "damping": {"path": "algorithm_params.damping", "values": [0.4, 0.6]},
            "onsager": {"path": "spreading.onsager_correction", "values": [False, True]},
            "init": {
                "kind": "composite",
                "values": {
                    "cold": {"algorithm_params.init_mode": "random"},
                    "warm_095": {"algorithm_params.init_mode": "teacher", "algorithm_params.init_overlap": 0.95},
                },
            },
            "alpha": {"path": "alpha", "values": [0.0, 0.1]},
        }
    }
    return ExperimentConfig(
        matrix=MatrixParams(N1=4, N2=4, M=1),
        training=TrainingParams(samples_per_alpha=1, max_steps=1, max_epochs=1),
        algorithm_key="bigamp_spreading",
        scan=ScanConfig(dimension="alpha", values=[0.0, 0.1]),
        algorithm_params=AlgorithmParams(use_compile=False, use_bf16=False),
        spreading=SpreadingConfig(f_distribution="rademacher", tensor_order=2, chunk_size=0),
        scan_spec=scan_spec,
        experiment_name="resource_mixed",
    )


def test_scan_resource_planner_builds_effective_params_per_non_alpha_group():
    config = _mixed_config()
    plan = build_scan_plan(config)
    resource_plan = build_scan_resource_execution_plan(
        scan_plan=plan,
        base_config=config,
        coordinator=ParallelCoordinator(estimator=MemoryEstimator(apply_calibration=False)),
        batching_spec=get_batching_specs()[config.algorithm_key],
    )

    assert len(resource_plan.groups) == 16
    assert resource_plan.num_work_items == 32
    assert resource_plan.preflight_errors == []
    sizes = {(g.estimation_params["N1"], g.estimation_params["M"]) for g in resource_plan.groups}
    assert sizes == {(4, 1), (8, 2)}
    assert all("scan_axis" not in batch.batch_axes for batch in resource_plan.batches)
    assert all({item.output_group_id for item in batch.work_items} for batch in resource_plan.batches)


def test_max_steps_axis_is_isolated_from_alpha_folding():
    config = ExperimentConfig(
        matrix=MatrixParams(N1=4, N2=4, M=1),
        training=TrainingParams(samples_per_alpha=1, max_steps=3, max_epochs=3),
        algorithm_key="bigamp",
        scan=ScanConfig(dimension="alpha", values=[0.2]),
        algorithm_params=AlgorithmParams(use_compile=False, use_bf16=False),
        scan_spec={
            "axes": {
                "alpha": {"path": "alpha", "values": [0.2]},
                "max_steps": {"path": "max_steps", "values": [1, 2, 3]},
            }
        },
    )

    resource_plan = build_scan_resource_execution_plan(
        scan_plan=build_scan_plan(config),
        base_config=config,
        coordinator=ParallelCoordinator(estimator=MemoryEstimator(apply_calibration=False)),
        batching_spec=get_batching_specs()[config.algorithm_key],
    )

    assert resource_plan.num_batches == 3
    assert all(batch.batch_axes == ["point"] for batch in resource_plan.batches)
    assert [batch.work_items[0].axis_values["max_steps"] for batch in resource_plan.batches] == [1, 2, 3]
