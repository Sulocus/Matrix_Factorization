from matrix_factorization.core.experiment.config import (
    AlgorithmParams,
    ExperimentConfig,
    MatrixParams,
    ScanConfig,
    TrainingParams,
)
from matrix_factorization.core.experiment.runner import ExperimentRunner


def test_canonical_scan_runtime_resource_plan_aligns_work_items_and_cube_points():
    config = ExperimentConfig(
        matrix=MatrixParams(N1=3, N2=3, M=1),
        training=TrainingParams(samples_per_alpha=1, max_steps=1, max_epochs=1),
        algorithm_key="bigamp",
        scan=ScanConfig(dimension="alpha", values=[0.0]),
        algorithm_params=AlgorithmParams(use_compile=False, use_bf16=False),
        scan_spec={
            "axes": {
                "size": {
                    "kind": "composite",
                    "values": {
                        "N3_M1": {"matrix.N1": 3, "matrix.N2": 3, "matrix.M": 1},
                        "N4_M1": {"matrix.N1": 4, "matrix.N2": 4, "matrix.M": 1},
                    },
                },
                "alpha": {"path": "alpha", "values": [0.0, 0.1]},
            }
        },
    )

    result = ExperimentRunner(device=None, verbose=False).run(
        config,
        output_options={
            "save_tensors": False,
            "enable_heatmap": False,
            "experiment_plan": {"preflight": "copied"},
        },
    )
    resource_plan = result.metadata.contract["runtime_resource_plan"]
    work_item_ids = {
        item["scan_point_id"]
        for batch in resource_plan["batches"]
        for item in batch["work_items"]
    }

    assert resource_plan["num_work_items"] == 4
    assert "resource_execution_plan" not in resource_plan
    assert work_item_ids == set(result.result_cube.points)
    assert resource_plan["groups"]
    assert resource_plan["max_estimated_allocated_gb"] >= 0.0
    assert all("dominant_stage" in batch for batch in resource_plan["batches"])
