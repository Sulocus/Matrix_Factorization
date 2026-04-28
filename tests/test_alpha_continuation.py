import pytest
import torch

from matrix_factorization.cli import load_yaml_config
from matrix_factorization.core.contracts import AlgorithmResult, AlgorithmStateView, get_continuation_specs
from matrix_factorization.core.experiment.config import (
    AlgorithmParams,
    ExperimentConfig,
    MatrixParams,
    ScanConfig,
    SpreadingConfig,
    TrainingParams,
)
from matrix_factorization.core.experiment.data_factory import DataFactory
from matrix_factorization.core.experiment.runner import ExperimentRunner
from matrix_factorization.core.planning import build_experiment_plan


def _continuation_config():
    scan_spec = {
        "axes": {
            "alpha": {"path": "alpha", "values": [0.0, 0.1, 0.2]},
        },
        "continuation": {
            "enabled": True,
            "axis": "alpha",
            "order": "descending",
            "state_transfer": "full_algorithm_state",
            "observation_policy": "nested_prefix",
            "strict_state": True,
            "adaptive_controller_state": "reset",
        },
    }
    return ExperimentConfig(
        matrix=MatrixParams(N1=4, N2=4, M=2),
        training=TrainingParams(samples_per_alpha=1, max_steps=1, max_epochs=1),
        algorithm_key="bigamp",
        scan=ScanConfig(dimension="alpha", values=[0.0, 0.1, 0.2]),
        algorithm_params=AlgorithmParams(
            damping=0.5,
            noise_var=1e-5,
            use_compile=False,
            use_bf16=False,
        ),
        teacher_key="standard",
        experiment_name="alpha_continuation_test",
        scan_spec=scan_spec,
    )


def test_continuation_spec_is_registered():
    spec = get_continuation_specs()["alpha_descending_full_state"]
    assert spec.axis == "alpha"
    assert spec.order == "descending"
    assert spec.state_transfer == "full_algorithm_state"


def test_validate_accepts_scan_continuation(tmp_path):
    path = tmp_path / "continuation.yaml"
    path.write_text(
        """
tensor_order: 2
algorithm: 1
teacher: 2
matrix: {N1: 4, N2: 4, M: 2}
scan:
  axes:
    alpha: {path: alpha, values: [0.0, 0.1, 0.2]}
  continuation:
    enabled: true
    axis: alpha
    order: descending
    state_transfer: full_algorithm_state
    observation_policy: nested_prefix
    strict_state: true
    adaptive_controller_state: reset
training: {samples_per_alpha: 1, max_steps: 1, max_epochs: 1}
algorithm_params: {use_compile: false, use_bf16: false}
output: {save_tensors: false, enable_heatmap: false}
""",
        encoding="utf-8",
    )
    config, _, raw_yaml = load_yaml_config(path)
    plan = build_experiment_plan(config, raw_yaml=raw_yaml, config_path=path)
    assert not plan.errors
    chain = {item["path"]: item for item in plan.parameter_chain()}
    assert chain["scan.continuation.enabled"]["consumption_status"] == "effective"


def test_continuation_rejects_algorithm_without_state_capability():
    config = _continuation_config()
    config.algorithm_key = "bigamp_tensor"
    plan = build_experiment_plan(config)
    assert any("scan.continuation.enabled=true 目前不支持 algorithm" in error for error in plan.errors)


def test_continuation_rejects_general_spreading_until_metrics_support():
    config = _continuation_config()
    config.algorithm_key = "bigamp_spreading"
    config.spreading = SpreadingConfig(allow_intra_connection=True)
    plan = build_experiment_plan(config)
    assert any("SuperGraphDataGeneral" in error for error in plan.errors)


def test_nested_prefix_masks_are_monotone():
    factory = DataFactory(torch.device("cpu"))
    masks = factory.create_masks(
        N1=5,
        N2=5,
        M=2,
        alpha_values=[0.4, 0.2, 0.0],
        seed=123,
        nested_prefix=True,
    )
    high = masks[0].flatten()
    low = masks[1].flatten()
    zero = masks[2].flatten()
    assert torch.all(low <= high)
    assert int(zero.sum().item()) == 0


def test_continuation_runs_alpha_descending_and_passes_state(monkeypatch):
    config = _continuation_config()
    calls = []

    monkeypatch.setattr(ExperimentRunner, "_get_algorithm", lambda self, cfg: object())

    def fake_run_algorithm_result(
        self,
        algorithm,
        config,
        data,
        step_callback,
        initial_state=None,
        return_continuation_state=False,
        continuation_context=None,
    ):
        alpha = float(data.alpha_values[0])
        calls.append({
            "alpha": alpha,
            "source": None if initial_state is None else initial_state.alpha,
            "return_state": return_continuation_state,
        })
        W = torch.full((1, 1, config.matrix.N1, config.matrix.M), alpha)
        X = torch.full((1, 1, config.matrix.M, config.matrix.N2), alpha)
        metrics = {
            "Q_Y_mean": alpha,
            "Q_Y_std": 0.0,
            "Q_W_mean": alpha,
            "Q_W_std": 0.0,
            "Q_X_mean": alpha,
            "Q_X_std": 0.0,
        }
        state = AlgorithmStateView(
            student_factors={"W": W[0], "X": X[0]},
            step_index=1,
            alpha=alpha,
            metadata={"test_state": True},
        )
        return AlgorithmResult(
            metrics_by_alpha={alpha: metrics},
            matrix_factors={"W_students": W, "X_students": X},
            continuation_state=state,
            metadata={
                "algorithm_key": "bigamp",
                "result_contract": "legacy_matrix_result",
                "result_kind": "matrix_factors",
                "result_source": "fake_continuation_test",
                "matrix_factors_available": True,
            },
        )

    monkeypatch.setattr(ExperimentRunner, "_run_algorithm_result", fake_run_algorithm_result)

    result = ExperimentRunner(device=torch.device("cpu"), verbose=False).run(
        config,
        output_options={"save_tensors": False, "enable_heatmap": False},
    )

    assert [call["alpha"] for call in calls] == [0.2, 0.1, 0.0]
    assert [call["source"] for call in calls] == [None, 0.2, 0.1]
    assert all(call["return_state"] for call in calls)
    assert result.scan_dimension == "scan"
    assert len(result.result_cube.points) == 3
    assert result.metadata.contract["continuation"]["enabled"] is True
    assert result.metadata.contract["continuation_points"]


def test_continuation_strict_state_requires_algorithm_state(monkeypatch):
    config = _continuation_config()
    monkeypatch.setattr(ExperimentRunner, "_get_algorithm", lambda self, cfg: object())

    def fake_missing_state(self, algorithm, config, data, step_callback, **kwargs):
        alpha = float(data.alpha_values[0])
        W = torch.zeros(1, 1, config.matrix.N1, config.matrix.M)
        X = torch.zeros(1, 1, config.matrix.M, config.matrix.N2)
        metrics = {"Q_Y_mean": 0.0, "Q_Y_std": 0.0, "Q_W_mean": 0.0, "Q_W_std": 0.0, "Q_X_mean": 0.0, "Q_X_std": 0.0}
        return AlgorithmResult(
            metrics_by_alpha={alpha: metrics},
            matrix_factors={"W_students": W, "X_students": X},
            metadata={"algorithm_key": "bigamp", "result_contract": "legacy_matrix_result"},
        )

    monkeypatch.setattr(ExperimentRunner, "_run_algorithm_result", fake_missing_state)

    with pytest.raises(RuntimeError, match="continuation_state"):
        ExperimentRunner(device=torch.device("cpu"), verbose=False).run(
            config,
            output_options={"save_tensors": False, "enable_heatmap": False},
        )
