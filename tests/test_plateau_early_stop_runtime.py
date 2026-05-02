import pytest
import torch

from matrix_factorization.core.experiment.config import (
    AlgorithmParams,
    ExperimentConfig,
    MatrixParams,
    ScanConfig,
    SpreadingConfig,
    TrainingParams,
)
import matrix_factorization.core.experiment.runner as runner_module
from matrix_factorization.core.experiment.runner import ExperimentRunner
import matrix_factorization.modules.algorithms.bigamp.spreading as spreading_module
from matrix_factorization.modules.algorithms.bigamp.spreading import BiGAMPSpreading


def _runtime_fixture(*, max_steps=6, window=1, min_steps=0, patience=1):
    config = ExperimentConfig(
        matrix=MatrixParams(N1=2, N2=2, M=1),
        training=TrainingParams(samples_per_alpha=1, max_steps=max_steps, max_epochs=max_steps),
        algorithm_key="bigamp_spreading",
        scan=ScanConfig(dimension="alpha", values=[0.5, 1.0]),
        spreading=SpreadingConfig(f_distribution="ising", onsager_correction=False, chunk_size=0),
        algorithm_params=AlgorithmParams(
            damping=0.5,
            noise_var=1e-6,
            use_compile=False,
            use_bf16=False,
            precision_profile="safe",
            use_metric_plateau_stop=True,
            plateau_check_interval=1,
            plateau_window_steps=window,
            plateau_patience=patience,
            plateau_abs_tol=1e-8,
            plateau_rel_tol=0.01,
            plateau_min_steps=min_steps,
            plateau_monitor="teacher_latent_overlap_qw_qx",
        ),
        teacher_key="standard",
    )
    device = torch.device("cpu")
    algorithm = BiGAMPSpreading(config, device=device)
    W_teacher = torch.tensor([[1.0], [-1.0]], device=device)
    X_teacher = torch.tensor([[0.5, -0.5]], device=device)
    data = algorithm.create_spreading_data(
        W_teacher,
        X_teacher,
        [0.5, 1.0],
        S=1,
        base_seed=17,
    )
    return algorithm, data, W_teacher, X_teacher


def test_runtime_plateau_breaks_loop_and_records_metadata(monkeypatch):
    algorithm, data, W_teacher, X_teacher = _runtime_fixture(max_steps=6, window=1, min_steps=0, patience=1)
    calls = []

    def no_op_step(**kwargs):
        calls.append(1)
        return (
            kwargs["W_flat"],
            kwargs["X_flat"],
            kwargs["W_var_flat"],
            kwargs["X_var_flat"],
            kwargs.get("prev_s"),
            kwargs.get("prev_svar"),
        )

    monkeypatch.setattr(spreading_module, "bigamp_step_disjoint_union_flat_legacy_fast", no_op_step)

    result = algorithm.train_batch_result(
        algorithm_key="bigamp_spreading",
        W_teacher=W_teacher,
        X_teacher=X_teacher,
        Y_teacher=torch.empty(0),
        masks=None,
        alpha_values=[0.5, 1.0],
        seed=17,
        spreading_data=data,
    )

    assert len(calls) == 2
    summary = result.diagnostics["metric_plateau_stop"]
    assert summary["enabled"] is True
    assert summary["teacher_assisted"] is True
    assert summary["any_stopped_early"] is True
    assert summary["batches"][0]["stop_reason"] == "metric_self_convergence"
    assert summary["batches"][0]["steps_run"] == 2
    metadata = result.metadata["execution_metadata"]["metric_plateau_stop"]
    assert metadata["teacher_assisted"] is True
    assert metadata["last_run"]["batches"][0]["stop_step"] == 2
    assert metadata["last_run"]["batches"][0]["config"]["strategy"] == "self_convergence_window_trend_decay"
    assert metadata["last_run"]["batches"][0]["per_alpha_status"][0]["ready_step"] == 2
    metrics = metadata["last_run"]["batches"][0]["per_alpha_status"][0]["metrics"]
    assert "Q_W" in metrics
    assert "window_abs_range" in metrics["Q_W"]
    assert "projected_abs_change" in metrics["Q_W"]
    assert "stable_trend" in metrics["Q_W"]


def test_runtime_plateau_ignores_deprecated_min_steps_gate(monkeypatch):
    algorithm, data, W_teacher, X_teacher = _runtime_fixture(max_steps=6, window=1, min_steps=100, patience=1)
    calls = []

    def no_op_step(**kwargs):
        calls.append(1)
        return (
            kwargs["W_flat"],
            kwargs["X_flat"],
            kwargs["W_var_flat"],
            kwargs["X_var_flat"],
            kwargs.get("prev_s"),
            kwargs.get("prev_svar"),
        )

    monkeypatch.setattr(spreading_module, "bigamp_step_disjoint_union_flat_legacy_fast", no_op_step)

    result = algorithm.train_batch_result(
        algorithm_key="bigamp_spreading",
        W_teacher=W_teacher,
        X_teacher=X_teacher,
        Y_teacher=torch.empty(0),
        masks=None,
        alpha_values=[0.5, 1.0],
        seed=17,
        spreading_data=data,
    )

    summary = result.diagnostics["metric_plateau_stop"]["batches"][0]
    assert len(calls) == 2
    assert summary["stop_step"] == 2
    assert summary["config"]["min_steps"] == 100
    assert summary["config"]["effective_min_steps"] == 0
    assert summary["config"]["effective_window_steps"] == 1


def test_runtime_plateau_nonfinite_state_raises(monkeypatch):
    algorithm, data, W_teacher, X_teacher = _runtime_fixture(max_steps=3, window=1, min_steps=1, patience=1)

    def nan_step(**kwargs):
        W_flat = kwargs["W_flat"].clone()
        W_flat.fill_(float("nan"))
        return (
            W_flat,
            kwargs["X_flat"],
            kwargs["W_var_flat"],
            kwargs["X_var_flat"],
            kwargs.get("prev_s"),
            kwargs.get("prev_svar"),
        )

    monkeypatch.setattr(spreading_module, "bigamp_step_disjoint_union_flat_legacy_fast", nan_step)

    with pytest.raises(FloatingPointError, match="non-finite"):
        algorithm.train_batch_alphas(
            W_teacher=W_teacher,
            X_teacher=X_teacher,
            Y_teacher=torch.empty(0),
            masks=None,
            alpha_values=[0.5, 1.0],
            seed=17,
            spreading_data=data,
        )


def test_runner_records_compute_calibration_metadata(monkeypatch):
    class FakeSampler:
        enabled = True

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.steps = []
            self.started = False

        def start(self):
            self.started = True

        def mark_step(self, step):
            self.steps.append(step)

        def stop(self, *, steps_run=None, configured_max_steps=None):
            return {
                "enabled": True,
                "available": True,
                "partial": False,
                "target_gpu_utilization": self.kwargs["target_gpu_utilization"],
                "warmup_steps": self.kwargs["warmup_steps"],
                "measure_steps": self.kwargs["measure_steps"],
                "min_samples": self.kwargs["min_samples"],
                "min_measure_seconds": self.kwargs["min_measure_seconds"],
                "num_samples": 2,
                "small_kernel_limited": False,
                "measurement_quality": "complete",
                "mean_gpu_utilization": 0.91,
                "peak_allocated_gb": 0.25,
                "steps_run": steps_run,
                "configured_max_steps": configured_max_steps,
                "sec_per_step": 0.01,
                "metadata_only": True,
            }

    def no_op_step(**kwargs):
        return (
            kwargs["W_flat"],
            kwargs["X_flat"],
            kwargs["W_var_flat"],
            kwargs["X_var_flat"],
            kwargs.get("prev_s"),
            kwargs.get("prev_svar"),
        )

    monkeypatch.setattr(runner_module, "GPUComputeSampler", FakeSampler)
    monkeypatch.setattr(spreading_module, "bigamp_step_disjoint_union_flat_legacy_fast", no_op_step)
    config = ExperimentConfig(
        matrix=MatrixParams(N1=2, N2=2, M=1),
        training=TrainingParams(samples_per_alpha=1, max_steps=2, max_epochs=2),
        algorithm_key="bigamp_spreading",
        scan=ScanConfig(dimension="alpha", values=[0.85, 0.90]),
        spreading=SpreadingConfig(f_distribution="ising", onsager_correction=False, chunk_size=0),
        algorithm_params=AlgorithmParams(
            damping=0.5,
            noise_var=1e-6,
            use_compile=False,
            use_bf16=False,
            precision_profile="safe",
            use_metric_plateau_stop=True,
            plateau_check_interval=1,
            plateau_window_steps=1,
            plateau_patience=1,
            plateau_abs_tol=1e-8,
            plateau_rel_tol=0.01,
            plateau_min_steps=2,
            plateau_monitor="teacher_latent_overlap_qw_qx",
            seed_partition_policy="partition_invariant",
        ),
        teacher_key="standard",
        scan_spec={"axes": {"alpha": {"path": "alpha", "values": [0.85, 0.90]}}},
    )

    result = ExperimentRunner(device=torch.device("cpu"), verbose=False).run(
        config,
        output_options={"save_tensors": False, "storage_mode": "lightweight", "enable_heatmap": False},
    )

    batch_summary = result.metadata.contract["algorithm_result_batches"][0]
    runtime = batch_summary["execution_metadata"]["compute_calibration_runtime"]
    assert runtime["mean_gpu_utilization"] == 0.91
    assert runtime["steps_run"] == 2
    assert batch_summary["batching_metadata"]["planner"] == "metric_plateau_compute_aware_alpha_local"
    resource_batch = result.metadata.contract["runtime_resource_plan"]["batches"][0]
    assert resource_batch["batching_metadata"]["compute_calibration"]["warmup_steps"] == 20
    assert resource_batch["batching_metadata"]["compute_calibration"]["min_samples"] == 5
    assert resource_batch["batching_metadata"]["compute_calibration"]["min_measure_seconds"] == 3.0
