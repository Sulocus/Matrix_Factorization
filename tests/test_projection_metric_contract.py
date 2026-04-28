import json
from types import SimpleNamespace

import pytest
import torch

from matrix_factorization.core.experiment.config import (
    AlgorithmParams,
    ExperimentConfig,
    MatrixParams,
    ScanConfig,
    TrainingParams,
)
from matrix_factorization.core.experiment.result import ExperimentMetadata, ExperimentResult
from matrix_factorization.modules.metrics.contract_compute import compute_matrix_metric_payload
from matrix_factorization.modules.metrics.overlap import (
    cos_overlap_root,
    projection_abs,
    projection_abs_diagnostics,
    sign_aligned_projection_abs,
)
from matrix_factorization.modules.metrics.qy_unobserved import compute_qy_split
from matrix_factorization.modules.metrics.spreading import compute_all_metrics_spreading
from matrix_factorization.modules.metrics.tensor_metrics import (
    compute_tensor_factor_projection_overlaps,
    compute_tensor_projection_abs,
)
from matrix_factorization.modules.teachers.random_spreading import SpreadingData, compute_sparse_Y


def test_projection_abs_uses_teacher_norm_squared_and_does_not_clip():
    teacher = torch.tensor([1.0, -2.0, 0.5])

    assert projection_abs(teacher, teacher) == pytest.approx(1.0)
    assert projection_abs(-teacher, teacher) == pytest.approx(1.0)
    assert projection_abs(2.0 * teacher, teacher) == pytest.approx(2.0)

    diagnostics = projection_abs_diagnostics(torch.ones(3), torch.zeros(3))
    assert diagnostics["value"] == 0.0
    assert diagnostics["degenerate_teacher_norm"] is True
    assert diagnostics["clipped"] is False
    assert diagnostics["formula"] == "absolute_projection_teacher_norm_squared"


def test_sign_aligned_projection_abs_quotients_channel_sign_gauge():
    W_teacher = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    W_student = W_teacher.clone()
    W_student[:, 1] *= -1.0

    X_teacher = torch.tensor([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]])
    X_student = X_teacher.clone()
    X_student[0, :] *= -1.0

    assert projection_abs(W_student, W_teacher) < 1.0
    assert projection_abs(X_student, X_teacher) < 1.0
    assert sign_aligned_projection_abs(W_student, W_teacher, latent_axis=-1) == pytest.approx(1.0)
    assert sign_aligned_projection_abs(X_student, X_teacher, latent_axis=0) == pytest.approx(1.0)
    assert sign_aligned_projection_abs(2.0 * W_student, W_teacher, latent_axis=-1) == pytest.approx(2.0)


def test_matrix_projection_payload_matches_hand_calculation():
    W_teacher = torch.tensor([[1.0], [2.0]])
    X_teacher = torch.tensor([[3.0, 4.0]])
    W_student = 2.0 * W_teacher
    X_student = X_teacher.clone()
    Y_teacher = W_teacher @ X_teacher
    mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    data = SimpleNamespace(
        spreading_data=None,
        W_teacher=W_teacher,
        X_teacher=X_teacher,
        Y_teacher=Y_teacher,
        M=1,
        masks=mask,
    )

    metrics = compute_matrix_metric_payload(
        W_student.unsqueeze(0),
        X_student.unsqueeze(0),
        data,
    )

    assert metrics["Q_W_mean"] == pytest.approx(5.0)
    assert metrics["Q_X_mean"] == pytest.approx(12.5)
    assert metrics["Q_W_SIGN_GAUGE_mean"] == pytest.approx(5.0)
    assert metrics["Q_X_SIGN_GAUGE_mean"] == pytest.approx(12.5)
    assert metrics["Q_W_PROJ_ABS_mean"] == pytest.approx(2.0)
    assert metrics["Q_X_PROJ_ABS_mean"] == pytest.approx(1.0)
    assert metrics["Q_Y_mean"] == pytest.approx(2.0)
    assert metrics["FIT_Y_mean"] == pytest.approx(0.0)
    assert metrics["NMSE_Y_mean"] == pytest.approx(1.0)
    assert metrics["Q_Y_PROJ_ABS_mean"] == pytest.approx(2.0)
    assert metrics["Q_Y_observed_mean"] == pytest.approx(2.0)
    assert metrics["Q_Y_unobserved_mean"] == pytest.approx(2.0)
    assert metrics["Q_W_COS_ROOT_mean"] == pytest.approx(1.0)
    assert metrics["Q_X_COS_ROOT_mean"] == pytest.approx(1.0)
    assert metrics["Q_W_mean"] != metrics["Q_W_COS_ROOT_mean"]


def test_qy_split_uses_projection_for_full_observed_and_unobserved():
    Y_teacher = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    Y_student = -3.0 * Y_teacher
    mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    split = compute_qy_split(Y_student, Y_teacher, mask)

    assert split["Q_Y_full"] == pytest.approx(3.0)
    assert split["Q_Y_observed"] == pytest.approx(3.0)
    assert split["Q_Y_unobserved"] == pytest.approx(3.0)


def test_spreading_perfect_teacher_has_observed_projection_one():
    W_teacher = torch.tensor([[1.0], [2.0]])
    X_teacher = torch.tensor([[3.0, 4.0]])
    i_idx = torch.tensor([0, 1])
    j_idx = torch.tensor([0, 1])
    F = torch.ones(2, 1)
    spreading_data = SpreadingData(
        i_idx=i_idx,
        j_idx=j_idx,
        F=F,
        Y_values=compute_sparse_Y(W_teacher, X_teacher, F, i_idx, j_idx),
        seed=7,
        M=1,
    )

    metrics = compute_all_metrics_spreading(
        W_teacher,
        X_teacher,
        W_teacher,
        X_teacher,
        spreading_data,
    )

    assert metrics["Q_Y"] == pytest.approx(1.0)
    assert metrics["Q_Y_observed"] == pytest.approx(1.0)
    assert metrics["Q_W"] == pytest.approx(2.5)
    assert metrics["Q_X"] == pytest.approx(12.5)
    assert metrics["Q_W_PROJ_ABS"] == pytest.approx(1.0)
    assert metrics["Q_X_PROJ_ABS"] == pytest.approx(1.0)


def test_tensor_projection_and_qn_hand_fixture():
    teacher_factors = [
        torch.tensor([[1.0], [2.0]]),
        torch.tensor([[3.0], [4.0]]),
        torch.tensor([[5.0], [6.0]]),
    ]
    student_factors = [2.0 * teacher_factors[0], teacher_factors[1], -teacher_factors[2]]

    assert compute_tensor_projection_abs(teacher_factors, teacher_factors) == pytest.approx(1.0)
    assert compute_tensor_projection_abs(teacher_factors, student_factors) == pytest.approx(2.0)

    qn_modes = compute_tensor_factor_projection_overlaps(teacher_factors, student_factors)
    assert qn_modes == pytest.approx([5.0, 12.5, -30.5])
    assert sum(qn_modes) / len(qn_modes) == pytest.approx(-13.0 / 3.0)


def test_tensor_observed_hyperedge_projection_perfect_recovery():
    teacher_measurements = torch.tensor([1.5, -2.0, 0.5])
    student_measurements = teacher_measurements.clone()

    assert projection_abs(student_measurements, teacher_measurements) == pytest.approx(1.0)


def test_metric_schema_records_projection_policy_and_legacy_qy_load(tmp_path):
    config = ExperimentConfig(
        matrix=MatrixParams(N1=2, N2=2, M=1),
        training=TrainingParams(samples_per_alpha=1, max_steps=1),
        algorithm_key="bigamp",
        scan=ScanConfig(dimension="alpha", values=[0.5]),
        algorithm_params=AlgorithmParams(use_compile=False, use_bf16=False),
        experiment_name="legacy_metric_schema",
        teacher_key="standard",
    )
    run_dir = tmp_path / "legacy_run"
    run_dir.mkdir()
    config.save(run_dir / "config.json")
    metadata = ExperimentMetadata.create_now()
    (run_dir / "metadata.json").write_text(json.dumps(metadata.to_dict()), encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps({
            "schema_version": 2,
            "experiment_id": "legacy_metric_schema",
            "scan_dimension": "alpha",
            "scan_values": ["0.5"],
            "results": {"0.5": {"metrics": {"Q_Y_mean": 0.25}}},
        }),
        encoding="utf-8",
    )

    loaded = ExperimentResult.load(run_dir)
    compatibility = loaded.metadata.contract["metric_schema_compatibility"]

    assert compatibility["loaded_schema_version"] == 2
    assert compatibility["q_y_mean_interpretation"] == "legacy_cosine_or_reconstruction_proxy"
    assert compatibility["new_schema_q_y_mean_interpretation"] == "absolute_projection"
    assert compatibility["new_old_q_y_mean_not_comparable"] is True
