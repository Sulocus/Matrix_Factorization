import torch

from matrix_factorization.modules.algorithms.bigamp.spreading import BiGAMPSpreading
from matrix_factorization.modules.algorithms.bigamp.f_gen import (
    compute_Y_super,
    generate_F_super,
)
from matrix_factorization.modules.graphs.supergraph import create_supergraph
from matrix_factorization.modules.metrics.spreading import (
    BatchMetricPayload,
    compute_all_metrics_spreading_parallel,
)
from matrix_factorization.modules.teachers.random_spreading import SpreadingDataParallel


def _small_spreading_fixture(device: torch.device) -> SpreadingDataParallel:
    torch.manual_seed(7)
    n1, n2, m, samples = 4, 4, 2, 2
    alpha_values = [0.25, 0.5]
    w_teacher = torch.randn(n1, m, device=device)
    x_teacher = torch.randn(m, n2, device=device)
    supergraph = create_supergraph(
        N1=n1,
        N2=n2,
        M=m,
        alpha_values=alpha_values,
        S=samples,
        base_seed=11,
        device=device,
    )
    f_super = generate_F_super(
        supergraph=supergraph,
        M=m,
        base_seed=13,
        device=device,
        f_distribution="ising",
    )
    y_super = compute_Y_super(
        W_teacher=w_teacher,
        X_teacher=x_teacher,
        supergraph=supergraph,
        F_super=f_super,
    )
    return SpreadingDataParallel(
        supergraph=supergraph,
        F_super=f_super,
        Y_super=y_super,
        M=m,
        alpha_values=torch.tensor(alpha_values, device=device),
        W_teacher=w_teacher,
        X_teacher=x_teacher,
        f_distribution="ising",
    )


def test_spreading_batch_metrics_perfect_teacher_recovery():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = _small_spreading_fixture(device)
    samples = data.S
    alphas = data.A
    w_students = data.W_teacher.unsqueeze(0).unsqueeze(0).expand(samples, alphas, -1, -1).clone()
    x_students = data.X_teacher.unsqueeze(0).unsqueeze(0).expand(samples, alphas, -1, -1).clone()

    metrics = compute_all_metrics_spreading_parallel(w_students, x_students, data)

    for key in [
        "Q_Y_mean",
        "Q_Y_observed_mean",
        "Q_Y_unobserved_mean",
        "Q_W_mean",
        "Q_X_mean",
        "Q_W_SIGN_ALIGNED_mean",
        "Q_X_SIGN_ALIGNED_mean",
        "Q_W_COS_ROOT_mean",
        "Q_X_COS_ROOT_mean",
        "Q_W_SCALE_GAUGE_mean",
        "Q_X_SCALE_GAUGE_mean",
        "Q_WX_SCALE_GAUGE_mean",
        "Q_W_replica_mean",
        "Q_X_replica_mean",
        "Q_W_prime_replica_mean",
        "Q_X_prime_replica_mean",
    ]:
        assert torch.allclose(metrics[key], torch.ones_like(metrics[key]), atol=1e-5), key
    assert torch.allclose(
        metrics["median_abs_log_g_mean"],
        torch.zeros_like(metrics["median_abs_log_g_mean"]),
        atol=1e-5,
    )


def test_spreading_batch_metrics_separate_coordinate_and_sign_aligned_overlap():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = _small_spreading_fixture(device)
    samples = data.S
    alphas = data.A
    w_students = data.W_teacher.unsqueeze(0).unsqueeze(0).expand(samples, alphas, -1, -1).clone()
    x_students = data.X_teacher.unsqueeze(0).unsqueeze(0).expand(samples, alphas, -1, -1).clone()
    w_students[..., 0] *= -1.0
    x_students[..., 0, :] *= -1.0

    metrics = compute_all_metrics_spreading_parallel(w_students, x_students, data)

    assert torch.all(metrics["Q_W_mean"] < metrics["Q_W_SIGN_ALIGNED_mean"])
    assert torch.all(metrics["Q_X_mean"] < metrics["Q_X_SIGN_ALIGNED_mean"])
    assert torch.allclose(metrics["Q_W_SIGN_ALIGNED_mean"], torch.ones_like(metrics["Q_W_SIGN_ALIGNED_mean"]), atol=1e-5)
    assert torch.allclose(metrics["Q_X_SIGN_ALIGNED_mean"], torch.ones_like(metrics["Q_X_SIGN_ALIGNED_mean"]), atol=1e-5)
    assert torch.allclose(metrics["Q_WX_SCALE_GAUGE_mean"], torch.ones_like(metrics["Q_WX_SCALE_GAUGE_mean"]), atol=1e-5)


def test_spreading_batch_metric_payload_materializes_once_per_alpha():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alpha_values = torch.tensor([0.1, 0.2], device=device)
    payload = BatchMetricPayload(
        alpha_values=alpha_values,
        metrics={
            "Q_Y_mean": torch.tensor([0.3, 0.4], device=device),
            "Q_W_mean": torch.tensor([0.5, 0.6], device=device),
        },
        metadata={"path": "unit"},
    )

    result = payload.to_metrics_by_alpha()

    assert result == {
        0.10000000149011612: {"Q_Y_mean": 0.30000001192092896, "Q_W_mean": 0.5},
        0.20000000298023224: {"Q_Y_mean": 0.4000000059604645, "Q_W_mean": 0.6000000238418579},
    }


def test_bigamp_spreading_train_batch_result_precomputes_batch_metrics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = _small_spreading_fixture(device)
    algorithm = object.__new__(BiGAMPSpreading)
    algorithm._contract_execution_metadata = {"path": "unit_fixture", "metadata_only": True}

    def fake_train_batch_alphas(**kwargs):
        alpha_count = len(kwargs["alpha_values"])
        samples = data.S
        return (
            data.W_teacher.unsqueeze(0).unsqueeze(0).expand(alpha_count, samples, -1, -1).clone(),
            data.X_teacher.unsqueeze(0).unsqueeze(0).expand(alpha_count, samples, -1, -1).clone(),
        )

    algorithm.train_batch_alphas = fake_train_batch_alphas

    result = BiGAMPSpreading.train_batch_result(
        algorithm,
        algorithm_key="bigamp_spreading",
        W_teacher=data.W_teacher,
        X_teacher=data.X_teacher,
        Y_teacher=torch.empty(0, device=device),
        masks=None,
        alpha_values=[float(value) for value in data.alpha_values.detach().cpu().tolist()],
        seed=17,
        spreading_data=data,
    )

    assert result.metadata["batch_metric_path"] == "gpu_resident_batch_metric_payload"
    assert set(result.metrics_by_alpha) == {float(value) for value in data.alpha_values.detach().cpu().tolist()}
    for metrics in result.metrics_by_alpha.values():
        assert abs(metrics["Q_Y_observed_mean"] - 1.0) < 1e-5
