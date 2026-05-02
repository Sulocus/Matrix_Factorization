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
    _cos_root_and_replica_by_alpha,
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


def _direct_supergraph_measurement_metrics(
    w_students: torch.Tensor,
    x_students: torch.Tensor,
    data: SpreadingDataParallel,
) -> dict[str, torch.Tensor]:
    """Reference implementation over stored supergraph edges only."""

    samples, alpha_count = int(w_students.shape[0]), int(w_students.shape[1])
    c_max = int(data.C_max)
    sqrt_m_inv = float(data.M) ** -0.5
    result = {
        "full_projection": torch.zeros(samples, alpha_count, device=w_students.device),
        "observed_projection": torch.zeros(samples, alpha_count, device=w_students.device),
        "unobserved_projection": torch.zeros(samples, alpha_count, device=w_students.device),
        "full_cos": torch.zeros(samples, alpha_count, device=w_students.device),
        "observed_cos": torch.zeros(samples, alpha_count, device=w_students.device),
        "unobserved_cos": torch.zeros(samples, alpha_count, device=w_students.device),
    }

    for sample_idx in range(samples):
        for alpha_idx in range(alpha_count):
            c_k = int(data.supergraph.get_active_edges(alpha_idx))
            x_t = x_students[sample_idx, alpha_idx].T
            for prefix, start, stop in [
                ("full", 0, c_max),
                ("observed", 0, c_k),
                ("unobserved", c_k, c_max),
            ]:
                if stop <= start:
                    continue
                i_idx = data.supergraph.i_idx[sample_idx, start:stop].long()
                j_idx = data.supergraph.j_idx[sample_idx, start:stop].long()
                f_values = data.F_super[sample_idx, start:stop].to(w_students.dtype)
                y_student = sqrt_m_inv * (
                    f_values
                    * w_students[sample_idx, alpha_idx, i_idx]
                    * x_t[j_idx]
                ).sum(dim=-1)
                y_teacher = data.Y_super[sample_idx, start:stop].to(w_students.dtype)
                dot = (y_student * y_teacher).sum()
                teacher_norm = (y_teacher * y_teacher).sum()
                student_norm = (y_student * y_student).sum()
                result[f"{prefix}_projection"][sample_idx, alpha_idx] = dot.abs() / (teacher_norm + 1e-12)
                result[f"{prefix}_cos"][sample_idx, alpha_idx] = dot / torch.sqrt(
                    teacher_norm * student_norm + 1e-12
                )
    return result


def _legacy_cos_root_and_replica_by_alpha(
    factors: torch.Tensor,
    teacher: torch.Tensor,
    *,
    use_left: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if use_left:
        grams = torch.matmul(factors, factors.transpose(-1, -2))
        teacher_gram = teacher @ teacher.T
        n, m = int(factors.shape[-2]), int(factors.shape[-1])
    else:
        grams = torch.matmul(factors.transpose(-1, -2), factors)
        teacher_gram = teacher.T @ teacher
        n, m = int(factors.shape[-1]), int(factors.shape[-2])
    flat = grams.reshape(grams.shape[0], -1).float()
    teacher_flat = teacher_gram.reshape(-1).float()
    teacher_norm = teacher_flat.norm() + 1e-12
    q = (flat * teacher_flat).sum(dim=1) / ((flat.norm(dim=1) * teacher_norm) + 1e-12)
    baseline = float(m) / float(m + n + 1)
    corrected = ((q - baseline) / (1.0 - baseline + 1e-12)).clamp(0.0, 1.0)
    cos_root = corrected.sqrt()
    normalized = flat / (flat.norm(dim=1, keepdim=True) + 1e-12)
    pair_cos = normalized @ normalized.T
    pair_corrected = ((pair_cos - baseline) / (1.0 - baseline + 1e-12)).clamp(0.0, 1.0)
    upper = torch.triu_indices(factors.shape[0], factors.shape[0], offset=1, device=factors.device)
    return cos_root.float(), pair_cos[upper[0], upper[1]].mean().float(), pair_corrected[upper[0], upper[1]].mean().float()


def test_spreading_cos_root_uses_low_rank_equivalent_formula():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(19)
    w_factors = torch.randn(3, 7, 4, device=device)
    w_teacher = torch.randn(7, 4, device=device)
    x_factors = torch.randn(3, 4, 9, device=device)
    x_teacher = torch.randn(4, 9, device=device)

    for factors, teacher, use_left in [
        (w_factors, w_teacher, True),
        (x_factors, x_teacher, False),
    ]:
        actual = _cos_root_and_replica_by_alpha(factors, teacher, use_left=use_left)
        expected = _legacy_cos_root_and_replica_by_alpha(factors, teacher, use_left=use_left)
        for actual_value, expected_value in zip(actual, expected):
            assert torch.allclose(actual_value, expected_value, atol=5e-4, rtol=5e-4)


def test_spreading_batch_metrics_perfect_teacher_recovery():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = _small_spreading_fixture(device)
    samples = data.S
    alphas = data.A
    w_students = data.W_teacher.unsqueeze(0).unsqueeze(0).expand(samples, alphas, -1, -1).clone()
    x_students = data.X_teacher.unsqueeze(0).unsqueeze(0).expand(samples, alphas, -1, -1).clone()

    metrics = compute_all_metrics_spreading_parallel(w_students, x_students, data)

    expected_unobserved = torch.tensor(
        [
            1.0 if data.supergraph.get_active_edges(alpha_idx) < data.C_max else 0.0
            for alpha_idx in range(alphas)
        ],
        device=device,
    )

    for key in [
        "Q_Y_mean",
        "Q_Y_observed_mean",
        "Q_Y_COS_mean",
        "Q_Y_observed_COS_mean",
        "Q_W_COS_ROOT_mean",
        "Q_X_COS_ROOT_mean",
        "Q_W_replica_mean",
        "Q_X_replica_mean",
        "Q_W_prime_replica_mean",
        "Q_X_prime_replica_mean",
    ]:
        assert torch.allclose(metrics[key], torch.ones_like(metrics[key]), atol=1e-5), key
    assert torch.allclose(metrics["Q_Y_unobserved_mean"], expected_unobserved, atol=1e-5)
    assert torch.allclose(metrics["Q_Y_unobserved_COS_mean"], expected_unobserved, atol=1e-5)
    w_self = (data.W_teacher * data.W_teacher).mean()
    x_self = (data.X_teacher * data.X_teacher).mean()
    assert torch.allclose(metrics["Q_W_mean"], torch.full_like(metrics["Q_W_mean"], float(w_self)), atol=1e-5)
    assert torch.allclose(metrics["Q_X_mean"], torch.full_like(metrics["Q_X_mean"], float(x_self)), atol=1e-5)
    assert torch.allclose(metrics["Q_W_SIGN_GAUGE_mean"], torch.full_like(metrics["Q_W_SIGN_GAUGE_mean"], float(w_self)), atol=1e-5)
    assert torch.allclose(metrics["Q_X_SIGN_GAUGE_mean"], torch.full_like(metrics["Q_X_SIGN_GAUGE_mean"], float(x_self)), atol=1e-5)
    assert torch.allclose(metrics["Q_W_SCALE_GAUGE_mean"], torch.full_like(metrics["Q_W_SCALE_GAUGE_mean"], float(w_self)), atol=1e-5)
    assert torch.allclose(metrics["Q_X_SCALE_GAUGE_mean"], torch.full_like(metrics["Q_X_SCALE_GAUGE_mean"], float(x_self)), atol=1e-5)
    assert torch.allclose(
        metrics["Q_WX_SCALE_GAUGE_mean"],
        torch.full_like(metrics["Q_WX_SCALE_GAUGE_mean"], float(0.5 * (w_self + x_self))),
        atol=1e-5,
    )
    assert torch.allclose(
        metrics["median_abs_log_k_mean"],
        torch.zeros_like(metrics["median_abs_log_k_mean"]),
        atol=1e-5,
    )


def test_spreading_full_qy_uses_stored_supergraph_not_equal_count_heldout():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = _small_spreading_fixture(device)
    torch.manual_seed(23)
    w_students = torch.randn(data.S, data.A, data.W_teacher.shape[0], data.M, device=device)
    x_students = torch.randn(data.S, data.A, data.M, data.X_teacher.shape[1], device=device)

    metrics = compute_all_metrics_spreading_parallel(w_students, x_students, data)
    expected = _direct_supergraph_measurement_metrics(w_students, x_students, data)

    assert torch.allclose(metrics["Q_Y_mean"], expected["full_projection"].mean(dim=0), atol=1e-5)
    assert torch.allclose(metrics["Q_Y_observed_mean"], expected["observed_projection"].mean(dim=0), atol=1e-5)
    assert torch.allclose(metrics["Q_Y_unobserved_mean"], expected["unobserved_projection"].mean(dim=0), atol=1e-5)
    assert torch.allclose(metrics["Q_Y_COS_mean"], expected["full_cos"].mean(dim=0), atol=1e-5)
    assert torch.allclose(metrics["Q_Y_observed_COS_mean"], expected["observed_cos"].mean(dim=0), atol=1e-5)
    assert torch.allclose(metrics["Q_Y_unobserved_COS_mean"], expected["unobserved_cos"].mean(dim=0), atol=1e-5)


def test_spreading_batch_metrics_accept_local_factors_with_global_alpha_domain():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = _small_spreading_fixture(device)
    torch.manual_seed(29)
    w_students = torch.randn(data.S, data.A, data.W_teacher.shape[0], data.M, device=device)
    x_students = torch.randn(data.S, data.A, data.M, data.X_teacher.shape[1], device=device)

    full_metrics = compute_all_metrics_spreading_parallel(w_students, x_students, data)
    local_metrics = compute_all_metrics_spreading_parallel(
        w_students[:, 1:],
        x_students[:, 1:],
        data,
        target_alpha_indices=[1],
    )

    assert torch.allclose(local_metrics["alpha_values"], full_metrics["alpha_values"][1:2])
    for key, value in full_metrics.items():
        if key == "alpha_values":
            continue
        if isinstance(value, torch.Tensor) and value.ndim == 1 and value.numel() == data.A:
            assert torch.allclose(local_metrics[key], value[1:2], atol=1e-5), key


def test_spreading_output_cosine_is_signed_and_separate_from_projection_scale():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = _small_spreading_fixture(device)
    samples = data.S
    alphas = data.A
    w_students = (
        2.0
        * data.W_teacher.unsqueeze(0).unsqueeze(0).expand(samples, alphas, -1, -1).clone()
    )
    x_students = data.X_teacher.unsqueeze(0).unsqueeze(0).expand(samples, alphas, -1, -1).clone()

    metrics = compute_all_metrics_spreading_parallel(w_students, x_students, data)
    expected_unobserved_projection = torch.tensor(
        [
            2.0 if data.supergraph.get_active_edges(alpha_idx) < data.C_max else 0.0
            for alpha_idx in range(alphas)
        ],
        device=device,
    )
    expected_unobserved_cos = torch.tensor(
        [
            1.0 if data.supergraph.get_active_edges(alpha_idx) < data.C_max else 0.0
            for alpha_idx in range(alphas)
        ],
        device=device,
    )

    assert torch.allclose(metrics["Q_Y_mean"], torch.full_like(metrics["Q_Y_mean"], 2.0), atol=1e-5)
    assert torch.allclose(metrics["Q_Y_observed_mean"], torch.full_like(metrics["Q_Y_observed_mean"], 2.0), atol=1e-5)
    assert torch.allclose(metrics["Q_Y_unobserved_mean"], expected_unobserved_projection, atol=1e-5)
    assert torch.allclose(metrics["Q_Y_COS_mean"], torch.ones_like(metrics["Q_Y_COS_mean"]), atol=1e-5)
    assert torch.allclose(metrics["Q_Y_observed_COS_mean"], torch.ones_like(metrics["Q_Y_observed_COS_mean"]), atol=1e-5)
    assert torch.allclose(metrics["Q_Y_unobserved_COS_mean"], expected_unobserved_cos, atol=1e-5)


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

    w_self = (data.W_teacher * data.W_teacher).mean()
    x_self = (data.X_teacher * data.X_teacher).mean()
    assert torch.all(metrics["Q_W_mean"] < metrics["Q_W_SIGN_GAUGE_mean"])
    assert torch.all(metrics["Q_X_mean"] < metrics["Q_X_SIGN_GAUGE_mean"])
    assert torch.allclose(metrics["Q_W_SIGN_GAUGE_mean"], torch.full_like(metrics["Q_W_SIGN_GAUGE_mean"], float(w_self)), atol=1e-5)
    assert torch.allclose(metrics["Q_X_SIGN_GAUGE_mean"], torch.full_like(metrics["Q_X_SIGN_GAUGE_mean"], float(x_self)), atol=1e-5)
    assert torch.allclose(
        metrics["Q_WX_SCALE_GAUGE_mean"],
        torch.full_like(metrics["Q_WX_SCALE_GAUGE_mean"], float(0.5 * (w_self + x_self))),
        atol=1e-5,
    )


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


def test_bigamp_spreading_train_batch_result_maps_folded_alpha_to_global_metrics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = _small_spreading_fixture(device)
    algorithm = object.__new__(BiGAMPSpreading)
    algorithm._contract_execution_metadata = {"path": "unit_fixture", "metadata_only": True}
    alpha = float(data.alpha_values[1].item())

    def fake_train_batch_alphas(**kwargs):
        alpha_count = len(kwargs["alpha_values"])
        samples = data.S
        assert alpha_count == 1
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
        alpha_values=[alpha],
        seed=17,
        spreading_data=data,
    )

    assert set(result.metrics_by_alpha) == {alpha}
    payload = result.diagnostics["batch_metric_payload"]
    assert payload["spreading_metric_domain"] == "scan_global_supergraph"
    assert payload["metric_alpha_indices"] == [1]
    assert abs(result.metrics_by_alpha[alpha]["Q_Y_mean"] - 1.0) < 1e-5
