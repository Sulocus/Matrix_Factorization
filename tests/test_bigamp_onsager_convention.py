"""Convention tests for BiG-AMP Onsager, variance, and damping semantics."""

import torch


def test_damping_beta_endpoints_freeze_and_accept():
    from matrix_factorization.modules.algorithms.bigamp.conventions import blend_new_old

    old = torch.tensor([1.0, -2.0])
    new = torch.tensor([3.0, 4.0])

    assert torch.equal(blend_new_old(new, old, 0.0), old)
    assert torch.equal(blend_new_old(new, old, 1.0), new)
    assert torch.allclose(blend_new_old(new, old, 0.25), torch.tensor([1.5, -0.5]))


def test_gaussian_posterior_update_matches_zero_mean_prior_formula():
    from matrix_factorization.modules.algorithms.bigamp.conventions import gaussian_posterior_update

    current = torch.tensor([0.5, -1.0])
    residual = torch.tensor([2.0, -3.0])
    precision = torch.tensor([4.0, 9.0])

    mean, var = gaussian_posterior_update(
        current,
        residual,
        precision,
        prior_precision=1.0,
        prior_variance=1.0,
    )

    expected_var = 1.0 / (1.0 + precision)
    expected_mean = expected_var * (precision * current + residual)
    assert torch.allclose(var, expected_var)
    assert torch.allclose(mean, expected_mean)


def test_flat_step_damping_zero_preserves_old_state_and_onsager_state():
    from matrix_factorization.modules.algorithms.bigamp.step import bigamp_step_disjoint_union_flat

    W = torch.tensor([[[0.3, -0.4]]])
    X = torch.tensor([[[0.2, 0.1]]])
    W_var = torch.full_like(W, 0.7)
    X_var = torch.full_like(X, 0.6)
    prev_s = torch.tensor([[7.0]])
    prev_svar = torch.tensor([[11.0]])

    W_out, X_out, W_var_out, X_var_out, s_state, svar_state = bigamp_step_disjoint_union_flat(
        W,
        X,
        W_var,
        X_var,
        Y_flat=torch.tensor([0.0]),
        F_flat=torch.tensor([[1.0, -1.0]]),
        i_offset=torch.tensor([0]),
        j_offset=torch.tensor([0]),
        alpha_mask_exp=torch.tensor([[True]]),
        S=1,
        N1=1,
        N2=1,
        damping=0.0,
        noise_var=1e-5,
        prior_precision_base=1.0,
        prior_variance=1.0,
        is_ising=True,
        prev_s=prev_s,
        prev_svar=prev_svar,
    )

    assert torch.allclose(W_out, W)
    assert torch.allclose(X_out, X)
    assert torch.allclose(W_var_out, W_var)
    assert torch.allclose(X_var_out, X_var)
    assert torch.equal(s_state, prev_s)
    assert torch.equal(svar_state, prev_svar)


def test_flat_adaptive_output_variance_includes_cross_variance_at_cold_start():
    from matrix_factorization.modules.algorithms.bigamp.step import bigamp_step_disjoint_union_flat_adaptive

    W = torch.zeros(1, 1, 2)
    X = torch.zeros(1, 1, 2)
    W_var = torch.ones_like(W)
    X_var = torch.ones_like(X)

    *_, pvar = bigamp_step_disjoint_union_flat_adaptive(
        W,
        X,
        W_var,
        X_var,
        Y_flat=torch.tensor([0.0]),
        F_flat=torch.tensor([[1.0, -1.0]]),
        i_offset=torch.tensor([0]),
        j_offset=torch.tensor([0]),
        alpha_mask_exp=torch.tensor([[True]]),
        S=1,
        N1=1,
        N2=1,
        noise_var=1e-5,
        prior_precision_base=1.0,
        prior_variance=1.0,
        is_ising=True,
    )

    assert torch.allclose(pvar, torch.tensor([[1.0]]))


def test_corrected_pvar_controls_cold_start_residual_scale():
    from matrix_factorization.modules.algorithms.bigamp.step import (
        forward_disjoint_union_flat_corrected,
        forward_disjoint_union_flat_legacy_fast,
    )

    torch.manual_seed(0)
    M = 50
    edges = 512
    W = torch.randn(1, edges, M) * 0.1
    X = torch.randn(1, edges, M) * 0.1
    W_var = torch.ones_like(W)
    X_var = torch.ones_like(X)
    F = torch.randint(0, 2, (edges, M), dtype=torch.float32) * 2 - 1
    idx = torch.arange(edges)
    mask = torch.ones(1, edges, dtype=torch.bool)

    _, zvar_legacy = forward_disjoint_union_flat_legacy_fast(
        W,
        X,
        W_var,
        X_var,
        F,
        idx,
        idx,
        mask,
        is_ising=True,
    )
    _, pvar_corrected = forward_disjoint_union_flat_corrected(
        W,
        X,
        W_var,
        X_var,
        F,
        idx,
        idx,
        mask,
        is_ising=True,
    )

    ratio = torch.median(pvar_corrected / zvar_legacy)
    assert 35.0 < float(ratio) < 70.0

    Y = torch.randn(edges)
    legacy_s = (Y.unsqueeze(0) / zvar_legacy).abs()
    corrected_s = (Y.unsqueeze(0) / pvar_corrected).abs()
    s_ratio = torch.median(legacy_s / corrected_s)
    assert 35.0 < float(s_ratio) < 70.0


def test_spreading_onsager_update_route_metadata():
    from matrix_factorization.core.experiment.config import (
        AlgorithmParams,
        ExperimentConfig,
        MatrixParams,
        ScanConfig,
        SpreadingConfig,
        TrainingParams,
    )
    from matrix_factorization.modules.algorithms.bigamp.spreading import BiGAMPSpreading

    def make_algorithm(*, onsager: bool, adaptive: bool):
        config = ExperimentConfig(
            matrix=MatrixParams(N1=3, N2=3, M=2),
            training=TrainingParams(samples_per_alpha=1, max_steps=1, max_epochs=1),
            algorithm_key="bigamp_spreading",
            scan=ScanConfig(dimension="alpha", values=[0.5]),
            spreading=SpreadingConfig(f_distribution="ising", onsager_correction=onsager, chunk_size=0),
            algorithm_params=AlgorithmParams(
                damping=0.05,
                adaptive_damping=adaptive,
                use_compile=False,
                use_bf16=False,
                precision_profile="safe",
                seed_partition_policy="partition_invariant",
            ),
        )
        return BiGAMPSpreading(config, torch.device("cpu"))

    assert make_algorithm(onsager=False, adaptive=False)._contract_execution_metadata[
        "onsager_update_route"
    ] == "legacy_no_onsager"
    assert make_algorithm(onsager=True, adaptive=False)._contract_execution_metadata[
        "onsager_update_route"
    ] == "corrected_fixed_onsager"
    assert make_algorithm(onsager=True, adaptive=True)._contract_execution_metadata[
        "onsager_update_route"
    ] == "corrected_adaptive_onsager"


def test_corrected_onsager_internal_alpha_batches_split_large_spreading_step():
    from matrix_factorization.core.experiment.config import (
        AlgorithmParams,
        ExperimentConfig,
        MatrixParams,
        ScanConfig,
        SpreadingConfig,
        TrainingParams,
    )
    from matrix_factorization.modules.algorithms.bigamp.spreading import BiGAMPSpreading

    alpha_values = [round(i * 0.1, 1) for i in range(41)]
    config = ExperimentConfig(
        matrix=MatrixParams(N1=200, N2=200, M=50),
        training=TrainingParams(samples_per_alpha=20, max_steps=1, max_epochs=1),
        algorithm_key="bigamp_spreading",
        scan=ScanConfig(dimension="alpha", values=alpha_values),
        spreading=SpreadingConfig(f_distribution="ising", onsager_correction=True, chunk_size=0),
        algorithm_params=AlgorithmParams(
            damping=0.05,
            adaptive_damping=False,
            use_compile=False,
            use_bf16=False,
            precision_profile="safe",
            seed_partition_policy="partition_invariant",
        ),
    )
    algorithm = BiGAMPSpreading(config, torch.device("cpu"))

    batches = algorithm._compute_internal_spreading_alpha_batches(alpha_values, sample_count=20)

    assert len(batches) > 1
    assert batches[0][0] == 0
    assert batches[-1][1] == len(alpha_values)
    for start, end, alpha_max in batches:
        assert end > start
        assert alpha_max == max(alpha_values[start:end])
        assert (
            algorithm._estimate_flat_step_elements(
                alpha_count=end - start,
                alpha_max=alpha_max,
                sample_count=20,
            )
            <= 160_000_000
            or end == start + 1
        )


def test_legacy_no_onsager_keeps_single_internal_alpha_batch():
    from matrix_factorization.core.experiment.config import (
        AlgorithmParams,
        ExperimentConfig,
        MatrixParams,
        ScanConfig,
        SpreadingConfig,
        TrainingParams,
    )
    from matrix_factorization.modules.algorithms.bigamp.spreading import BiGAMPSpreading

    alpha_values = [round(i * 0.1, 1) for i in range(41)]
    config = ExperimentConfig(
        matrix=MatrixParams(N1=200, N2=200, M=50),
        training=TrainingParams(samples_per_alpha=20, max_steps=1, max_epochs=1),
        algorithm_key="bigamp_spreading",
        scan=ScanConfig(dimension="alpha", values=alpha_values),
        spreading=SpreadingConfig(f_distribution="ising", onsager_correction=False, chunk_size=0),
        algorithm_params=AlgorithmParams(
            damping=0.5,
            adaptive_damping=False,
            use_compile=False,
            use_bf16=False,
            precision_profile="safe",
            seed_partition_policy="partition_invariant",
        ),
    )
    algorithm = BiGAMPSpreading(config, torch.device("cpu"))

    batches = algorithm._compute_internal_spreading_alpha_batches(alpha_values, sample_count=20)

    assert batches == [(0, len(alpha_values), max(alpha_values))]


def test_tensor_variance_includes_cross_variance_at_cold_start():
    from matrix_factorization.modules.algorithms.bigamp.tensor_step import compute_variance_tensor

    factors = [torch.zeros(1, 2), torch.zeros(1, 2)]
    factor_vars = [torch.ones(1, 2), torch.ones(1, 2)]
    F = torch.tensor([[1.0, -1.0]])
    indices = [torch.tensor([0]), torch.tensor([0])]

    pvar = compute_variance_tensor(factors, factor_vars, F, indices, is_ising=True)

    assert torch.allclose(pvar, torch.tensor([1.0]))


def test_adaptive_spreading_cold_start_initialization_runs():
    from matrix_factorization.core.experiment.config import (
        AlgorithmParams,
        ExperimentConfig,
        MatrixParams,
        ScanConfig,
        SpreadingConfig,
        TrainingParams,
    )
    from matrix_factorization.modules.algorithms.bigamp.spreading import BiGAMPSpreading

    device = torch.device("cpu")
    config = ExperimentConfig(
        matrix=MatrixParams(N1=3, N2=3, M=2),
        training=TrainingParams(samples_per_alpha=1, max_steps=1, max_epochs=1),
        algorithm_key="bigamp_spreading",
        scan=ScanConfig(dimension="alpha", values=[0.5]),
        spreading=SpreadingConfig(f_distribution="ising", onsager_correction=True, chunk_size=0),
        algorithm_params=AlgorithmParams(
            damping=0.05,
            adaptive_damping=True,
            init_mode="random",
            use_compile=False,
            use_bf16=False,
            precision_profile="safe",
            seed_partition_policy="partition_invariant",
        ),
    )
    algorithm = BiGAMPSpreading(config, device)
    W_teacher = torch.randn(3, 2)
    X_teacher = torch.randn(2, 3)
    data = algorithm.create_spreading_data(W_teacher, X_teacher, alpha_values=[0.5], S=1, base_seed=7)

    W_hat, X_hat = algorithm.train_full_parallel(data, verbose=False, batch_alpha_values=[0.5], base_seed=11)

    assert W_hat.shape == (1, 1, 3, 2)
    assert X_hat.shape == (1, 1, 2, 3)
