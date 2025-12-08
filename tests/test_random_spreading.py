"""
Physical correctness verification tests for Random Spreading module.

Tests verify:
1. F generation determinism and statistics
2. Y computation correctness (vectorized vs naive)
3. BiG-AMP message passing correctness
4. Q_Y evaluation consistency
5. Phase transition behavior (optional, slow)
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smf.modules.teachers.random_spreading import (
    SpreadingData,
    generate_spreading_coefficients,
    generate_spreading_coefficients_per_edge,
    compute_sparse_Y,
    compute_sparse_Y_batched,
    RandomSpreadingTeacher,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Get available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def small_dims():
    """Small dimensions for quick tests."""
    return {'N1': 10, 'N2': 12, 'M': 5}


@pytest.fixture
def medium_dims():
    """Medium dimensions for statistical tests."""
    return {'N1': 100, 'N2': 100, 'M': 20}


# ============================================================================
# Test 1: F Generation Determinism
# ============================================================================

class TestFDeterminism:
    """Test that F generation is deterministic."""

    def test_same_seed_same_f(self, device, small_dims):
        """Same seed should produce identical F."""
        M = small_dims['M']
        seed = 42

        i_idx = torch.tensor([0, 1, 2, 3], device=device)
        j_idx = torch.tensor([5, 6, 7, 8], device=device)

        F1 = generate_spreading_coefficients(i_idx, j_idx, M, seed, device)
        F2 = generate_spreading_coefficients(i_idx, j_idx, M, seed, device)

        assert torch.allclose(F1, F2), "Same seed should give identical F"

    def test_different_seed_different_f(self, device, small_dims):
        """Different seeds should produce different F."""
        M = small_dims['M']

        i_idx = torch.tensor([0, 1, 2, 3], device=device)
        j_idx = torch.tensor([5, 6, 7, 8], device=device)

        F1 = generate_spreading_coefficients(i_idx, j_idx, M, seed=42, device=device)
        F2 = generate_spreading_coefficients(i_idx, j_idx, M, seed=43, device=device)

        assert not torch.allclose(F1, F2), "Different seeds should give different F"

    def test_per_edge_determinism(self, device, small_dims):
        """Per-edge generation should be order-independent."""
        M = small_dims['M']
        seed = 42

        # Original order
        i_idx1 = torch.tensor([0, 1, 2], device=device)
        j_idx1 = torch.tensor([3, 4, 5], device=device)

        # Reversed order
        i_idx2 = torch.tensor([2, 1, 0], device=device)
        j_idx2 = torch.tensor([5, 4, 3], device=device)

        F1 = generate_spreading_coefficients_per_edge(i_idx1, j_idx1, M, seed, device)
        F2 = generate_spreading_coefficients_per_edge(i_idx2, j_idx2, M, seed, device)

        # F2[0] should equal F1[2] (same edge (2,5))
        assert torch.allclose(F1[2], F2[0], atol=1e-6), \
            "Per-edge generation should be order-independent"
        assert torch.allclose(F1[1], F2[1], atol=1e-6), \
            "Per-edge generation should be order-independent"
        assert torch.allclose(F1[0], F2[2], atol=1e-6), \
            "Per-edge generation should be order-independent"

    def test_empty_edges(self, device):
        """Empty edge list should return empty tensor."""
        i_idx = torch.empty(0, dtype=torch.long, device=device)
        j_idx = torch.empty(0, dtype=torch.long, device=device)

        F = generate_spreading_coefficients(i_idx, j_idx, M=10, seed=42, device=device)

        assert F.shape == (0, 10), f"Expected shape (0, 10), got {F.shape}"


# ============================================================================
# Test 2: F Statistics
# ============================================================================

class TestFStatistics:
    """Test that F follows N(0,1) distribution."""

    def test_mean_approximately_zero(self, device):
        """F should have mean approximately 0."""
        M = 100
        C = 10000  # Many edges for statistics

        i_idx = torch.arange(C, device=device)
        j_idx = torch.zeros(C, dtype=torch.long, device=device)

        F = generate_spreading_coefficients(i_idx, j_idx, M, seed=42, device=device)

        mean = F.mean().item()
        assert abs(mean) < 0.05, f"Mean should be ~0, got {mean}"

    def test_std_approximately_one(self, device):
        """F should have std approximately 1."""
        M = 100
        C = 10000

        i_idx = torch.arange(C, device=device)
        j_idx = torch.zeros(C, dtype=torch.long, device=device)

        F = generate_spreading_coefficients(i_idx, j_idx, M, seed=42, device=device)

        std = F.std().item()
        assert abs(std - 1.0) < 0.05, f"Std should be ~1, got {std}"

    def test_distribution_shape(self, device):
        """F should follow Gaussian distribution."""
        M = 50
        C = 5000

        i_idx = torch.arange(C, device=device)
        j_idx = torch.zeros(C, dtype=torch.long, device=device)

        F = generate_spreading_coefficients(i_idx, j_idx, M, seed=42, device=device)

        # Check percentiles match Gaussian
        F_flat = F.flatten().cpu().numpy()

        # 68-95-99.7 rule
        within_1std = np.mean(np.abs(F_flat) < 1.0)
        within_2std = np.mean(np.abs(F_flat) < 2.0)
        within_3std = np.mean(np.abs(F_flat) < 3.0)

        assert 0.63 < within_1std < 0.73, f"68% within 1σ, got {within_1std:.2%}"
        assert 0.90 < within_2std < 0.98, f"95% within 2σ, got {within_2std:.2%}"
        assert within_3std > 0.99, f"99.7% within 3σ, got {within_3std:.2%}"


# ============================================================================
# Test 3: Y Computation Correctness
# ============================================================================

class TestYComputation:
    """Test Y computation correctness."""

    def test_small_scale_manual(self, device):
        """Small-scale manual verification."""
        # Small dimensions for manual verification
        N1, N2, M = 3, 4, 2

        # Fixed teacher matrices
        W = torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ], device=device)

        X = torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ], device=device)

        # Observed edges: (0,0), (1,2), (2,3)
        i_idx = torch.tensor([0, 1, 2], device=device)
        j_idx = torch.tensor([0, 2, 3], device=device)

        # Fixed F for verification
        F = torch.tensor([
            [0.5, 1.5],   # F for edge (0,0)
            [-0.5, 0.5],  # F for edge (1,2)
            [1.0, -1.0],  # F for edge (2,3)
        ], device=device)

        alpha_scale = 1.0 / (M ** 0.5)

        # Manual computation
        # Y[0,0] = (1/√2) × (F[0,0,0]×W[0,0]×X[0,0] + F[0,0,1]×W[0,1]×X[1,0])
        #        = (1/√2) × (0.5×1.0×1.0 + 1.5×2.0×5.0)
        #        = (1/√2) × (0.5 + 15.0) = 15.5/√2
        Y_00_expected = alpha_scale * (0.5 * 1.0 * 1.0 + 1.5 * 2.0 * 5.0)

        # Y[1,2] = (1/√2) × (F[1,2,0]×W[1,0]×X[0,2] + F[1,2,1]×W[1,1]×X[1,2])
        #        = (1/√2) × (-0.5×3.0×3.0 + 0.5×4.0×7.0)
        #        = (1/√2) × (-4.5 + 14.0) = 9.5/√2
        Y_12_expected = alpha_scale * (-0.5 * 3.0 * 3.0 + 0.5 * 4.0 * 7.0)

        # Y[2,3] = (1/√2) × (F[2,3,0]×W[2,0]×X[0,3] + F[2,3,1]×W[2,1]×X[1,3])
        #        = (1/√2) × (1.0×5.0×4.0 + (-1.0)×6.0×8.0)
        #        = (1/√2) × (20.0 - 48.0) = -28.0/√2
        Y_23_expected = alpha_scale * (1.0 * 5.0 * 4.0 + (-1.0) * 6.0 * 8.0)

        Y_expected = torch.tensor(
            [Y_00_expected, Y_12_expected, Y_23_expected],
            device=device
        )

        # Vectorized computation
        Y_computed = compute_sparse_Y(W, X, F, i_idx, j_idx)

        assert torch.allclose(Y_computed, Y_expected, atol=1e-5), \
            f"Y mismatch:\nExpected: {Y_expected}\nComputed: {Y_computed}"

    def test_vectorized_vs_naive_loop(self, device, medium_dims):
        """Compare vectorized implementation against naive loop."""
        N1, N2, M = medium_dims['N1'], medium_dims['N2'], medium_dims['M']
        num_edges = 500
        seed = 42

        # Random teacher
        torch.manual_seed(seed)
        W = torch.randn(N1, M, device=device)
        X = torch.randn(M, N2, device=device)

        # Random edges
        i_idx = torch.randint(0, N1, (num_edges,), device=device)
        j_idx = torch.randint(0, N2, (num_edges,), device=device)

        F = generate_spreading_coefficients(i_idx, j_idx, M, seed, device)

        # Vectorized
        Y_vectorized = compute_sparse_Y(W, X, F, i_idx, j_idx)

        # Naive loop
        alpha_scale = 1.0 / (M ** 0.5)
        Y_naive = torch.zeros(num_edges, device=device)
        for c in range(num_edges):
            i, j = i_idx[c].item(), j_idx[c].item()
            Y_naive[c] = alpha_scale * (F[c, :] * W[i, :] * X[:, j]).sum()

        max_diff = (Y_vectorized - Y_naive).abs().max().item()
        assert torch.allclose(Y_vectorized, Y_naive, atol=1e-5), \
            f"Vectorized vs naive mismatch: max diff = {max_diff}"

    def test_batched_computation(self, device, small_dims):
        """Test batched Y computation."""
        N1, N2, M = small_dims['N1'], small_dims['N2'], small_dims['M']
        S = 3  # Batch size
        num_edges = 20
        seed = 42

        torch.manual_seed(seed)
        W = torch.randn(S, N1, M, device=device)
        X = torch.randn(S, M, N2, device=device)

        i_idx = torch.randint(0, N1, (num_edges,), device=device)
        j_idx = torch.randint(0, N2, (num_edges,), device=device)

        F = generate_spreading_coefficients(i_idx, j_idx, M, seed, device)

        # Batched computation
        Y_batched = compute_sparse_Y_batched(W, X, F, i_idx, j_idx)

        # Individual computation
        for s in range(S):
            Y_single = compute_sparse_Y(W[s], X[s], F, i_idx, j_idx)
            assert torch.allclose(Y_batched[s], Y_single, atol=1e-5), \
                f"Batch sample {s} mismatch"

    def test_empty_edges_y(self, device, small_dims):
        """Empty edges should return empty Y."""
        N1, N2, M = small_dims['N1'], small_dims['N2'], small_dims['M']

        W = torch.randn(N1, M, device=device)
        X = torch.randn(M, N2, device=device)

        i_idx = torch.empty(0, dtype=torch.long, device=device)
        j_idx = torch.empty(0, dtype=torch.long, device=device)
        F = torch.empty(0, M, device=device)

        Y = compute_sparse_Y(W, X, F, i_idx, j_idx)

        assert Y.shape == (0,), f"Expected shape (0,), got {Y.shape}"


# ============================================================================
# Test 4: RandomSpreadingTeacher Class
# ============================================================================

class TestRandomSpreadingTeacher:
    """Test RandomSpreadingTeacher class."""

    def test_create_wx(self, device, small_dims):
        """Test W, X creation."""
        N1, N2, M = small_dims['N1'], small_dims['N2'], small_dims['M']

        teacher = RandomSpreadingTeacher(spreading_seed=12345)
        W, X = teacher.create(N1, N2, M, device, seed=42)

        assert W.shape == (N1, M), f"W shape mismatch: {W.shape}"
        assert X.shape == (M, N2), f"X shape mismatch: {X.shape}"

        # Check scale (should be 1/√M)
        # Use looser tolerance for small sample sizes
        expected_var = 1.0 / M
        w_var = W.var().item()
        x_var = X.var().item()

        # Allow 50% tolerance for small matrices (N1=10, N2=12, M=5)
        assert abs(w_var - expected_var) < 0.5 * expected_var, \
            f"W variance should be ~{expected_var}, got {w_var}"
        assert abs(x_var - expected_var) < 0.5 * expected_var, \
            f"X variance should be ~{expected_var}, got {x_var}"

    def test_create_with_spreading(self, device, small_dims):
        """Test full spreading creation."""
        N1, N2, M = small_dims['N1'], small_dims['N2'], small_dims['M']
        num_edges = 30

        teacher = RandomSpreadingTeacher(spreading_seed=12345)

        i_idx = torch.randint(0, N1, (num_edges,), device=device)
        j_idx = torch.randint(0, N2, (num_edges,), device=device)

        W, X, spreading_data = teacher.create_with_spreading(
            N1, N2, M, i_idx, j_idx, device, seed=42
        )

        # Check shapes
        assert W.shape == (N1, M)
        assert X.shape == (M, N2)
        assert spreading_data.F.shape == (num_edges, M)
        assert spreading_data.Y_values.shape == (num_edges,)
        assert spreading_data.num_edges == num_edges

    def test_spreading_reproducibility(self, device, small_dims):
        """Same seeds should produce identical results."""
        N1, N2, M = small_dims['N1'], small_dims['N2'], small_dims['M']
        num_edges = 30

        i_idx = torch.randint(0, N1, (num_edges,), device=device)
        j_idx = torch.randint(0, N2, (num_edges,), device=device)

        teacher1 = RandomSpreadingTeacher(spreading_seed=12345)
        teacher2 = RandomSpreadingTeacher(spreading_seed=12345)

        W1, X1, sd1 = teacher1.create_with_spreading(
            N1, N2, M, i_idx, j_idx, device, seed=42
        )
        W2, X2, sd2 = teacher2.create_with_spreading(
            N1, N2, M, i_idx, j_idx, device, seed=42
        )

        assert torch.allclose(W1, W2)
        assert torch.allclose(X1, X2)
        assert torch.allclose(sd1.F, sd2.F)
        assert torch.allclose(sd1.Y_values, sd2.Y_values)

    def test_regenerate_spreading_data(self, device, small_dims):
        """Test regenerating spreading data for new edges."""
        N1, N2, M = small_dims['N1'], small_dims['N2'], small_dims['M']

        teacher = RandomSpreadingTeacher(spreading_seed=12345)

        # First set of edges
        i_idx1 = torch.randint(0, N1, (20,), device=device)
        j_idx1 = torch.randint(0, N2, (20,), device=device)

        W, X, sd1 = teacher.create_with_spreading(
            N1, N2, M, i_idx1, j_idx1, device, seed=42
        )

        # Different set of edges
        i_idx2 = torch.randint(0, N1, (30,), device=device)
        j_idx2 = torch.randint(0, N2, (30,), device=device)

        sd2 = teacher.regenerate_spreading_data(W, X, i_idx2, j_idx2)

        assert sd2.num_edges == 30
        assert sd2.F.shape == (30, M)
        assert sd2.Y_values.shape == (30,)

        # Verify Y is consistent with regenerated F
        Y_recomputed = compute_sparse_Y(W, X, sd2.F, sd2.i_idx, sd2.j_idx)
        assert torch.allclose(sd2.Y_values, Y_recomputed, atol=1e-5)


# ============================================================================
# Test 5: SpreadingData Operations
# ============================================================================

class TestSpreadingData:
    """Test SpreadingData dataclass."""

    def test_to_device(self, device, small_dims):
        """Test moving to different device."""
        M = small_dims['M']
        num_edges = 10

        # Create on CPU
        sd_cpu = SpreadingData(
            i_idx=torch.arange(num_edges),
            j_idx=torch.arange(num_edges),
            F=torch.randn(num_edges, M),
            Y_values=torch.randn(num_edges),
            seed=42,
            M=M,
        )

        # Move to target device
        sd_device = sd_cpu.to(device)

        # Compare device types (cuda vs cpu), ignoring device index
        assert sd_device.F.device.type == device.type
        assert sd_device.Y_values.device.type == device.type
        assert sd_device.i_idx.device.type == device.type
        assert sd_device.j_idx.device.type == device.type

    def test_clone(self, device, small_dims):
        """Test cloning."""
        M = small_dims['M']
        num_edges = 10

        sd = SpreadingData(
            i_idx=torch.arange(num_edges, device=device),
            j_idx=torch.arange(num_edges, device=device),
            F=torch.randn(num_edges, M, device=device),
            Y_values=torch.randn(num_edges, device=device),
            seed=42,
            M=M,
        )

        sd_clone = sd.clone()

        # Modify original
        sd.F[0, 0] = 999.0

        # Clone should be unaffected
        assert sd_clone.F[0, 0] != 999.0


# ============================================================================
# Test 6: BiG-AMP Message Passing Correctness
# ============================================================================

from smf.modules.algorithms.bigamp_spreading import (
    _scatter_add_2d,
    _bigamp_spreading_step_single,
)


class TestScatterAdd:
    """Test scatter_add operations."""

    def test_scatter_add_dim0(self, device):
        """Test scatter along dimension 0 (rows)."""
        # src: 3 edges, 2 hidden dims
        src = torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ], device=device)

        # Aggregate edges 0 and 2 to row 0, edge 1 to row 1
        idx = torch.tensor([0, 1, 0], device=device)

        result = _scatter_add_2d(src, idx, dim_size=3, dim=0)

        expected = torch.tensor([
            [1.0 + 5.0, 2.0 + 6.0],  # row 0: edges 0 and 2
            [3.0, 4.0],               # row 1: edge 1
            [0.0, 0.0],               # row 2: no edges
        ], device=device)

        assert torch.allclose(result, expected), \
            f"Scatter dim=0 mismatch:\nExpected:\n{expected}\nGot:\n{result}"

    def test_scatter_add_dim1(self, device):
        """Test scatter along dimension 1 (columns)."""
        # src: 2 hidden dims, 3 edges -> need to transpose for dim=1
        src = torch.tensor([
            [1.0, 3.0, 5.0],  # hidden dim 0
            [2.0, 4.0, 6.0],  # hidden dim 1
        ], device=device)

        # Aggregate edges 0 and 2 to col 0, edge 1 to col 1
        idx = torch.tensor([0, 1, 0], device=device)

        result = _scatter_add_2d(src, idx, dim_size=3, dim=1)

        expected = torch.tensor([
            [1.0 + 5.0, 3.0, 0.0],  # hidden dim 0
            [2.0 + 6.0, 4.0, 0.0],  # hidden dim 1
        ], device=device)

        assert torch.allclose(result, expected), \
            f"Scatter dim=1 mismatch:\nExpected:\n{expected}\nGot:\n{result}"


class TestBiGAMPMessagePassing:
    """Test BiG-AMP message passing correctness."""

    def test_r_W_computation(self, device, small_dims):
        """
        Test r_W computation: r_W[i,μ] = (1/√M) Σ_{j:(i,j)∈obs} F[ij,μ] s[ij] x[μ,j]

        Compare vectorized scatter_add against naive loop.
        """
        N1, N2, M = small_dims['N1'], small_dims['N2'], small_dims['M']
        num_edges = 50
        seed = 42

        torch.manual_seed(seed)

        # Random data
        x_hat = torch.randn(M, N2, device=device)
        i_idx = torch.randint(0, N1, (num_edges,), device=device)
        j_idx = torch.randint(0, N2, (num_edges,), device=device)
        F = torch.randn(num_edges, M, device=device)
        s_values = torch.randn(num_edges, device=device)

        alpha_scale = 1.0 / (M ** 0.5)

        # Vectorized (using scatter_add)
        x_selected = x_hat[:, j_idx].T  # (C, M)
        r_W_contrib = F * s_values.unsqueeze(1) * x_selected
        r_W_vectorized = alpha_scale * _scatter_add_2d(r_W_contrib, i_idx, N1, dim=0)

        # Naive loop
        r_W_naive = torch.zeros(N1, M, device=device)
        for c in range(num_edges):
            i, j = i_idx[c].item(), j_idx[c].item()
            r_W_naive[i, :] += F[c, :] * s_values[c] * x_hat[:, j]
        r_W_naive *= alpha_scale

        max_diff = (r_W_vectorized - r_W_naive).abs().max().item()
        assert torch.allclose(r_W_vectorized, r_W_naive, atol=1e-5), \
            f"r_W mismatch: max diff = {max_diff}"

    def test_r_X_computation(self, device, small_dims):
        """
        Test r_X computation: r_X[μ,j] = (1/√M) Σ_{i:(i,j)∈obs} F[ij,μ] w[i,μ] s[ij]

        Compare vectorized scatter_add against naive loop.
        """
        N1, N2, M = small_dims['N1'], small_dims['N2'], small_dims['M']
        num_edges = 50
        seed = 42

        torch.manual_seed(seed)

        # Random data
        w_hat = torch.randn(N1, M, device=device)
        i_idx = torch.randint(0, N1, (num_edges,), device=device)
        j_idx = torch.randint(0, N2, (num_edges,), device=device)
        F = torch.randn(num_edges, M, device=device)
        s_values = torch.randn(num_edges, device=device)

        alpha_scale = 1.0 / (M ** 0.5)

        # Vectorized
        w_selected = w_hat[i_idx, :]  # (C, M)
        r_X_contrib = F * s_values.unsqueeze(1) * w_selected  # (C, M)
        r_X_vectorized = alpha_scale * _scatter_add_2d(r_X_contrib.T, j_idx, N2, dim=1)

        # Naive loop
        r_X_naive = torch.zeros(M, N2, device=device)
        for c in range(num_edges):
            i, j = i_idx[c].item(), j_idx[c].item()
            r_X_naive[:, j] += F[c, :] * w_hat[i, :] * s_values[c]
        r_X_naive *= alpha_scale

        max_diff = (r_X_vectorized - r_X_naive).abs().max().item()
        assert torch.allclose(r_X_vectorized, r_X_naive, atol=1e-5), \
            f"r_X mismatch: max diff = {max_diff}"

    def test_tau_W_computation(self, device, small_dims):
        """
        Test τ_W computation:
        τ_W[i,μ] = (1/M) Σ_{j:(i,j)∈obs} F²[ij,μ] × (1/V[ij]) × x²[μ,j]
        """
        N1, N2, M = small_dims['N1'], small_dims['N2'], small_dims['M']
        num_edges = 50
        seed = 42

        torch.manual_seed(seed)

        x_hat = torch.randn(M, N2, device=device)
        i_idx = torch.randint(0, N1, (num_edges,), device=device)
        j_idx = torch.randint(0, N2, (num_edges,), device=device)
        F = torch.randn(num_edges, M, device=device)
        V_values = torch.rand(num_edges, device=device) + 0.1  # Ensure positive

        alpha_scale = 1.0 / (M ** 0.5)
        F_sq = F ** 2

        # Vectorized
        x_sq_sel = x_hat[:, j_idx].T ** 2  # (C, M)
        inv_V = 1.0 / V_values
        tau_W_contrib = F_sq * inv_V.unsqueeze(1) * x_sq_sel
        tau_W_vectorized = (alpha_scale ** 2) * _scatter_add_2d(tau_W_contrib, i_idx, N1, dim=0)

        # Naive loop
        tau_W_naive = torch.zeros(N1, M, device=device)
        for c in range(num_edges):
            i, j = i_idx[c].item(), j_idx[c].item()
            tau_W_naive[i, :] += (F[c, :] ** 2) * (1.0 / V_values[c]) * (x_hat[:, j] ** 2)
        tau_W_naive *= (alpha_scale ** 2)

        max_diff = (tau_W_vectorized - tau_W_naive).abs().max().item()
        assert torch.allclose(tau_W_vectorized, tau_W_naive, atol=1e-5), \
            f"tau_W mismatch: max diff = {max_diff}"

    def test_forward_z_hat_computation(self, device, small_dims):
        """
        Test forward pass z_hat:
        z_hat[c] = (1/√M) Σ_μ F[c,μ] w[i,μ] x[μ,j]
        """
        N1, N2, M = small_dims['N1'], small_dims['N2'], small_dims['M']
        num_edges = 30
        seed = 42

        torch.manual_seed(seed)

        w_hat = torch.randn(N1, M, device=device)
        x_hat = torch.randn(M, N2, device=device)
        i_idx = torch.randint(0, N1, (num_edges,), device=device)
        j_idx = torch.randint(0, N2, (num_edges,), device=device)
        F = torch.randn(num_edges, M, device=device)

        alpha_scale = 1.0 / (M ** 0.5)

        # Vectorized
        w_selected = w_hat[i_idx, :]
        x_selected = x_hat[:, j_idx].T
        z_hat_vectorized = alpha_scale * (F * w_selected * x_selected).sum(dim=1)

        # Naive loop
        z_hat_naive = torch.zeros(num_edges, device=device)
        for c in range(num_edges):
            i, j = i_idx[c].item(), j_idx[c].item()
            z_hat_naive[c] = alpha_scale * (F[c, :] * w_hat[i, :] * x_hat[:, j]).sum()

        max_diff = (z_hat_vectorized - z_hat_naive).abs().max().item()
        assert torch.allclose(z_hat_vectorized, z_hat_naive, atol=1e-5), \
            f"z_hat mismatch: max diff = {max_diff}"


class TestBiGAMPSpreadingIntegration:
    """Integration tests for BiG-AMP spreading."""

    def test_single_step_runs(self, device, small_dims):
        """Test that a single step runs without error."""
        N1, N2, M = small_dims['N1'], small_dims['N2'], small_dims['M']
        num_edges = 30

        torch.manual_seed(42)

        w_hat = torch.randn(N1, M, device=device) * 0.1
        x_hat = torch.randn(M, N2, device=device) * 0.1
        w_var = torch.ones(N1, M, device=device) * 0.2
        x_var = torch.ones(M, N2, device=device) * 0.2

        i_idx = torch.randint(0, N1, (num_edges,), device=device)
        j_idx = torch.randint(0, N2, (num_edges,), device=device)
        F = torch.randn(num_edges, M, device=device)
        Y_values = torch.randn(num_edges, device=device)

        spreading_data = SpreadingData(
            i_idx=i_idx,
            j_idx=j_idx,
            F=F,
            Y_values=Y_values,
            seed=42,
            M=M,
        )

        alpha_scale = 1.0 / (M ** 0.5)

        # Run single step
        w_hat_new, x_hat_new, w_var_new, x_var_new = _bigamp_spreading_step_single(
            w_hat, x_hat, w_var, x_var,
            spreading_data,
            alpha_scale=alpha_scale,
            damping=0.5,
            noise_var=1e-10,
            M=M,
            N1=N1,
            N2=N2,
        )

        # Check shapes unchanged
        assert w_hat_new.shape == (N1, M)
        assert x_hat_new.shape == (M, N2)
        assert w_var_new.shape == (N1, M)
        assert x_var_new.shape == (M, N2)

        # Check variances are positive and bounded
        assert (w_var_new > 0).all()
        assert (x_var_new > 0).all()
        assert (w_var_new <= 1.0).all()
        assert (x_var_new <= 1.0).all()

    def test_convergence_with_perfect_init(self, device, small_dims):
        """
        Test that starting from perfect initialization stays perfect.

        If w_hat = W_true, x_hat = X_true initially, the estimates
        should remain close after one step.
        """
        N1, N2, M = small_dims['N1'], small_dims['N2'], small_dims['M']
        num_edges = N1 * M // 2  # Reasonable observation density

        # Create teacher
        teacher = RandomSpreadingTeacher(spreading_seed=12345)
        i_idx = torch.randint(0, N1, (num_edges,), device=device)
        j_idx = torch.randint(0, N2, (num_edges,), device=device)

        W_true, X_true, spreading_data = teacher.create_with_spreading(
            N1, N2, M, i_idx, j_idx, device, seed=42
        )

        # Initialize at true values
        w_hat = W_true.clone()
        x_hat = X_true.clone()
        w_var = torch.ones(N1, M, device=device) * 1e-6  # Low variance = high confidence
        x_var = torch.ones(M, N2, device=device) * 1e-6

        alpha_scale = 1.0 / (M ** 0.5)

        # Run one step
        w_hat_new, x_hat_new, _, _ = _bigamp_spreading_step_single(
            w_hat, x_hat, w_var, x_var,
            spreading_data,
            alpha_scale=alpha_scale,
            damping=0.9,  # High damping to stay close
            noise_var=1e-10,
            M=M,
            N1=N1,
            N2=N2,
        )

        # Should stay very close to true values
        w_diff = (w_hat_new - W_true).abs().mean().item()
        x_diff = (x_hat_new - X_true).abs().mean().item()

        assert w_diff < 0.1, f"W drifted too much: mean diff = {w_diff}"
        assert x_diff < 0.1, f"X drifted too much: mean diff = {x_diff}"


# ============================================================================
# Test 7: Q_Y Evaluation Consistency
# ============================================================================

from smf.modules.metrics.spreading import (
    compute_qy_spreading,
    compute_mse_spreading,
    compute_all_metrics_spreading,
    compute_qy_with_wrong_f,
)


class TestQYSpreadingEvaluation:
    """Test Q_Y evaluation for spreading model."""

    def test_perfect_recovery_gives_qy_one(self, device, small_dims):
        """
        When student = teacher, Q_Y should be 1.0 (with same F).
        """
        N1, N2, M = small_dims['N1'], small_dims['N2'], small_dims['M']
        num_edges = 50

        teacher = RandomSpreadingTeacher(spreading_seed=12345)
        i_idx = torch.randint(0, N1, (num_edges,), device=device)
        j_idx = torch.randint(0, N2, (num_edges,), device=device)

        W, X, spreading_data = teacher.create_with_spreading(
            N1, N2, M, i_idx, j_idx, device, seed=42
        )

        # Student = Teacher (perfect recovery)
        W_student = W.clone()
        X_student = X.clone()

        Q_Y = compute_qy_spreading(W_student, X_student, spreading_data)

        assert abs(Q_Y - 1.0) < 1e-5, \
            f"Perfect recovery should give Q_Y ≈ 1.0, got {Q_Y}"

    def test_wrong_f_gives_low_qy(self, device, small_dims):
        """
        Using different F should give low Q_Y even with perfect W, X.
        """
        N1, N2, M = small_dims['N1'], small_dims['N2'], small_dims['M']
        num_edges = 100  # More edges for statistical stability

        teacher = RandomSpreadingTeacher(spreading_seed=12345)
        i_idx = torch.randint(0, N1, (num_edges,), device=device)
        j_idx = torch.randint(0, N2, (num_edges,), device=device)

        W, X, spreading_data = teacher.create_with_spreading(
            N1, N2, M, i_idx, j_idx, device, seed=42
        )

        # Correct F gives Q_Y = 1
        Q_Y_correct = compute_qy_spreading(W, X, spreading_data)
        assert Q_Y_correct > 0.999

        # Wrong F should give low Q_Y
        Q_Y_wrong = compute_qy_with_wrong_f(W, X, spreading_data, wrong_seed=99999)

        # With random F, Q_Y should be close to 0 (uncorrelated)
        assert abs(Q_Y_wrong) < 0.3, \
            f"Wrong F should give low |Q_Y|, got {Q_Y_wrong}"

    def test_mse_perfect_recovery(self, device, small_dims):
        """MSE should be 0 for perfect recovery."""
        N1, N2, M = small_dims['N1'], small_dims['N2'], small_dims['M']
        num_edges = 50

        teacher = RandomSpreadingTeacher(spreading_seed=12345)
        i_idx = torch.randint(0, N1, (num_edges,), device=device)
        j_idx = torch.randint(0, N2, (num_edges,), device=device)

        W, X, spreading_data = teacher.create_with_spreading(
            N1, N2, M, i_idx, j_idx, device, seed=42
        )

        MSE = compute_mse_spreading(W, X, spreading_data)

        assert MSE < 1e-10, f"Perfect recovery should give MSE ≈ 0, got {MSE}"

    def test_all_metrics_spreading(self, device, small_dims):
        """Test compute_all_metrics_spreading returns expected keys."""
        N1, N2, M = small_dims['N1'], small_dims['N2'], small_dims['M']
        num_edges = 50

        teacher = RandomSpreadingTeacher(spreading_seed=12345)
        i_idx = torch.randint(0, N1, (num_edges,), device=device)
        j_idx = torch.randint(0, N2, (num_edges,), device=device)

        W_t, X_t, spreading_data = teacher.create_with_spreading(
            N1, N2, M, i_idx, j_idx, device, seed=42
        )

        # Student slightly different from teacher
        torch.manual_seed(123)
        W_s = W_t + torch.randn_like(W_t) * 0.1
        X_s = X_t + torch.randn_like(X_t) * 0.1

        metrics = compute_all_metrics_spreading(
            W_s, X_s, W_t, X_t, spreading_data
        )

        # Check all expected keys present
        expected_keys = {'Q_Y', 'Q_W', 'Q_X', 'Q_W_prime', 'Q_X_prime', 'MSE'}
        assert set(metrics.keys()) == expected_keys, \
            f"Missing keys: {expected_keys - set(metrics.keys())}"

        # Check ranges
        assert 0 <= metrics['Q_Y'] <= 1, f"Q_Y out of range: {metrics['Q_Y']}"
        assert 0 <= metrics['Q_W_prime'] <= 1, f"Q_W' out of range"
        assert 0 <= metrics['Q_X_prime'] <= 1, f"Q_X' out of range"
        assert metrics['MSE'] >= 0, f"MSE should be non-negative"

    def test_batched_qy_computation(self, device, small_dims):
        """Test Q_Y computation with batched student matrices."""
        N1, N2, M = small_dims['N1'], small_dims['N2'], small_dims['M']
        S = 3  # Batch size
        num_edges = 50

        teacher = RandomSpreadingTeacher(spreading_seed=12345)
        i_idx = torch.randint(0, N1, (num_edges,), device=device)
        j_idx = torch.randint(0, N2, (num_edges,), device=device)

        W_t, X_t, spreading_data = teacher.create_with_spreading(
            N1, N2, M, i_idx, j_idx, device, seed=42
        )

        # Create batched student = replicated teacher
        W_s = W_t.unsqueeze(0).expand(S, -1, -1).clone()
        X_s = X_t.unsqueeze(0).expand(S, -1, -1).clone()

        Q_Y = compute_qy_spreading(W_s, X_s, spreading_data)

        # All samples are perfect, so mean Q_Y should be 1
        assert abs(Q_Y - 1.0) < 1e-5, \
            f"Batched perfect recovery should give Q_Y ≈ 1.0, got {Q_Y}"


class TestQYGradualRecovery:
    """Test that Q_Y correlates with recovery quality."""

    def test_qy_increases_with_noise_decrease(self, device, small_dims):
        """
        As noise level decreases, Q_Y should increase.
        """
        N1, N2, M = small_dims['N1'], small_dims['N2'], small_dims['M']
        num_edges = 100

        teacher = RandomSpreadingTeacher(spreading_seed=12345)
        i_idx = torch.randint(0, N1, (num_edges,), device=device)
        j_idx = torch.randint(0, N2, (num_edges,), device=device)

        W_t, X_t, spreading_data = teacher.create_with_spreading(
            N1, N2, M, i_idx, j_idx, device, seed=42
        )

        noise_levels = [1.0, 0.5, 0.1, 0.01, 0.0]
        qy_values = []

        for noise_scale in noise_levels:
            torch.manual_seed(999)
            W_s = W_t + torch.randn_like(W_t) * noise_scale
            X_s = X_t + torch.randn_like(X_t) * noise_scale

            Q_Y = compute_qy_spreading(W_s, X_s, spreading_data)
            qy_values.append(Q_Y)

        # Q_Y should be monotonically increasing as noise decreases
        for i in range(len(qy_values) - 1):
            assert qy_values[i] <= qy_values[i + 1] + 0.05, \
                f"Q_Y should increase as noise decreases: {qy_values}"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
