# tests/test_tensor_step.py
"""Tests for N-dimensional tensor step functions."""

import pytest
import math
import torch


class TestForwardPassTensor:
    """Test forward_pass_tensor function."""

    def test_forward_pass_n2_matches_manual(self):
        """Test n=2 forward pass against manual calculation."""
        from matrix_factorization.modules.algorithms.bigamp.tensor_step import forward_pass_tensor
        
        # n=2, C=2, M=3
        factors = [
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),  # (2, 3) = (N0, M)
            torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),  # (2, 3) = (N1, M)
        ]
        F = torch.tensor([[1.0, 1.0, 1.0], [-1.0, 1.0, -1.0]])  # (C=2, M=3)
        indices = [torch.tensor([0, 1]), torch.tensor([1, 0])]  # 2 edges
        
        Z = forward_pass_tensor(factors, F, indices)
        
        # Manual: edge0 = (1/√3) * sum([1,2,3] * [0.4,0.5,0.6] * [1,1,1])
        #       = (1/√3) * (0.4 + 1.0 + 1.8) = 3.2/√3
        expected_0 = (1/math.sqrt(3)) * (1*0.4*1 + 2*0.5*1 + 3*0.6*1)
        assert abs(Z[0].item() - expected_0) < 1e-5

    def test_forward_pass_n3(self):
        """Test n=3 forward pass."""
        from matrix_factorization.modules.algorithms.bigamp.tensor_step import forward_pass_tensor
        
        factors = [
            torch.ones(2, 4),   # (N0, M)
            torch.ones(3, 4) * 2,  # (N1, M)
            torch.ones(4, 4) * 0.5,  # (N2, M)
        ]
        F = torch.ones(5, 4)  # (C=5, M=4)
        indices = [
            torch.tensor([0, 1, 0, 1, 0]),
            torch.tensor([0, 1, 2, 0, 1]),
            torch.tensor([0, 1, 2, 3, 0]),
        ]
        
        Z = forward_pass_tensor(factors, F, indices)
        
        # Each edge: (1/√4) * sum(1 * 2 * 0.5 * 1) for 4 dims = (1/2) * 4 = 2.0
        expected = 2.0
        assert abs(Z[0].item() - expected) < 1e-5


class TestVarianceTensor:
    """Test compute_variance_tensor function."""
    
    def test_variance_n3_positive(self):
        """Test n=3 variance is positive."""
        from matrix_factorization.modules.algorithms.bigamp.tensor_step import compute_variance_tensor
        
        factors = [torch.ones(2, 3), torch.ones(2, 3) * 2, torch.ones(2, 3) * 0.5]
        factor_vars = [torch.ones(2, 3) * 0.1] * 3
        F = torch.ones(1, 3)
        indices = [torch.tensor([0]), torch.tensor([1]), torch.tensor([0])]
        
        V = compute_variance_tensor(factors, factor_vars, F, indices)
        
        assert V.shape == (1,)
        assert (V > 0).all()

    def test_variance_ising_optimization(self):
        """Test Ising optimization (F²=1)."""
        from matrix_factorization.modules.algorithms.bigamp.tensor_step import compute_variance_tensor
        
        factors = [torch.randn(5, 4), torch.randn(5, 4)]
        factor_vars = [torch.ones(5, 4) * 0.2] * 2
        F = torch.tensor([[1, -1, 1, -1], [-1, 1, -1, 1]]).float()  # Ising
        indices = [torch.tensor([0, 1]), torch.tensor([2, 3])]
        
        V_normal = compute_variance_tensor(factors, factor_vars, F, indices, is_ising=False)
        V_opt = compute_variance_tensor(factors, factor_vars, F, indices, is_ising=True)
        
        # For true Ising F, both should give same result
        assert torch.allclose(V_normal, V_opt, atol=1e-5)


class TestTensorStep:
    """Test complete tensor_step function."""

    def test_tensor_step_shapes(self):
        """Test that tensor_step returns correct shapes."""
        from matrix_factorization.modules.algorithms.bigamp.tensor_step import tensor_step
        
        n, N, M, C = 3, 10, 5, 20
        factors = [torch.randn(N, M) for _ in range(n)]
        factor_vars = [torch.ones(N, M) for _ in range(n)]
        Y = torch.randn(C)
        F = torch.randn(C, M)
        indices = [torch.randint(0, N, (C,)) for _ in range(n)]
        
        new_factors, new_vars, s, svar = tensor_step(
            factors, factor_vars, Y, F, indices,
            damping=0.5, noise_var=1e-6
        )
        
        assert len(new_factors) == n
        assert all(f.shape == (N, M) for f in new_factors)
        assert len(new_vars) == n
        assert all(v.shape == (N, M) for v in new_vars)
        assert s.shape == (C,)
        assert svar.shape == (C,)

    def test_tensor_step_decreases_error(self):
        """Test that multiple steps decrease reconstruction error."""
        from matrix_factorization.modules.algorithms.bigamp.tensor_step import tensor_step, forward_pass_tensor
        
        torch.manual_seed(42)
        n, N, M, C = 2, 20, 5, 50
        
        # Teacher
        teacher = [torch.randn(N, M) * 0.1 for _ in range(n)]
        indices = [torch.randint(0, N, (C,)) for _ in range(n)]
        F = (torch.randint(0, 2, (C, M)) * 2 - 1).float()
        Y = forward_pass_tensor(teacher, F, indices)
        
        # Student (random init)
        factors = [torch.randn(N, M) * 0.1 for _ in range(n)]
        factor_vars = [torch.ones(N, M) for _ in range(n)]
        
        # Initial error
        Y_init = forward_pass_tensor(factors, F, indices)
        mse_init = ((Y - Y_init) ** 2).mean().item()
        
        # Run steps
        for _ in range(20):
            factors, factor_vars, _, _ = tensor_step(
                factors, factor_vars, Y, F, indices,
                damping=0.5, noise_var=1e-6
            )
        
        # Final error
        Y_final = forward_pass_tensor(factors, F, indices)
        mse_final = ((Y - Y_final) ** 2).mean().item()
        
        # Error should decrease
        assert mse_final < mse_init
