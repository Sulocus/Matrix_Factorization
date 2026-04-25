
import torch
import unittest
from matrix_factorization.modules.metrics.tensor_metrics import (
    compute_factor_gram_overlap,
    compute_tensor_cosine,
)

class TestTensorMetrics(unittest.TestCase):
    def test_tensor_cosine_correctness(self):
        """
        Verify that optimized tensor cosine matches brute-force calculation.
        Use small dimensions to keep brute-force feasible.
        """
        N = 10
        M = 3
        order = 3
        device = torch.device('cpu')
        
        # Create random factors
        torch.manual_seed(42)
        factors_a = [torch.randn(N, M) for _ in range(order)]
        factors_b = [torch.randn(N, M) for _ in range(order)]
        
        # 1. Optimized Calculation
        optimized_sim = compute_tensor_cosine(factors_a, factors_b)
        
        # 2. Brute Force Calculation
        # Reconstruct Tensor A and B
        # T_ijk = sum_m A_im * B_jm * C_km
        tensor_a = self._reconstruct_tensor(factors_a)
        tensor_b = self._reconstruct_tensor(factors_b)
        
        # Flatten and compute cosine
        vec_a = tensor_a.flatten()
        vec_b = tensor_b.flatten()
        
        brute_force_sim = torch.dot(vec_a, vec_b) / (torch.norm(vec_a) * torch.norm(vec_b))
        
        print(f"\n--- Tensor Metric Validation ---")
        print(f"Optimized:   {optimized_sim:.6f}")
        print(f"Brute Force: {brute_force_sim:.6f}")
        
        self.assertAlmostEqual(optimized_sim, brute_force_sim.item(), places=5)

    def test_factor_gram_overlap_is_cp_permutation_invariant(self):
        """
        CP factors are equivalent under shared latent-column permutations.
        Factor diagnostics used for heatmaps must respect that gauge symmetry.
        """
        N = 10
        M = 3
        order = 3

        torch.manual_seed(123)
        factors = [torch.randn(N, M) for _ in range(order)]
        permutation = torch.tensor([2, 0, 1])
        permuted = [factor[:, permutation] for factor in factors]

        self.assertAlmostEqual(
            compute_tensor_cosine(factors, permuted),
            1.0,
            places=5,
        )
        self.assertAlmostEqual(
            compute_factor_gram_overlap(factors, permuted),
            1.0,
            places=5,
        )

    def _reconstruct_tensor(self, factors):
        # Brute force Einsum reconstruction for Order 3
        # Supports general order via recursion or explicit formula
        order = len(factors)
        N, M = factors[0].shape
        
        if order == 3:
            # A_im, B_jm, C_km -> ijk
            return torch.einsum('im,jm,km->ijk', factors[0], factors[1], factors[2])
        elif order == 2:
            return torch.einsum('im,jm->ij', factors[0], factors[1])
        else:
            raise NotImplementedError("Brute force helper only for order 2 or 3")

if __name__ == '__main__':
    unittest.main()
