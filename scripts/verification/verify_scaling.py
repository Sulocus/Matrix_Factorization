import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import torch
import math
from matrix_factorization.modules.algorithms.bigamp.tensor_supergraph import create_tensor_superdata
from matrix_factorization.modules.algorithms.bigamp.tensor_spreading_parallel import forward_pass_tensor_super

class MockSuperGraph:
    def __init__(self, S, C_max, alpha_mask):
        self.S = S
        self.C_max = C_max
        self.alpha_mask = alpha_mask
        self.indices = torch.randint(0, 50, (S, C_max, 3)).long()

    def get_offset_indices(self):
        # Flatten indices
        flat_indices = self.indices.reshape(-1, 3)
        return [flat_indices[:, d].long() for d in range(3)]

def verify_scaling():
    S = 10
    M = 10
    N = 50
    C_max = 200 # approximate for alpha=4
    
    # 1. Setup Mock Data
    print("=== 1. Setup ===")
    dims = [N, N, N]
    alpha_mask = torch.ones(S, C_max) # All valid for simplicity
    supergraph = MockSuperGraph(S, C_max, alpha_mask)
    
    # Teacher ~ N(0, 1/M)
    teacher_factors = [torch.randn(N, M) * (1.0 / math.sqrt(M)) for _ in range(3)]
    
    # 2. Generate Y
    print("=== 2. Generating Y from Teacher (expecting O(1/M^1.5) or O(1/M)?) ===")
    # Replicate logic from create_tensor_superdata roughly or use it if possible
    # But better to use the actual function if we can mock dependecies.
    # Let's manually compute Y to verify theoretical expectation.
    
    indices = supergraph.indices
    F = (torch.randint(0, 2, (S, C_max, M), dtype=torch.int8) * 2 - 1).float()
    
    # Gather teacher factors
    # (S, C, M)
    X0 = teacher_factors[0][indices[:, :, 0]]
    X1 = teacher_factors[1][indices[:, :, 1]]
    X2 = teacher_factors[2][indices[:, :, 2]]
    
    # Product
    # Teacher ~ 0.316. Product ~ 0.0316. 
    prod = X0 * X1 * X2 
    
    # Alpha scale
    alpha_scale = 1.0 / math.sqrt(M) # 0.316
    
    # Y = alpha_scale * Sum(F * prod)
    # Sum over M=10 terms. Each term ~ O(0.0316). F is random sign.
    # Sum variance = 10 * Var(term).
    Y = alpha_scale * (F * prod).sum(dim=2)
    
    print(f"Teacher Std: {teacher_factors[0].std():.6f}")
    print(f"Product Mean: {prod.mean():.6e}, Std: {prod.std():.6e}")
    print(f"Y Mean: {Y.mean():.6e}, Std: {Y.std():.6e}, Max: {Y.max():.6e}")
    
    # 3. Student Initialization (Rescaled)
    print("\n=== 3. Student Initialization (Rescaled to 1/sqrt(M)) ===")
    student_factors = [torch.randn(N, M) * (1.0 / math.sqrt(M)) for _ in range(3)]
    
    # Expand student factors for forward pass simulation
    # In real code, factors are (A, S*C_max, M) if we use 'factors' in step_fn?
    # No, step_fn takes 'factors' as list of tensors.
    # In parallel mode, factors are usually shape (A=1 for now, S*N, M).
    # But let's simplify: just use N, M factors and gather them.
    
    SX0 = student_factors[0][indices[:, :, 0]]
    SX1 = student_factors[1][indices[:, :, 1]]
    SX2 = student_factors[2][indices[:, :, 2]]
    
    sprod = SX0 * SX1 * SX2
    Z_hat = alpha_scale * (F * sprod).sum(dim=2)
    
    print(f"Student Factor Std: {student_factors[0].std():.6f}")
    print(f"Z_hat Mean: {Z_hat.mean():.6e}, Std: {Z_hat.std():.6e}, Max: {Z_hat.max():.6e}")
    
    # 4. MSE Calculation
    MSE = ((Y - Z_hat)**2).mean()
    print(f"\nMSE: {MSE.item():.6f}")
    
    # === 3b. Verify forward_pass_tensor_super function ===
    print("\n=== 3b. Verify forward_pass_tensor_super function ===")
    
    # Needs: factors, F_flat, offset_indices, S, N_dims
    # Flatten F
    F_flat = F.reshape(-1, M) # (S*C_max, M)
    offset_indices = supergraph.get_offset_indices() # List of (S*C_max,)
    
    # Factors needs to be list of (S*N, M) or similar? 
    # tensor_step_super expects factors as [N_d_tensor, ...] ?
    # In parallel mode: factors are (1, S*N, M). 
    # forward_pass_tensor_super checks: A = factors[0].shape[0]
    
    # Reshape mock factors to match parallel mode expected input
    # Mock factors were (N, M). We need (1, S*N, M) full expanded?
    # No, usually we just pass the factors. In parallel mode, factors are maintained as (A, S*N, M).
    # Here A=1 (mock).
    
    # We need to construct full factors (A, S*N, M)
    # But wait, our mock factors 'student_factors' are just (N, M) * 3.
    # In parallel mode, each sample has its own copy if independent, 
    # OR we share factors? 
    # Code: factors = [torch.randn(A, S * N, M) ...]
    
    # Let's construct correct shape inputs
    A = 1
    full_factors = []
    for d in range(3):
         # shape (A, S*N, M). 
         # Simplest: repeat student_factors[d] S times?
         # student_factors[d] is (N, M)
         f = student_factors[d].unsqueeze(0).repeat(1, S, 1).reshape(1, S*N, M)
         full_factors.append(f)
    
    N_dims = [N, N, N]
    
    # Call function
    # Note: offset_indices in parallel code are (S*C*A) ? No.
    # get_offset_indices returns indices into (S*N).
    # indices: (S, C, 3). Flatten -> (S*C, 3).
    # The offset indices should point to the correct sample block.
    # sample s, node i -> index = s*N + i.
    
    # We need to fixing mock offset_indices
    # Current MockSuperGraph indices are in range [0, 50).
    # They need to be offset by s*N.
    
    real_indices = supergraph.indices.clone() # (S, C, 3)
    for s in range(S):
        real_indices[s] += s * N
            
    flat_indices = real_indices.reshape(-1, 3)
    real_offset_indices = [flat_indices[:, d].long() for d in range(3)]
    
    z_out = forward_pass_tensor_super(
        full_factors, F_flat, real_offset_indices, S, N_dims
    )
    # z_out shape: (A, S*C_max) -> (1, S*C_max)
    
    # Compare z_out with Manual Z_hat
    # Manual Z_hat was (S, C, M) -> sum -> (S, C). Flatten -> (S*C).
    # But wait, manual Z_hat used un-offset indices into (N, M) factors.
    # Function uses offset indices into (S*N, M) factors.
    # If factors are repeated, result should be identical.
    
    z_out_flat = z_out.flatten()
    z_manual_flat = Z_hat.flatten()
    
    diff = (z_out_flat - z_manual_flat).abs().max()
    print(f"Function Z_out Mean: {z_out_flat.mean():.6e}, Std: {z_out_flat.std():.6e}")
    print(f"Difference from Manual: {diff:.6e}")
    
    if diff > 1e-5:
        print("!!! FUNCTION IMPLEMENTATION MISMATCH !!!")
    else:
        print("Function matches manual calculation.")

if __name__ == "__main__":
    verify_scaling()
