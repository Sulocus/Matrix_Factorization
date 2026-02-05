
import sys
import os
import torch
import math
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from matrix_factorization.modules.teachers.random_spreading import RandomSpreadingTeacher
from matrix_factorization.modules.algorithms.bigamp.step import bigamp_step_disjoint_union_flat

def verify_bipartite():
    print("=== Bipartite Scaling Verification ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    N = 1000
    M = 10
    alpha = 1.0 # ratio
    C_max = int(alpha * N * M)
    S = 1 # samples
    
    # 1. Teacher Gen
    print("\n--- Teacher Generation ---")
    teacher = RandomSpreadingTeacher(spreading_seed=123)
    W, X = teacher.create(N, N, M, device, seed=42)
    
    print(f"Teacher W Mean: {W.mean():.4f}, Std: {W.std():.4f}, Var: {W.var():.4f}")
    print(f"Teacher X Mean: {X.mean():.4f}, Std: {X.std():.4f}, Var: {X.var():.4f}")
    
    # Check interaction
    # Y = 1/sqrt(M) * F * W * X
    F = torch.randn(S, C_max, M, device=device)
    # Mock indices
    i_idx = torch.randint(0, N, (S, C_max), device=device)
    j_idx = torch.randint(0, N, (S, C_max), device=device)
    
    alpha_scale = 1.0/math.sqrt(M)
    
    # Compute Y manually
    Y_vals = []
    for s in range(S):
         w_sel = W[i_idx[s]]
         x_sel = X.T[j_idx[s]]
         y = alpha_scale * (F[s] * w_sel * x_sel).sum(dim=1)
         Y_vals.append(y)
    Y = torch.cat(Y_vals)
    print(f"Y Mean: {Y.mean():.4f}, Std: {Y.std():.4f}, Var: {Y.var():.4f}")
    
    # 2. Student Update
    print("\n--- Student Update Step 0 ---")
    # Init like Bipartite code: randn * 0.1
    # Create flat structures
    A=1
    W_flat = torch.randn(A, S*N, M, device=device) * 0.1
    X_flat = torch.randn(A, S*N, M, device=device) * 0.1
    W_var_flat = torch.ones(A, S*N, M, device=device)
    X_var_flat = torch.ones(A, S*N, M, device=device)
    
    # Prepare Inputs
    F_flat = F.reshape(S*C_max, M)
    Y_flat = Y.reshape(S*C_max)
    
    # Offsets
    i_offset = i_idx.reshape(-1)
    j_offset = j_idx.reshape(-1)
    
    alpha_mask = torch.ones(A, S*C_max, device=device, dtype=torch.bool)
    
    print(f"Student Init W Mean: {W_flat.mean():.4f}, Std: {W_flat.std():.4f}")
    print(f"Prior Precision Assumption: M = {M}")
    
    # Run Step
    W_new, X_new, W_v_new, X_v_new, s = bigamp_step_disjoint_union_flat(
        W_flat, X_flat, W_var_flat, X_var_flat,
        Y_flat, F_flat, i_offset, j_offset, alpha_mask,
        S, N, N, damping=1.0, noise_var=1e-5, is_rademacher=False
    )
    
    print("\n--- After Step 0 ---")
    print(f"Student New W Mean: {W_new.mean():.4f}, Std: {W_new.std():.4f}")
    print(f"Student New Var Mean: {W_v_new.mean():.4f} (Expected ~1/M if M-term dominates)")
    
    return

if __name__ == "__main__":
    verify_bipartite()
