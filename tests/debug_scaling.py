
import torch
import math
from matrix_factorization.modules.algorithms.bigamp.tensor_supergraph import create_tensor_supergraph, create_tensor_superdata
from matrix_factorization.modules.algorithms.bigamp.tensor_step_super import tensor_step_super

def probe_scaling():
    print("=== Probing Tensor Algorithm Scaling ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Setup Full Connectivity Case
    N = 50 
    M = 10
    S = 10
    # Full connectivity: C = N*N*N. But our formula is C = alpha * N * M.
    # To get effective full connectivity approx, alpha = N^2/M? 
    # User said alpha=4 is full connectivity (Avg Degree=N) which is WRONG for N^3 case but let's stick to their definition.
    # Let's test EXACTLY the case Alpha=4.0 with N=50, M=10 -> Degree=20. (Not full, but dense-ish)
    
    dims = (N, N, N)
    alpha = 4.0
    
    print(f"Config: N={N}, M={M}, Alpha={alpha}")
    
    torch.manual_seed(42)
    
    # 2. Creates
    supergraph = create_tensor_supergraph(dims, [alpha], M, S, 1, device)
    print(f"Edges (C_max) = {supergraph.C_max}")
    print(f"Avg Degree = {supergraph.C_max / N:.2f}")
    
    # 3. Create Teacher Factors (Unit Variance)
    teacher_factors = [torch.randn(N, M, device=device) for _ in range(3)]
    
    # 4. Create Y using current logic
    superdata = create_tensor_superdata(supergraph, teacher_factors, 'rademacher', 2)
    Y_flat = superdata.Y_super.reshape(-1)
    
    Y_mean = Y_flat.mean().item()
    Y_std = Y_flat.std().item()
    print(f"Y Stats: Mean={Y_mean:.4f}, Std={Y_std:.4f} (Expected ~1.0?)")
    
    # 5. Run One Step of AMP
    factors = [torch.randn(1, S*N, M, device=device) for _ in range(3)]
    factor_vars = [torch.ones(1, S*N, M, device=device) for _ in range(3)]
    
    # Call step
    new_factors, new_vars, s_values = tensor_step_super(
        factors=factors,
        factor_vars=factor_vars,
        Y_flat=Y_flat,
        F_flat=superdata.F_super.reshape(-1, M),
        offset_indices=supergraph.get_offset_indices(),
        S=S,
        N_dims=[N]*3,
        M=M,
        alpha_mask=superdata.alpha_mask_exp,
        damping=0.5,
        noise_var=1e-5,
        is_rademacher=True,
        onsager_correction=False
    )
    
    # 6. Analyze S (Residuals)
    # S = (Y - Z) / (V + noise)
    # If scaling is wrong, S will be huge or tiny.
    s_mean = s_values.mean().item()
    s_std = s_values.std().item()
    print(f"Residual (s) Stats: Mean={s_mean:.4f}, Std={s_std:.4f}")
    
    # 7. Analyze Update (Tau, R)
    # We can't easily see internal r/tau inside the function without modifying it.
    # But we can see the output new_factors.
    # If output exploded, we know.
    
    f_new_std = new_factors[0].std().item()
    print(f"New Factor Std: {f_new_std:.4f} (Expected ~1.0)")
    
    if f_new_std < 0.01:
        print("FAIL: Gradient Vanished (Stuck at 0)")
    elif f_new_std > 10.0:
        print("FAIL: Gradient Exploded")
    else:
        print("PASS: Gradient Healthy")

if __name__ == "__main__":
    probe_scaling()
