import torch
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from smf.modules.algorithms.bigamp_spreading_parallel import bigamp_step_disjoint_union_flat

def run_reproduction():
    print("=== Reproduction Script: BigAMP Spreading Scatter Crash ===")
    
    # Parameters matching the crash (600x600, M=150, S=20, Alpha=4.0)
    N1 = 600
    N2 = 600
    M = 150
    S = 20
    # Alpha=4.0 implies C_max = 4 * N * M / ? No, C = Alpha * N * M
    # Check logic: C = alpha * M * N1? Or M * N1? 
    # Usually M measurements per signal element? No, M is measurements.
    # Standard CS: M = alpha * N.
    # But here logic is: C_max = alpha_max * M * N1 is wrong.
    # Let's check bigamp_spreading_parallel.py logic:
    # C_max = max(1, int(alpha_max * M * N1)) <- Line 322 in memory_estimator.
    # Wait, usually C is number of edges.
    # If standard Spreading: M measurement nodes, N variable nodes.
    # Each measurement connected to L variables? Or dense?
    # Spreading usually random sparse.
    # Let's trust the error trace or memory estimator: "buf1... (1, 7200000, 150)"
    # 7,200,000 = 20 * 360,000. S=20. So C_max = 360,000.
    # If N=600, N*N=360,000. It looks like fully connected or close to it?
    # Or maybe it's 600*600 matrix.
    
    SC = 7200000
    A = 1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create dummy tensors
    print("Allocating tensors...")
    W_flat = torch.randn(A, S*N1, M, device=device, dtype=torch.float32)
    X_flat = torch.randn(A, S*N2, M, device=device, dtype=torch.float32)
    W_var = torch.ones_like(W_flat)
    X_var = torch.ones_like(X_flat)
    
    Y_flat = torch.randn(SC, device=device)
    F_flat = torch.randn(SC, M, device=device)
    
    # Random indices mapping to valid range
    i_offset = torch.randint(0, S*N1, (SC,), device=device)
    j_offset = torch.randint(0, S*N2, (SC,), device=device)
    
    alpha_mask = torch.ones(A, SC, device=device)
    
    print("Compiling function...")
    compiled_step = torch.compile(bigamp_step_disjoint_union_flat, mode='default', fullgraph=False)
    
    print("Running Step 1 (should trigger compilation)...")
    t0 = time.time()
    try:
        with torch.no_grad():
            compiled_step(
                W_flat, X_flat, W_var, X_var,
                Y_flat, F_flat,
                i_offset, j_offset,
                alpha_mask,
                S, N1, N2,
                0.5, 1e-10
            )
        torch.cuda.synchronize()
        print(f"Step 1 done in {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"Step 1 FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_reproduction()
