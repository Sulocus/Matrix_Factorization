
import pytest
import torch
import logging
from matrix_factorization.modules.algorithms.bigamp.tensor_spreading_parallel import BiGAMPTensorSpreadingParallel

pytest.skip(
    "local-GPU tensor Onsager diagnostic: large direct algorithm run, not default pytest",
    allow_module_level=True,
)

logging.basicConfig(level=logging.WARN)

def test_onsager_impact():
    print("=== Testing Onsager Correction Impact ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    N = 200
    M = 50
    # Lower steps to save time, seeing trend is enough
    MAX_STEPS = 50 
    
    # Setup Teacher
    torch.manual_seed(42)
    W = torch.randn(N, M, device=device) * 1.0 # Unit Variance
    X = torch.randn(M, N, device=device) * 1.0
    
    alpha_val = [10.0] # Use 10.0 to see if Onsager improves the q=0.14 result
    
    def run_trial(use_onsager):
        print(f"\n--- Running with Onsager={use_onsager} ---")
        algorithm = BiGAMPTensorSpreadingParallel(
            tensor_order=3,
            dims=(N, N, N),
            M=M,
            S=10,
            max_steps=MAX_STEPS,
            onsager_correction=use_onsager,
            device=device
        )
        teacher_factors = algorithm._create_teacher_factors(W, X)
        
        result = algorithm._train_full_parallel(
            teacher_factors=teacher_factors,
            alpha_values=alpha_val,
            seed=100,
            device=device
        )
        
        q_y = result['Q_Y_per_alpha'][0]
        q_full = result['Q_Full_Tensor'][0] if 'Q_Full_Tensor' in result else 0.0
        print(f"Result (Onsager={use_onsager}): Q_Y={q_y:.4f}, Q_Full={q_full:.4f}")
        return q_full

    # Baseline (False)
    q_no = run_trial(False)
    
    # Test (True)
    q_yes = run_trial(True)
    
    print("\n=== Summary ===")
    print(f"No Onsager: {q_no:.4f}")
    print(f"Onsager:    {q_yes:.4f}")
    
    if q_yes > q_no + 0.05:
        print("✅ Onsager significantly improved convergence.")
    elif q_yes < q_no - 0.05:
        print("❌ Onsager degraded performance.")
    else:
        print("⏸️ No significant difference observed.")

if __name__ == "__main__":
    test_onsager_impact()
