
import torch
import logging
from matrix_factorization.modules.algorithms.bigamp.tensor_spreading_parallel import BiGAMPTensorSpreadingParallel
from matrix_factorization.modules.metrics.tensor_metrics import compute_tensor_cosine

logging.basicConfig(level=logging.INFO)

def test_convergence():
    print("=== Testing Convergence at Alpha=4.0 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # N=200, M=50, Order=3, S=10
    N = 200
    M = 50
    algorithm = BiGAMPTensorSpreadingParallel(
        tensor_order=3,
        dims=(N, N, N),
        M=M,
        S=10,
        max_steps=200, # Standard steps
        onsager_correction=False, # Strict user constraint
        device=device
    )
    
    print("Creating Teacher...")
    torch.manual_seed(42)
    W = torch.randn(N, M, device=device) * (1.0) # Unit Variance for Order 3
    X = torch.randn(M, N, device=device) * (1.0) # Transposed shape (M, N)
    
    # Train at User-Defined Limit Alpha=4.0
    print("Training at Alpha=4.0...")
    alpha_values = [4.0]
    
    try:
        teacher_factors = algorithm._create_teacher_factors(W, X)
        
        # Hook for progress
        def step_reporter(step, total, metrics=None):
            if step % 10 == 0 or step == 1:
                print(f"Step {step}/{total}")
        
        result = algorithm._train_full_parallel(
            teacher_factors=teacher_factors,
            alpha_values=alpha_values,
            seed=100,
            device=device,
            step_callback=step_reporter
        )
        
        q_full = result['Q_Full_Tensor'][0]
        q_y = result['Q_Y_per_alpha'][0]
        C_max = result['C_max']
        
        print(f"DEBUG: C_max = {C_max} (Edges)")
        print(f"DEBUG: Variables = {3 * N * M}")
        print(f"DEBUG: Ratio = {C_max / (3 * N * M):.2f}")
        
        print(f"Result Q_Full at Alpha=4.0: {q_full:.4f}")
        print(f"Result Q_Y at Alpha=4.0:    {q_y:.4f}")
        
        if q_full > 0.05: # Lower threshold for initial detection
            print("✅ PASS: Algorithm converged (Q > 0.05)")
        else:
            print("❌ FAIL: Algorithm stuck at 0 (Q <= 0.05)")
            
    except Exception as e:
        print(f"CRASHED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_convergence()
