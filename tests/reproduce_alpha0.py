
import torch
import logging
from matrix_factorization.modules.algorithms.bigamp.tensor_spreading_parallel import BiGAMPTensorSpreadingParallel

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_alpha_zero():
    print("=== Testing Alpha=0 Anomaly & Physics Fix ===")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Config matching user's case roughly
    # N=200, M=50, Order=3, S=10
    N = 200
    M = 50
    algorithm = BiGAMPTensorSpreadingParallel(
        tensor_order=3,
        dims=(N, N, N),
        M=M,
        S=10,
        max_steps=10, # Short run
        device=device
    )
    
    # Create Teacher
    print("Creating Teacher...")
    torch.manual_seed(42)
    W = torch.randn(N, M, device=device) * (1.0/M**0.5)
    X = torch.randn(M, N, device=device) * (1.0/M**0.5) # Corrected Shape (M, N)
    
    # Train at Alpha=0
    print("Training at Alpha=0...")
    alpha_values = [0.0]
    
    teacher_factors = algorithm._create_teacher_factors(W, X)
    
    result = algorithm._train_full_parallel(
        teacher_factors=teacher_factors,
        alpha_values=alpha_values,
        seed=100,
        device=device
    )
    
    q_y = result['Q_Y_per_alpha'][0]
    print(f"Result Q_Y at Alpha=0: {q_y}")
    
    # Check new Metric if present
    if 'Q_Full_Tensor' in result:
        print(f"Result Q_Full_Tensor: {result['Q_Full_Tensor']}")
    else:
        print("Q_Full_Tensor not found (Integration pending?)")
    
    # Verify expectations
    if q_y > 0.01:
        print("❌ FAIL: Q_Y is high at Alpha=0 (Expected ~0.0)")
    else:
        print("✅ PASS: Q_Y is low at Alpha=0")

if __name__ == "__main__":
    test_alpha_zero()
