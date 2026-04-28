"""
Quick test script for General mode chunked processing.
Tests both chunked and legacy modes.
"""
import torch
import sys
sys.path.insert(0, '/home/sucia/Sparse-Matrix')

from matrix_factorization.core.experiment.config import (
    ExperimentConfig, MatrixParams, TrainingParams, 
    ScanConfig, SpreadingConfig, AlgorithmParams
)
from matrix_factorization.modules.algorithms.bigamp.spreading import BiGAMPSpreading

def test_general_chunked():
    """Test General mode with chunked processing."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Small-scale test config
    config = ExperimentConfig(
        matrix=MatrixParams(N1=100, N2=100, M=25),
        training=TrainingParams(samples_per_alpha=5, max_steps=50),
        algorithm_key='bigamp_spreading',
        scan=ScanConfig(dimension='alpha', values=[0.5, 1.0, 1.5]),
        spreading=SpreadingConfig(
            f_distribution='ising',
            allow_intra_connection=True,  # Enable General mode
            chunk_size=1024,  # Small chunk for testing
        ),
        algorithm_params=AlgorithmParams(damping=0.5, use_compile=False),
    )
    
    print(f"\n=== Test 1: Chunked Mode (chunk_size={config.spreading.chunk_size}) ===")
    algo = BiGAMPSpreading(config, device)
    print(f"  Algorithm initialized. chunk_size={algo.chunk_size}")
    print(f"  allow_intra_connection={algo.allow_intra_connection}")
    
    # Create test data
    W_teacher = torch.randn(100, 25, device=device)
    X_teacher = torch.randn(25, 100, device=device)
    
    spreading_data = algo.create_spreading_data(
        W_teacher, X_teacher,
        alpha_values=[0.5, 1.0, 1.5],
        S=5,
        base_seed=42
    )
    print(f"  SpreadingData created. C_max={spreading_data.supergraph.C_max}")
    
    # Run training
    W_hat, X_hat = algo._train_full_parallel_general(spreading_data, verbose=True)
    print(f"  Training complete!")
    print(f"  W_hat shape: {W_hat.shape}")
    print(f"  X_hat shape: {X_hat.shape}")
    
    # Quick sanity check
    assert W_hat.shape[1] == 3, "Should have 3 alpha values"
    assert W_hat.shape[2] == 100, "Should have N1=100"
    print("  ✓ Shape check passed!")
    
    print("\n=== Test 2: Legacy Mode (chunk_size=0) ===")
    config_legacy = ExperimentConfig(
        matrix=MatrixParams(N1=100, N2=100, M=25),
        training=TrainingParams(samples_per_alpha=5, max_steps=50),
        algorithm_key='bigamp_spreading',
        scan=ScanConfig(dimension='alpha', values=[0.5, 1.0, 1.5]),
        spreading=SpreadingConfig(
            f_distribution='ising',
            allow_intra_connection=True,
            chunk_size=0,  # Disable chunking
        ),
        algorithm_params=AlgorithmParams(damping=0.5, use_compile=False),
    )
    
    algo_legacy = BiGAMPSpreading(config_legacy, device)
    print(f"  chunk_size={algo_legacy.chunk_size} (legacy)")
    
    spreading_data_legacy = algo_legacy.create_spreading_data(
        W_teacher, X_teacher,
        alpha_values=[0.5, 1.0, 1.5],
        S=5,
        base_seed=42
    )
    
    W_hat_legacy, X_hat_legacy = algo_legacy._train_full_parallel_general(
        spreading_data_legacy, verbose=False
    )
    print(f"  Training complete!")
    print("  ✓ Legacy mode works!")
    
    print("\n=== All tests passed! ===")

if __name__ == '__main__':
    test_general_chunked()
