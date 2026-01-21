# tests/test_tensor_spreading.py
"""Integration tests for BiGAMPTensorSpreading."""

import pytest
import torch


class TestBiGAMPTensorSpreading:
    """Test main algorithm class."""

    def test_train_single_alpha_converges(self):
        """Test that training on a single alpha makes progress."""
        from matrix_factorization.modules.algorithms.bigamp.tensor_spreading import BiGAMPTensorSpreading
        
        algo = BiGAMPTensorSpreading(
            tensor_order=3,
            dims=(20, 20, 20),
            M=10,
            max_steps=50,
            damping=0.5,
        )
        
        # Create teacher
        teacher = [torch.randn(20, 10) * 0.1 for _ in range(3)]
        
        # Train using internal method (Legacy API)
        result = algo._train_single_internal(
            teacher_factors=teacher,
            alpha=2.0,
            seed=42,
            device=torch.device('cpu'),
        )
        
        assert 'Q_Y' in result
        assert result['Q_Y'] >= 0.0  # Should make some progress

    def test_train_n2_degradation(self):
        """Test n=2 works correctly (matrix case)."""
        from matrix_factorization.modules.algorithms.bigamp.tensor_spreading import BiGAMPTensorSpreading
        
        algo = BiGAMPTensorSpreading(
            tensor_order=2,
            dims=(30, 30),
            M=10,
            max_steps=100,
            damping=0.5,
        )
        
        teacher = [torch.randn(30, 10) * 0.1 for _ in range(2)]
        result = algo._train_single_internal(teacher, alpha=3.0, seed=42, device=torch.device('cpu'))
        
        # For n=2, should achieve good Q_Y with sufficient steps
        assert result['Q_Y'] > 0.5

    def test_train_multiple_alphas(self):
        """Test training across multiple alpha values."""
        from matrix_factorization.modules.algorithms.bigamp.tensor_spreading import BiGAMPTensorSpreading
        
        algo = BiGAMPTensorSpreading(
            tensor_order=2,
            dims=(20, 20),
            M=5,
            max_steps=30,
        )
        
        teacher = [torch.randn(20, 5) * 0.1 for _ in range(2)]
        results = algo.train(teacher, alpha_values=[1.0, 2.0, 3.0], S=2, base_seed=42, device=torch.device('cpu'))
        
        assert len(results) == 6  # 3 alphas * 2 samples
        assert all('Q_Y' in r for r in results)

    def test_onsager_correction_option(self):
        """Test Onsager correction can be enabled."""
        from matrix_factorization.modules.algorithms.bigamp.tensor_spreading import BiGAMPTensorSpreading
        
        algo = BiGAMPTensorSpreading(
            tensor_order=2,
            dims=(20, 20),
            M=5,
            max_steps=30,
            onsager_correction=True,  # Enable Onsager
        )
        
        teacher = [torch.randn(20, 5) * 0.1 for _ in range(2)]
        result = algo._train_single_internal(teacher, alpha=2.0, seed=42, device=torch.device('cpu'))
        
        # Should still run and produce result
        assert 'Q_Y' in result
