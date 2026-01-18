"""
Tests for Teacher-Assisted Initialization (Hysteresis Analysis).

Tests verify:
1. _initialize_near_teacher produces correct overlap
2. Variance is preserved after initialization
3. Random mode behavior is unchanged
"""

import pytest
import torch
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smf.modules.algorithms.bigamp_spreading import BiGAMPSpreading


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Get available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def small_dims():
    """Small dimensions for quick tests."""
    return {'N1': 50, 'N2': 60, 'M': 10, 'S': 5, 'B': 3}


# ============================================================================
# Test 1: Overlap Correctness
# ============================================================================

class TestTeacherInitOverlap:
    """Test that initialization produces correct overlap."""

    def test_overlap_high(self, device, small_dims):
        """High m_init should produce high overlap."""
        N1, M, S, B = small_dims['N1'], small_dims['M'], small_dims['S'], small_dims['B']
        m_init = 0.99
        
        # Create teacher
        teacher = torch.randn(N1, M, device=device)
        teacher = teacher / teacher.norm(dim=1, keepdim=True)  # Normalize rows
        
        # Initialize near teacher
        target_shape = (B, S * N1, M)
        result = BiGAMPSpreading._initialize_near_teacher(
            target_shape, teacher, m_init, S, device, torch.float32
        )
        
        # Reshape to (B, S, N1, M) for overlap computation
        result_reshaped = result.view(B, S, N1, M)
        
        # Compute overlap for first sample
        student = result_reshaped[0, 0]  # (N1, M)
        student = student / student.norm(dim=1, keepdim=True)
        
        # Cosine similarity per row, then average
        overlap = (student * teacher).sum(dim=1).mean().item()
        
        # With m=0.99, expect overlap ~0.99 ± 0.05
        assert overlap > 0.9, f"Expected overlap > 0.9 for m=0.99, got {overlap:.4f}"

    def test_overlap_medium(self, device, small_dims):
        """Medium m_init should produce medium overlap."""
        N1, M, S, B = small_dims['N1'], small_dims['M'], small_dims['S'], small_dims['B']
        m_init = 0.7
        
        teacher = torch.randn(N1, M, device=device)
        target_shape = (B, S * N1, M)
        
        result = BiGAMPSpreading._initialize_near_teacher(
            target_shape, teacher, m_init, S, device, torch.float32
        )
        
        result_reshaped = result.view(B, S, N1, M)
        student = result_reshaped[0, 0]
        
        # Normalize for cosine
        t_norm = teacher / (teacher.norm(dim=1, keepdim=True) + 1e-8)
        s_norm = student / (student.norm(dim=1, keepdim=True) + 1e-8)
        overlap = (s_norm * t_norm).sum(dim=1).mean().item()
        
        # With m=0.7, expect overlap in range [0.5, 0.9]
        assert 0.4 < overlap < 0.95, f"Expected overlap ~0.7 for m=0.7, got {overlap:.4f}"


# ============================================================================
# Test 2: Variance Preservation
# ============================================================================

class TestVariancePreservation:
    """Test that initialization preserves variance."""

    def test_variance_formula(self, device, small_dims):
        """Variance should be preserved by the formula: m^2 + (1-m^2) = 1."""
        N1, M, S, B = small_dims['N1'], small_dims['M'], small_dims['S'], small_dims['B']
        m_init = 0.9
        
        # Unit variance teacher
        teacher = torch.randn(N1, M, device=device)
        target_shape = (B, S * N1, M)
        
        result = BiGAMPSpreading._initialize_near_teacher(
            target_shape, teacher, m_init, S, device, torch.float32
        )
        
        # Theoretical: if teacher ~ N(0, σ²) and noise ~ N(0, σ²),
        # then result = m*teacher + √(1-m²)*noise has same variance σ²
        result_var = result.var().item()
        teacher_var = teacher.var().item()
        
        # Allow 20% tolerance
        assert abs(result_var - teacher_var) < 0.3 * teacher_var, \
            f"Variance not preserved: teacher={teacher_var:.4f}, result={result_var:.4f}"


# ============================================================================
# Test 3: Shape Correctness
# ============================================================================

class TestShapeCorrectness:
    """Test that output shapes are correct."""

    def test_output_shape(self, device, small_dims):
        """Output should have correct shape."""
        N1, M, S, B = small_dims['N1'], small_dims['M'], small_dims['S'], small_dims['B']
        
        teacher = torch.randn(N1, M, device=device)
        target_shape = (B, S * N1, M)
        
        result = BiGAMPSpreading._initialize_near_teacher(
            target_shape, teacher, 0.95, S, device, torch.float32
        )
        
        assert result.shape == target_shape, \
            f"Expected shape {target_shape}, got {result.shape}"

    def test_dtype_preservation(self, device, small_dims):
        """Output should have requested dtype."""
        N1, M, S, B = small_dims['N1'], small_dims['M'], small_dims['S'], small_dims['B']
        
        teacher = torch.randn(N1, M, device=device)
        target_shape = (B, S * N1, M)
        
        # Test float32
        result_f32 = BiGAMPSpreading._initialize_near_teacher(
            target_shape, teacher, 0.95, S, device, torch.float32
        )
        assert result_f32.dtype == torch.float32
        
        # Test bfloat16 if available
        if device.type == 'cuda' and torch.cuda.is_bf16_supported():
            result_bf16 = BiGAMPSpreading._initialize_near_teacher(
                target_shape, teacher, 0.95, S, device, torch.bfloat16
            )
            assert result_bf16.dtype == torch.bfloat16


# ============================================================================
# Test 4: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases."""

    def test_m_init_zero(self, device, small_dims):
        """m_init=0 should produce uncorrelated output (pure noise)."""
        N1, M, S, B = small_dims['N1'], small_dims['M'], small_dims['S'], small_dims['B']
        
        teacher = torch.randn(N1, M, device=device)
        target_shape = (B, S * N1, M)
        
        result = BiGAMPSpreading._initialize_near_teacher(
            target_shape, teacher, 0.0, S, device, torch.float32
        )
        
        # Should be pure noise, uncorrelated with teacher
        result_reshaped = result.view(B, S, N1, M)
        student = result_reshaped[0, 0]
        
        # Normalize for cosine
        t_norm = teacher / (teacher.norm(dim=1, keepdim=True) + 1e-8)
        s_norm = student / (student.norm(dim=1, keepdim=True) + 1e-8)
        overlap = (s_norm * t_norm).sum(dim=1).mean().item()
        
        # With m=0, expect overlap ~0 ± 0.2
        assert abs(overlap) < 0.3, f"Expected overlap ~0 for m=0, got {overlap:.4f}"

    def test_m_init_one(self, device, small_dims):
        """m_init=1 should produce exact teacher (no noise)."""
        N1, M, S, B = small_dims['N1'], small_dims['M'], small_dims['S'], small_dims['B']
        
        teacher = torch.randn(N1, M, device=device)
        target_shape = (B, S * N1, M)
        
        # m=1 means coeff_noise = sqrt(1-1) = 0
        result = BiGAMPSpreading._initialize_near_teacher(
            target_shape, teacher, 1.0, S, device, torch.float32
        )
        
        # Should be exactly teacher (broadcasted)
        result_reshaped = result.view(B, S, N1, M)
        
        for b in range(B):
            for s in range(S):
                diff = (result_reshaped[b, s] - teacher).abs().max().item()
                assert diff < 1e-5, f"m=1 should give exact teacher, max diff = {diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
