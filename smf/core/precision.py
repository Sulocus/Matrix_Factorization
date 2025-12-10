"""
Mixed Precision Utilities for BiG-AMP.

Phase 3 Optimization: BF16 storage + FP32 accumulation strategy.

Strategy:
- BF16 for storage (W, X, F tensors) - 2x memory reduction
- FP32 for accumulation (sum, variance) - numerical stability

This provides significant memory savings while maintaining the numerical
precision required for BiG-AMP convergence.

Requirements:
- PyTorch 1.10+ for BF16 support
- Ampere+ GPU (RTX 30xx, A100, etc.) for hardware BF16 acceleration
  - On older GPUs, BF16 falls back to software emulation (slower)
"""

import torch
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class PrecisionConfig:
    """Configuration for mixed precision training."""
    use_bf16: bool = True  # Enable BF16 storage
    force_fp32_accumulation: bool = True  # Always use FP32 for sum/mean
    
    # Automatically detect hardware support
    @classmethod
    def auto_detect(cls) -> 'PrecisionConfig':
        """Create config with automatic hardware detection."""
        # Check if BF16 is hardware-supported
        bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        return cls(use_bf16=bf16_supported, force_fp32_accumulation=True)


class MixedPrecisionContext:
    """
    Context manager and utilities for mixed precision training.
    
    Usage:
        ctx = MixedPrecisionContext(use_bf16=True)
        W = ctx.to_storage(W)  # Convert to BF16 for storage
        result = ctx.safe_sum(W * X, dim=-1)  # Sum in FP32 for precision
    """
    
    def __init__(self, use_bf16: bool = True):
        """
        Initialize mixed precision context.
        
        Args:
            use_bf16: Whether to use BF16 for storage. Falls back to FP32 if
                      hardware doesn't support BF16.
        """
        # Check hardware support
        self.bf16_available = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.use_bf16 = use_bf16 and self.bf16_available
        
        # Set dtypes
        self.storage_dtype = torch.bfloat16 if self.use_bf16 else torch.float32
        self.compute_dtype = torch.float32  # Always FP32 for accumulation
        
        if use_bf16 and not self.bf16_available:
            import warnings
            warnings.warn(
                "BF16 requested but not available on this hardware. "
                "Falling back to FP32. BF16 requires Ampere+ GPU (RTX 30xx, A100, etc.)"
            )
    
    def to_storage(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert tensor to storage dtype (BF16 or FP32).
        
        Use this for storing large tensors like W, X, F.
        """
        if tensor.dtype in (torch.float32, torch.float64):
            return tensor.to(self.storage_dtype)
        return tensor  # Don't convert int8 (Rademacher F) or other types
    
    def to_compute(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert tensor to compute dtype (FP32) for accumulation.
        
        Use this before sum(), mean(), or other reduction operations.
        """
        return tensor.to(self.compute_dtype)
    
    def safe_sum(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Sum with FP32 accumulation for numerical stability.
        
        Critical for BiG-AMP variance calculations where BF16 precision
        may cause numerical issues.
        
        Args:
            tensor: Input tensor (any dtype)
            dim: Dimension to sum over
            
        Returns:
            Sum result in storage_dtype (converted back from FP32)
        """
        # Convert to FP32 for accumulation
        result = tensor.float().sum(dim=dim)
        # Convert back to storage dtype
        return result.to(self.storage_dtype) if self.use_bf16 else result
    
    def safe_mean(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Mean with FP32 accumulation."""
        result = tensor.float().mean(dim=dim)
        return result.to(self.storage_dtype) if self.use_bf16 else result
    
    def safe_var(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Variance with FP32 accumulation."""
        result = tensor.float().var(dim=dim)
        return result.to(self.storage_dtype) if self.use_bf16 else result
    
    def safe_matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Matrix multiplication with automatic dtype handling.
        
        On Ampere+ GPUs, BF16 matmul is hardware-accelerated.
        Result is kept in storage dtype.
        """
        # PyTorch handles BF16 matmul efficiently on supported hardware
        return torch.matmul(a, b)
    
    def clamp(self, tensor: torch.Tensor, min_val: float = None, max_val: float = None) -> torch.Tensor:
        """Clamp values, preserving dtype."""
        return torch.clamp(tensor, min=min_val, max=max_val)
    
    def zeros(self, *shape, device: torch.device = None) -> torch.Tensor:
        """Create zeros tensor in storage dtype."""
        return torch.zeros(*shape, device=device, dtype=self.storage_dtype)
    
    def ones(self, *shape, device: torch.device = None) -> torch.Tensor:
        """Create ones tensor in storage dtype."""
        return torch.ones(*shape, device=device, dtype=self.storage_dtype)
    
    def randn(self, *shape, device: torch.device = None) -> torch.Tensor:
        """Create random normal tensor in storage dtype."""
        return torch.randn(*shape, device=device, dtype=self.storage_dtype)
    
    def get_memory_multiplier(self) -> float:
        """
        Get memory multiplication factor compared to FP32.
        
        Returns:
            0.5 for BF16 (2x memory reduction), 1.0 for FP32
        """
        return 0.5 if self.use_bf16 else 1.0
    
    def __repr__(self) -> str:
        return f"MixedPrecisionContext(use_bf16={self.use_bf16}, storage={self.storage_dtype})"


# Global singleton for easy access
_global_precision_ctx: Optional[MixedPrecisionContext] = None


def get_precision_context(use_bf16: bool = None) -> MixedPrecisionContext:
    """
    Get or create the global precision context.
    
    Args:
        use_bf16: Override BF16 setting. If None, auto-detect hardware support.
        
    Returns:
        MixedPrecisionContext instance
    """
    global _global_precision_ctx
    
    if _global_precision_ctx is None or use_bf16 is not None:
        if use_bf16 is None:
            # Auto-detect
            use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        _global_precision_ctx = MixedPrecisionContext(use_bf16=use_bf16)
    
    return _global_precision_ctx


def reset_precision_context():
    """Reset the global precision context."""
    global _global_precision_ctx
    _global_precision_ctx = None
