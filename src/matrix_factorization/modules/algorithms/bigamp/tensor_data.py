"""
Data structures for N-dimensional tensor BigAMP Spreading.

This module defines the core data structures for n-dimensional tensor
CP decomposition using the spreading model.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch


@dataclass
class TensorHypergraph:
    """
    N-uniform hypergraph for tensor spreading.
    
    In n-dimensional tensor decomposition, each hyperedge connects
    exactly n nodes (one from each factor dimension).
    
    Attributes:
        order: n - tensor order (n=2 for matrix, n=3 for 3-tensor, etc.)
        indices: List of n tensors, each of shape (C,) containing
                 node indices for that dimension
        dims: Tuple of n integers (N_1, ..., N_n) - factor dimensions
    
    Example:
        For a 3-dimensional tensor with dims=(10, 10, 10) and 5 hyperedges:
        - order = 3
        - indices = [tensor([0,1,2,3,4]), tensor([5,6,7,8,9]), tensor([0,2,4,6,8])]
        - Each hyperedge c connects nodes indices[0][c], indices[1][c], indices[2][c]
    """
    order: int
    indices: List[torch.Tensor]  # n tensors of shape (C,)
    dims: Tuple[int, ...]
    
    @property
    def C(self) -> int:
        """Number of hyperedges."""
        if not self.indices or len(self.indices) == 0:
            return 0
        return self.indices[0].shape[0] if self.indices[0].numel() > 0 else 0
    
    def to(self, device: torch.device) -> 'TensorHypergraph':
        """Move all tensors to specified device."""
        return TensorHypergraph(
            order=self.order,
            indices=[idx.to(device) for idx in self.indices],
            dims=self.dims,
        )
    
    def clone(self) -> 'TensorHypergraph':
        """Create a deep copy."""
        return TensorHypergraph(
            order=self.order,
            indices=[idx.clone() for idx in self.indices],
            dims=self.dims,
        )


@dataclass
class TensorSpreadingData:
    """
    Complete data package for tensor spreading training.
    
    Contains the hypergraph structure, spreading coefficients F,
    and observed Y values for a single (alpha, sample) configuration.
    
    Attributes:
        hypergraph: TensorHypergraph defining the observation structure
        F: (C, M) spreading coefficients
        Y: (C,) observed values
        M: latent dimension
        alpha: observation density parameter
        seed: random seed used for generation
    """
    hypergraph: TensorHypergraph
    F: torch.Tensor              # (C, M)
    Y: torch.Tensor              # (C,)
    M: int
    alpha: float
    seed: int
    
    def to(self, device: torch.device) -> 'TensorSpreadingData':
        """Move all tensors to specified device."""
        return TensorSpreadingData(
            hypergraph=self.hypergraph.to(device),
            F=self.F.to(device),
            Y=self.Y.to(device),
            M=self.M,
            alpha=self.alpha,
            seed=self.seed,
        )
