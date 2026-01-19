# tests/test_tensor_data.py
"""Tests for N-dimensional tensor data structures."""

import pytest
import torch
from matrix_factorization.modules.algorithms.bigamp.tensor_data import TensorHypergraph


class TestTensorHypergraph:
    """Test TensorHypergraph dataclass."""

    def test_hypergraph_creation(self):
        """Test basic hypergraph creation."""
        indices = [
            torch.tensor([0, 1, 0]),  # dim 0 indices
            torch.tensor([1, 0, 1]),  # dim 1 indices
            torch.tensor([2, 2, 0]),  # dim 2 indices
        ]
        hg = TensorHypergraph(order=3, indices=indices, dims=(3, 3, 3))
        
        assert hg.order == 3
        assert hg.C == 3
        assert len(hg.indices) == 3
        assert hg.dims == (3, 3, 3)

    def test_hypergraph_to_device(self):
        """Test moving hypergraph to device."""
        indices = [torch.tensor([0, 1]), torch.tensor([1, 0])]
        hg = TensorHypergraph(order=2, indices=indices, dims=(2, 2))
        
        device = torch.device('cpu')
        hg_moved = hg.to(device)
        
        assert hg_moved.indices[0].device.type == 'cpu'
        assert hg_moved.order == 2

    def test_hypergraph_empty(self):
        """Test empty hypergraph."""
        indices = [torch.tensor([]), torch.tensor([])]
        hg = TensorHypergraph(order=2, indices=indices, dims=(10, 10))
        
        assert hg.C == 0
