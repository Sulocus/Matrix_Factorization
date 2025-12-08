"""
Graph/mask generation module.

Available methods:
- random: Pure random GPU-based generation
- uniform: Bi-regular graph using Dinic algorithm
- low_loop: MCMC-optimized low short-cycle graphs
- combined: Flexible method selection
"""

from .base import GraphBase
from .random import RandomGraph
from .uniform import UniformGraph
from .low_loop import LowLoopGraph
from .combined import CombinedGraph

# Backward compatibility alias (dinic.py was removed as it was identical to uniform.py)
DinicGraph = UniformGraph

__all__ = [
    'GraphBase',
    'RandomGraph',
    'UniformGraph',
    'DinicGraph',  # Alias for UniformGraph
    'LowLoopGraph',
    'CombinedGraph',
]
