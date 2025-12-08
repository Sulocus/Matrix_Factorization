"""
Graph/mask generation module.

Available methods:
- random: Pure random GPU-based generation
- uniform: Bi-regular graph using Dinic algorithm
- dinic: Strict bi-regular using max-flow algorithm
- low_loop: MCMC-optimized low short-cycle graphs
- combined: Flexible method selection
"""

from .base import GraphBase
from .random import RandomGraph
from .uniform import UniformGraph
from .dinic import DinicGraph
from .low_loop import LowLoopGraph
from .combined import CombinedGraph

__all__ = [
    'GraphBase',
    'RandomGraph',
    'UniformGraph',
    'DinicGraph',
    'LowLoopGraph',
    'CombinedGraph',
]
