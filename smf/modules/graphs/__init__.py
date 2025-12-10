"""
Graph/mask generation module.

Available methods:
- random: Pure random GPU-based generation
"""

from .base import GraphBase
from .generator import GraphGenerator
from .supergraph import SuperGraphData

__all__ = [
    'GraphBase',
    'GraphGenerator',
    'SuperGraphData',
]
