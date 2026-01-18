"""
Pluggable modules for SMF framework.

Module categories:
- algorithms: Training algorithms (BiG-AMP, AGD, etc.)
- graphs: Graph/mask generation methods
- teachers: Teacher model initialization
- metrics: Evaluation metrics
- outputs: Output handlers (plotting, storage)

All modules are auto-registered via decorators. Adding a new module only
requires using the @register_* decorator - no manual list updates needed.
"""

from .registry import (
    get_algorithm,
    get_graph,
    get_teacher,
    get_metric,
    get_output,
    list_algorithms,
    list_graphs,
    list_teachers,
    list_metrics,
    list_outputs,
)

# ============================================================
# Import ALL modules to trigger registration
# Adding a new module? Just add the import here.
# ============================================================

# Algorithms
from .algorithms import bigamp, agd
from .algorithms.combined import CombinedAlgorithm

# Graphs
from .graphs import random, uniform, low_loop
from .graphs.combined import CombinedGraph

# Teachers
from .teachers import standard, scaled_variance, orthogonal
from .teachers.combined import CombinedTeacher

# Outputs
from .outputs import plotting, storage
from .outputs.combined import CombinedOutput

# ============================================================
# Convenience functions for getting valid keys (used by llm_advisor)
# ============================================================

def get_valid_algorithm_keys():
    """Get all registered algorithm keys."""
    return {info.key for info in list_algorithms()}

def get_valid_graph_keys():
    """Get all registered graph keys."""
    return {info.key for info in list_graphs()}

def get_valid_teacher_keys():
    """Get all registered teacher keys."""
    return {info.key for info in list_teachers()}

def get_valid_metric_keys():
    """Get all registered metric keys."""
    return {info.key for info in list_metrics()}

def get_valid_output_keys():
    """Get all registered output keys."""
    return {info.key for info in list_outputs()}


__all__ = [
    # Registry getters
    'get_algorithm',
    'get_graph',
    'get_teacher',
    'get_metric',
    'get_output',
    # Registry listers
    'list_algorithms',
    'list_graphs',
    'list_teachers',
    'list_metrics',
    'list_outputs',
    # Key validators (for llm_advisor)
    'get_valid_algorithm_keys',
    'get_valid_graph_keys',
    'get_valid_teacher_keys',
    'get_valid_metric_keys',
    'get_valid_output_keys',
]
