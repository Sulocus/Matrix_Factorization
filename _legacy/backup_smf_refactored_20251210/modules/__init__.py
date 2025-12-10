"""
Pluggable modules for SMF framework.

Module categories:
- algorithms: Training algorithms
- graphs: Graph/mask generation methods
- teachers: Teacher model initialization
- metrics: Evaluation metrics
- outputs: Output handlers

All modules are auto-registered via decorators.
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
# ============================================================

# Algorithms
from .algorithms import unified

# Graphs
from .graphs import generator

# Teachers
from .teachers import generator, scaled_variance, base

# Metrics (usually imported via package logic, but explicit here is safer)
# from .metrics import ... (metrics init usually handles this)

# Outputs
from .outputs import plotting, storage
from .outputs.combined import CombinedOutput

# ============================================================
# Convenience functions
# ============================================================

def get_valid_algorithm_keys():
    return {info.key for info in list_algorithms()}

def get_valid_graph_keys():
    return {info.key for info in list_graphs()}

def get_valid_teacher_keys():
    return {info.key for info in list_teachers()}

def get_valid_metric_keys():
    return {info.key for info in list_metrics()}

def get_valid_output_keys():
    return {info.key for info in list_outputs()}


__all__ = [
    'get_algorithm', 'get_graph', 'get_teacher', 'get_metric', 'get_output',
    'list_algorithms', 'list_graphs', 'list_teachers', 'list_metrics', 'list_outputs',
    'get_valid_algorithm_keys', 'get_valid_graph_keys', 'get_valid_teacher_keys',
    'get_valid_metric_keys', 'get_valid_output_keys',
]
