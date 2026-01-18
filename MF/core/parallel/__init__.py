"""
Smart Parallel Module for SMF.

This module provides unified memory estimation, parallel execution coordination,
and runtime memory monitoring for SMF algorithms.

Example usage:
    from MF.core.parallel import (
        ParallelCoordinator,
        MemoryEstimator,
        AllocationPresets,
        EstimationParams,
    )
    
    # Create coordinator with conservative allocation
    coordinator = ParallelCoordinator(
        estimator=MemoryEstimator(),
        config=AllocationPresets.CONSERVATIVE
    )
    
    # Plan execution
    params = EstimationParams(
        N1=600, N2=600, M=150, S=20,
        alpha_values=[0.0, 0.5, 1.0, 1.5, 2.0],
        algorithm_key='bigamp_spreading_parallel'
    )
    plan = coordinator.plan_execution(params)
    
    # Execute
    results = coordinator.execute(plan, algorithm)
"""

from .execution_modes import (
    ParallelMode,
    EstimationParams,
    MemoryEstimate,
    BatchConfig,
    AllocationConfig,
    AllocationPresets,
    ExecutionPlan,
)

# Delay imports to avoid circular dependencies
def get_memory_estimator():
    """Get MemoryEstimator class (lazy import)."""
    from .memory_estimator import MemoryEstimator
    return MemoryEstimator

def get_parallel_coordinator():
    """Get ParallelCoordinator class (lazy import)."""
    from .parallel_coordinator import ParallelCoordinator
    return ParallelCoordinator

def get_memory_guard():
    """Get MemoryGuard class (lazy import)."""
    from .memory_guard import MemoryGuard, OOMRecoveryHandler
    return MemoryGuard, OOMRecoveryHandler


__all__ = [
    # Enums and data structures
    'ParallelMode',
    'EstimationParams',
    'MemoryEstimate',
    'BatchConfig',
    'AllocationConfig',
    'AllocationPresets',
    'ExecutionPlan',
    # Lazy loaders
    'get_memory_estimator',
    'get_parallel_coordinator',
    'get_memory_guard',
    # Adapters
    'BaseAlgorithmAdapter',
    'DenseAlgorithmAdapter',
    'SpreadingAlgorithmAdapter',
    'get_adapter',
    'create_adapter',
    'get_memory_estimate',
    'compare_f_distributions',
]

# Import adapters directly (no circular dependency issues)
from .algorithm_adapters import (
    BaseAlgorithmAdapter,
    DenseAlgorithmAdapter,
    SpreadingAlgorithmAdapter,
    get_adapter,
    create_adapter,
    get_memory_estimate,
    compare_f_distributions,
)

