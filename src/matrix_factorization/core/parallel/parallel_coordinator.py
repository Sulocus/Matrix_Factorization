"""
Parallel Coordinator for Smart Parallel Module.

Provides intelligent execution planning and coordination for parallel algorithms,
with support for:
- Multiple parallel modes (LINEAR, SAMPLE_PARALLEL, ALPHA_PARALLEL, HYBRID, FULL_PARALLEL)
- Dynamic batch sizing based on memory constraints
- Conservative allocation with large safety buffers (user-requested feature)
- Runtime memory monitoring integration
"""
from typing import List, Callable, Optional, Any, Dict
from dataclasses import replace
import hashlib
import json
import logging
import gc
import torch

from .execution_modes import (
    ParallelMode,
    EstimationParams,
    BatchConfig,
    AllocationConfig,
    AllocationPresets,
    ExecutionPlan,
)
from .memory_estimator import MemoryEstimator
from .resource_execution import build_resource_execution_plan

logger = logging.getLogger(__name__)


class ParallelCoordinator:
    """
    Parallel execution coordinator.
    
    Plans and executes algorithms with intelligent memory management and
    automatic batching based on GPU memory constraints.
    
    Design Philosophy:
    - Precise estimation: Calculate memory requirements accurately
    - Conservative allocation: Use only 50-60% of available memory (user-requested)
    - Runtime monitoring: Detect and recover from memory pressure
    
    Example:
        coordinator = ParallelCoordinator(
            estimator=MemoryEstimator(),
            config=AllocationPresets.CONSERVATIVE  # Uses 55% allocation
        )
        
        params = EstimationParams(...)
        plan = coordinator.plan_execution(params)
        results = coordinator.execute(plan, algorithm)
    """
    
    def __init__(
        self,
        estimator: Optional[MemoryEstimator] = None,
        config: Optional[AllocationConfig] = None,
    ):
        """
        Initialize coordinator.
        
        Args:
            estimator: Memory estimator instance. Creates default if None.
            config: Allocation configuration. Uses CONSERVATIVE preset if None.
        """
        self.estimator = estimator or MemoryEstimator()
        self.config = config or AllocationPresets.CONSERVATIVE
        
        # Runtime state
        self.current_plan: Optional[ExecutionPlan] = None
        self.current_batch_idx: int = 0
        self._abort_flag: bool = False
        self._memory_guard = None  # Set by set_memory_guard()
        
        # Statistics
        self.stats = {
            "plans_created": 0,
            "batches_executed": 0,
            "oom_recoveries": 0,
            "replans_created": 0,
        }
    
    def plan_execution(self, params: EstimationParams) -> ExecutionPlan:
        """
        Create execution plan for given parameters.
        
        Uses conservative allocation based on config.allocation_ratio to ensure
        sufficient buffer for torch.compile dynamic caches and memory spikes.
        
        Args:
            params: Estimation parameters
            
        Returns:
            ExecutionPlan with batches and estimated memory
        """
        available_gb = self._get_available_memory()
        
        # Apply maximum allocation limit if configured
        if self.config.max_allocation_gb is not None:
            available_gb = min(available_gb, self.config.max_allocation_gb)
        
        # Conservative allocation: use only allocation_ratio of available memory
        target_gb = available_gb * self.config.allocation_ratio
        
        logger.info(
            f"Planning execution: available={available_gb:.1f}GB, "
            f"target={target_gb:.1f}GB (ratio={self.config.allocation_ratio:.0%})"
        )
        # DEBUG: Print to terminal
        
        
        S = params.S
        A = len(params.alpha_values)
        
        # Strategy 1: Try full parallel
        full_estimate = self.estimator.estimate(params)
        replan_metadata = self._replan_metadata(params)
        
        if full_estimate.total_gb <= target_gb:
            plan = ExecutionPlan(
                mode=ParallelMode.FULL_PARALLEL,
                batches=[BatchConfig(
                    sample_range=(0, S),
                    alpha_range=(0, A),
                    estimated_memory_gb=full_estimate.total_gb,
                    alpha_values=params.alpha_values,
                    memory_breakdown=dict(full_estimate.breakdown),
                )],
                total_estimated_memory_gb=full_estimate.total_gb,
                allocation_config=self.config,
                algorithm_key=params.algorithm_key,
                gpu_model=self.estimator.gpu_model,
                available_memory_gb=available_gb,
                **replan_metadata,
            )
            logger.info(f"Selected FULL_PARALLEL mode: {full_estimate.total_gb:.1f}GB")
            self.stats["plans_created"] += 1
            return self._attach_plan_provenance(plan, params)
        
        # Strategy 2: Fixed S, split Alpha (prefer keeping Sample parallel)
        batches = self._compute_alpha_batches(params, target_gb)
        
        if batches:
            total_mem = max(b.estimated_memory_gb for b in batches)
            plan = ExecutionPlan(
                mode=ParallelMode.HYBRID,
                batches=batches,
                total_estimated_memory_gb=total_mem,
                allocation_config=self.config,
                algorithm_key=params.algorithm_key,
                gpu_model=self.estimator.gpu_model,
                available_memory_gb=available_gb,
                **replan_metadata,
            )
            logger.info(
                f"Selected HYBRID mode: {len(batches)} batches, "
                f"peak={total_mem:.1f}GB"
            )
            self.stats["plans_created"] += 1
            return self._attach_plan_provenance(plan, params)
        
        # Strategy 3 intentionally does not reduce S/sample count yet.
        # The current runner records sample_range but does not pass sample offsets
        # into algorithms.  Planning with a smaller S would under-estimate memory
        # while still executing the full sample count, which is unsafe.  Actual
        # sample/student folding must be added only after the algorithm contract
        # exposes sample_offset and partition-invariant random streams.

        # Strategy 4: Fallback to one alpha per batch, full S preserved.
        logger.warning("Falling back to LINEAR mode due to memory constraints")
        linear_plan = self._create_linear_plan(params, available_gb)
        
        # Strategy 5: Check whether every real runner batch
        # (full S, single alpha) fits. If the largest one does not fit, the
        # data is too large for this GPU under the current precision/algorithm.
        if linear_plan.batches:
            max_batch_mem = max(b.estimated_memory_gb for b in linear_plan.batches)
            if max_batch_mem > target_gb:
                # Even linear mode can't fit - problem is too large
                total_gb = self._get_total_memory()
                worst_batch = max(linear_plan.batches, key=lambda b: b.estimated_memory_gb)
                error_msg = (
                    f"ERROR: Problem size too large for GPU!\n"
                    f"  Largest single-alpha full-S batch required: {max_batch_mem:.2f} GB\n"
                    f"  Available allocation: {target_gb:.2f} GB (ratio={self.config.allocation_ratio:.0%})\n"
                    f"  GPU available: {available_gb:.2f} GB (after reserved)\n"
                    f"  GPU total: {total_gb:.2f} GB\n"
                    f"  Parameters: N1={params.N1}, N2={params.N2}, M={params.M}\n"
                    f"  Worst alpha batch: {worst_batch.alpha_values}\n"
                    f"\n"
                    f"  Suggestions:\n"
                    f"    1. Reduce matrix size (N1, N2, or M)\n"
                    f"    2. Use BF16 precision (halves memory usage)\n"
                    f"    3. Use a GPU with more VRAM\n"
                    f"    4. Try 'mf resume' after freeing GPU memory"
                )
                logger.error(error_msg)
                raise MemoryError(error_msg)
        
        return linear_plan

    def plan_resource_execution(self, scan_plan, params: EstimationParams, batching_spec=None):
        """Build a WorkItem-level resource plan from the current batch plan.

        This is the strict scan/resource bridge.  It wraps the existing
        ExecutionPlan without changing algorithm math, and it records whether
        planned sample ranges are actually honored by the runner/algorithm.
        """
        execution_plan = self.plan_execution(params)
        sample_range_honored = bool(getattr(batching_spec, "sample_range_honored", False))
        metadata_only = bool(getattr(batching_spec, "metadata_only", True))
        return execution_plan, build_resource_execution_plan(
            scan_plan=scan_plan,
            execution_plan=execution_plan,
            algorithm_key=params.algorithm_key,
            samples_per_alpha=params.S,
            seed_partition_policy=getattr(execution_plan, "seed_partition_policy", "legacy"),
            sample_range_honored=sample_range_honored,
            metadata_only=metadata_only,
            calibration_source=self._calibration_source(params),
        )
    
    def execute(
        self,
        plan: ExecutionPlan,
        algorithm: Any,
        step_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Execute plan with the given algorithm.
        
        Args:
            plan: Execution plan from plan_execution()
            algorithm: Algorithm instance with run_batch() method
            step_callback: Optional callback(batch_idx, step, metrics)
            
        Returns:
            Dict mapping alpha values to results
        """
        self.current_plan = plan
        self.current_batch_idx = 0
        self._abort_flag = False
        
        results = {}
        
        logger.info(f"Executing plan: {plan.mode.name}, {plan.num_batches} batches")
        
        # Start memory guard if configured
        if self._memory_guard:
            self._memory_guard.start()
        
        try:
            for batch_idx, batch in enumerate(plan.batches):
                if self._abort_flag:
                    logger.warning(f"Execution aborted at batch {batch_idx}")
                    break
                
                self.current_batch_idx = batch_idx
                
                # Execute single batch
                batch_result = self._execute_batch(
                    batch, algorithm, step_callback
                )
                results.update(batch_result)
                
                # Inter-batch cleanup
                self._cleanup_between_batches()
                
                self.stats["batches_executed"] += 1
                
        finally:
            if self._memory_guard:
                self._memory_guard.stop()
            
            self.current_plan = None
        
        return results
    
    def abort_current_batch(self) -> None:
        """Signal to abort current batch execution."""
        self._abort_flag = True
        logger.warning("Abort signal received")
    
    def replan_with_safety(self, factor: float = 0.7, failed_batch_idx: Optional[int] = None) -> ExecutionPlan:
        """
        Create a more conservative plan after OOM.

        Automatic replan is allowed only when the active effective seed policy
        is partition-invariant and the current plan carries the original
        EstimationParams snapshot needed to reconstruct a new plan.
        
        Args:
            factor: Multiplier for allocation ratio (< 1.0)
            
        Returns:
            New execution plan with reduced memory usage
        """
        if self.current_plan is None:
            raise RuntimeError("No current plan to replan from")

        algorithm_key = self.current_plan.algorithm_key
        if self.current_plan.automatic_rebatch_allowed:
            if not (0.0 < factor < 1.0):
                raise ValueError("replan factor must be between 0 and 1.")
            if not self.current_plan.estimation_params:
                raise NotImplementedError(
                    f"Automatic replan requires original EstimationParams for algorithm '{algorithm_key}'."
                )

            params = EstimationParams.from_dict(self.current_plan.estimation_params)
            parent_plan = self.current_plan
            replan_config = replace(
                self.config,
                max_allocation_gb=parent_plan.available_memory_gb * factor,
            )
            replanner = ParallelCoordinator(estimator=self.estimator, config=replan_config)
            new_plan = replanner.plan_execution(params)
            new_plan.parent_plan_id = parent_plan.plan_id
            new_plan.replan_attempt = parent_plan.replan_attempt + 1
            new_plan.replan_implemented = True
            new_plan.replan_provenance.update({
                "source": "oom_replan",
                "parent_plan_id": parent_plan.plan_id,
                "failed_batch_idx": self.current_batch_idx if failed_batch_idx is None else int(failed_batch_idx),
                "replan_factor": float(factor),
                "previous_mode": parent_plan.mode.name,
                "previous_num_batches": parent_plan.num_batches,
                "previous_total_estimated_memory_gb": float(parent_plan.total_estimated_memory_gb),
                "metadata_only": True,
            })
            self.current_plan = new_plan
            self.stats["replans_created"] += 1
            return new_plan

        policy_key = self.current_plan.replan_policy_key or "<missing effective seed policy>"
        raise RuntimeError(
            f"Automatic OOM replan is disabled for algorithm '{algorithm_key}' "
            f"(seed_policy={policy_key}). The current seed policy is not guaranteed "
            "partition-invariant, so shrinking or regrouping batches could change random "
            "streams. Use checkpoint/resume or define a partition-invariant SeedPolicySpec "
            "before enabling automatic rebatch."
        )
    
    @staticmethod
    def _replan_metadata(params: EstimationParams) -> Dict[str, Any]:
        from matrix_factorization.core.contracts import get_effective_seed_policy_summary

        seed_policy = get_effective_seed_policy_summary(
            params.algorithm_key,
            getattr(params, "seed_partition_policy", "legacy"),
        )
        return {
            "seed_partition_policy": seed_policy["requested_policy"],
            "replan_policy_key": seed_policy["policy_key"],
            "automatic_rebatch_allowed": bool(seed_policy["automatic_rebatch_allowed"]),
            "replan_implemented": False,
        }

    def set_memory_guard(self, guard) -> None:
        """Set the memory guard for runtime monitoring."""
        self._memory_guard = guard
    
    def _compute_alpha_batches(
        self, 
        params: EstimationParams, 
        target_gb: float
    ) -> Optional[List[BatchConfig]]:
        """
        Compute alpha batches using greedy algorithm.
        
        Groups alphas to fill target memory, using dynamic sizing based on
        alpha_max per batch (since memory scales with alpha).
        """
        alpha_values = sorted(params.alpha_values)
        if not alpha_values:
            return None
        
        batches = []
        current_start = 0
        n = len(alpha_values)
        
        while current_start < n:
            # Greedy: try to fit as many alphas as possible
            best_end = current_start + 1
            
            for end in range(n, current_start, -1):
                batch_alphas = alpha_values[current_start:end]
                batch_alpha_max = max(batch_alphas) if batch_alphas else 0.1
                
                # Skip zero alpha (no edges, negligible memory)
                if batch_alpha_max <= 0:
                    best_end = end
                    break
                
                batch_params = replace(
                    params,
                    alpha_values=batch_alphas,
                )
                
                estimate = self.estimator.estimate(batch_params)
                
                if estimate.total_gb <= target_gb:
                    logger.info(
                        f"Batch selected: alphas={len(batch_alphas)}, α_max={batch_alpha_max:.2f}, "
                        f"estimate={estimate.total_gb:.1f}GB <= target={target_gb:.1f}GB"
                    )
                    batches.append(BatchConfig(
                        sample_range=(0, params.S),
                        alpha_range=(current_start, end),
                        estimated_memory_gb=estimate.total_gb,
                        alpha_values=batch_alphas,
                        memory_breakdown=dict(estimate.breakdown),
                    ))
                    current_start = end
                    break
            else:
                # Cannot fit even single alpha - need to reduce S or fail
                return None
        
        return batches if batches else None

    def _calibration_source(self, params: EstimationParams) -> str:
        calibration = getattr(self.estimator, "_calibration_data", {})
        gpu_model = getattr(self.estimator, "gpu_model", "cpu")
        if gpu_model in calibration and params.algorithm_key in calibration[gpu_model]:
            return f"calibrated:{gpu_model}"
        return "theory_unchecked"
    
    def _create_linear_plan(
        self, 
        params: EstimationParams, 
        available_gb: float
    ) -> ExecutionPlan:
        """Create fully sequential execution plan."""
        replan_metadata = self._replan_metadata(params)
        batches = []
        
        for i, alpha in enumerate(params.alpha_values):
            single_params = replace(params, alpha_values=[alpha])
            estimate = self.estimator.estimate(single_params)
            
            batches.append(BatchConfig(
                sample_range=(0, params.S),
                alpha_range=(i, i + 1),
                estimated_memory_gb=estimate.total_gb,
                alpha_values=[alpha],
                memory_breakdown=dict(estimate.breakdown),
            ))
        
        total_mem = max(b.estimated_memory_gb for b in batches) if batches else 0
        
        plan = ExecutionPlan(
            mode=ParallelMode.LINEAR,
            batches=batches,
            total_estimated_memory_gb=total_mem,
            allocation_config=self.config,
            algorithm_key=params.algorithm_key,
            gpu_model=self.estimator.gpu_model,
            available_memory_gb=available_gb,
            **replan_metadata,
        )
        return self._attach_plan_provenance(plan, params)

    @staticmethod
    def _attach_plan_provenance(plan: ExecutionPlan, params: EstimationParams) -> ExecutionPlan:
        params_snapshot = params.to_dict()
        batch_summary = [
            {
                "sample_range": [int(batch.sample_range[0]), int(batch.sample_range[1])],
                "alpha_range": [int(batch.alpha_range[0]), int(batch.alpha_range[1])],
                "alpha_values": [float(alpha) for alpha in batch.alpha_values],
                "estimated_memory_gb": float(batch.estimated_memory_gb),
            }
            for batch in plan.batches
        ]
        plan_payload = {
            "mode": plan.mode.name,
            "algorithm_key": plan.algorithm_key,
            "estimation_params": params_snapshot,
            "batches": batch_summary,
            "seed_partition_policy": plan.seed_partition_policy,
            "replan_policy_key": plan.replan_policy_key,
        }
        digest = hashlib.blake2b(
            json.dumps(plan_payload, sort_keys=True).encode("utf-8"),
            digest_size=8,
        ).hexdigest()
        plan.plan_id = f"plan_{digest}"
        plan.estimation_params = params_snapshot
        plan.replan_provenance = {
            "source": "initial_plan",
            "mode": plan.mode.name,
            "num_batches": len(plan.batches),
            "batch_summary": batch_summary,
            "allocation_ratio": float(plan.allocation_config.allocation_ratio),
            "available_memory_gb": float(plan.available_memory_gb),
            "total_estimated_memory_gb": float(plan.total_estimated_memory_gb),
            "metadata_only": True,
        }
        return plan
    
    def _execute_batch(
        self,
        batch: BatchConfig,
        algorithm: Any,
        step_callback: Optional[Callable],
    ) -> Dict[str, Any]:
        """Execute a single batch."""
        logger.debug(
            f"Executing batch: alphas={batch.alpha_range}, "
            f"samples={batch.sample_range}, "
            f"estimated={batch.estimated_memory_gb:.1f}GB"
        )
        
        # Call algorithm's batch execution method
        if hasattr(algorithm, 'run_batch'):
            return algorithm.run_batch(
                alpha_values=batch.alpha_values,
                sample_range=batch.sample_range,
                step_callback=step_callback,
            )
        elif hasattr(algorithm, 'train_batch_alphas'):
            # Legacy interface
            return algorithm.train_batch_alphas(
                alpha_values=batch.alpha_values,
                step_callback=step_callback,
            )
        else:
            raise TypeError(
                f"Algorithm {type(algorithm).__name__} must have "
                "'run_batch' or 'train_batch_alphas' method"
            )
    
    def _cleanup_between_batches(self) -> None:
        """Clean up GPU memory between batches."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _get_available_memory(self, use_total: bool = False) -> float:
        """Get available GPU memory in GB.
        
        Args:
            use_total: If True, return physical total memory (for OOM recovery).
                       If False, return available memory (total - reserved).
        
        Subtracts fixed CUDA overhead (~500MB) for context, compile cache, etc.
        This is the ONLY place where overhead is accounted for.
        """
        if use_total:
            # Use physical total memory - useful for OOM recovery replanning
            raw_memory = self.estimator.get_total_memory() or 8.0
        else:
            raw_memory = self.estimator.get_available_memory() or 8.0  # Default for CPU
        CUDA_OVERHEAD_GB = 0.5  # Fixed 500MB for CUDA context
        return max(raw_memory - CUDA_OVERHEAD_GB, 1.0)
    
    def _get_total_memory(self) -> float:
        """Get total GPU memory in GB (physical total, not available)."""
        return self.estimator.get_total_memory() or 8.0


class BatchExecutionContext:
    """
    Context manager for batch execution with automatic cleanup.
    
    Usage:
        with BatchExecutionContext(coordinator, batch):
            # Execute batch operations
            result = algorithm.run(batch)
    """
    
    def __init__(self, coordinator: ParallelCoordinator, batch: BatchConfig):
        self.coordinator = coordinator
        self.batch = batch
        self._initial_memory = 0.0
    
    def __enter__(self):
        """Record initial memory state."""
        if torch.cuda.is_available():
            self._initial_memory = torch.cuda.memory_allocated() / (1024**3)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up and record memory usage."""
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
            
            # Record actual usage for calibration
            if self.coordinator.estimator and peak_memory > 0:
                logger.debug(
                    f"Batch memory: estimated={self.batch.estimated_memory_gb:.2f}GB, "
                    f"actual_peak={peak_memory:.2f}GB"
                )
            
            # Reset peak stats for next batch
            torch.cuda.reset_peak_memory_stats()
        
        # Clean up
        self.coordinator._cleanup_between_batches()
        
        return False  # Don't suppress exceptions
