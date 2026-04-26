"""
Experiment Runner - The Core Execution Engine.

Responsibilities:
1. Execute experiments according to configuration
2. Handle scan modes (alpha, steps, etc.)
3. Manage memory and batching
4. Collect and save results
5. Broadcast execution events (Progress Bridge)

Supports nested scans:
- Outer loop: N, M (different matrix sizes)
- Inner loop: alpha, steps (within each size)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, TYPE_CHECKING
from pathlib import Path
from enum import Enum
import hashlib
import inspect
import json
import time
import logging
import torch

from .config import ExperimentConfig, MatrixParams, ScanConfig
from .result import ExperimentResult, SingleRunResult, ExperimentMetadata, Checkpoint
from .data_factory import DataFactory, ExperimentData
from ..contracts import AlgorithmResult

# Parallel execution support
from ..parallel import (
    get_parallel_coordinator,
    get_memory_estimator,
    EstimationParams,
    AllocationPresets,
)
from ..parallel.batch_checkpoint import CheckpointManager, config_to_dict
from ..parallel.memory_guard import MemoryAbortException

if TYPE_CHECKING:
    from ...modules.algorithms.base import AlgorithmBase

logger = logging.getLogger(__name__)


class ProgressEventType(Enum):
    """Types of progress events."""
    EXPERIMENT_START = "experiment_start"
    EXPERIMENT_END = "experiment_end"
    BATCH_START = "batch_start"
    BATCH_END = "batch_end"
    STEP_UPDATE = "step_update"
    POINT_START = "point_start"
    POINT_COMPLETE = "point_complete"
    EXECUTION_PLAN = "execution_plan"
    ERROR = "error"


@dataclass
class ProgressEvent:
    """Event payload for progress updates."""
    type: ProgressEventType
    payload: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def batch_idx(self) -> Optional[int]:
        return self.payload.get('batch_idx')
        
    @property
    def step(self) -> Optional[int]:
        return self.payload.get('step')


@dataclass
class BatchPlan:
    """Plan for executing a batch of scan points."""
    start_idx: int
    end_idx: int
    values: List[Any]
    estimated_memory_gb: float


class ExperimentRunner:
    """
    Main experiment execution engine.
    
    Features:
    1. Unified execution for all algorithms
    2. Automatic memory-based batching
    3. Support for nested scans (N/M outer, alpha/steps inner)
    4. Checkpoint support for step scans
    5. Event-driven progress reporting (Progress Bridge)
    
    Usage:
        runner = ExperimentRunner()
        result = runner.run(config, observer=my_observer)
    """
    
    def __init__(
        self,
        device: torch.device = None,
        max_memory_gb: float = None,
        verbose: bool = True,
    ):
        """
        Initialize runner.
        
        Args:
            device: Target device (default: CUDA if available)
            max_memory_gb: Maximum GPU memory to use (default: auto-detect)
            verbose: Print progress information (legacy console output)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        
        # Auto-detect memory limit
        if max_memory_gb is None and torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.max_memory_gb = total_mem * 0.85  # Leave 15% buffer
        else:
            self.max_memory_gb = max_memory_gb or 24.0
        
        self.data_factory = DataFactory(self.device)
        self._algorithm_cache = {}
        self._warned_private_metric_channel = set()
        
        # Initialize parallel coordinator for intelligent batching
        ParallelCoordinator = get_parallel_coordinator()
        MemoryEstimator = get_memory_estimator()
        self.parallel_coordinator = ParallelCoordinator(
            estimator=MemoryEstimator(),
            config=AllocationPresets.CONSERVATIVE,
        )
        
        # Enable MemoryGuard for runtime OOM protection
        from ..parallel.memory_guard import create_monitored_guard
        self._memory_guard = create_monitored_guard(self.parallel_coordinator)
    
    def run(
        self,
        config: ExperimentConfig,
        observer: Optional[Callable[[ProgressEvent], None]] = None,
        # Legacy callbacks (mapped to observer internally or ignored if observer present)
        step_callback: Optional[Callable[[int, int], None]] = None,
        point_callback: Optional[Callable[[int, int, Any], None]] = None,
        # Resume support
        resume_results: Optional[Dict[float, Dict]] = None,
        # Output options for checkpoint (rsb_ordering, save_tensors, etc.)
        output_options: Optional[Dict[str, Any]] = None,
        # Raw YAML config string for checkpoint
        raw_yaml: str = "",
    ) -> ExperimentResult:
        """
        Run a single experiment.
        
        Args:
            config: Experiment configuration
            observer: Callback for ProgressEvents (recommended)
            step_callback: Legacy callback (step, max_steps)
            point_callback: Legacy callback (idx, total, value)
            resume_results: Pre-completed results from checkpoint (metrics only)
            output_options: Output options dict (rsb_ordering, save_tensors, uniform_colormap)
                           Saved to checkpoint for resume
            raw_yaml: Complete original YAML config string for checkpoint
            
        Returns:
            ExperimentResult with all results
        """
        # Adapter for legacy callbacks if no observer provided
        if observer is None and (step_callback or point_callback):
            observer = self._create_legacy_observer(step_callback, point_callback)

        # Normalize config (handle legacy Config object)
        if not hasattr(config, 'scan'):
            # Construct synthetic ScanConfig from legacy AlphaConfig
            vals = config.alpha.get_values()
            # If quick run or default, it's usually an alpha scan
            # But wait, does legacy support steps scan? 
            # Legacy Runner decided based on config.scan? No, legacy Config has alpha.
            # We assume alpha scan for legacy config unless specified otherwise
            # create a mock object if ScanConfig import fails or for simplicity
            class MockScan:
                dimension = 'alpha'
                values = vals
                num_points = len(vals)
                is_steps_scan = False
                is_alpha_scan = True
            
            # Monkey-patch config for this run scope (or wrapper)
            # Since we can't easily modify the object's class, we'll just set the attribute
            # This is safe in python given we just need it for reading
            config.scan = MockScan()
            
            # Also ensure matrix shortcut exists if we accessed it (we fixed that though)
            # But let's check spreading
            if not hasattr(config, 'spreading') and hasattr(config, 'algorithm_key'):
                 if 'spreading' in config.algorithm_key:
                      # Legacy might have spreading config elsewhere or default?
                      # We'll assume None/Default if missing
                      config.spreading = None

            # Patch seeds
            if not hasattr(config, 'seeds'):
                class MockSeeds:
                    base_seed = getattr(config.training, 'seed', 42)
                    teacher_seed = 12345
                    spreading_seed = 99999
                    student_seed = 0
                config.seeds = MockSeeds()

            # Patch algorithm_params -> algorithm
            if not hasattr(config, 'algorithm_params') and hasattr(config, 'algorithm'):
                config.algorithm_params = config.algorithm

        self._emit(observer, ProgressEventType.EXPERIMENT_START, {
            'experiment_name': getattr(config, 'experiment_name', 'unnamed_experiment'),
            'config': config,
            'matrix': f"{config.matrix.N1}x{config.matrix.N2}, M={config.matrix.M}",
            'scan': f"{config.scan.dimension} ({len(config.scan.values)} points)"
        })
        
        if self.verbose:
            print(f"Starting experiment: {getattr(config, 'experiment_name', 'unnamed_experiment')}")
            print(f"  Algorithm: {config.algorithm_key}")
            print(f"  Matrix: {config.matrix.N1}x{config.matrix.N2}, M={config.matrix.M}")

        if self._should_use_canonical_scan_executor(config):
            return self._run_canonical_scan(
                config=config,
                observer=observer,
                output_options=output_options,
                raw_yaml=raw_yaml,
            )
        
        try:
            # Start MemoryGuard for OOM protection
            if hasattr(self, '_memory_guard') and self._memory_guard:
                self._memory_guard.start()
            
            # Create Teacher data (shared across all scan points)
            W_teacher, X_teacher, Y_teacher = self.data_factory.create_teacher(
                N1=config.matrix.N1,
                N2=config.matrix.N2,
                M=config.matrix.M,
                teacher_key=getattr(config, 'teacher_key', 'standard'),
                seed=config.seeds.teacher_seed,
                init_distribution=getattr(getattr(config, 'teacher', None), 'init_distribution', 'gaussian'),
            )
            
            # Create result container with raw data
            result = ExperimentResult(
                experiment_id=getattr(config, 'experiment_name', 'unnamed_experiment'),
                config=config,
                scan_dimension=config.scan.dimension,
                scan_values=config.scan.values,
                metadata=ExperimentMetadata.create_now(),
                # Raw data for post-hoc analysis
                W_teacher=W_teacher,
                X_teacher=X_teacher,
                Y_teacher=Y_teacher,
            )
            if output_options and output_options.get('experiment_plan'):
                import copy
                result.metadata.contract = copy.deepcopy(output_options['experiment_plan'])
            runtime_extensions = self._build_runtime_extension_executor(
                result.metadata.contract,
                config.algorithm_key,
            )
            self._runtime_extensions = runtime_extensions
            runtime_state = self._initial_runtime_state(
                config=config,
                W_teacher=W_teacher,
                X_teacher=X_teacher,
                Y_teacher=Y_teacher,
            )
            runtime_state = runtime_extensions.dispatch(
                "before_initialize",
                runtime_state,
                self._runtime_context(config, "before_initialize"),
            )
            
            # Get algorithm
            algorithm = self._get_algorithm(config)
            result.metadata.contract["algorithm_config_trace"] = getattr(
                algorithm,
                "_contract_config_trace",
                self._algorithm_config_trace(config, None),
            )
            
            # Route to appropriate scan handler
            if getattr(config.scan, 'is_steps_scan', config.scan.dimension == 'steps'):
                self._run_steps_scan(config, algorithm, result, observer)
            else:
                self._run_standard_scan(config, algorithm, result, observer, resume_results, output_options, raw_yaml)
            
            runtime_extensions.dispatch(
                "after_run",
                runtime_state,
                self._runtime_context(config, "after_run"),
            )
            runtime_extensions.run_analyzers(result)
            result.metadata.contract["runtime_extension_report"] = runtime_extensions.report.to_dict()
            if hasattr(self, '_memory_guard') and self._memory_guard:
                result.metadata.contract["memory_guard_runtime"] = self._memory_guard.snapshot()
            self._emit(observer, ProgressEventType.EXPERIMENT_END, {'result': result})
            return result

        except Exception as e:
            self._emit(observer, ProgressEventType.ERROR, {'error': str(e)})
            raise e
        finally:
            # Stop MemoryGuard
            if hasattr(self, '_memory_guard') and self._memory_guard:
                self._memory_guard.stop()

    def _should_use_canonical_scan_executor(self, config: ExperimentConfig) -> bool:
        """Return True when scan.axes requires grouping beyond alpha-only."""
        if getattr(config, "_disable_canonical_scan_executor", False):
            return False
        scan_spec = getattr(config, "scan_spec", None)
        if not isinstance(scan_spec, dict) or "axes" not in scan_spec:
            return False
        axes = scan_spec.get("axes") or {}
        return set(axes.keys()) != {"alpha"}

    def _run_canonical_scan(
        self,
        config: ExperimentConfig,
        observer: Optional[Callable[[ProgressEvent], None]] = None,
        output_options: Optional[Dict[str, Any]] = None,
        raw_yaml: str = "",
    ) -> ExperimentResult:
        """Execute canonical multi-axis scan by isolated effective-config groups."""
        import copy
        import gc
        from .result import ResultCube
        from matrix_factorization.core.contracts import get_batching_specs
        from matrix_factorization.core.scan_planning import build_scan_plan
        from matrix_factorization.core.parallel.resource_execution import (
            build_scan_resource_execution_plan,
            effective_config_for_scan_points,
        )

        scan_plan = build_scan_plan(config)
        resource_plan = build_scan_resource_execution_plan(
            scan_plan=scan_plan,
            base_config=config,
            coordinator=self.parallel_coordinator,
            batching_spec=get_batching_specs().get(config.algorithm_key),
        )
        if resource_plan.preflight_errors:
            raise ValueError(
                "canonical scan resource preflight failed:\n"
                + "\n".join(f"- {item}" for item in resource_plan.preflight_errors)
            )
        aggregate = ExperimentResult(
            experiment_id=getattr(config, "experiment_name", "canonical_scan"),
            config=config,
            scan_dimension="scan",
            scan_values=scan_plan.point_ids(),
            metadata=ExperimentMetadata.create_now(),
        )
        if output_options and output_options.get("experiment_plan"):
            aggregate.metadata.contract = copy.deepcopy(output_options["experiment_plan"])
        aggregate.metadata.contract["scan_plan"] = scan_plan.to_dict()
        aggregate.metadata.contract["runtime_resource_plan"] = resource_plan.to_dict()
        aggregate.result_cube = ResultCube(
            axes={axis.key: axis.to_dict() for axis in scan_plan.axes},
            metric_semantics=aggregate.metric_semantics(),
        )
        points_by_id = {point.point_id: point for point in scan_plan.points}
        shared_teacher_shape = None
        teacher_shape_mismatch = False

        for resource_batch in resource_plan.batches:
            batch_points = [points_by_id[item.scan_point_id] for item in resource_batch.work_items]
            effective_config = effective_config_for_scan_points(config, batch_points)
            setattr(effective_config, "_disable_canonical_scan_executor", True)
            child_runner = ExperimentRunner(device=self.device, verbose=self.verbose)
            child_output_options = copy.deepcopy(output_options) if output_options else None
            child_result = child_runner.run(
                effective_config,
                observer=observer,
                output_options=child_output_options,
                raw_yaml=raw_yaml,
            )
            child_teacher = getattr(child_result, "W_teacher", None)
            if child_teacher is not None:
                child_shape = tuple(child_teacher.shape)
                if shared_teacher_shape is None and not teacher_shape_mismatch:
                    shared_teacher_shape = child_shape
                    aggregate.W_teacher = child_result.W_teacher
                    aggregate.X_teacher = child_result.X_teacher
                    aggregate.Y_teacher = child_result.Y_teacher
                elif shared_teacher_shape != child_shape:
                    teacher_shape_mismatch = True
                    aggregate.W_teacher = None
                    aggregate.X_teacher = None
                    aggregate.Y_teacher = None
            for point in batch_points:
                child_key = self._child_result_key_for_scan_point(point, effective_config)
                if child_key not in child_result.results:
                    raise ValueError(
                        f"canonical scan point {point.point_id} expected child result {child_key!r}; "
                        f"available: {list(child_result.results)}"
                    )
                child_single = child_result.results[child_key]
                single = copy.copy(child_single)
                single.scan_value = point.point_id
                aggregate.add_result(point.point_id, single)
                aggregate.result_cube.add_point(
                    point.point_id,
                    point.coordinates,
                    child_single.metrics,
                    overrides=point.overrides,
                    effective_config_hash=point.effective_config_hash,
                    group_id=point.group_id,
                    metric_contract=child_single.metric_contract,
                )
            del child_runner
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._emit(observer, ProgressEventType.EXPERIMENT_END, {"result": aggregate})
        return aggregate

    def _effective_config_for_scan_group(self, base_config: ExperimentConfig, points: List[Any]) -> ExperimentConfig:
        import copy

        config = copy.deepcopy(base_config)
        group_overrides = dict(points[0].overrides)
        for path, value in group_overrides.items():
            if path == "training.max_steps" and len(points) > 1:
                continue
            self._set_config_path(config, path, value)
        if any(point.max_steps is not None for point in points):
            step_values = sorted({int(point.max_steps) for point in points if point.max_steps is not None})
            config.scan = ScanConfig(dimension="steps", values=step_values)
            if points[0].alpha is not None:
                config.algorithm_params.default_alpha = float(points[0].alpha)
            if step_values:
                config.training.max_steps = max(step_values)
        else:
            alpha_values = sorted({float(point.alpha) for point in points if point.alpha is not None})
            if not alpha_values:
                alpha_values = [float(config.algorithm_params.default_alpha)]
            config.scan = ScanConfig(dimension="alpha", values=alpha_values)
        config.scan_spec = {"axes": {"alpha": {"path": "alpha", "values": list(config.scan.values)}}}
        config.experiment_name = f"{base_config.experiment_name}_{points[0].group_id}".replace("|", "_").replace("=", "-")
        return config

    @staticmethod
    def _set_config_path(config: ExperimentConfig, path: str, value: Any) -> None:
        if path == "alpha":
            return
        target = config
        parts = path.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)

    @staticmethod
    def _child_result_key_for_scan_point(point: Any, config: ExperimentConfig) -> Any:
        if getattr(config.scan, "dimension", "") == "alpha" and getattr(point, "alpha", None) is not None:
            return float(point.alpha)
        if getattr(point, "max_steps", None) is not None:
            return int(point.max_steps)
        if getattr(point, "alpha", None) is not None:
            return float(point.alpha)
        return float(config.algorithm_params.default_alpha)
    
    def run_scaling_sweep(
        self,
        base_config: ExperimentConfig,
        matrix_sizes: List[tuple],  # [(N1, N2, M), ...]
        output_dir: Optional[Path] = None,
        observer: Optional[Callable[[ProgressEvent], None]] = None,
        output_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[tuple, ExperimentResult]:
        """
        Run nested scan: outer loop over matrix sizes, inner loop over alpha/steps.
        """
        results = {}
        
        for i, (N1, N2, M) in enumerate(matrix_sizes):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Scaling sweep {i+1}/{len(matrix_sizes)}: N={N1}, M={M}")
            
            # Create config for this size
            config = ExperimentConfig(
                matrix=MatrixParams(N1=N1, N2=N2, M=M),
                training=base_config.training,
                algorithm_key=base_config.algorithm_key,
                scan=base_config.scan,
                seeds=base_config.seeds,
                algorithm_params=base_config.algorithm_params,
                spreading=base_config.spreading,
                teacher=base_config.teacher,
                experiment_name=f"{base_config.experiment_name}_N{N1}_M{M}",
                teacher_key=base_config.teacher_key,
                notes=f"Scaling sweep: N={N1}, M={M}",
            )
            
            # Run experiment
            result = self.run(config, observer=observer, output_options=output_options)
            results[(N1, N2, M)] = result
            
            # Save intermediate results
            if output_dir:
                result.save_unified(output_dir / f"N{N1}_M{M}.pt")
        
        return results
    
    def _run_standard_scan(
        self,
        config: ExperimentConfig,
        algorithm: 'AlgorithmBase',
        result: ExperimentResult,
        observer: Optional[Callable[[ProgressEvent], None]],
        resume_results: Optional[Dict[float, Dict]] = None,
        output_options: Optional[Dict[str, Any]] = None,
        raw_yaml: str = "",
    ):
        """
        Run standard scan (alpha, samples, seed).
        Uses ParallelCoordinator and emits detailed Batch events.
        """
        scan_values = config.scan.values
        total_points = len(scan_values)
        
        params = self._estimation_params_for_config(config, scan_values)

        # Get execution plan from ParallelCoordinator.  If the scan planner is
        # available, also attach WorkItem-level metadata so alpha/sample/scan
        # folding decisions are explicit before deeper runner refactors.
        resource_execution_plan = None
        scan_points_by_alpha = {}
        try:
            from matrix_factorization.core.contracts import get_batching_specs
            from matrix_factorization.core.scan_planning import build_scan_plan

            scan_plan = build_scan_plan(config)
            scan_points_by_alpha = {
                float(point.alpha): point
                for point in scan_plan.points
                if point.alpha is not None
            }
            result.result_cube.axes = {axis.key: axis.to_dict() for axis in scan_plan.axes}
            batching_spec = get_batching_specs().get(config.algorithm_key)
            plan, resource_execution_plan = self.parallel_coordinator.plan_resource_execution(
                scan_plan,
                params,
                batching_spec=batching_spec,
            )
        except Exception as exc:
            logger.warning("Falling back to alpha-only execution plan: %s", exc)
            plan = self.parallel_coordinator.plan_execution(params)
        result.metadata.contract["runtime_resource_plan"] = self._runtime_resource_plan_report(
            config,
            plan,
            resource_execution_plan=resource_execution_plan,
        )
        
        # Phase 3 Tensor Parallel: Let all alphas be processed in single batch
        # The tensor_spreading_parallel._train_full_parallel() handles true Alpha+Sample parallelism
        # No override needed - use the default plan from ParallelCoordinator

        
        # Initialize checkpoint manager. CLI/web runs pass a run-scoped path.
        checkpoint_path = None
        if output_options:
            checkpoint_path = output_options.get('checkpoint_path')
        ckpt_mgr = CheckpointManager(Path(checkpoint_path)) if checkpoint_path else CheckpointManager()
        
        # Prepare config dict for checkpoint
        config_dict = config_to_dict(config)
        
        # Check if this is a resume or fresh start
        if resume_results:
            # Resume mode: use existing results, don't delete checkpoint
            completed_alphas: List[float] = list(resume_results.keys())
            checkpoint_results: Dict[float, Dict] = dict(resume_results)
            if self.verbose:
                print(f"  📂 Resuming: {len(completed_alphas)} alphas already done")
        else:
            # Fresh start: delete old checkpoint
            ckpt_mgr.delete()
            completed_alphas: List[float] = []
            checkpoint_results: Dict[float, Dict] = {}
        
        # Calculate max_alpha for Physics-Aware ETA in UI
        # Pass batch structure to UI so it can predict time accurately
        batch_assignments = []
        for b in plan.batches:
            if b.alpha_values:
                # (start_index, end_index, alpha_max)
                # Note: indices are cumulative points, but for ETA we just need relative weight (alpha_max)
                # We'll use dummy expected indices for now, matching the list order
                start = 0 
                end = len(b.alpha_values)
                alpha_max = max(b.alpha_values) if b.alpha_values else 1.0
                batch_assignments.append((start, end, alpha_max))

        # Emit Execution Plan immediately so the UI knows total batches and workload
        self._emit(observer, ProgressEventType.EXECUTION_PLAN, {
            'batches': batch_assignments,
            'total_batches': plan.num_batches,
            'mode': plan.mode.name,
            'algorithm_key': config.algorithm_key  # Phase 4: Enable algorithm-aware ETA
        })

        if self.verbose:
            print(f"  Execution plan: {plan.mode.name}, {plan.num_batches} batch(es)")
        
        total_batches = plan.num_batches
        
        # Execute batches according to plan
        # Execute batches according to plan
        global_point_idx = 0
        skipped_msg_printed = False
        
        for batch_idx, batch in enumerate(plan.batches):
            batch_start_time = time.time()
            planned_batch_alpha_values = self._batch_alpha_values(batch)
            
            # Skip if all alphas in this batch are already completed (resume mode)
            remaining_alphas = [a for a in planned_batch_alpha_values if a not in completed_alphas]
            if not remaining_alphas:
                if self.verbose and not skipped_msg_printed:
                    print("  ⏭️ Skipping completed batch(es)...")
                    skipped_msg_printed = True
                global_point_idx += len(planned_batch_alpha_values)
                continue
            skipped_alphas_in_batch = len(planned_batch_alpha_values) - len(remaining_alphas)
            if skipped_alphas_in_batch:
                global_point_idx += skipped_alphas_in_batch
            batch_alpha_values = remaining_alphas
            
            # Reset skip flag when we encounter a batch to run
            skipped_msg_printed = False
            
            # Emit Batch Start Event
            self._emit(observer, ProgressEventType.BATCH_START, {
                'batch_idx': batch_idx,
                'total_batches': total_batches,
                'alpha_values': batch_alpha_values,
                'estimated_memory_gb': batch.estimated_memory_gb,
                'steps_per_alpha': config.training.max_steps,
            })
            
            # if self.verbose:
            #     alpha_range = f"{min(batch_alpha_values):.2f}-{max(batch_alpha_values):.2f}"
            #     print(f"  Batch {batch_idx+1}/{total_batches}: alpha {alpha_range}")
            
            try:
                # Create data for this batch
                data = self.data_factory.create(config, alpha_values=batch_alpha_values)
                
                # Check abort after data creation (OOM may occur during data setup)
                if self._memory_guard and self._memory_guard.is_running:
                    self._memory_guard.check_abort()
                
                # Create a localized step callback for this batch
                def internal_step_callback(step, total, metrics=None):
                    # Check memory abort flag every step (no algorithm modification needed)
                    if self._memory_guard and self._memory_guard.is_running:
                        self._memory_guard.check_abort()
                    
                    self._emit(observer, ProgressEventType.STEP_UPDATE, {
                        'step': step, 
                        'total': total,
                        'batch_idx': batch_idx,
                        'metrics': metrics
                    })

                # Run algorithm for this batch
                if output_options:
                    setattr(algorithm, 'heatmap_metric', output_options.get('heatmap_metric', 'Q_Y'))
                batch_algorithm_result = self._run_algorithm_result(
                    algorithm=algorithm,
                    config=config,
                    data=data,
                    step_callback=internal_step_callback,
                )
                self._record_algorithm_result_summary(
                    result=result,
                    batch_idx=batch_idx,
                    alpha_values=batch_alpha_values,
                    algorithm_result=batch_algorithm_result,
                )
                W_students, X_students = self._matrix_factors_from_result(batch_algorithm_result)
                
                # Process each alpha in the batch
                for alpha_idx, alpha in enumerate(batch_alpha_values):
                    start_time = time.time()
                    
                    # Report Point Start (for granular UI)
                    self._emit(observer, ProgressEventType.POINT_START, {
                        'point_idx': global_point_idx + 1,
                        'total_points': total_points,
                        'value': alpha
                    })
                    
                    W_single = self._slice_alpha_factor(W_students, alpha_idx)
                    X_single = self._slice_alpha_factor(X_students, alpha_idx)
                    
                    # Create single-alpha data for metrics
                    single_data = ExperimentData(
                        W_teacher=data.W_teacher,
                        X_teacher=data.X_teacher,
                        Y_teacher=data.Y_teacher,
                        masks=data.masks[alpha_idx:alpha_idx+1] if data.masks is not None and data.masks.dim() == 3 else data.masks,
                        spreading_data=data.spreading_data,
                        alpha_values=[alpha],
                    )
                    
                    # Compute metrics
                    W_metrics = W_single.unsqueeze(0) if W_single is not None and W_single.dim() == 3 else W_single
                    X_metrics = X_single.unsqueeze(0) if X_single is not None and X_single.dim() == 3 else X_single
                    metrics = self._compute_metrics(
                        W_students=W_metrics,
                        X_students=X_metrics,
                        data=single_data,
                        algorithm=algorithm,
                        algorithm_result=batch_algorithm_result,
                    )
                    metric_contract = self._validate_metric_payload(
                        config.algorithm_key,
                        metrics,
                        source=self._metric_source(single_data, batch_algorithm_result),
                    ).to_dict()
                    
                    # Store result
                    single_result = SingleRunResult(
                        scan_value=alpha,
                        metrics=metrics,
                        W_students=W_single,
                        X_students=X_single,
                        mask=None, # Optimization: Don't save masks to save space unless needed
                        duration_seconds=time.time() - start_time,
                        metric_contract=metric_contract,
                    )
                    result.add_result(alpha, single_result)
                    point = scan_points_by_alpha.get(float(alpha))
                    if point is not None:
                        result.result_cube.add_point(
                            point.point_id,
                            point.coordinates,
                            metrics,
                            overrides=point.overrides,
                            effective_config_hash=point.effective_config_hash,
                            group_id=point.group_id,
                            metric_contract=metric_contract,
                        )
                    
                    global_point_idx += 1
                    self._emit(observer, ProgressEventType.POINT_COMPLETE, {
                        'point_idx': global_point_idx,
                        'total_points': total_points,
                        'value': alpha,
                        'metrics': metrics
                    })
                
                # Emit Batch End Event
                self._emit(observer, ProgressEventType.BATCH_END, {
                    'batch_idx': batch_idx, 
                    'duration': time.time() - batch_start_time
                })
                self._dispatch_after_batch_runtime_extensions(
                    config=config,
                    batch_idx=batch_idx,
                    alpha_values=batch_alpha_values,
                    algorithm_result=batch_algorithm_result,
                    result=result,
                )
                
                # Save checkpoint after successful batch
                for alpha in batch_alpha_values:
                    if alpha not in completed_alphas:
                        completed_alphas.append(alpha)
                        # Save metrics (not tensors)
                        if alpha in result.results:
                            r = result.results[alpha]
                            checkpoint_results[alpha] = {
                                'metrics': r.metrics if hasattr(r, 'metrics') else {},
                                'duration_seconds': r.duration_seconds if hasattr(r, 'duration_seconds') else 0.0,
                            }
                
                ckpt_mgr.save(config_dict, completed_alphas, checkpoint_results, output_options, raw_yaml)
                
                # Clean up batch data before next batch
                del data
                del batch_algorithm_result
                
            except (MemoryAbortException, torch.cuda.OutOfMemoryError) as e:
                # OOM batch recovery: save checkpoint and exit gracefully
                # This is the SAFEST approach - clean process restart via 'mf resume'
                # ensures all torch.compile caches are properly cleared
                abort_kind = (
                    "CUDA OOM"
                    if isinstance(e, torch.cuda.OutOfMemoryError)
                    else "memory guard abort"
                )
                logger.warning(f"Batch {batch_idx} aborted due to {abort_kind}: {e}")
                
                if self.verbose:
                    print(f"\n  ⚠️ OOM detected in batch {batch_idx+1}")
                
                # Clean up partial data to ensure valid checkpoint
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                
                # Save checkpoint with completed alphas
                ckpt_mgr.save(config_dict, completed_alphas, checkpoint_results, output_options, raw_yaml)
                ckpt_mgr.flush()
                
                # Count remaining work
                remaining_count = total_points - len(completed_alphas)
                
                if self.verbose:
                    print("\n" + "=" * 60)
                    print("💾 Progress saved to checkpoint!")
                    print(f"   Completed: {len(completed_alphas)}/{total_points} alphas")
                    print(f"   Remaining: {remaining_count} alphas")
                    print("\n💡 To continue, run: smf resume")
                    print("   This restarts the process with clean GPU memory.")
                    print("=" * 60)
                
                # Exit gracefully instead of trying to continue
                # This ensures clean memory state on next resume
                import sys
                sys.exit(0)
            
            # Force memory cleanup between batches (all algorithms)
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        
        # Cleanup checkpoints on successful completion
        if len(completed_alphas) == total_points:
            ckpt_mgr.flush()
            ckpt_mgr.delete()
    
    def _run_steps_scan(
        self,
        config: ExperimentConfig,
        algorithm: 'AlgorithmBase',
        result: ExperimentResult,
        observer: Optional[Callable[[ProgressEvent], None]],
    ):
        """
        Run steps scan (convergence curve).
        Emits events simulating a single batch with multiple checkpoints.
        """
        step_values = sorted(config.scan.values)
        total_points = len(step_values)
        
        # Treat the entire steps scan as one "Logical Batch" for UI consistency
        # Or maybe separate batches? No, one batch is better for continuity.
        # Let's say it's Batch 0/1.
        
        default_alpha = config.algorithm_params.__dict__.get('default_alpha', 1.0)
        
        self._emit(observer, ProgressEventType.BATCH_START, {
            'batch_idx': 0,
            'total_batches': 1,
            'alpha_values': [default_alpha],
            'estimated_memory_gb': 0.0, # TODO: Estimate
            'steps_per_alpha': max(step_values), # Max steps is the final target
            'mode': 'steps_scan'
        })

        data = self.data_factory.create(config, alpha_values=[default_alpha])
        checkpoint = None
        prev_steps = 0
        batch_start_time = time.time()
        
        # Step callback wrapper
        def internal_step_callback(step, total, metrics=None):
            self._emit(observer, ProgressEventType.STEP_UPDATE, {
                'step': step, 
                'total': total,
                'batch_idx': 0,
                'metrics': metrics
            })
        
        for idx, max_steps in enumerate(step_values):
            start_time = time.time()
            additional_steps = max_steps - prev_steps
            
            self._emit(observer, ProgressEventType.POINT_START, {
                'point_idx': idx + 1,
                'total_points': total_points,
                'value': max_steps
            })
            
            # Run algorithm through the formal result contract.
            step_algorithm_result, new_checkpoint = self._run_algorithm_with_checkpoint_result(
                algorithm=algorithm,
                config=config,
                data=data,
                additional_steps=additional_steps,
                checkpoint=checkpoint,
                step_callback=internal_step_callback,
            )
            self._record_algorithm_result_summary(
                result=result,
                batch_idx=idx,
                alpha_values=[default_alpha],
                algorithm_result=step_algorithm_result,
            )
            W_students, X_students = self._matrix_factors_from_result(step_algorithm_result)
            
            checkpoint = new_checkpoint
            prev_steps = max_steps
            
            metrics = self._compute_metrics(
                W_students=W_students,
                X_students=X_students,
                data=data,
                algorithm=algorithm,
                algorithm_result=step_algorithm_result,
            )
            metric_contract = self._validate_metric_payload(
                config.algorithm_key,
                metrics,
                source=self._metric_source(data, step_algorithm_result),
            ).to_dict()
            
            single_result = SingleRunResult(
                scan_value=max_steps,
                metrics=metrics,
                W_students=W_students,
                X_students=X_students,
                duration_seconds=time.time() - start_time,
                metric_contract=metric_contract,
            )
            result.add_result(max_steps, single_result)
            
            self._emit(observer, ProgressEventType.POINT_COMPLETE, {
                'point_idx': idx + 1,
                'total_points': total_points,
                'value': max_steps,
                'metrics': metrics
            })
            
        self._emit(observer, ProgressEventType.BATCH_END, {
            'batch_idx': 0,
            'duration': time.time() - batch_start_time,
        })
        self._dispatch_after_batch_runtime_extensions(
            config=config,
            batch_idx=0,
            alpha_values=[default_alpha],
            algorithm_result=step_algorithm_result if step_values else None,
            result=result,
        )

    def _run_algorithm(
        self,
        algorithm: 'AlgorithmBase',
        config: ExperimentConfig,
        data: ExperimentData,
        step_callback: Optional[Callable],
    ) -> tuple:
        """Run algorithm and return (W_students, X_students)."""
        # Include tensor algorithm in spreading family (uses step_callback)
        is_spreading_family = 'spreading' in config.algorithm_key or 'tensor' in config.algorithm_key
        if is_spreading_family:
            W_students, X_students = algorithm.train_batch_alphas(
                W_teacher=data.W_teacher,
                X_teacher=data.X_teacher,
                Y_teacher=data.Y_teacher,
                masks=None,
                alpha_values=data.alpha_values,
                seed=config.seeds.base_seed,
                step_callback=step_callback,
            )
        else:
            W_students, X_students = algorithm.train_batch_alphas(
                W_teacher=data.W_teacher,
                X_teacher=data.X_teacher,
                Y_teacher=data.Y_teacher,
                masks=data.masks,
                alpha_values=data.alpha_values,
                seed=config.seeds.base_seed,
                progress_callback=step_callback,
            )
        return W_students, X_students

    @staticmethod
    def _estimation_params_for_config(
        config: ExperimentConfig,
        scan_values: List[Any],
    ) -> EstimationParams:
        """Build memory-estimation parameters from the effective config."""
        f_dist = 'rademacher'
        if config.spreading:
            f_dist = config.spreading.f_distribution

        allow_intra = getattr(config.spreading, 'allow_intra_connection', False) if config.spreading else False
        tensor_order = getattr(config.spreading, 'tensor_order', 2) if config.spreading else 2
        tensor_dims = None
        if tensor_order >= 2:
            tensor_dims_list = [config.matrix.N1, config.matrix.N2]
            for _ in range(2, tensor_order):
                tensor_dims_list.append(config.matrix.N1)
            tensor_dims = tuple(tensor_dims_list)
        return EstimationParams(
            N1=config.matrix.N1,
            N2=config.matrix.N2,
            M=config.matrix.M,
            S=config.training.samples_per_alpha,
            alpha_values=[float(v) for v in scan_values],
            algorithm_key=config.algorithm_key,
            use_compile=config.algorithm_params.use_compile,
            use_bf16=config.algorithm_params.use_bf16,
            f_distribution=f_dist,
            adaptive_damping=config.algorithm_params.adaptive_damping,
            allow_intra_connection=allow_intra,
            tensor_order=tensor_order,
            tensor_dims=tensor_dims,
            seed_partition_policy=config.algorithm_params.seed_partition_policy,
            chunk_size=getattr(config.spreading, "chunk_size", None) if config.spreading else None,
        )

    def _runtime_resource_plan_report(
        self,
        config: ExperimentConfig,
        plan: Any,
        resource_execution_plan: Any = None,
    ) -> Dict[str, Any]:
        """Serialize the actual runner-level execution plan as metadata only."""
        allocation = getattr(plan, "allocation_config", None)
        algorithm_params = getattr(config, "algorithm_params", None)
        spreading = getattr(config, "spreading", None)
        return {
            "algorithm_key": config.algorithm_key,
            "device": str(self.device),
            "gpu_model": getattr(plan, "gpu_model", ""),
            "available_memory_gb": float(getattr(plan, "available_memory_gb", 0.0) or 0.0),
            "total_estimated_memory_gb": float(getattr(plan, "total_estimated_memory_gb", 0.0) or 0.0),
            "mode": getattr(getattr(plan, "mode", None), "name", str(getattr(plan, "mode", ""))),
            "num_batches": int(getattr(plan, "num_batches", 0) or 0),
            "allocation": {
                "allocation_ratio": getattr(allocation, "allocation_ratio", None),
                "max_allocation_gb": getattr(allocation, "max_allocation_gb", None),
                "warning_threshold": getattr(allocation, "warning_threshold", None),
                "critical_threshold": getattr(allocation, "critical_threshold", None),
                "safety_margin": getattr(allocation, "safety_margin", None),
            },
            "config_effective": {
                "use_compile": getattr(algorithm_params, "use_compile", None),
                "compile_fallback_policy": getattr(algorithm_params, "compile_fallback_policy", None),
                "use_bf16": getattr(algorithm_params, "use_bf16", None),
                "dtype_fallback_policy": getattr(algorithm_params, "dtype_fallback_policy", None),
                "use_tf32": getattr(algorithm_params, "use_tf32", None),
                "seed_partition_policy": getattr(algorithm_params, "seed_partition_policy", None),
                "spreading.chunk_size": getattr(spreading, "chunk_size", None) if spreading else None,
                "spreading.tensor_order": getattr(spreading, "tensor_order", None) if spreading else None,
            },
            "seed_policy": self._effective_runtime_seed_policy(config),
            "replan_safety": {
                "seed_partition_policy": getattr(plan, "seed_partition_policy", "legacy"),
                "replan_policy_key": getattr(plan, "replan_policy_key", ""),
                "automatic_rebatch_allowed": bool(getattr(plan, "automatic_rebatch_allowed", False)),
                "replan_implemented": bool(getattr(plan, "replan_implemented", False)),
                "plan_id": getattr(plan, "plan_id", ""),
                "parent_plan_id": getattr(plan, "parent_plan_id", ""),
                "replan_attempt": int(getattr(plan, "replan_attempt", 0)),
                "estimation_params": dict(getattr(plan, "estimation_params", {}) or {}),
                "replan_provenance": dict(getattr(plan, "replan_provenance", {}) or {}),
            },
            "batches": [
                {
                    "batch_index": idx,
                    "sample_range": list(batch.sample_range),
                    "sample_range_honored_by_runner": False,
                    "alpha_range": list(batch.alpha_range),
                    "alpha_values": [float(value) for value in self._batch_alpha_values(batch)],
                    "batch_axes": list(getattr(batch, "batch_axes", []) or []),
                    "calibration_source": getattr(batch, "calibration_source", "theory_unchecked"),
                    "raw_peak_allocated_gb": float(getattr(batch, "raw_peak_allocated_gb", 0.0) or 0.0),
                    "device_peak_gb": float(getattr(batch, "device_peak_gb", 0.0) or 0.0),
                    "dominant_stage": getattr(batch, "dominant_stage", ""),
                    "confidence": float(getattr(batch, "confidence", 0.0) or 0.0),
                    "work_items": [
                        item.to_dict() if hasattr(item, "to_dict") else dict(item)
                        for item in (getattr(batch, "work_items", []) or [])
                    ],
                    "estimated_memory_gb": float(batch.estimated_memory_gb),
                    "memory_breakdown": {
                        str(key): float(value)
                        for key, value in (getattr(batch, "memory_breakdown", {}) or {}).items()
                    },
                    "persistent_tensors": dict(getattr(batch, "persistent_tensors", {}) or {}),
                    "transient_peak_tensors": dict(getattr(batch, "transient_peak_tensors", {}) or {}),
                }
                for idx, batch in enumerate(getattr(plan, "batches", []) or [])
            ],
            "resource_execution_plan": (
                resource_execution_plan.to_dict() if resource_execution_plan is not None else None
            ),
            "metadata_only": True,
            "notes": "Runner-level resource metadata; WorkItems drive alpha selection for alpha scans but do not change algorithm formulas.",
        }

    @staticmethod
    def _batch_alpha_values(batch: Any) -> List[float]:
        work_items = getattr(batch, "work_items", None) or []
        values = []
        for item in work_items:
            alpha = getattr(item, "alpha", None)
            if alpha is not None and float(alpha) not in values:
                values.append(float(alpha))
        if values:
            return values
        return [float(value) for value in getattr(batch, "alpha_values", []) or []]

    @staticmethod
    def _effective_runtime_seed_policy(config: ExperimentConfig) -> Dict[str, Any]:
        from matrix_factorization.core.contracts import get_effective_seed_policy_summary

        algorithm_params = getattr(config, "algorithm_params", None)
        requested_policy = getattr(algorithm_params, "seed_partition_policy", "legacy")
        return get_effective_seed_policy_summary(config.algorithm_key, requested_policy)

    def _run_algorithm_result(
        self,
        algorithm: 'AlgorithmBase',
        config: ExperimentConfig,
        data: ExperimentData,
        step_callback: Optional[Callable],
    ) -> AlgorithmResult:
        """Run algorithm through the formal AlgorithmResult interface."""
        is_spreading_family = 'spreading' in config.algorithm_key or 'tensor' in config.algorithm_key
        masks = None if is_spreading_family else data.masks
        if hasattr(algorithm, 'train_batch_result'):
            call_kwargs = self._filter_callable_kwargs(
                algorithm.train_batch_result,
                {
                    "algorithm_key": config.algorithm_key,
                    "W_teacher": data.W_teacher,
                    "X_teacher": data.X_teacher,
                    "Y_teacher": data.Y_teacher,
                    "masks": masks,
                    "alpha_values": data.alpha_values,
                    "seed": config.seeds.base_seed,
                    "progress_callback": None if is_spreading_family else step_callback,
                    "step_callback": step_callback if is_spreading_family else None,
                    "runtime_step_callback": self._build_runtime_step_callback(config),
                },
            )
            return algorithm.train_batch_result(**call_kwargs)

        W_students, X_students = self._run_algorithm(
            algorithm=algorithm,
            config=config,
            data=data,
            step_callback=step_callback,
        )
        return self._coerce_algorithm_result(
            config=config,
            algorithm=algorithm,
            W_students=W_students,
            X_students=X_students,
        )

    @staticmethod
    def _matrix_factors_from_result(
        algorithm_result: AlgorithmResult,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not algorithm_result.matrix_factors:
            return None, None
        return (
            algorithm_result.matrix_factors.get("W_students"),
            algorithm_result.matrix_factors.get("X_students"),
        )

    @staticmethod
    def _slice_alpha_factor(
        factor: Optional[torch.Tensor],
        alpha_idx: int,
    ) -> Optional[torch.Tensor]:
        if factor is None:
            return None
        if factor.dim() == 4:
            return factor[alpha_idx]
        return factor

    def _coerce_algorithm_result(
        self,
        config: ExperimentConfig,
        algorithm: 'AlgorithmBase',
        W_students: Optional[torch.Tensor],
        X_students: Optional[torch.Tensor],
    ) -> AlgorithmResult:
        """Wrap legacy algorithm outputs in the formal AlgorithmResult contract."""
        from ...modules.registry import get_algorithm_spec

        spec = get_algorithm_spec(config.algorithm_key)
        batch_metrics = getattr(algorithm, '_batch_metrics', None)
        metrics_by_alpha = {}
        if isinstance(batch_metrics, dict):
            metrics_by_alpha = {
                float(alpha): dict(metrics)
                for alpha, metrics in batch_metrics.items()
            }

        metadata = {
            "algorithm_key": config.algorithm_key,
            "result_contract": spec.result_contract,
            "result_source": "legacy_runner_algorithm_result_adapter",
        }
        if spec.result_contract == "legacy_tensor_metrics_only":
            artifacts = {}
            if any("overlap_matrix" in metrics for metrics in metrics_by_alpha.values()):
                artifacts["overlap_matrix"] = "per_alpha_metric_payload"
            return AlgorithmResult.from_metrics_only(
                metrics_by_alpha=metrics_by_alpha,
                artifacts=artifacts,
                metadata=dict(
                    metadata,
                    result_source="legacy_tensor_metrics_only_runner_adapter",
                    matrix_factors_available=False,
                ),
            )

        if W_students is None or X_students is None:
            raise ValueError(
                f"Algorithm '{config.algorithm_key}' uses result_contract={spec.result_contract} "
                "and must return real W_students/X_students. Use a metrics-only "
                "AlgorithmSpec result_contract for tensor/artifact-only outputs."
            )

        return AlgorithmResult(
            metrics_by_alpha=metrics_by_alpha,
            matrix_factors={"W_students": W_students, "X_students": X_students},
            metadata=dict(
                metadata,
                result_kind="matrix_factors",
                result_source="legacy_matrix_runner_adapter",
                matrix_factors_available=True,
            ),
        )

    def _run_algorithm_with_checkpoint_result(
        self,
        algorithm: 'AlgorithmBase',
        config: ExperimentConfig,
        data: ExperimentData,
        additional_steps: int,
        checkpoint: Optional[Checkpoint],
        step_callback: Optional[Callable],
    ) -> tuple[AlgorithmResult, Checkpoint]:
        """Run a step-scan segment and wrap its output in AlgorithmResult."""
        if checkpoint is not None:
            total_steps = checkpoint.step + additional_steps
        else:
            total_steps = additional_steps

        if hasattr(algorithm, 'train_batch_result'):
            is_spreading_family = 'spreading' in config.algorithm_key or 'tensor' in config.algorithm_key
            masks = None if is_spreading_family else data.masks
            call_kwargs = self._filter_callable_kwargs(
                algorithm.train_batch_result,
                {
                    "algorithm_key": config.algorithm_key,
                    "W_teacher": data.W_teacher,
                    "X_teacher": data.X_teacher,
                    "Y_teacher": data.Y_teacher,
                    "masks": masks,
                    "alpha_values": data.alpha_values,
                    "seed": config.seeds.base_seed,
                    "max_steps": total_steps,
                    "progress_callback": None if is_spreading_family else step_callback,
                    "step_callback": step_callback if is_spreading_family else None,
                    "runtime_step_callback": self._build_runtime_step_callback(config),
                },
            )
            algorithm_result = algorithm.train_batch_result(**call_kwargs)
            W_students, X_students = self._matrix_factors_from_result(algorithm_result)
            new_checkpoint = Checkpoint(
                step=total_steps,
                W_state=W_students.clone() if W_students is not None else None,
                X_state=X_students.clone() if X_students is not None else None,
            )
            return algorithm_result, new_checkpoint

        W_students, X_students, new_checkpoint = self._run_algorithm_with_checkpoint(
            algorithm=algorithm,
            config=config,
            data=data,
            additional_steps=additional_steps,
            checkpoint=checkpoint,
            step_callback=step_callback,
        )
        algorithm_result = self._coerce_algorithm_result(
            config=config,
            algorithm=algorithm,
            W_students=W_students,
            X_students=X_students,
        )
        return algorithm_result, new_checkpoint
    
    def _run_algorithm_with_checkpoint(
        self,
        algorithm: 'AlgorithmBase',
        config: ExperimentConfig,
        data: ExperimentData,
        additional_steps: int,
        checkpoint: Optional[Checkpoint],
        step_callback: Optional[Callable],
    ) -> tuple:
        """Run algorithm for step scanning."""
        if checkpoint is not None:
            total_steps = checkpoint.step + additional_steps
        else:
            total_steps = additional_steps
        
        # Note: We pass total_steps as max_steps, so callbacks will show e.g. 2000/3000
        # If we wanted purely incremental, we'd need to adjust the callback logic.
        
        # Include tensor algorithm in spreading family (uses step_callback)
        is_spreading_family = 'spreading' in config.algorithm_key or 'tensor' in config.algorithm_key
        if is_spreading_family:
            W_students, X_students = algorithm.train_batch_alphas(
                W_teacher=data.W_teacher,
                X_teacher=data.X_teacher,
                Y_teacher=data.Y_teacher,
                masks=None,
                alpha_values=data.alpha_values,
                seed=config.seeds.base_seed,
                max_steps=total_steps,
                step_callback=step_callback,
            )
        else:
            W_students, X_students = algorithm.train_batch_alphas(
                W_teacher=data.W_teacher,
                X_teacher=data.X_teacher,
                Y_teacher=data.Y_teacher,
                masks=data.masks,
                alpha_values=data.alpha_values,
                seed=config.seeds.base_seed,
                max_steps=total_steps,
                progress_callback=step_callback,
            )
        
        new_checkpoint = Checkpoint(
            step=total_steps,
            W_state=W_students.clone() if W_students is not None else None,
            X_state=X_students.clone() if X_students is not None else None,
        )
        
        return W_students, X_students, new_checkpoint
    
    def _compute_metrics(
        self,
        W_students: torch.Tensor,
        X_students: torch.Tensor,
        data: ExperimentData,
        algorithm: Optional['AlgorithmBase'] = None,
        algorithm_result: Optional[AlgorithmResult] = None,
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        if algorithm_result is not None:
            current_alpha = data.alpha_values[0]
            contract_metrics = self._metrics_for_alpha(algorithm_result.metrics_by_alpha, current_alpha)
            if contract_metrics is not None:
                return contract_metrics
            if not algorithm_result.matrix_factors:
                raise RuntimeError(
                    f"AlgorithmResult for algorithm metrics-only path did not provide "
                    f"metrics for alpha={current_alpha}. The active runner path must "
                    "not recover missing metrics from private _batch_metrics."
                )

        # === 优先使用算法内部计算的 Metrics (针对 Tensor 等复杂模式) ===
        if algorithm_result is None and algorithm is not None:
            # Check for batch pre-computed metrics
            if hasattr(algorithm, '_batch_metrics') and algorithm._batch_metrics:
                if type(algorithm) not in self._warned_private_metric_channel:
                    logger.warning(
                        "Algorithm %s exposed metrics through legacy _batch_metrics; "
                        "runner wrapped it into AlgorithmResult for the active path.",
                        type(algorithm).__name__,
                    )
                    self._warned_private_metric_channel.add(type(algorithm))
                # data.alpha_values usually contains 1 alpha in this context
                current_alpha = data.alpha_values[0]
                # Try exact match first
                if current_alpha in algorithm._batch_metrics:
                    return algorithm._batch_metrics[current_alpha]
                # Try fuzzy match (float precision)
                for a, m in algorithm._batch_metrics.items():
                    if abs(a - current_alpha) < 1e-6:
                        return m
            
            # Check for single-run last result (fallback)
            if hasattr(algorithm, '_last_result') and algorithm._last_result:
                res = algorithm._last_result
                if 'alpha' in res and abs(res['alpha'] - data.alpha_values[0]) < 1e-6:
                    # Convert single result keys to mean/std format
                    metrics = {}
                    for k, v in res.items():
                        if k not in ['alpha', 'sample', 'alpha_idx']:
                            metrics[f"{k}_mean"] = float(v)
                            metrics[f"{k}_std"] = 0.0
                    return metrics

        try:
            from ...modules.metrics.contract_compute import compute_metric_payload

            return compute_metric_payload(
                W_students=W_students,
                X_students=X_students,
                data=data,
            )
        except ImportError as e:
            raise RuntimeError(
                "Metric computation adapter failed to load or execute. "
                "Do not replace failed metric computation with zero-valued "
                "placeholder metrics."
            ) from e

    @staticmethod
    def _metrics_for_alpha(
        metrics_by_alpha: Dict[float, Dict[str, Any]],
        alpha: Any,
    ) -> Optional[Dict[str, Any]]:
        if not metrics_by_alpha:
            return None
        try:
            current_alpha = float(alpha)
        except (TypeError, ValueError):
            current_alpha = alpha
        if current_alpha in metrics_by_alpha:
            return metrics_by_alpha[current_alpha]
        for stored_alpha, metrics in metrics_by_alpha.items():
            try:
                if abs(float(stored_alpha) - float(current_alpha)) < 1e-6:
                    return metrics
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _validate_metric_payload(
        algorithm_key: str,
        metrics: Dict[str, Any],
        source: str = "unknown",
    ):
        """Fail if an algorithm emits undeclared metric keys on the main path."""
        from matrix_factorization.modules.metrics.spec_adapter import MetricSpecAdapter

        return MetricSpecAdapter.validate_payload(algorithm_key, metrics, source=source)

    @staticmethod
    def _metric_source(
        data: ExperimentData,
        algorithm_result: Optional[AlgorithmResult] = None,
    ) -> str:
        """Describe which legacy path produced this metric payload."""
        if algorithm_result is not None:
            current_alpha = data.alpha_values[0] if data.alpha_values else None
            if ExperimentRunner._metrics_for_alpha(algorithm_result.metrics_by_alpha, current_alpha) is not None:
                return "algorithm_result"
        if data.spreading_data is not None:
            return "runner_spreading_metrics"
        return "runner_matrix_metrics"

    def _get_algorithm(self, config: ExperimentConfig) -> 'AlgorithmBase':
        """Get or create algorithm instance."""
        key = config.algorithm_key
        cache_key = (key, self._algorithm_cache_signature(config))
        if cache_key not in self._algorithm_cache:
            from ...modules.registry import get_algorithm
            algo_config = self._build_algorithm_config(config)
            module_info = get_algorithm(key)
            algorithm = module_info.cls(algo_config, self.device)
            algorithm._contract_config_trace = self._algorithm_config_trace(config, algo_config)
            self._algorithm_cache[cache_key] = algorithm
        return self._algorithm_cache[cache_key]

    @staticmethod
    def _algorithm_cache_signature(config: ExperimentConfig) -> str:
        payload = config.to_dict() if hasattr(config, "to_dict") else str(config)
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    @staticmethod
    def _algorithm_config_trace(config: ExperimentConfig, algo_config: Any) -> Dict[str, Any]:
        spreading = getattr(config, "spreading", None)
        algorithm_params = getattr(config, "algorithm_params", None)
        training = getattr(config, "training", None)
        trace = {
            "algorithm_key": config.algorithm_key,
            "cache_signature": ExperimentRunner._algorithm_cache_signature(config),
            "matrix": {
                "N1": config.matrix.N1,
                "N2": config.matrix.N2,
                "M": config.matrix.M,
            },
            "training": {
                "samples_per_alpha": getattr(training, "samples_per_alpha", None),
                "max_steps": getattr(training, "max_steps", None),
                "max_epochs": getattr(training, "max_epochs", None),
                "seed": getattr(config.seeds, "base_seed", None),
            },
            "algorithm_params": {
                "damping": getattr(algorithm_params, "damping", None),
                "noise_var": getattr(algorithm_params, "noise_var", None),
                "learning_rate": getattr(algorithm_params, "learning_rate", None),
                "use_compile": getattr(algorithm_params, "use_compile", None),
                "compile_fallback_policy": getattr(algorithm_params, "compile_fallback_policy", None),
                "use_bf16": getattr(algorithm_params, "use_bf16", None),
                "dtype_fallback_policy": getattr(algorithm_params, "dtype_fallback_policy", None),
                "use_tf32": getattr(algorithm_params, "use_tf32", None),
                "seed_partition_policy": getattr(algorithm_params, "seed_partition_policy", None),
                "adaptive_damping": getattr(algorithm_params, "adaptive_damping", None),
                "init_mode": getattr(algorithm_params, "init_mode", None),
                "init_overlap": getattr(algorithm_params, "init_overlap", None),
            },
            "spreading": {
                "f_distribution": getattr(spreading, "f_distribution", None),
                "onsager_correction": getattr(spreading, "onsager_correction", None),
                "allow_intra_connection": getattr(spreading, "allow_intra_connection", None),
                "seed": getattr(spreading, "seed", None),
                "chunk_size": getattr(spreading, "chunk_size", None),
                "tensor_order": getattr(spreading, "tensor_order", None),
            },
            "mock_config_shape": {
                "has_algorithm_alias": hasattr(algo_config, "algorithm") if algo_config is not None else None,
                "has_algorithm_params_alias": hasattr(algo_config, "algorithm_params") if algo_config is not None else None,
                "has_spreading": hasattr(algo_config, "spreading") if algo_config is not None else None,
            },
            "metadata_only": True,
        }
        return trace

    @staticmethod
    def _build_runtime_extension_executor(contract: Dict[str, Any], algorithm_key: str):
        from matrix_factorization.modules.interventions import RuntimeExtensionExecutor

        return RuntimeExtensionExecutor.from_contract(contract, algorithm_key=algorithm_key)

    @staticmethod
    def _initial_runtime_state(
        config: ExperimentConfig,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
        Y_teacher: torch.Tensor,
    ) -> 'AlgorithmStateView':
        from matrix_factorization.core.contracts import AlgorithmStateView

        return AlgorithmStateView(
            teacher_factors={"W_teacher": W_teacher, "X_teacher": X_teacher},
            teacher={"W_teacher": W_teacher, "X_teacher": X_teacher, "Y_teacher": Y_teacher},
            metadata={"scan_dimension": config.scan.dimension},
        )

    @staticmethod
    def _runtime_context(config: ExperimentConfig, hook: str, metadata: Optional[Dict[str, Any]] = None):
        from matrix_factorization.modules.interventions import HookPoint, RuntimeHookContext

        return RuntimeHookContext(
            algorithm_key=config.algorithm_key,
            hook=HookPoint(hook),
            config=config,
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def _filter_callable_kwargs(callable_obj: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        signature = inspect.signature(callable_obj)
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            return dict(kwargs)
        allowed = set(signature.parameters)
        return {key: value for key, value in kwargs.items() if key in allowed}

    def _build_runtime_step_callback(self, config: ExperimentConfig):
        runtime_extensions = getattr(self, "_runtime_extensions", None)
        if runtime_extensions is None:
            return None

        has_after_step_hook = any(
            getattr(item, "spec", item).trigger == "after_step"
            for item in [
                *getattr(runtime_extensions, "interventions", []),
                *getattr(runtime_extensions, "probes", []),
            ]
        )
        if not has_after_step_hook:
            return None

        def runtime_step_callback(state):
            from matrix_factorization.modules.interventions import HookPoint, RuntimeHookContext

            runtime_extensions.dispatch(
                "after_step",
                state,
                RuntimeHookContext(
                    algorithm_key=config.algorithm_key,
                    hook=HookPoint.AFTER_STEP,
                    config=config,
                    alpha=getattr(state, "alpha", None),
                    step_index=getattr(state, "step_index", None),
                    metadata={"runtime_hook_source": "algorithm_step"},
                ),
            )

        return runtime_step_callback

    @staticmethod
    def _record_algorithm_result_summary(
        *,
        result: ExperimentResult,
        batch_idx: int,
        alpha_values: List[Any],
        algorithm_result: Optional[AlgorithmResult],
    ) -> None:
        if algorithm_result is None:
            return
        metadata = dict(getattr(algorithm_result, "metadata", {}) or {})
        summary = {
            "batch_idx": batch_idx,
            "alpha_values": list(alpha_values),
            "available_outputs": algorithm_result.available_outputs(),
            "metrics_by_alpha_count": len(getattr(algorithm_result, "metrics_by_alpha", {}) or {}),
            "metrics_by_alpha_keys": sorted({
                key
                for metrics in (getattr(algorithm_result, "metrics_by_alpha", {}) or {}).values()
                for key in (metrics or {})
            }),
            "algorithm_key": metadata.get("algorithm_key"),
            "result_contract": metadata.get("result_contract"),
            "result_kind": metadata.get("result_kind"),
            "result_source": metadata.get("result_source"),
            "matrix_factors_available": metadata.get("matrix_factors_available"),
            "metadata_keys": sorted(metadata),
        }
        for key in [
            "execution_metadata",
            "tensor_execution",
            "internal_alpha_batch_plan",
        ]:
            value = metadata.get(key)
            if isinstance(value, dict):
                summary[key] = dict(value)
        result.metadata.contract.setdefault("algorithm_result_batches", []).append(summary)

    def _dispatch_after_batch_runtime_extensions(
        self,
        *,
        config: ExperimentConfig,
        batch_idx: int,
        alpha_values: List[Any],
        algorithm_result: Optional[AlgorithmResult],
        result: ExperimentResult,
    ) -> None:
        runtime_extensions = getattr(self, "_runtime_extensions", None)
        if runtime_extensions is None:
            return
        from matrix_factorization.core.contracts import AlgorithmStateView

        metric_keys = sorted({
            key
            for alpha in alpha_values
            if alpha in result.results
            for key in (result.results[alpha].metrics or {})
        })
        state_metadata = {
            "batch_idx": batch_idx,
            "alpha_values": list(alpha_values),
            "metric_keys": metric_keys,
            "algorithm_result_outputs": (
                algorithm_result.available_outputs()
                if algorithm_result is not None else []
            ),
        }
        if algorithm_result is not None:
            algorithm_metadata = dict(getattr(algorithm_result, "metadata", {}) or {})
            state_metadata["algorithm_result_metadata_keys"] = sorted(algorithm_metadata)
            state_metadata["result_contract"] = algorithm_metadata.get("result_contract")
            state_metadata["result_kind"] = algorithm_metadata.get("result_kind")
            state_metadata["result_source"] = algorithm_metadata.get("result_source")
            state_metadata["batching_source"] = algorithm_metadata.get("batching_source")
            state_metadata["metrics_by_alpha_count"] = len(getattr(algorithm_result, "metrics_by_alpha", {}) or {})
            state_metadata["metrics_by_alpha_keys"] = sorted({
                key
                for metrics in (getattr(algorithm_result, "metrics_by_alpha", {}) or {}).values()
                for key in (metrics or {})
            })
            if isinstance(algorithm_metadata.get("execution_metadata"), dict):
                state_metadata["execution_metadata_keys"] = sorted(algorithm_metadata["execution_metadata"])
            if isinstance(algorithm_metadata.get("tensor_execution"), dict):
                state_metadata["tensor_execution_keys"] = sorted(algorithm_metadata["tensor_execution"])
            if isinstance(algorithm_metadata.get("internal_alpha_batch_plan"), dict):
                internal_plan = algorithm_metadata["internal_alpha_batch_plan"]
                alpha_batches = internal_plan.get("alpha_batches")
                state_metadata["internal_alpha_batch_plan_keys"] = sorted(internal_plan)
                state_metadata["internal_alpha_batch_plan_summary"] = {
                    "planner": internal_plan.get("planner"),
                    "num_batches": internal_plan.get("num_batches"),
                    "alpha_batch_count": len(alpha_batches) if isinstance(alpha_batches, list) else None,
                    "probe_enabled": internal_plan.get("probe_enabled"),
                    "seed_partition_sensitive": internal_plan.get("seed_partition_sensitive"),
                    "max_alphas_per_batch": internal_plan.get("max_alphas_per_batch"),
                    "metadata_only": bool(internal_plan.get("metadata_only", True)),
                }

        state = AlgorithmStateView(metadata=state_metadata)
        runtime_extensions.dispatch(
            "after_batch",
            state,
            self._runtime_context(
                config,
                "after_batch",
                metadata={"batch_idx": batch_idx, "alpha_values": list(alpha_values)},
            ),
        )
    
    def _build_algorithm_config(self, config: ExperimentConfig) -> Any:
        # Create a mock config object that algorithms expect
        from dataclasses import dataclass as dc
        
        @dc
        class MatrixConfig:
            N1: int = config.matrix.N1
            N2: int = config.matrix.N2
            M: int = config.matrix.M
        
        @dc
        class AlgoConfig:
            damping: float = config.algorithm_params.damping
            noise_var: float = config.algorithm_params.noise_var
            learning_rate: float = config.algorithm_params.learning_rate
            use_compile: bool = config.algorithm_params.use_compile
            compile_fallback_policy: str = config.algorithm_params.compile_fallback_policy
            use_bf16: bool = config.algorithm_params.use_bf16
            dtype_fallback_policy: str = config.algorithm_params.dtype_fallback_policy
            use_tf32: bool = config.algorithm_params.use_tf32
            seed_partition_policy: str = config.algorithm_params.seed_partition_policy
            adaptive_damping: bool = config.algorithm_params.adaptive_damping
        
        @dc
        class TrainConfig:
            max_steps: int = config.training.max_steps
            max_epochs: int = config.training.max_epochs
            samples_per_alpha: int = config.training.samples_per_alpha
            seed: int = config.seeds.base_seed
        
        @dc
        class SpreadConfig:
            f_distribution: str = config.spreading.f_distribution if config.spreading else "rademacher"
            onsager_correction: bool = config.spreading.onsager_correction if config.spreading else False
            allow_intra_connection: bool = config.spreading.allow_intra_connection if config.spreading else False
            seed: int = getattr(config.spreading, "seed", config.seeds.spreading_seed) if config.spreading else config.seeds.spreading_seed
            chunk_size: int = getattr(config.spreading, "chunk_size", 131072) if config.spreading else 131072
            tensor_order: int = getattr(config.spreading, 'tensor_order', 2) if config.spreading else 2
        
        @dc
        class MockConfig:
            matrix = MatrixConfig()
            algorithm = AlgoConfig()
            algorithm_params = config.algorithm_params
            training = TrainConfig()
            spreading = SpreadConfig()
        
        return MockConfig()

    def _emit(self, observer: Optional[Callable[[ProgressEvent], None]], type: ProgressEventType, payload: Dict[str, Any]):
        """Emit a progress event safely."""
        if observer:
            try:
                observer(ProgressEvent(type=type, payload=payload))
            except Exception as e:
                # Don't let UI errors crash the experiment
                logger.error(f"Error in progress observer: {e}")

    def _create_legacy_observer(self, step_cb, point_cb):
        """Create an observer that forwards to legacy callbacks."""
        def observer(event: ProgressEvent):
            if event.type == ProgressEventType.STEP_UPDATE and step_cb:
                step = event.payload.get('step', 0)
                total = event.payload.get('total', 1)
                step_cb(step, total)
            elif event.type == ProgressEventType.POINT_COMPLETE and point_cb:
                idx = event.payload.get('point_idx', 0)
                total = event.payload.get('total_points', 1)
                val = event.payload.get('value')
                point_cb(idx, total, val)
        return observer
