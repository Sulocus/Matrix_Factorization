"""
Experiment Runner - The Core Execution Engine.

Responsibilities:
1. Execute experiments according to configuration
2. Handle scan modes (alpha, steps, etc.)
3. Manage memory and batching
4. Collect and save results

Supports nested scans:
- Outer loop: N, M (different matrix sizes)
- Inner loop: alpha, steps (within each size)
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable, TYPE_CHECKING
from pathlib import Path
import time
import logging
import torch

from .config import ExperimentConfig, ScanConfig, ScanDimension, MatrixParams
from .result import ExperimentResult, SingleRunResult, ExperimentMetadata, Checkpoint
from .data_factory import DataFactory, ExperimentData

if TYPE_CHECKING:
    from ...modules.algorithms.base import AlgorithmBase

logger = logging.getLogger(__name__)


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
    5. Progress callbacks
    
    Usage:
        runner = ExperimentRunner()
        result = runner.run(config)
        result.save('results/my_experiment/')
        
        # Nested scan (multiple matrix sizes)
        results = runner.run_scaling_sweep(
            base_config=config,
            matrix_sizes=[(200, 200, 50), (400, 400, 100), (600, 600, 150)],
        )
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
            verbose: Print progress information
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
    
    def run(
        self,
        config: ExperimentConfig,
        step_callback: Optional[Callable[[int, int], None]] = None,
        point_callback: Optional[Callable[[int, int, Any], None]] = None,
    ) -> ExperimentResult:
        """
        Run a single experiment.
        
        Args:
            config: Experiment configuration
            step_callback: Called after each training step (step, max_steps)
            point_callback: Called after each scan point (idx, total, value)
            
        Returns:
            ExperimentResult with all results
        """
        if self.verbose:
            print(f"Starting experiment: {config.experiment_name}")
            print(f"  Algorithm: {config.algorithm_key}")
            print(f"  Matrix: {config.N1}x{config.N2}, M={config.M}")
            print(f"  Scan: {config.scan.dimension} with {config.scan.num_points} points")
        
        # Create Teacher data (shared across all scan points)
        W_teacher, X_teacher, Y_teacher = self.data_factory.create_teacher(
            N1=config.N1,
            N2=config.N2,
            M=config.M,
            teacher_key=config.teacher_key,
            seed=config.seeds.teacher_seed,
        )
        
        # Create result container with raw data
        result = ExperimentResult(
            experiment_id=config.experiment_name,
            config=config,
            scan_dimension=config.scan.dimension,
            scan_values=config.scan.values,
            metadata=ExperimentMetadata.create_now(),
            # Raw data for post-hoc analysis
            W_teacher=W_teacher,
            X_teacher=X_teacher,
            Y_teacher=Y_teacher,
        )
        
        # Get algorithm
        algorithm = self._get_algorithm(config)
        
        # Route to appropriate scan handler
        if config.scan.is_steps_scan:
            self._run_steps_scan(config, algorithm, result, step_callback, point_callback)
        else:
            self._run_standard_scan(config, algorithm, result, step_callback, point_callback)
        
        return result
    
    def run_scaling_sweep(
        self,
        base_config: ExperimentConfig,
        matrix_sizes: List[tuple],  # [(N1, N2, M), ...]
        output_dir: Optional[Path] = None,
        step_callback: Optional[Callable] = None,
    ) -> Dict[tuple, ExperimentResult]:
        """
        Run nested scan: outer loop over matrix sizes, inner loop over alpha/steps.
        
        This is for scaling analysis (how does behavior change with N, M).
        
        Args:
            base_config: Base configuration (scan config will be reused)
            matrix_sizes: List of (N1, N2, M) tuples to sweep
            output_dir: Optional directory to save results
            step_callback: Progress callback
            
        Returns:
            Dict mapping (N1, N2, M) to ExperimentResult
        """
        results = {}
        
        for i, (N1, N2, M) in enumerate(matrix_sizes):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Scaling sweep {i+1}/{len(matrix_sizes)}: N={N1}, M={M}")
                print(f"{'='*60}")
            
            # Create config for this size
            config = ExperimentConfig(
                matrix=MatrixParams(N1=N1, N2=N2, M=M),
                training=base_config.training,
                algorithm_key=base_config.algorithm_key,
                scan=base_config.scan,
                seeds=base_config.seeds,
                algorithm_params=base_config.algorithm_params,
                spreading=base_config.spreading,
                experiment_name=f"{base_config.experiment_name}_N{N1}_M{M}",
                teacher_key=base_config.teacher_key,
                notes=f"Scaling sweep: N={N1}, M={M}",
            )
            
            # Run experiment
            result = self.run(config, step_callback=step_callback)
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
        step_callback: Optional[Callable],
        point_callback: Optional[Callable],
    ):
        """
        Run standard scan (alpha, samples, seed).
        
        Each scan point is independent - data is recreated for each.
        """
        scan_values = config.scan.values
        total_points = len(scan_values)
        
        for idx, scan_value in enumerate(scan_values):
            start_time = time.time()
            
            if self.verbose:
                print(f"  Point {idx+1}/{total_points}: {config.scan.dimension}={scan_value}")
            
            # Create data for this scan point
            if config.scan.is_alpha_scan:
                alpha_values = [scan_value]
            else:
                alpha_values = config.alpha_values
            
            data = self.data_factory.create(config, alpha_values=alpha_values)
            
            # Run algorithm
            W_students, X_students = self._run_algorithm(
                algorithm=algorithm,
                config=config,
                data=data,
                step_callback=step_callback,
            )
            
            # Compute metrics
            metrics = self._compute_metrics(
                W_students=W_students,
                X_students=X_students,
                data=data,
            )
            
            # Extract mask for saving (for Q_Y_unobserved computation)
            mask_to_save = None
            observation_indices = None
            if data.masks is not None:
                mask_to_save = data.masks[0] if data.masks.dim() == 3 else data.masks
            if data.spreading_data is not None:
                # Extract observation indices from SuperGraph
                sg = data.spreading_data.supergraph
                observation_indices = {
                    'i_idx': sg.i_idx.cpu() if hasattr(sg, 'i_idx') else None,
                    'j_idx': sg.j_idx.cpu() if hasattr(sg, 'j_idx') else None,
                    'edge_counts': sg.edge_counts.cpu() if hasattr(sg, 'edge_counts') else None,
                }
            
            # Store result with raw data
            single_result = SingleRunResult(
                scan_value=scan_value,
                metrics=metrics,
                W_students=W_students if config.scan.num_points <= 10 else None,
                X_students=X_students if config.scan.num_points <= 10 else None,
                mask=mask_to_save,
                observation_indices=observation_indices,
                duration_seconds=time.time() - start_time,
            )
            result.add_result(scan_value, single_result)
            
            if point_callback:
                point_callback(idx + 1, total_points, scan_value)
            
            # Clear cache between points
            torch.cuda.empty_cache()
    
    def _run_steps_scan(
        self,
        config: ExperimentConfig,
        algorithm: 'AlgorithmBase',
        result: ExperimentResult,
        step_callback: Optional[Callable],
        point_callback: Optional[Callable],
    ):
        """
        Run steps scan (convergence curve).
        
        Uses checkpointing to efficiently scan multiple step counts.
        Data is created once and reused.
        """
        step_values = sorted(config.scan.values)  # Must be sorted ascending
        total_points = len(step_values)
        
        # Create data once (reused across all step counts)
        # For steps scan, use a single alpha (typically specified elsewhere)
        default_alpha = config.algorithm_params.__dict__.get('default_alpha', 1.0)
        data = self.data_factory.create(config, alpha_values=[default_alpha])
        
        # Initialize student once
        checkpoint = None
        prev_steps = 0
        
        for idx, max_steps in enumerate(step_values):
            start_time = time.time()
            
            if self.verbose:
                print(f"  Point {idx+1}/{total_points}: steps={max_steps}")
            
            # Calculate additional steps needed
            additional_steps = max_steps - prev_steps
            
            # Run algorithm (continue from checkpoint)
            W_students, X_students, new_checkpoint = self._run_algorithm_with_checkpoint(
                algorithm=algorithm,
                config=config,
                data=data,
                additional_steps=additional_steps,
                checkpoint=checkpoint,
                step_callback=step_callback,
            )
            
            # Update checkpoint for next iteration
            checkpoint = new_checkpoint
            prev_steps = max_steps
            
            # Compute metrics
            metrics = self._compute_metrics(
                W_students=W_students,
                X_students=X_students,
                data=data,
            )
            
            # Store result
            single_result = SingleRunResult(
                scan_value=max_steps,
                metrics=metrics,
                duration_seconds=time.time() - start_time,
            )
            result.add_result(max_steps, single_result)
            
            if point_callback:
                point_callback(idx + 1, total_points, max_steps)
    
    def _run_algorithm(
        self,
        algorithm: 'AlgorithmBase',
        config: ExperimentConfig,
        data: ExperimentData,
        step_callback: Optional[Callable],
    ) -> tuple:
        """Run algorithm and return (W_students, X_students)."""
        # For now, use existing train_batch_alphas interface
        # TODO: Migrate to new run_single() interface
        
        if config.is_spreading_algorithm:
            # Spreading algorithm
            W_students, X_students = algorithm.train_batch_alphas(
                W_teacher=data.W_teacher,
                X_teacher=data.X_teacher,
                Y_teacher=data.Y_teacher,
                masks=None,  # Not used for spreading
                alpha_values=data.alpha_values,
                seed=config.seeds.base_seed,
                step_callback=step_callback,
            )
        else:
            # Dense algorithm (AGD, BiGAMP)
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
    
    def _run_algorithm_with_checkpoint(
        self,
        algorithm: 'AlgorithmBase',
        config: ExperimentConfig,
        data: ExperimentData,
        additional_steps: int,
        checkpoint: Optional[Checkpoint],
        step_callback: Optional[Callable],
    ) -> tuple:
        """
        Run algorithm with checkpoint support for step scanning.
        
        TODO: Implement proper checkpoint support in algorithms.
        For now, this is a placeholder that re-runs from scratch.
        """
        # Temporarily run full training (proper checkpoint support TBD)
        W_students, X_students = self._run_algorithm(
            algorithm=algorithm,
            config=config,
            data=data,
            step_callback=step_callback,
        )
        
        # Create checkpoint for next iteration
        new_checkpoint = Checkpoint(
            step=additional_steps,
            W_state=W_students,
            X_state=X_students,
        )
        
        return W_students, X_students, new_checkpoint
    
    def _compute_metrics(
        self,
        W_students: torch.Tensor,
        X_students: torch.Tensor,
        data: ExperimentData,
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        # Import metrics computation
        try:
            from ...modules.metrics.overlap import (
                gram_overlap_normalized,
                compute_qy,
            )
            
            # Handle different tensor shapes
            # W_students could be (A, S, N1, M) or (S, N1, M)
            if W_students.dim() == 4:
                # Average over alpha dimension for metrics
                W_for_metrics = W_students.mean(dim=0)
                X_for_metrics = X_students.mean(dim=0)
            else:
                W_for_metrics = W_students
                X_for_metrics = X_students
            
            # Compute Q_W and Q_X for each sample
            S = W_for_metrics.shape[0]
            Q_W_list = []
            Q_X_list = []
            Q_Y_list = []
            
            Y_teacher = data.W_teacher @ data.X_teacher
            
            for s in range(S):
                # Q_W' (Gram overlap with baseline correction)
                Q_W = gram_overlap_normalized(W_for_metrics[s], data.W_teacher, use_left=True)
                Q_W_list.append(Q_W)
                
                # Q_X' (Gram overlap with baseline correction)
                Q_X = gram_overlap_normalized(X_for_metrics[s], data.X_teacher, use_left=False)
                Q_X_list.append(Q_X)
                
                # Q_Y (cosine similarity)
                Y_student = W_for_metrics[s] @ X_for_metrics[s]
                Q_Y = compute_qy(Y_student, Y_teacher)
                Q_Y_list.append(Q_Y)
            
            import numpy as np
            return {
                'Q_W_mean': float(np.mean(Q_W_list)),
                'Q_W_std': float(np.std(Q_W_list, ddof=1)) if len(Q_W_list) > 1 else 0.0,
                'Q_X_mean': float(np.mean(Q_X_list)),
                'Q_X_std': float(np.std(Q_X_list, ddof=1)) if len(Q_X_list) > 1 else 0.0,
                'Q_Y_mean': float(np.mean(Q_Y_list)),
                'Q_Y_std': float(np.std(Q_Y_list, ddof=1)) if len(Q_Y_list) > 1 else 0.0,
            }
        except ImportError as e:
            # Fallback if metrics module not available
            print(f"Warning: metrics import failed: {e}")
            return {'Q_W_mean': 0.0, 'Q_X_mean': 0.0, 'Q_Y_mean': 0.0}
    
    def _get_algorithm(self, config: ExperimentConfig) -> 'AlgorithmBase':
        """Get or create algorithm instance."""
        key = config.algorithm_key
        
        if key not in self._algorithm_cache:
            # Create algorithm using registry
            from ...modules.registry import get_algorithm
            
            # Build config-compatible object for algorithm
            algo_config = self._build_algorithm_config(config)
            
            # get_algorithm returns ModuleInfo, need to instantiate
            module_info = get_algorithm(key)
            algorithm = module_info.cls(algo_config, self.device)
            self._algorithm_cache[key] = algorithm
        
        return self._algorithm_cache[key]
    
    def _build_algorithm_config(self, config: ExperimentConfig) -> Any:
        """Build config object for algorithm initialization."""
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
        
        @dc
        class TrainConfig:
            max_steps: int = config.training.max_steps
            max_epochs: int = config.training.max_epochs
            samples_per_alpha: int = config.training.samples_per_alpha
            seed: int = config.seeds.base_seed
        
        @dc
        class SpreadConfig:
            f_distribution: str = config.spreading.f_distribution if config.spreading else "rademacher"
            seed: int = config.seeds.spreading_seed
        
        @dc
        class MockConfig:
            matrix = MatrixConfig()
            algorithm = AlgoConfig()
            training = TrainConfig()
            spreading = SpreadConfig()
        
        return MockConfig()
