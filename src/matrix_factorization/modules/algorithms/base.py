"""
Base class for training algorithms.

Provides two interfaces:
1. Legacy: train_single_alpha(), train_batch_alphas() - for backward compatibility
2. New: run_single() - for external framework control
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional, Callable, TYPE_CHECKING
import torch

from ...core.config import Config

if TYPE_CHECKING:
    from ...core.experiment.data_factory import ExperimentData
    from ...core.experiment.result import Checkpoint
    from ...core.contracts import AlgorithmResult


class AlgorithmBase(ABC):
    """
    Base class for training algorithms.
    
    Two usage patterns:
    
    1. Legacy (for backward compatibility):
        result = algorithm.train_batch_alphas(W, X, Y, masks, alphas, seed)
    
    2. New (recommended, with external framework):
        data = data_factory.create(config)
        result = algorithm.run_single(data, params)
    """

    def __init__(self, config: Config, device: torch.device):
        """
        Initialize algorithm.

        Args:
            config: Experiment configuration
            device: Torch device
        """
        self.config = config
        self.device = device

    # =========================================================================
    # New Interface (for external framework control)
    # =========================================================================

    def run_single(
        self,
        data: 'ExperimentData',
        max_steps: Optional[int] = None,
        checkpoint: Optional['Checkpoint'] = None,
        step_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional['Checkpoint']]:
        """
        Execute a single training run.
        
        This is the new unified interface for external framework control.
        The algorithm receives prepared data and just computes.
        
        Args:
            data: ExperimentData containing all inputs
            max_steps: Override max steps (for step scanning)
            checkpoint: Optional checkpoint to resume from
            step_callback: Optional progress callback (step, max_steps)
            
        Returns:
            (W_students, X_students, checkpoint)
            - W_students: (A, S, N1, M) or (S, N1, M)
            - X_students: (A, S, M, N2) or (S, M, N2)
            - checkpoint: State for resumption (optional)
        """
        # Default implementation: delegate to legacy interface
        # Subclasses can override for optimized implementation
        
        if data.spreading_data is not None:
            # Spreading algorithm
            W, X = self.train_batch_alphas(
                W_teacher=data.W_teacher,
                X_teacher=data.X_teacher,
                Y_teacher=data.Y_teacher,
                masks=None,
                alpha_values=data.alpha_values,
                seed=42,
                step_callback=step_callback,
            )
        else:
            # Dense algorithm
            W, X = self.train_batch_alphas(
                W_teacher=data.W_teacher,
                X_teacher=data.X_teacher,
                Y_teacher=data.Y_teacher,
                masks=data.masks,
                alpha_values=data.alpha_values,
                seed=42,
                progress_callback=step_callback,
            )
        
        # Create checkpoint for potential resumption
        from ...core.experiment.result import Checkpoint
        new_checkpoint = Checkpoint(
            step=max_steps or self._get_max_steps(),
            W_state=W,
            X_state=X,
        )
        
        return W, X, new_checkpoint

    def _get_max_steps(self) -> int:
        """Get max_steps from config."""
        if hasattr(self.config.training, 'max_steps'):
            return self.config.training.max_steps
        elif hasattr(self.config.training, 'max_epochs'):
            return self.config.training.max_epochs
        return 1000

    # =========================================================================
    # Legacy Interface (for backward compatibility)
    # =========================================================================

    @abstractmethod
    def train_single_alpha(
        self,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
        Y_teacher: torch.Tensor,
        mask: torch.Tensor,
        alpha: float,
        seed: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Train for a single alpha value.

        Args:
            W_teacher: Teacher W matrix (N1, M)
            X_teacher: Teacher X matrix (M, N2)
            Y_teacher: Teacher Y = W @ X (N1, N2)
            mask: Observation mask (N1, N2)
            alpha: Sparsity parameter
            seed: Random seed for student initialization

        Returns:
            W_student: Trained W matrices (S, N1, M)
            X_student: Trained X matrices (S, M, N2)
        """
        pass

    def train_batch_alphas(
        self,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
        Y_teacher: torch.Tensor,
        masks: torch.Tensor,
        alpha_values: list[float],
        seed: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        step_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Train for multiple alpha values in parallel (if supported).

        Default implementation runs single_alpha in sequence.
        Subclasses can override for parallel processing.

        Args:
            masks: Shape (num_alphas, N1, N2)
            alpha_values: List of alpha values
            progress_callback: Legacy callback
            step_callback: New callback (step, max_steps)

        Returns:
            W_student: (num_alphas, S, N1, M)
            X_student: (num_alphas, S, M, N2)
        """
        results_W = []
        results_X = []

        callback = step_callback or progress_callback

        for i, alpha in enumerate(alpha_values):
            W, X = self.train_single_alpha(
                W_teacher, X_teacher, Y_teacher,
                masks[i] if masks is not None else None,
                alpha, seed + i * 1000
            )
            results_W.append(W)
            results_X.append(X)

        return torch.stack(results_W), torch.stack(results_X)

    def train_batch_result(
        self,
        *,
        algorithm_key: str,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
        Y_teacher: torch.Tensor,
        masks: Optional[torch.Tensor],
        alpha_values: list[float],
        seed: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        step_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs: Any,
    ) -> 'AlgorithmResult':
        """Return the formal AlgorithmResult for a batch of alpha values.

        The default implementation wraps the legacy train_batch_alphas return
        without changing algorithm behavior. Subclasses can override this when
        they natively produce tensor-only metrics or richer artifacts.
        """
        call_kwargs = self._filter_train_batch_kwargs({
            "W_teacher": W_teacher,
            "X_teacher": X_teacher,
            "Y_teacher": Y_teacher,
            "masks": masks,
            "alpha_values": alpha_values,
            "seed": seed,
            "progress_callback": progress_callback,
            "step_callback": step_callback,
            **kwargs,
        })
        W_students, X_students = self.train_batch_alphas(**call_kwargs)
        return self.coerce_legacy_batch_result(
            algorithm_key=algorithm_key,
            W_students=W_students,
            X_students=X_students,
        )

    def coerce_legacy_batch_result(
        self,
        *,
        algorithm_key: str,
        W_students: Optional[torch.Tensor],
        X_students: Optional[torch.Tensor],
    ) -> 'AlgorithmResult':
        """Wrap legacy W/X and private batch metrics in AlgorithmResult."""
        from ...core.contracts import AlgorithmResult
        from ..registry import get_algorithm_spec

        spec = get_algorithm_spec(algorithm_key)
        batch_metrics = getattr(self, '_batch_metrics', None)
        metrics_by_alpha: Dict[float, Dict[str, Any]] = {}
        if isinstance(batch_metrics, dict):
            metrics_by_alpha = {
                float(alpha): dict(metrics)
                for alpha, metrics in batch_metrics.items()
            }

        metadata = {
            "algorithm_key": algorithm_key,
            "result_contract": spec.result_contract,
        }
        execution_metadata = getattr(self, "_contract_execution_metadata", None)
        if isinstance(execution_metadata, dict):
            metadata["execution_metadata"] = dict(execution_metadata)
        if spec.result_contract == "legacy_tensor_metrics_only":
            artifacts = {}
            if any("overlap_matrix" in metrics for metrics in metrics_by_alpha.values()):
                artifacts["overlap_matrix"] = "per_alpha_metric_payload"
            return AlgorithmResult.from_metrics_only(
                metrics_by_alpha=metrics_by_alpha,
                artifacts=artifacts,
                metadata=metadata,
            )

        if W_students is None or X_students is None:
            raise ValueError(
                f"Algorithm '{algorithm_key}' uses result_contract={spec.result_contract} "
                "and must return real W_students/X_students. Use a metrics-only "
                "AlgorithmSpec result_contract for tensor/artifact-only outputs."
            )

        return AlgorithmResult(
            metrics_by_alpha=metrics_by_alpha,
            matrix_factors={"W_students": W_students, "X_students": X_students},
            metadata=dict(metadata, result_kind="matrix_factors"),
        )

    def _filter_train_batch_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Only pass keyword arguments accepted by the concrete legacy method."""
        import inspect

        signature = inspect.signature(self.train_batch_alphas)
        parameters = signature.parameters
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if accepts_var_kwargs:
            return kwargs
        return {
            key: value
            for key, value in kwargs.items()
            if key in parameters
        }

    def supports_batch_training(self) -> bool:
        """Check if algorithm supports efficient batch training."""
        return False

    def estimate_memory_per_alpha(self, N1: int, N2: int, M: int, S: int) -> float:
        """
        Estimate GPU memory needed per alpha value (in GB).

        Subclasses should override with accurate estimates.
        """
        # Default conservative estimate
        student_params = 2 * (S * N1 * M + S * M * N2)
        intermediate = 10 * S * N1 * N2
        total_elements = student_params + intermediate
        return total_elements * 4 / (1024**3)  # 4 bytes per float32
