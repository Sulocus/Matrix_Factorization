"""
Base class for training algorithms.

Provides two interfaces:
1. Legacy: train_single_alpha(), train_batch_alphas() - for backward compatibility
2. New: run_single() - for external framework control
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Callable, Any, TYPE_CHECKING
import torch

from ...core.config import Config
from ..graphs.base import GraphBase
from ..teachers.base import TeacherBase

if TYPE_CHECKING:
    from ...core.experiment.data_factory import ExperimentData
    from ...core.experiment.result import Checkpoint


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

