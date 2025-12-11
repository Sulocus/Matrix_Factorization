"""
AGD (Alternating Gradient Descent) algorithm for matrix factorization.

Migrated from Wang/agd/train_parallel.py
"""

from typing import Tuple, Optional, Callable
import torch

from ..registry import register_algorithm
from .base import AlgorithmBase
from ...core.config import Config


@register_algorithm(
    key="agd",
    name="Alternating Gradient Descent",
    description="Gradient descent, stable but slower (~20k epochs)",
    default_params={
        'learning_rate': 1e-2,
        'weight_decay': 0.0,
        'use_early_stop': False,
        'target_loss_threshold': 1e-8,
        'relative_change_threshold': 1e-7,
        'early_stop_check_interval': 100,
        'early_stop_patience': 5,
    },
)
class AGDAlgorithm(AlgorithmBase):
    """
    Alternating Gradient Descent for masked matrix factorization.

    Updates W and X alternately using gradient descent on masked MSE loss.
    Slower than BiG-AMP (~20k epochs vs ~1k steps) but more stable.
    """

    def __init__(self, config: Config, device: torch.device):
        super().__init__(config, device)
        self.lr = config.algorithm.learning_rate
        self.max_epochs = config.training.max_epochs
        self.S = config.training.samples_per_alpha

        # Early stop settings
        self.use_early_stop = getattr(config.algorithm, 'use_early_stop', False)
        self.target_loss = getattr(config.algorithm, 'target_loss_threshold', 1e-8)
        self.relative_threshold = getattr(config.algorithm, 'relative_change_threshold', 1e-7)
        self.check_interval = getattr(config.algorithm, 'early_stop_check_interval', 100)
        self.patience = getattr(config.algorithm, 'early_stop_patience', 5)

        # BF16 settings
        self.use_bf16 = device.type == 'cuda'
        self.compute_dtype = torch.bfloat16 if self.use_bf16 else torch.float32

    def train_single_alpha(
        self,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
        Y_teacher: torch.Tensor,
        mask: torch.Tensor,
        alpha: float,
        seed: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train AGD for single alpha value."""
        N1, M = W_teacher.shape
        N2 = X_teacher.shape[1]
        S = self.S
        device = self.device
        lr = self.lr

        alpha_scale = 1.0 / (M ** 0.5)
        scale = 1.0 / (M ** 0.5)

        # Ensure mask has batch dimension
        A = mask.unsqueeze(0) if mask.dim() == 2 else mask  # (1, N1, N2)
        Y_teacher_b = Y_teacher.unsqueeze(0)  # (1, N1, N2)

        # Initialize student
        torch.manual_seed(seed)
        W = torch.randn((S, N1, M), device=device, dtype=torch.float32) * scale
        X = torch.randn((S, M, N2), device=device, dtype=torch.float32) * scale

        # Early stop tracking
        from collections import deque
        loss_history = deque(maxlen=self.patience) if self.use_early_stop else None

        # Training loop
        for step in range(self.max_epochs):
            # Use autocast for BF16 acceleration
            with torch.autocast(device_type=device.type, dtype=self.compute_dtype,
                                enabled=self.use_bf16):
                # W update
                Y_student = alpha_scale * torch.matmul(W, X)
                Mres = (Y_teacher_b - Y_student) * A
                grad_W = -2.0 * alpha_scale * torch.matmul(Mres, X.transpose(1, 2))

            W = W - lr * grad_W.float()

            with torch.autocast(device_type=device.type, dtype=self.compute_dtype,
                                enabled=self.use_bf16):
                # X update with updated W
                Y_student2 = alpha_scale * torch.matmul(W, X)
                Mres2 = (Y_teacher_b - Y_student2) * A
                grad_X = -2.0 * alpha_scale * torch.matmul(W.transpose(1, 2), Mres2)

            X = X - lr * grad_X.float()

            # Report progress
            if progress_callback:
                progress_callback(step + 1, self.max_epochs)

            # Early stop check
            if self.use_early_stop and (step + 1) % self.check_interval == 0:
                with torch.no_grad():
                    Y_check = alpha_scale * torch.matmul(W, X)
                    R_check = (Y_teacher_b - Y_check) * A
                    current_loss = float(torch.sum(R_check ** 2, dim=(1, 2)).mean().item())

                    if current_loss < self.target_loss:
                        break

                    if loss_history is not None:
                        loss_history.append(current_loss)
                        if len(loss_history) >= self.patience:
                            losses = list(loss_history)
                            max_loss, min_loss = max(losses), min(losses)
                            if max_loss > 1e-12:
                                relative_change = (max_loss - min_loss) / max_loss
                                if relative_change < self.relative_threshold:
                                    break

        return W.float(), X.float()

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
        sample_callback: Optional[Callable] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train AGD for multiple alphas in parallel."""
        N1, M = W_teacher.shape
        N2 = X_teacher.shape[1]
        S = self.S
        device = self.device
        lr = self.lr
        num_alphas = len(alpha_values)

        alpha_scale = 1.0 / (M ** 0.5)
        scale = 1.0 / (M ** 0.5)

        # masks: (num_alphas, N1, N2) -> (num_alphas, 1, N1, N2)
        A_all = masks.unsqueeze(1)
        Y_teacher_b = Y_teacher.unsqueeze(0).unsqueeze(0)  # (1, 1, N1, N2)

        # Initialize student - (num_alphas, S, N1, M)
        torch.manual_seed(seed)
        W = torch.randn((num_alphas, S, N1, M), device=device, dtype=torch.float32) * scale
        X = torch.randn((num_alphas, S, M, N2), device=device, dtype=torch.float32) * scale

        for step in range(self.max_epochs):
            with torch.autocast(device_type=device.type, dtype=self.compute_dtype,
                                enabled=self.use_bf16):
                # W update
                Y_student = alpha_scale * torch.matmul(W, X)
                Mres = (Y_teacher_b - Y_student) * A_all
                grad_W = -2.0 * alpha_scale * torch.matmul(Mres, X.transpose(-2, -1))

            W = W - lr * grad_W.float()

            with torch.autocast(device_type=device.type, dtype=self.compute_dtype,
                                enabled=self.use_bf16):
                # X update
                Y_student2 = alpha_scale * torch.matmul(W, X)
                Mres2 = (Y_teacher_b - Y_student2) * A_all
                grad_X = -2.0 * alpha_scale * torch.matmul(W.transpose(-2, -1), Mres2)

            X = X - lr * grad_X.float()

            # Report progress (step_callback is the new interface, progress_callback for backward compat)
            callback = step_callback or progress_callback
            if callback:
                callback(step + 1, self.max_epochs)

        return W.float(), X.float()

    def supports_batch_training(self) -> bool:
        """AGD supports batch training."""
        return True

    def estimate_memory_per_alpha(self, N1: int, N2: int, M: int, S: int) -> float:
        """
        Estimate GPU memory needed per alpha value.

        AGD needs:
        - W, X student parameters: 2 * (S * N1 * M + S * M * N2)
        - Y_student, Mres, grad intermediates: ~6 * S * N1 * N2
        - Mask: S * N1 * N2
        Total: ~9 S*N1*N2 tensors + parameters
        """
        student_params = 2 * (S * N1 * M + S * M * N2)
        intermediate = 9 * S * N1 * N2
        total_elements = student_params + intermediate
        return total_elements * 4 / (1024**3)  # 4 bytes per float32
