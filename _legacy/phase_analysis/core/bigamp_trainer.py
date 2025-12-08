"""
BiG-AMP Training Interface for Phase Analyzer

Provides a clean interface to call the BiG-AMP training from Main_bigamp_optimized.py
for use in Mode 2 and Mode 3 analysis.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from tqdm.auto import tqdm

# Add parent directory to path to import from Main_bigamp_optimized
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))


@dataclass
class TrainingConfig:
    """Configuration for BiG-AMP training"""
    N1: int = 200
    N2: int = 200
    M: int = 50
    steps: int = 200
    samples_per_alpha: int = 5
    damping: float = 0.5
    noise_var: float = 1e-6
    seed: int = 42
    device: str = 'auto'  # 'auto', 'cuda', 'mps', 'cpu'

    def get_device(self) -> torch.device:
        if self.device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(self.device)


@dataclass
class TrainingResult:
    """Result from training at a single alpha value"""
    alpha: float
    Q_Y_mean: float
    Q_Y_std: float
    Q_W_mean: float
    Q_W_std: float
    Q_X_mean: float
    Q_X_std: float
    Q_W_prime_mean: float
    Q_W_prime_std: float
    Q_X_prime_mean: float
    Q_X_prime_std: float
    Gen_Error_mean: float
    Gen_Error_std: float


class BiGAMPTrainer:
    """
    BiG-AMP trainer for phase transition analysis.

    Usage:
        trainer = BiGAMPTrainer(config)
        results = trainer.train(alpha_values)
        # results is a list of TrainingResult objects
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.get_device()
        self._setup_device()

        # Teacher model (created once, reused for all alphas)
        self.Wt = None
        self.Xt = None
        self.Y_teacher = None

    def _setup_device(self):
        """Setup device-specific optimizations"""
        if self.device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def _create_teacher(self):
        """Create teacher model W_true and X_true"""
        self._set_seed(self.config.seed)
        N1, N2, M = self.config.N1, self.config.N2, self.config.M
        scale = 1.0 / (M ** 0.5)

        self.Wt = torch.randn((N1, M), device=self.device, dtype=torch.float32) * scale
        self.Xt = torch.randn((M, N2), device=self.device, dtype=torch.float32) * scale
        self.Y_teacher = self.Wt @ self.Xt

    def _sample_mask(self, alpha: float, seed: int) -> torch.Tensor:
        """Generate random observation mask"""
        N1, N2, M = self.config.N1, self.config.N2, self.config.M
        torch.manual_seed(seed)

        c = alpha * M
        prob = min(c / N2, 1.0)
        mask = (torch.rand((N1, N2), device=self.device) < prob).float()
        return mask

    @torch.no_grad()
    def _gram_overlap_cosine(self, A: torch.Tensor, B: torch.Tensor, use_left: bool = True) -> float:
        """Compute Gram matrix overlap using cosine similarity"""
        if use_left:
            G_A = A @ A.T
            G_B = B @ B.T
        else:
            G_A = A.T @ A
            G_B = B.T @ B

        G_A_flat = G_A.flatten()
        G_B_flat = G_B.flatten()

        dot = (G_A_flat * G_B_flat).sum()
        norm_A = G_A_flat.norm()
        norm_B = G_B_flat.norm()

        return float(dot / (norm_A * norm_B + 1e-12))

    def _train_single_alpha(self, alpha: float, seed: int) -> Dict:
        """Train BiG-AMP for a single alpha value"""
        N1, N2, M = self.config.N1, self.config.N2, self.config.M
        S = self.config.samples_per_alpha
        steps = self.config.steps
        damping = self.config.damping
        noise_var = self.config.noise_var
        alpha_scale = 1.0 / (M ** 0.5)
        scale = 1.0 / (M ** 0.5)

        # Generate mask
        A = self._sample_mask(alpha, seed).unsqueeze(0)  # (1, N1, N2)

        # Initialize estimates
        torch.manual_seed(seed + 10000)
        w_hat = torch.randn((S, N1, M), device=self.device) * scale
        x_hat = torch.randn((S, M, N2), device=self.device) * scale
        w_var = torch.ones_like(w_hat) * (1.0 / M)
        x_var = torch.ones_like(x_hat) * (1.0 / M)

        Y_teacher = self.Y_teacher

        # BiG-AMP iterations
        for _ in range(steps):
            # Forward pass
            z_hat = alpha_scale * torch.matmul(w_hat, x_hat)
            w_sq = w_hat ** 2
            x_sq = x_hat ** 2
            p_var = (alpha_scale ** 2) * (torch.matmul(w_sq, x_var) + torch.matmul(w_var, x_sq))
            V = torch.clamp(p_var + noise_var, min=1e-8)
            residual = (Y_teacher - z_hat) * A
            s = residual / V

            # Update W
            tau_W = (alpha_scale ** 2) * torch.matmul(A / V, x_sq.transpose(-2, -1))
            tau_W = torch.clamp(tau_W, min=1e-8)
            w_var_new = 1.0 / (M + tau_W)
            r_W = alpha_scale * torch.matmul(s, x_hat.transpose(-2, -1))
            w_hat_new = w_hat + w_var_new * r_W
            w_hat = damping * w_hat + (1 - damping) * w_hat_new
            w_var = torch.clamp(damping * w_var + (1 - damping) * w_var_new, min=1e-8, max=1.0)

            # Update X
            z_hat2 = alpha_scale * torch.matmul(w_hat, x_hat)
            w_sq2 = w_hat ** 2
            p_var2 = (alpha_scale ** 2) * (torch.matmul(w_sq2, x_var) + torch.matmul(w_var, x_sq))
            V2 = torch.clamp(p_var2 + noise_var, min=1e-8)
            residual2 = (Y_teacher - z_hat2) * A
            s2 = residual2 / V2

            tau_X = (alpha_scale ** 2) * torch.matmul(w_sq2.transpose(-2, -1), A / V2)
            tau_X = torch.clamp(tau_X, min=1e-8)
            x_var_new = 1.0 / (M + tau_X)
            r_X = alpha_scale * torch.matmul(w_hat.transpose(-2, -1), s2)
            x_hat_new = x_hat + x_var_new * r_X
            x_hat = damping * x_hat + (1 - damping) * x_hat_new
            x_var = torch.clamp(damping * x_var + (1 - damping) * x_var_new, min=1e-8, max=1.0)

        # Evaluate
        return self._evaluate(w_hat, x_hat)

    @torch.no_grad()
    def _evaluate(self, W: torch.Tensor, X: torch.Tensor) -> Dict:
        """Evaluate training results"""
        S = W.shape[0]
        trial_results = []

        for s in range(S):
            W_s, X_s = W[s], X[s]

            Q_W = self._gram_overlap_cosine(W_s, self.Wt, use_left=True)
            Q_X = self._gram_overlap_cosine(X_s, self.Xt, use_left=False)
            Q_W_prime = (Q_W + 1.0) / 2.0
            Q_X_prime = (Q_X + 1.0) / 2.0

            Yp = W_s @ X_s
            Q_Y = float((self.Y_teacher.flatten() * Yp.flatten()).sum() /
                       (self.Y_teacher.norm() * Yp.norm() + 1e-12))
            gen_error = float(torch.mean((self.Y_teacher - Yp) ** 2))

            trial_results.append({
                'Q_W': Q_W, 'Q_X': Q_X,
                'Q_W_prime': Q_W_prime, 'Q_X_prime': Q_X_prime,
                'Q_Y': Q_Y, 'Gen_Error': gen_error
            })

        # Aggregate
        metrics = {}
        for key in trial_results[0].keys():
            vals = [r[key] for r in trial_results]
            metrics[f'{key}_mean'] = float(np.mean(vals))
            metrics[f'{key}_std'] = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

        return metrics

    def train(self, alpha_values: List[float], verbose: bool = True) -> List[TrainingResult]:
        """
        Train BiG-AMP for multiple alpha values.

        Args:
            alpha_values: List of alpha values to train
            verbose: Whether to show progress bar

        Returns:
            List of TrainingResult objects
        """
        # Create teacher model once
        if self.Wt is None:
            self._create_teacher()

        results = []
        iterator = tqdm(alpha_values, desc="Training", disable=not verbose)

        for alpha in iterator:
            if verbose:
                iterator.set_postfix({'alpha': f'{alpha:.2f}'})

            seed = self.config.seed + int(alpha * 1000)
            metrics = self._train_single_alpha(alpha, seed)

            result = TrainingResult(
                alpha=alpha,
                Q_Y_mean=metrics['Q_Y_mean'],
                Q_Y_std=metrics['Q_Y_std'],
                Q_W_mean=metrics['Q_W_mean'],
                Q_W_std=metrics['Q_W_std'],
                Q_X_mean=metrics['Q_X_mean'],
                Q_X_std=metrics['Q_X_std'],
                Q_W_prime_mean=metrics['Q_W_prime_mean'],
                Q_W_prime_std=metrics['Q_W_prime_std'],
                Q_X_prime_mean=metrics['Q_X_prime_mean'],
                Q_X_prime_std=metrics['Q_X_prime_std'],
                Gen_Error_mean=metrics['Gen_Error_mean'],
                Gen_Error_std=metrics['Gen_Error_std'],
            )
            results.append(result)

            # Clear cache
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        return results

    def get_Q_Y_array(self, results: List[TrainingResult]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract Q_Y values from results as numpy arrays"""
        alphas = np.array([r.alpha for r in results])
        Q_Y = np.array([r.Q_Y_mean for r in results])
        return alphas, Q_Y

    def scan_alpha_range(self, alpha_step: float = 0.2,
                          saturation_threshold: float = 0.95,
                          max_alpha: float = 6.0,
                          sparse_step_multiplier: float = 3.0,
                          verbose: bool = True) -> Tuple[float, List[TrainingResult]]:
        """
        Smart alpha range scan: dense before saturation, sparse after saturation.

        Continues to max_alpha for complete plot, but uses sparser sampling
        after Q_Y saturates to reduce computation.

        Args:
            alpha_step: Initial alpha step (used before saturation)
            saturation_threshold: Q_Y value to consider saturated
            max_alpha: Maximum alpha value for complete plot
            sparse_step_multiplier: How much to increase step after saturation
            verbose: Show progress

        Returns:
            (saturation_alpha, results): Alpha where saturation starts and all results
        """
        if self.Wt is None:
            self._create_teacher()

        results = []
        alpha = 0.0
        saturation_alpha = None
        saturation_count = 0
        current_step = alpha_step

        if verbose:
            print(f"[Alpha Range Scan] step={alpha_step}, sparse_mult={sparse_step_multiplier}")
            print(f"  Saturation threshold: {saturation_threshold}")
            print(f"  Max alpha: {max_alpha}")

        while alpha <= max_alpha:
            seed = self.config.seed + int(alpha * 1000)
            metrics = self._train_single_alpha(alpha, seed)

            result = TrainingResult(
                alpha=alpha,
                Q_Y_mean=metrics['Q_Y_mean'],
                Q_Y_std=metrics['Q_Y_std'],
                Q_W_mean=metrics['Q_W_mean'],
                Q_W_std=metrics['Q_W_std'],
                Q_X_mean=metrics['Q_X_mean'],
                Q_X_std=metrics['Q_X_std'],
                Q_W_prime_mean=metrics['Q_W_prime_mean'],
                Q_W_prime_std=metrics['Q_W_prime_std'],
                Q_X_prime_mean=metrics['Q_X_prime_mean'],
                Q_X_prime_std=metrics['Q_X_prime_std'],
                Gen_Error_mean=metrics['Gen_Error_mean'],
                Gen_Error_std=metrics['Gen_Error_std'],
            )
            results.append(result)

            mode_str = "(sparse)" if saturation_alpha is not None else "(dense)"
            if verbose:
                print(f"  alpha={alpha:.2f}: Q_Y={metrics['Q_Y_mean']:.4f} {mode_str}")

            # Check saturation transition
            if saturation_alpha is None:
                if metrics['Q_Y_mean'] >= saturation_threshold:
                    saturation_count += 1
                    if saturation_count >= 2:
                        saturation_alpha = alpha
                        current_step = alpha_step * sparse_step_multiplier
                        if verbose:
                            print(f"  -> Saturation detected! Switching to sparse sampling (step={current_step:.2f})")
                else:
                    saturation_count = 0

            alpha += current_step

            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        if saturation_alpha is None:
            saturation_alpha = max_alpha  # Never saturated

        if verbose:
            print(f"  Total points: {len(results)}, saturation at alpha={saturation_alpha:.2f}")

        return saturation_alpha, results


class EpochScanner:
    """
    扫描不同epoch数，找到足够收敛的epoch值

    策略：
    - 从低epoch开始（快速）
    - 逐步增加epoch直到Q_Y变化稳定
    - 当相邻epoch的Q_Y差异 < tolerance时认为收敛
    """

    def __init__(self, config: TrainingConfig):
        self.base_config = config

    def scan(self, alpha_values: List[float],
             epoch_levels: List[int] = None,
             tolerance: float = 0.05,
             verbose: bool = True) -> Tuple[int, Dict]:
        """
        扫描不同epoch找到收敛点

        Args:
            alpha_values: 用于测试的alpha值
            epoch_levels: 要测试的epoch列表（默认 [50, 100, 200, 400]）
            tolerance: Q_Y差异容忍度（相对变化）
            verbose: 显示进度

        Returns:
            (converged_epoch, scan_results): 收敛的epoch值和所有扫描结果
        """
        if epoch_levels is None:
            epoch_levels = [50, 100, 200, 400]

        if verbose:
            print(f"[Epoch Convergence Scan]")
            print(f"  Testing epochs: {epoch_levels}")
            print(f"  Tolerance: {tolerance*100:.1f}%")
            print(f"  Test alphas: {[f'{a:.1f}' for a in alpha_values]}")

        scan_results = {}
        prev_Q_Y = None
        converged_epoch = epoch_levels[-1]  # 默认用最大的

        for steps in epoch_levels:
            config = TrainingConfig(
                N1=self.base_config.N1,
                N2=self.base_config.N2,
                M=self.base_config.M,
                steps=steps,
                samples_per_alpha=self.base_config.samples_per_alpha,
                damping=self.base_config.damping,
                noise_var=self.base_config.noise_var,
                seed=self.base_config.seed,
            )

            trainer = BiGAMPTrainer(config)
            results = trainer.train(alpha_values, verbose=False)
            Q_Y_array = np.array([r.Q_Y_mean for r in results])

            scan_results[steps] = {
                'results': results,
                'Q_Y': Q_Y_array,
            }

            if verbose:
                print(f"\n  Steps={steps}:")
                for i, a in enumerate(alpha_values):
                    print(f"    alpha={a:.1f}: Q_Y={Q_Y_array[i]:.4f}")

            # Check convergence
            if prev_Q_Y is not None:
                # 计算相对变化
                diff = np.abs(Q_Y_array - prev_Q_Y)
                # 只看Q_Y > 0.1的点（避免除零和噪声）
                mask = prev_Q_Y > 0.1
                if mask.any():
                    rel_diff = diff[mask] / prev_Q_Y[mask]
                    max_rel_diff = rel_diff.max()

                    if verbose:
                        print(f"    Max relative change: {max_rel_diff*100:.2f}%")

                    if max_rel_diff < tolerance:
                        converged_epoch = steps
                        if verbose:
                            print(f"  -> Converged at {steps} steps!")
                        break

            prev_Q_Y = Q_Y_array.copy()

        return converged_epoch, scan_results


def train_and_get_Q_Y(alpha_values: List[float],
                      N1: int = 200, N2: int = 200, M: int = 50,
                      steps: int = 200, samples: int = 5,
                      verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function: Train and return Q_Y values.

    Args:
        alpha_values: Alpha values to train
        N1, N2, M: Matrix dimensions
        steps: BiG-AMP steps
        samples: Samples per alpha
        verbose: Show progress

    Returns:
        (alphas, Q_Y) as numpy arrays
    """
    config = TrainingConfig(
        N1=N1, N2=N2, M=M,
        steps=steps,
        samples_per_alpha=samples
    )

    trainer = BiGAMPTrainer(config)
    results = trainer.train(list(alpha_values), verbose=verbose)

    return trainer.get_Q_Y_array(results)


# Quick test
if __name__ == "__main__":
    print("Testing BiGAMPTrainer...")

    # Small test
    config = TrainingConfig(N1=200, N2=200, M=50, steps=50, samples_per_alpha=2)
    trainer = BiGAMPTrainer(config)

    alphas = [0.0, 1.0, 2.0]
    results = trainer.train(alphas)

    print("\nResults:")
    for r in results:
        print(f"  alpha={r.alpha:.1f}: Q_Y={r.Q_Y_mean:.4f} +/- {r.Q_Y_std:.4f}")

    print("\nTest complete!")
