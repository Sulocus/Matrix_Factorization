"""
Mode 3: Precise Phase Transition Analysis

Goal: Determine the exact phase transition point alpha_c with high precision.

Strategy:
1. Initial localization: Use Mode 2 to roughly locate alpha_c
2. Focused refinement: Narrow range progressively (alpha_c +/- 0.5 -> +/- 0.05)
3. Epoch escalation: Train with increasing steps [200, 500, 1000, 2000, ...]
4. Convergence detection: Stop when |delta_alpha_c| < threshold for N rounds
5. Extrapolation: Fit alpha_c vs 1/steps to get thermodynamic limit

Output:
- alpha_c with confidence interval
- Convergence history
- Extrapolated alpha_c (steps -> infinity)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
import matplotlib.pyplot as plt
from scipy import optimize

from .bigamp_trainer import BiGAMPTrainer, TrainingConfig, TrainingResult
from .gradient_adaptive_sampler import GradientAdaptiveSampler


@dataclass
class PhaseTransitionResult:
    """Result from phase transition detection at a single epoch level"""
    steps: int
    alpha_c: float
    alpha_c_std: float  # Uncertainty estimate
    phase_zone: Tuple[float, float]
    max_gradient: float
    gradient_at_alpha_c: float
    Q_Y_at_alpha_c: float


@dataclass
class PreciseAnalysisResult:
    """Final result from precise phase analysis"""
    # Final estimates
    alpha_c_final: float
    alpha_c_uncertainty: float

    # Extrapolated (thermodynamic limit)
    alpha_c_extrapolated: float
    extrapolation_quality: float  # R^2 of fit

    # History
    history: List[PhaseTransitionResult]

    # Convergence info
    converged: bool
    convergence_round: int
    total_rounds: int

    # Config
    N1: int
    N2: int
    M: int


class PrecisePhaseAnalyzer:
    """
    Mode 3: Precise determination of phase transition point.

    Uses iterative refinement with increasing epochs to converge
    on the exact alpha_c value.
    """

    def __init__(self,
                 N1: int = 200,
                 N2: int = 200,
                 M: int = 50,
                 samples_per_alpha: int = 5):
        self.N1 = N1
        self.N2 = N2
        self.M = M
        self.samples_per_alpha = samples_per_alpha

        # Analysis history
        self.history: List[PhaseTransitionResult] = []

    def _find_alpha_c(self,
                      alphas: np.ndarray,
                      Q_Y: np.ndarray,
                      smooth_sigma: float = 0.5) -> Tuple[float, float, float]:
        """
        Find alpha_c from Q_Y curve using gradient analysis.

        Returns:
            (alpha_c, max_gradient, Q_Y_at_alpha_c)
        """
        # Smooth the curve
        from scipy.ndimage import gaussian_filter1d
        Q_Y_smooth = gaussian_filter1d(Q_Y, sigma=smooth_sigma)

        # Compute gradient
        gradient = np.gradient(Q_Y_smooth, alphas)

        # Find maximum gradient point
        max_idx = np.argmax(gradient)
        alpha_c = alphas[max_idx]
        max_gradient = gradient[max_idx]
        Q_Y_at_alpha_c = Q_Y[max_idx]

        return alpha_c, max_gradient, Q_Y_at_alpha_c

    def _estimate_phase_zone(self,
                              alphas: np.ndarray,
                              Q_Y: np.ndarray,
                              alpha_c: float) -> Tuple[float, float]:
        """
        Estimate phase transition zone around alpha_c.

        Uses the region where gradient > 50% of max gradient.
        """
        sampler = GradientAdaptiveSampler(alphas, Q_Y, smooth_sigma=0.5)
        gradient = sampler.gradient

        max_grad = gradient.max()
        threshold = max_grad * 0.5

        # Find zone where gradient > threshold
        above_threshold = gradient > threshold
        indices = np.where(above_threshold)[0]

        if len(indices) > 0:
            left_idx = indices[0]
            right_idx = indices[-1]
            return (alphas[left_idx], alphas[right_idx])
        else:
            # Fallback: use alpha_c +/- 0.3
            return (alpha_c - 0.3, alpha_c + 0.3)

    def _train_and_analyze(self,
                            alpha_range: Tuple[float, float],
                            n_points: int,
                            steps: int,
                            verbose: bool = True) -> PhaseTransitionResult:
        """
        Train at given alpha range and analyze phase transition.
        """
        # Generate alpha points
        alphas = np.linspace(alpha_range[0], alpha_range[1], n_points)

        # Train
        config = TrainingConfig(
            N1=self.N1, N2=self.N2, M=self.M,
            steps=steps,
            samples_per_alpha=self.samples_per_alpha
        )
        trainer = BiGAMPTrainer(config)
        results = trainer.train(list(alphas), verbose=verbose)

        # Extract Q_Y
        Q_Y = np.array([r.Q_Y_mean for r in results])
        Q_Y_std = np.array([r.Q_Y_std for r in results])

        # Find alpha_c
        alpha_c, max_gradient, Q_Y_at_alpha_c = self._find_alpha_c(alphas, Q_Y)

        # Estimate uncertainty from Q_Y std
        idx = np.argmin(np.abs(alphas - alpha_c))
        alpha_c_std = Q_Y_std[idx] / max_gradient if max_gradient > 0 else 0.1

        # Estimate phase zone
        phase_zone = self._estimate_phase_zone(alphas, Q_Y, alpha_c)

        return PhaseTransitionResult(
            steps=steps,
            alpha_c=alpha_c,
            alpha_c_std=alpha_c_std,
            phase_zone=phase_zone,
            max_gradient=max_gradient,
            gradient_at_alpha_c=max_gradient,
            Q_Y_at_alpha_c=Q_Y_at_alpha_c
        )

    def _extrapolate_alpha_c(self,
                              history: List[PhaseTransitionResult]) -> Tuple[float, float]:
        """
        Extrapolate alpha_c to steps -> infinity.

        Uses linear fit: alpha_c = alpha_c_inf + A / steps

        Returns:
            (alpha_c_extrapolated, R_squared)
        """
        if len(history) < 3:
            return history[-1].alpha_c, 0.0

        steps = np.array([h.steps for h in history])
        alpha_c = np.array([h.alpha_c for h in history])

        # Fit: alpha_c = a + b / steps
        # i.e., alpha_c = a + b * x where x = 1/steps
        x = 1.0 / steps
        y = alpha_c

        # Weighted by inverse variance
        weights = np.array([1.0 / (h.alpha_c_std**2 + 1e-6) for h in history])

        # Weighted linear regression
        def model(params, x):
            return params[0] + params[1] * x

        def residual(params, x, y, w):
            return np.sqrt(w) * (y - model(params, x))

        # Initial guess
        p0 = [alpha_c[-1], 0.0]

        try:
            result = optimize.least_squares(residual, p0, args=(x, y, weights))
            alpha_c_inf = result.x[0]

            # Compute R^2
            y_pred = model(result.x, x)
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        except:
            alpha_c_inf = alpha_c[-1]
            r_squared = 0.0

        return alpha_c_inf, max(0, r_squared)

    def _auto_step_levels(self) -> List[int]:
        """
        Automatically determine step_levels based on matrix size.

        Larger matrices need more BiG-AMP iterations to converge.
        Reference: Main_bigamp_optimized.py uses 5000 steps for N=10000.
        """
        N = max(self.N1, self.N2)

        if N <= 500:
            # Small matrices: quick convergence
            return [200, 400, 800, 1600]
        elif N <= 2000:
            # Medium matrices
            return [500, 1000, 2000, 4000]
        elif N <= 5000:
            # Large matrices
            return [1000, 2000, 3000, 5000]
        else:
            # Very large matrices (N > 5000)
            # Main_bigamp_optimized.py uses 5000 for N=10000
            # Use practical step levels that don't take forever
            return [1000, 2000, 4000, 6000]

    def analyze(self,
                initial_range: Tuple[float, float] = (0.0, 4.0),
                step_levels: List[int] = None,
                convergence_threshold: float = 0.02,
                convergence_patience: int = 2,
                n_points_initial: int = 21,
                n_points_refined: int = 31,
                verbose: bool = True) -> PreciseAnalysisResult:
        """
        Run precise phase analysis.

        Args:
            initial_range: Initial alpha range to scan
            step_levels: List of step counts to use (auto-scaled if None)
            convergence_threshold: Stop when |delta_alpha_c| < this
            convergence_patience: Converged if stable for this many rounds
            n_points_initial: Points for initial scan
            n_points_refined: Points for refined scans
            verbose: Show progress

        Returns:
            PreciseAnalysisResult
        """
        if step_levels is None:
            step_levels = self._auto_step_levels()

        self.history = []
        converged = False
        convergence_round = len(step_levels)
        stable_count = 0
        prev_alpha_c = None
        current_range = initial_range

        if verbose:
            print("=" * 60)
            print("Mode 3: Precise Phase Transition Analysis")
            print("=" * 60)
            print(f"Matrix: N1={self.N1}, N2={self.N2}, M={self.M}")
            print(f"Step levels: {step_levels}")
            print(f"Convergence threshold: {convergence_threshold}")
            print()

        for round_idx, steps in enumerate(step_levels):
            if verbose:
                print(f"[Round {round_idx + 1}/{len(step_levels)}] Steps={steps}")
                print("-" * 40)

            # Determine number of points
            n_points = n_points_initial if round_idx == 0 else n_points_refined

            # Train and analyze
            result = self._train_and_analyze(
                current_range, n_points, steps, verbose=verbose
            )
            self.history.append(result)

            if verbose:
                print(f"  alpha_c = {result.alpha_c:.4f} +/- {result.alpha_c_std:.4f}")
                print(f"  Phase zone: [{result.phase_zone[0]:.2f}, {result.phase_zone[1]:.2f}]")
                print(f"  Q_Y at alpha_c: {result.Q_Y_at_alpha_c:.4f}")
                print()

            # Check convergence
            if prev_alpha_c is not None:
                delta = abs(result.alpha_c - prev_alpha_c)
                if verbose:
                    print(f"  Delta from previous: {delta:.4f}")

                if delta < convergence_threshold:
                    stable_count += 1
                    if stable_count >= convergence_patience:
                        converged = True
                        convergence_round = round_idx + 1
                        if verbose:
                            print(f"  -> CONVERGED at round {convergence_round}!")
                        break
                else:
                    stable_count = 0

            prev_alpha_c = result.alpha_c

            # Narrow range for next round (focus on phase zone)
            margin = (result.phase_zone[1] - result.phase_zone[0]) * 0.3
            new_left = max(0, result.phase_zone[0] - margin)
            new_right = result.phase_zone[1] + margin
            current_range = (new_left, new_right)

            if verbose:
                print(f"  Next range: [{current_range[0]:.2f}, {current_range[1]:.2f}]")
                print()

        # Extrapolate
        alpha_c_extrapolated, r_squared = self._extrapolate_alpha_c(self.history)

        # Final estimate
        alpha_c_final = self.history[-1].alpha_c
        alpha_c_uncertainty = self.history[-1].alpha_c_std

        if verbose:
            print("=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"Final alpha_c: {alpha_c_final:.4f} +/- {alpha_c_uncertainty:.4f}")
            print(f"Extrapolated (steps->inf): {alpha_c_extrapolated:.4f}")
            print(f"Extrapolation R^2: {r_squared:.4f}")
            print(f"Converged: {converged} (round {convergence_round})")
            print("=" * 60)

        return PreciseAnalysisResult(
            alpha_c_final=alpha_c_final,
            alpha_c_uncertainty=alpha_c_uncertainty,
            alpha_c_extrapolated=alpha_c_extrapolated,
            extrapolation_quality=r_squared,
            history=self.history,
            converged=converged,
            convergence_round=convergence_round,
            total_rounds=len(self.history),
            N1=self.N1,
            N2=self.N2,
            M=self.M
        )

    def plot_convergence(self,
                          result: PreciseAnalysisResult,
                          save_path: Optional[Path] = None):
        """
        Plot convergence history.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        history = result.history
        steps = [h.steps for h in history]
        alpha_c = [h.alpha_c for h in history]
        alpha_c_std = [h.alpha_c_std for h in history]
        Q_Y_at_c = [h.Q_Y_at_alpha_c for h in history]
        max_grad = [h.max_gradient for h in history]

        # 1. Alpha_c convergence
        ax1 = axes[0, 0]
        ax1.errorbar(steps, alpha_c, yerr=alpha_c_std, fmt='o-', capsize=5,
                     markersize=8, linewidth=2, color='#2563eb')
        ax1.axhline(result.alpha_c_extrapolated, color='red', linestyle='--',
                    label=f'Extrapolated: {result.alpha_c_extrapolated:.4f}')
        ax1.set_xlabel('Steps', fontsize=12)
        ax1.set_ylabel(r'$\alpha_c$', fontsize=12)
        ax1.set_title(r'$\alpha_c$ Convergence', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Alpha_c vs 1/steps (for extrapolation)
        ax2 = axes[0, 1]
        inv_steps = [1.0 / s for s in steps]
        ax2.scatter(inv_steps, alpha_c, s=80, c='#2563eb', zorder=5)
        # Fit line
        if len(steps) >= 2:
            x_fit = np.linspace(0, max(inv_steps), 100)
            slope = (alpha_c[-1] - alpha_c[0]) / (inv_steps[-1] - inv_steps[0]) if inv_steps[-1] != inv_steps[0] else 0
            y_fit = result.alpha_c_extrapolated + slope * x_fit
            ax2.plot(x_fit, y_fit, 'r--', linewidth=1.5,
                     label=f'R^2={result.extrapolation_quality:.3f}')
        ax2.set_xlabel('1 / Steps', fontsize=12)
        ax2.set_ylabel(r'$\alpha_c$', fontsize=12)
        ax2.set_title('Extrapolation to Steps -> Infinity', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Q_Y at alpha_c
        ax3 = axes[1, 0]
        ax3.plot(steps, Q_Y_at_c, 'o-', markersize=8, linewidth=2, color='#16a34a')
        ax3.set_xlabel('Steps', fontsize=12)
        ax3.set_ylabel(r'$Q_Y$ at $\alpha_c$', fontsize=12)
        ax3.set_title(r'$Q_Y$ at Phase Transition', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 4. Max gradient
        ax4 = axes[1, 1]
        ax4.plot(steps, max_grad, 'o-', markersize=8, linewidth=2, color='#dc2626')
        ax4.set_xlabel('Steps', fontsize=12)
        ax4.set_ylabel('Max Gradient', fontsize=12)
        ax4.set_title('Gradient Sharpness', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        plt.suptitle(f'Mode 3: Precise Phase Analysis\n'
                     f'N={result.N1}, M={result.M} | '
                     f'Final: {result.alpha_c_final:.4f} +/- {result.alpha_c_uncertainty:.4f}',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Plot saved: {save_path}")

        plt.close(fig)

    def save_results(self,
                      result: PreciseAnalysisResult,
                      save_dir: Path):
        """
        Save results to JSON.
        """
        save_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "config": {
                "N1": result.N1,
                "N2": result.N2,
                "M": result.M,
                "samples_per_alpha": self.samples_per_alpha
            },
            "result": {
                "alpha_c_final": result.alpha_c_final,
                "alpha_c_uncertainty": result.alpha_c_uncertainty,
                "alpha_c_extrapolated": result.alpha_c_extrapolated,
                "extrapolation_quality": result.extrapolation_quality,
                "converged": result.converged,
                "convergence_round": result.convergence_round,
                "total_rounds": result.total_rounds
            },
            "history": [
                {
                    "steps": h.steps,
                    "alpha_c": h.alpha_c,
                    "alpha_c_std": h.alpha_c_std,
                    "phase_zone": list(h.phase_zone),
                    "max_gradient": h.max_gradient,
                    "Q_Y_at_alpha_c": h.Q_Y_at_alpha_c
                }
                for h in result.history
            ]
        }

        json_path = save_dir / f"mode3_precise_{result.N1}x{result.N2}_M{result.M}.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Results saved: {json_path}")

        return json_path


# Convenience function
def run_precise_analysis(
    N1: int = 200,
    N2: int = 200,
    M: int = 50,
    samples: int = 5,
    step_levels: List[int] = None,
    verbose: bool = True,
    save_dir: Path = None
) -> PreciseAnalysisResult:
    """
    Run Mode 3 precise phase analysis.

    Args:
        N1, N2, M: Matrix dimensions
        samples: Samples per alpha
        step_levels: List of step counts (auto-scaled based on N if None)
        verbose: Show progress
        save_dir: Directory to save results

    Returns:
        PreciseAnalysisResult
    """
    # step_levels will be auto-scaled in analyzer.analyze() if None
    analyzer = PrecisePhaseAnalyzer(N1=N1, N2=N2, M=M, samples_per_alpha=samples)
    result = analyzer.analyze(step_levels=step_levels, verbose=verbose)

    if save_dir is None:
        save_dir = Path(__file__).parent.parent / "Result" / f"{N1}_{N2}_{M}"

    # Save results
    analyzer.save_results(result, save_dir)

    # Plot
    plot_path = save_dir / f"mode3_convergence_{N1}x{N2}_M{M}.png"
    analyzer.plot_convergence(result, plot_path)

    return result
