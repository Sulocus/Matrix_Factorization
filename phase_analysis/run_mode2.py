#!/usr/bin/env python3
"""
Mode 2: Gradient-Adaptive Phase Analysis

Complete workflow:
1. Epoch convergence scan - find sufficient steps for convergence
2. Alpha range scan - smart detection of saturation point
3. Coarse training - uniform scan to locate phase transition
4. Gradient analysis - detect phase zone and generate smart alphas
5. Fine training - high-density sampling in phase zone
6. Visualization - comprehensive result plots
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

from core.bigamp_trainer import BiGAMPTrainer, TrainingConfig, EpochScanner
from core.gradient_adaptive_sampler import GradientAdaptiveSampler


def run_mode2(
    N1: int = 200,
    N2: int = 200,
    M: int = 50,
    coarse_step: float = 0.2,
    max_alpha: float = 6.0,
    phase_fraction: float = 0.6,
    samples_per_alpha: int = 3,
    skip_epoch_scan: bool = False,
    fixed_steps: int = None,
    verbose: bool = True,
    save_dir: Path = None
):
    """
    Run complete Mode 2 analysis.

    Args:
        N1, N2, M: Matrix dimensions
        coarse_step: Step size for coarse alpha scan
        max_alpha: Maximum alpha value
        phase_fraction: Fraction of points to put in phase zone
        samples_per_alpha: Number of trials per alpha
        skip_epoch_scan: Skip epoch convergence scan
        fixed_steps: Use fixed steps instead of scanning
        verbose: Show progress
        save_dir: Directory to save results
    """
    if save_dir is None:
        save_dir = Path(__file__).parent / "Result" / f"{N1}_{N2}_{M}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MODE 2: Gradient-Adaptive Phase Analysis")
    print("=" * 70)
    print(f"Matrix: N1={N1}, N2={N2}, M={M}")
    print(f"Save directory: {save_dir}")
    print()

    # =========================================================================
    # Step 1: Epoch Convergence Scan
    # =========================================================================
    if fixed_steps is not None:
        optimal_steps = fixed_steps
        print(f"[Step 1] Using fixed steps: {fixed_steps}")
    elif skip_epoch_scan:
        optimal_steps = 200
        print(f"[Step 1] Skipped - using default steps: {optimal_steps}")
    else:
        print("[Step 1] Epoch Convergence Scan")
        print("-" * 50)

        base_config = TrainingConfig(
            N1=N1, N2=N2, M=M,
            steps=50,  # Will be overridden
            samples_per_alpha=samples_per_alpha
        )
        scanner = EpochScanner(base_config)

        # Test at a few key alpha points
        test_alphas = [0.5, 1.0, 1.5, 2.0, 2.5]
        optimal_steps, scan_results = scanner.scan(
            test_alphas,
            epoch_levels=[50, 100, 200, 400],
            tolerance=0.05,
            verbose=verbose
        )
        print(f"\n-> Optimal steps: {optimal_steps}")

    # =========================================================================
    # Step 2: Alpha Range Scan with Smart Saturation Detection
    # =========================================================================
    print()
    print("[Step 2] Alpha Range Scan")
    print("-" * 50)

    config = TrainingConfig(
        N1=N1, N2=N2, M=M,
        steps=optimal_steps,
        samples_per_alpha=samples_per_alpha
    )
    trainer = BiGAMPTrainer(config)

    saturation_alpha, range_results = trainer.scan_alpha_range(
        alpha_step=coarse_step,
        saturation_threshold=0.95,
        max_alpha=max_alpha,
        sparse_step_multiplier=3.0,
        verbose=verbose
    )

    # Extract Q_Y for coarse scan
    coarse_alphas = np.array([r.alpha for r in range_results])
    coarse_Q_Y = np.array([r.Q_Y_mean for r in range_results])

    print(f"\n-> Saturation at alpha={saturation_alpha:.2f}")
    print(f"-> Total scan points: {len(coarse_alphas)}")

    # =========================================================================
    # Step 3: Gradient Analysis and Phase Zone Detection
    # =========================================================================
    print()
    print("[Step 3] Gradient Analysis")
    print("-" * 50)

    sampler = GradientAdaptiveSampler(coarse_alphas, coarse_Q_Y, smooth_sigma=1.0)

    # Find phase zone
    result = sampler.redistribute_zone_based(
        n_points=len(coarse_alphas),
        phase_fraction=phase_fraction
    )

    print(f"Phase zone: [{result.phase_zone[0]:.2f}, {result.phase_zone[1]:.2f}]")
    print(f"Phase center: {(result.phase_zone[0] + result.phase_zone[1]) / 2:.2f}")

    # Analyze distribution
    dist = sampler.analyze_distribution(result.adaptive_alphas)
    print(f"\nSampling distribution:")
    for region, info in dist['regions'].items():
        print(f"  {region}: {info['count']} pts, density={info['relative_density']:.2f}x")

    # =========================================================================
    # Step 4: Fine Training with Adaptive Alphas
    # =========================================================================
    print()
    print("[Step 4] Fine Training with Adaptive Alphas")
    print("-" * 50)

    config_fine = TrainingConfig(
        N1=N1, N2=N2, M=M,
        steps=optimal_steps * 2,  # Double steps for fine training
        samples_per_alpha=samples_per_alpha
    )
    trainer_fine = BiGAMPTrainer(config_fine)

    adaptive_results = trainer_fine.train(list(result.adaptive_alphas), verbose=verbose)
    adaptive_Q_Y = np.array([r.Q_Y_mean for r in adaptive_results])

    print(f"\n-> Fine training complete: {len(result.adaptive_alphas)} points")

    # =========================================================================
    # Step 5: Generate Visualization
    # =========================================================================
    print()
    print("[Step 5] Generating Visualization")
    print("-" * 50)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Main comparison plot
    ax1 = axes[0, 0]
    ax1.plot(coarse_alphas, coarse_Q_Y, 'b-o', markersize=5, linewidth=1,
             label=f'Coarse scan ({len(coarse_alphas)} pts, {optimal_steps} steps)', alpha=0.7)
    ax1.plot(result.adaptive_alphas, adaptive_Q_Y, 'r-s', markersize=5, linewidth=1.5,
             label=f'Adaptive fine ({len(result.adaptive_alphas)} pts, {optimal_steps*2} steps)')
    ax1.axvspan(result.phase_zone[0], result.phase_zone[1], alpha=0.15, color='yellow',
                label=f'Phase zone [{result.phase_zone[0]:.1f}, {result.phase_zone[1]:.1f}]')
    ax1.axvline(saturation_alpha, color='gray', linestyle=':', alpha=0.5,
                label=f'Saturation (alpha={saturation_alpha:.1f})')
    ax1.set_xlabel('alpha', fontsize=11)
    ax1.set_ylabel('Q_Y', fontsize=11)
    ax1.set_title('Q_Y vs Alpha: Uniform vs Adaptive Sampling', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.1, max_alpha + 0.1)

    # 2. Gradient profile
    ax2 = axes[0, 1]
    ax2.plot(sampler.alphas, sampler.gradient, 'g-o', linewidth=1.5, markersize=5)
    ax2.axvspan(result.phase_zone[0], result.phase_zone[1], alpha=0.2, color='red',
                label='Phase zone')
    ax2.axvline(result.phase_zone[0], color='r', linestyle='--', alpha=0.5)
    ax2.axvline(result.phase_zone[1], color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('alpha', fontsize=11)
    ax2.set_ylabel('Normalized Gradient', fontsize=11)
    ax2.set_title('Gradient Profile (from coarse scan)', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. Sampling density histogram
    ax3 = axes[1, 0]
    bins = np.linspace(0, max_alpha, int(max_alpha / coarse_step) + 1)
    uniform_hist, _ = np.histogram(coarse_alphas, bins=bins)
    adaptive_hist, bin_edges = np.histogram(result.adaptive_alphas, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = (bins[1] - bins[0]) * 0.4

    ax3.bar(bin_centers - width/2, uniform_hist, width, label='Uniform (coarse)',
            alpha=0.7, color='blue')
    ax3.bar(bin_centers + width/2, adaptive_hist, width, label='Adaptive',
            alpha=0.7, color='red')
    ax3.axvline(result.phase_zone[0], color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(result.phase_zone[1], color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('alpha', fontsize=11)
    ax3.set_ylabel('Sample Count', fontsize=11)
    ax3.set_title('Sampling Density Comparison', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # 4. Point distribution scatter
    ax4 = axes[1, 1]
    ax4.scatter(coarse_alphas, np.zeros_like(coarse_alphas) + 0.2,
                c='blue', s=40, label='Uniform (coarse)', alpha=0.7)
    ax4.scatter(result.adaptive_alphas, np.zeros_like(result.adaptive_alphas) - 0.2,
                c='red', s=40, label='Adaptive', alpha=0.7)
    ax4.axvline(result.phase_zone[0], color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(result.phase_zone[1], color='gray', linestyle='--', alpha=0.5)
    ax4.axvspan(result.phase_zone[0], result.phase_zone[1], alpha=0.1, color='gray')
    ax4.set_xlabel('alpha', fontsize=11)
    ax4.set_ylim(-0.5, 0.5)
    ax4.set_yticks([0.2, -0.2])
    ax4.set_yticklabels(['Uniform', 'Adaptive'])
    ax4.set_title('Alpha Point Distribution', fontsize=12)
    ax4.legend(fontsize=9, loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'Mode 2: Gradient-Adaptive Phase Analysis\n'
                 f'N1={N1}, N2={N2}, M={M} | Phase center: {(result.phase_zone[0] + result.phase_zone[1]) / 2:.2f}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = save_dir / f"mode2_analysis_{timestamp}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {fig_path}")
    plt.close()

    # =========================================================================
    # Step 6: Save Results to JSON
    # =========================================================================
    results_data = {
        "config": {
            "N1": N1, "N2": N2, "M": M,
            "coarse_steps": optimal_steps,
            "fine_steps": optimal_steps * 2,
            "samples_per_alpha": samples_per_alpha,
            "coarse_step": coarse_step,
            "max_alpha": max_alpha,
            "phase_fraction": phase_fraction
        },
        "phase_analysis": {
            "phase_zone": [float(result.phase_zone[0]), float(result.phase_zone[1])],
            "phase_center": float((result.phase_zone[0] + result.phase_zone[1]) / 2),
            "saturation_alpha": float(saturation_alpha),
            "distribution": {
                region: {
                    "count": info["count"],
                    "relative_density": float(info["relative_density"])
                }
                for region, info in dist['regions'].items()
            }
        },
        "coarse_results": {
            "alphas": coarse_alphas.tolist(),
            "Q_Y": coarse_Q_Y.tolist()
        },
        "adaptive_results": {
            "alphas": result.adaptive_alphas.tolist(),
            "Q_Y": adaptive_Q_Y.tolist()
        }
    }

    json_path = save_dir / f"mode2_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"Results saved: {json_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Phase zone: [{result.phase_zone[0]:.2f}, {result.phase_zone[1]:.2f}]")
    print(f"Phase center: {(result.phase_zone[0] + result.phase_zone[1]) / 2:.2f}")
    print(f"Saturation alpha: {saturation_alpha:.2f}")
    print(f"Coarse scan: {len(coarse_alphas)} points ({optimal_steps} steps)")
    print(f"Adaptive fine: {len(result.adaptive_alphas)} points ({optimal_steps * 2} steps)")
    print(f"\nPhase zone density boost: {dist['regions']['phase']['relative_density']:.2f}x")
    print()
    print(f"Output files:")
    print(f"  - {fig_path}")
    print(f"  - {json_path}")
    print("=" * 70)

    return results_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mode 2: Gradient-Adaptive Phase Analysis")
    parser.add_argument("--N1", type=int, default=200, help="Matrix dimension N1")
    parser.add_argument("--N2", type=int, default=200, help="Matrix dimension N2")
    parser.add_argument("--M", type=int, default=50, help="Latent dimension M")
    parser.add_argument("--coarse-step", type=float, default=0.2, help="Coarse scan step")
    parser.add_argument("--max-alpha", type=float, default=6.0, help="Maximum alpha")
    parser.add_argument("--samples", type=int, default=3, help="Samples per alpha")
    parser.add_argument("--skip-epoch-scan", action="store_true", help="Skip epoch convergence scan")
    parser.add_argument("--steps", type=int, default=None, help="Fixed steps (overrides epoch scan)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")

    args = parser.parse_args()

    run_mode2(
        N1=args.N1,
        N2=args.N2,
        M=args.M,
        coarse_step=args.coarse_step,
        max_alpha=args.max_alpha,
        samples_per_alpha=args.samples,
        skip_epoch_scan=args.skip_epoch_scan,
        fixed_steps=args.steps,
        verbose=not args.quiet
    )
