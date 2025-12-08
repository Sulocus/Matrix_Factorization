#!/usr/bin/env python3
"""
Mode 2 Simple: Gradient-Adaptive Phase Analysis

Simplified output matching the main Result format:
- Single JSON with config + results (same format as Main_bigamp_optimized.py)
- Single PNG with 4-panel plot (Q_Y, Q_W'/Q_X', Gen_Error, Q_W/Q_X)

Usage:
    python run_mode2_simple.py                    # Default 200x200 M=50
    python run_mode2_simple.py --N1 2000 --M 100  # Custom dimensions
    python run_mode2_simple.py --steps 200        # Custom steps
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
import json
import time
import argparse

from core.bigamp_trainer import BiGAMPTrainer, TrainingConfig, EpochScanner
from core.gradient_adaptive_sampler import GradientAdaptiveSampler


def run_mode2_simple(
    N1: int = 200,
    N2: int = 200,
    M: int = 50,
    coarse_step: float = 0.2,
    max_alpha: float = 6.0,
    phase_fraction: float = 0.7,  # 70% points in phase zone
    samples_per_alpha: int = 5,
    steps: int = None,
    verbose: bool = True
):
    """
    Run Mode 2 and output in standard format.

    Returns results in the same format as Main_bigamp_optimized.py
    """
    # Output directory (same as main Result)
    result_dir = Path(__file__).parent.parent / "Result" / f"{N1}_{N2}_{M}"
    result_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Mode 2: Gradient-Adaptive Phase Analysis")
    print("=" * 60)
    print(f"Matrix: N1={N1}, N2={N2}, M={M}")

    start_time = time.time()

    # =========================================================================
    # Step 1: Determine optimal steps if not specified
    # =========================================================================
    if steps is None:
        print("\n[Step 1] Epoch Convergence Scan...")
        base_config = TrainingConfig(N1=N1, N2=N2, M=M, steps=50, samples_per_alpha=2)
        scanner = EpochScanner(base_config)
        test_alphas = [0.5, 1.0, 1.5, 2.0, 2.5]
        steps, _ = scanner.scan(test_alphas, epoch_levels=[50, 100, 200], tolerance=0.05, verbose=False)
        print(f"  -> Optimal steps: {steps}")
    else:
        print(f"\n[Step 1] Using fixed steps: {steps}")

    # =========================================================================
    # Step 2: Coarse alpha scan with smart saturation
    # =========================================================================
    print("\n[Step 2] Alpha Range Scan...")
    config = TrainingConfig(N1=N1, N2=N2, M=M, steps=steps, samples_per_alpha=samples_per_alpha)
    trainer = BiGAMPTrainer(config)

    saturation_alpha, coarse_results = trainer.scan_alpha_range(
        alpha_step=coarse_step,
        saturation_threshold=0.95,
        max_alpha=max_alpha,
        sparse_step_multiplier=3.0,
        verbose=verbose
    )

    coarse_alphas = np.array([r.alpha for r in coarse_results])
    coarse_Q_Y = np.array([r.Q_Y_mean for r in coarse_results])

    print(f"  -> Saturation at alpha={saturation_alpha:.2f}")
    print(f"  -> Coarse points: {len(coarse_alphas)}")

    # =========================================================================
    # Step 3: Gradient analysis and adaptive alpha generation
    # =========================================================================
    print("\n[Step 3] Gradient Analysis...")
    sampler = GradientAdaptiveSampler(coarse_alphas, coarse_Q_Y, smooth_sigma=1.0)
    # 增加采样点数：至少31点，相变区占70%
    adaptive_result = sampler.redistribute_zone_based(
        n_points=max(31, int(len(coarse_alphas) * 1.5)),
        phase_fraction=phase_fraction
    )
    print(f"  -> Phase zone: [{adaptive_result.phase_zone[0]:.2f}, {adaptive_result.phase_zone[1]:.2f}]")
    print(f"  -> Adaptive points: {len(adaptive_result.adaptive_alphas)}")

    # =========================================================================
    # Step 4: Fine training with adaptive alphas
    # =========================================================================
    print("\n[Step 4] Fine Training...")
    fine_steps = steps * 2  # Double steps for fine training
    config_fine = TrainingConfig(N1=N1, N2=N2, M=M, steps=fine_steps, samples_per_alpha=samples_per_alpha)
    trainer_fine = BiGAMPTrainer(config_fine)

    final_results = trainer_fine.train(list(adaptive_result.adaptive_alphas), verbose=verbose)

    total_time = time.time() - start_time

    # =========================================================================
    # Step 5: Format output (same as Main_bigamp_optimized.py)
    # =========================================================================
    print("\n[Step 5] Saving Results...")

    alpha_values = [r.alpha for r in final_results]

    # JSON format matching Main_bigamp_optimized.py
    output_data = {
        "config": {
            "N1": N1,
            "N2": N2,
            "M": M,
            "steps": fine_steps,
            "samples_per_alpha": samples_per_alpha,
            "mode": "mode2_adaptive",
            "phase_zone": [float(adaptive_result.phase_zone[0]), float(adaptive_result.phase_zone[1])],
            "total_time": total_time
        },
        "alpha_values": alpha_values,
        "results": {}
    }

    for r in final_results:
        output_data["results"][str(r.alpha)] = {
            "Q_W_mean": r.Q_W_mean,
            "Q_W_std": r.Q_W_std,
            "Q_X_mean": r.Q_X_mean,
            "Q_X_std": r.Q_X_std,
            "Q_W_prime_mean": r.Q_W_prime_mean,
            "Q_W_prime_std": r.Q_W_prime_std,
            "Q_X_prime_mean": r.Q_X_prime_mean,
            "Q_X_prime_std": r.Q_X_prime_std,
            "Q_Y_mean": r.Q_Y_mean,
            "Q_Y_std": r.Q_Y_std,
            "Gen_Error_mean": r.Gen_Error_mean,
            "Gen_Error_std": r.Gen_Error_std,
        }

    # Save JSON
    json_filename = f"mode2_adaptive_steps{fine_steps}_batch{samples_per_alpha}.json"
    json_path = result_dir / json_filename
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"  JSON: {json_path}")

    # =========================================================================
    # Step 6: Generate plot (same format as Main_bigamp_optimized.py)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    MAIN_COLOR = '#2563eb'
    SECONDARY_COLOR = '#dc2626'

    # Plot 1: Q_Y
    ax1 = axes[0, 0]
    qy_mean = [r.Q_Y_mean for r in final_results]
    qy_std = [r.Q_Y_std for r in final_results]
    ax1.errorbar(alpha_values, qy_mean, yerr=qy_std, fmt='o-', color=MAIN_COLOR,
                 capsize=3, markersize=6, linewidth=2, label='Q_Y')
    ax1.axvspan(adaptive_result.phase_zone[0], adaptive_result.phase_zone[1],
                alpha=0.15, color='yellow', label='Phase zone')
    ax1.set_xlabel(r'$\tilde{\alpha}$', fontsize=12)
    ax1.set_ylabel('Q_Y', fontsize=12)
    ax1.set_title('Y Overlap (Q_Y)', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    # Plot 2: Q_W' and Q_X'
    ax2 = axes[0, 1]
    qw_mean = [r.Q_W_prime_mean for r in final_results]
    qx_mean = [r.Q_X_prime_mean for r in final_results]
    ax2.plot(alpha_values, qw_mean, 'o-', color=MAIN_COLOR, markersize=6, linewidth=2, label="Q_W'")
    ax2.plot(alpha_values, qx_mean, 's-', color=SECONDARY_COLOR, markersize=6, linewidth=2, label="Q_X'")
    ax2.set_xlabel(r'$\tilde{\alpha}$', fontsize=12)
    ax2.set_ylabel("Q' (normalized)", fontsize=12)
    ax2.set_title("Normalized Gram Overlaps", fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.45, 1.05)

    # Plot 3: Generalization Error
    ax3 = axes[1, 0]
    ge_mean = [r.Gen_Error_mean for r in final_results]
    ax3.semilogy(alpha_values, ge_mean, 'o-', color=MAIN_COLOR, markersize=6, linewidth=2)
    ax3.set_xlabel(r'$\tilde{\alpha}$', fontsize=12)
    ax3.set_ylabel('Generalization Error (log)', fontsize=12)
    ax3.set_title('Generalization Error', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Q_W and Q_X (cosine)
    ax4 = axes[1, 1]
    qw_cos = [r.Q_W_mean for r in final_results]
    qx_cos = [r.Q_X_mean for r in final_results]
    ax4.plot(alpha_values, qw_cos, 'o-', color=MAIN_COLOR, markersize=6, linewidth=2, label='Q_W')
    ax4.plot(alpha_values, qx_cos, 's-', color=SECONDARY_COLOR, markersize=6, linewidth=2, label='Q_X')
    ax4.set_xlabel(r'$\tilde{\alpha}$', fontsize=12)
    ax4.set_ylabel('Q (cosine)', fontsize=12)
    ax4.set_title('Gram Overlaps (Cosine)', fontsize=14, fontweight='bold')
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3)

    phase_center = (adaptive_result.phase_zone[0] + adaptive_result.phase_zone[1]) / 2
    plt.suptitle(f'Mode 2 Adaptive: {N1}x{N2}, M={M}, Steps={fine_steps}\n'
                 f'Phase zone: [{adaptive_result.phase_zone[0]:.2f}, {adaptive_result.phase_zone[1]:.2f}], '
                 f'center={phase_center:.2f}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save PNG
    png_filename = f"Mode2_Adaptive_Steps{fine_steps}_batch{samples_per_alpha}.png"
    png_path = result_dir / png_filename
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  PNG: {png_path}")
    plt.close(fig)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Phase zone: [{adaptive_result.phase_zone[0]:.2f}, {adaptive_result.phase_zone[1]:.2f}]")
    print(f"Phase center: {phase_center:.2f}")
    print(f"Total points: {len(final_results)}")
    print(f"Time: {total_time:.1f}s")
    print(f"\nOutput:")
    print(f"  {json_path}")
    print(f"  {png_path}")

    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mode 2: Gradient-Adaptive Phase Analysis")
    parser.add_argument("--N1", type=int, default=200, help="Matrix dimension N1")
    parser.add_argument("--N2", type=int, default=None, help="Matrix dimension N2 (default: same as N1)")
    parser.add_argument("--M", type=int, default=50, help="Latent dimension M")
    parser.add_argument("--steps", type=int, default=None, help="BiG-AMP steps (auto if not specified)")
    parser.add_argument("--samples", type=int, default=5, help="Samples per alpha")
    parser.add_argument("--max-alpha", type=float, default=6.0, help="Maximum alpha")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")

    args = parser.parse_args()
    N2 = args.N2 if args.N2 is not None else args.N1

    run_mode2_simple(
        N1=args.N1,
        N2=N2,
        M=args.M,
        max_alpha=args.max_alpha,
        samples_per_alpha=args.samples,
        steps=args.steps,
        verbose=not args.quiet
    )
