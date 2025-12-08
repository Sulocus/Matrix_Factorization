"""
Test Mode 2 with REAL BiG-AMP training data

This test actually runs the BiG-AMP algorithm to generate real Q_Y values,
then applies the gradient-adaptive sampling to redistribute alpha points.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

from core.bigamp_trainer import BiGAMPTrainer, TrainingConfig
from core.gradient_adaptive_sampler import GradientAdaptiveSampler


def test_mode2_with_real_data():
    """Test Mode 2 sampler with real BiG-AMP training data"""
    print("=" * 60)
    print("Mode 2 Test with REAL BiG-AMP Training")
    print("=" * 60)

    # Step 1: Run coarse training to get Q_Y curve
    print("\n[Step 1] Coarse training (21 uniform alpha points)...")

    config = TrainingConfig(
        N1=200, N2=200, M=50,
        steps=100,  # Low steps for coarse scan
        samples_per_alpha=3
    )
    trainer = BiGAMPTrainer(config)

    # Coarse uniform sampling
    coarse_alphas = np.linspace(0, 4, 21)
    results = trainer.train(list(coarse_alphas), verbose=True)
    coarse_Q_Y = np.array([r.Q_Y_mean for r in results])

    print(f"\nCoarse scan complete:")
    print(f"  Alpha range: [{coarse_alphas[0]:.1f}, {coarse_alphas[-1]:.1f}]")
    print(f"  Points: {len(coarse_alphas)}")
    print(f"  Q_Y range: [{coarse_Q_Y.min():.4f}, {coarse_Q_Y.max():.4f}]")

    # Step 2: Analyze gradient and find phase zone
    print("\n[Step 2] Gradient analysis and phase zone detection...")

    sampler = GradientAdaptiveSampler(coarse_alphas, coarse_Q_Y, smooth_sigma=1.0)

    max_grad_idx = np.argmax(sampler.gradient)
    print(f"  Max gradient at: alpha = {sampler.alphas[max_grad_idx]:.2f}")
    print(f"  Max gradient value: {sampler.gradient[max_grad_idx]:.4f}")

    # Step 3: Generate adaptive alpha distribution
    print("\n[Step 3] Generate adaptive sampling points...")

    result = sampler.redistribute_zone_based(n_points=len(coarse_alphas), phase_fraction=0.6)

    print(f"  Phase zone detected: [{result.phase_zone[0]:.2f}, {result.phase_zone[1]:.2f}]")
    print(f"  Adaptive points generated: {len(result.adaptive_alphas)}")

    # Analyze distribution
    dist = sampler.analyze_distribution(result.adaptive_alphas)
    print(f"\n  Sampling distribution:")
    print(f"    Total points: {dist['total_points']}")
    for region, info in dist['regions'].items():
        print(f"    {region}: {info['count']} pts, rel_density={info['relative_density']:.2f}x")

    # Step 4: Run fine training with adaptive alphas
    print("\n[Step 4] Fine training with adaptive alpha points...")

    config_fine = TrainingConfig(
        N1=200, N2=200, M=50,
        steps=200,  # Higher steps for fine training
        samples_per_alpha=3
    )
    trainer_fine = BiGAMPTrainer(config_fine)

    # Train with adaptive alphas
    adaptive_results = trainer_fine.train(list(result.adaptive_alphas), verbose=True)
    adaptive_Q_Y = np.array([r.Q_Y_mean for r in adaptive_results])

    print(f"\nFine training complete:")
    print(f"  Points: {len(result.adaptive_alphas)}")
    print(f"  Q_Y range: [{adaptive_Q_Y.min():.4f}, {adaptive_Q_Y.max():.4f}]")

    # Step 5: Visualization
    print("\n[Step 5] Generating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Compare uniform vs adaptive sampling
    ax1 = axes[0, 0]
    ax1.scatter(coarse_alphas, coarse_Q_Y, c='blue', s=60, marker='o',
                label=f'Uniform coarse ({len(coarse_alphas)} pts, 100 steps)', zorder=5)
    ax1.scatter(result.adaptive_alphas, adaptive_Q_Y, c='red', s=60, marker='x',
                label=f'Adaptive fine ({len(result.adaptive_alphas)} pts, 200 steps)', zorder=6)
    ax1.axvspan(result.phase_zone[0], result.phase_zone[1], alpha=0.2, color='yellow', label='Phase zone')
    ax1.set_xlabel('alpha')
    ax1.set_ylabel('Q_Y')
    ax1.set_title('REAL BiG-AMP Results: Uniform vs Adaptive Sampling')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Gradient profile
    ax2 = axes[0, 1]
    ax2.plot(sampler.alphas, sampler.gradient, 'g-o', linewidth=1.5, markersize=5)
    ax2.axvline(result.phase_zone[0], color='r', linestyle='--', alpha=0.5)
    ax2.axvline(result.phase_zone[1], color='r', linestyle='--', alpha=0.5)
    ax2.axvspan(result.phase_zone[0], result.phase_zone[1], alpha=0.2, color='red')
    ax2.set_xlabel('alpha')
    ax2.set_ylabel('Normalized Gradient')
    ax2.set_title('Gradient Profile (from coarse scan)')
    ax2.grid(True, alpha=0.3)

    # 3. Sampling density histogram
    ax3 = axes[1, 0]
    bins = np.linspace(0, 4, 21)
    uniform_hist, _ = np.histogram(coarse_alphas, bins=bins)
    adaptive_hist, bin_edges = np.histogram(result.adaptive_alphas, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = 0.08

    ax3.bar(bin_centers - width/2, uniform_hist, width, label='Uniform', alpha=0.7, color='blue')
    ax3.bar(bin_centers + width/2, adaptive_hist, width, label='Adaptive', alpha=0.7, color='red')
    ax3.axvline(result.phase_zone[0], color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(result.phase_zone[1], color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('alpha')
    ax3.set_ylabel('Sample Count')
    ax3.set_title('Sampling Density Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Point distribution scatter
    ax4 = axes[1, 1]
    ax4.scatter(coarse_alphas, np.zeros_like(coarse_alphas) + 0.2,
                c='blue', s=50, label='Uniform', alpha=0.7)
    ax4.scatter(result.adaptive_alphas, np.zeros_like(result.adaptive_alphas) - 0.2,
                c='red', s=50, label='Adaptive', alpha=0.7)
    ax4.axvline(result.phase_zone[0], color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(result.phase_zone[1], color='gray', linestyle='--', alpha=0.5)
    ax4.axvspan(result.phase_zone[0], result.phase_zone[1], alpha=0.1, color='gray')
    ax4.set_xlabel('alpha')
    ax4.set_ylim(-0.5, 0.5)
    ax4.set_yticks([0.2, -0.2])
    ax4.set_yticklabels(['Uniform', 'Adaptive'])
    ax4.set_title('Alpha Point Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Mode 2: Gradient-Adaptive Sampling with REAL BiG-AMP Data\n'
                 f'N1={config.N1}, N2={config.N2}, M={config.M}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save
    save_dir = Path(__file__).parent.parent / "Result"
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "mode2_real_bigamp.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {save_path}")
    plt.close()

    # Validation
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    # Check phase zone is in expected range [1.5, 2.5]
    phase_center = (result.phase_zone[0] + result.phase_zone[1]) / 2
    if 1.5 <= phase_center <= 2.5:
        print(f"[OK] Phase center {phase_center:.2f} in expected range [1.5, 2.5]")
    else:
        print(f"[X] Phase center {phase_center:.2f} NOT in expected range [1.5, 2.5]")

    # Check density boost
    phase_density = dist['regions']['phase']['relative_density']
    if phase_density >= 2.0:
        print(f"[OK] Phase zone density boost: {phase_density:.2f}x >= 2.0x")
    else:
        print(f"[X] Phase zone density boost: {phase_density:.2f}x < 2.0x")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

    return True


if __name__ == "__main__":
    test_mode2_with_real_data()
