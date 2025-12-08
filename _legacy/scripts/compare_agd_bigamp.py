#!/usr/bin/env python3
"""
AGD vs BiG-AMP Algorithm Comparison

This script compares the results of two different algorithms:
- AGD (Alternating Gradient Descent) from Main_multi_alpha.py
- BiG-AMP (Bilinear Generalized Approximate Message Passing) from bigamp_optimized.py

Purpose: Verify that the physical phenomena observed (e.g., M/N linear relationship)
are NOT artifacts of the BiG-AMP algorithm, but genuine physical properties of the system.

Usage:
    # Mode 1: Compare from JSON files
    python compare_agd_bigamp.py --mode from_json \
        --bigamp-json path/to/bigamp_results.json \
        --agd-json path/to/agd_results.json

    # Mode 2: Run both algorithms and compare (slower)
    python compare_agd_bigamp.py --mode run_both

Author: Claude Code
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Tuple, Optional

# ============================================================
# Configuration
# ============================================================

# Default comparison parameters
DEFAULT_TOLERANCE = 0.05  # 5% tolerance for Q_Y difference

# For run_both mode
DEFAULT_N1 = 200
DEFAULT_N2 = 200
DEFAULT_M = 50
DEFAULT_ALPHA_START = 0.0
DEFAULT_ALPHA_STOP = 2.0
DEFAULT_ALPHA_STEP = 0.1
DEFAULT_BIGAMP_STEPS = 5000
DEFAULT_AGD_EPOCHS = 50000  # AGD needs ~10x more iterations
DEFAULT_SAMPLES = 10

# Output directory
RESULT_DIR = Path("Result_comparison")


# ============================================================
# JSON Loading
# ============================================================

def load_json_results(json_path: str) -> dict:
    """Load results from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def extract_qy_from_results(results: dict) -> Tuple[List[float], List[float], List[float]]:
    """
    Extract alpha values, Q_Y means, and Q_Y_unobserved means from results.

    Handles different JSON formats:
    - Direct dict: {alpha: {metrics}}
    - Nested dict: {results: {config: {alpha: {metrics}}}}
    """
    alphas = []
    qy_means = []
    qy_unobs_means = []

    # Try to detect format
    if 'results' in results:
        inner = results['results']
        if isinstance(inner, dict):
            # Check if this is already alpha->metrics format
            first_key = list(inner.keys())[0]
            first_val = inner[first_key]
            # If first value has 'Q_Y_mean', it's alpha->metrics format
            if isinstance(first_val, dict) and 'Q_Y_mean' in first_val:
                data = inner
            else:
                # Multi-size format: {config_key: {alpha: metrics}}
                data = first_val
        else:
            data = results
    else:
        data = results

    # Extract metrics
    for key, metrics in sorted(data.items(), key=lambda x: float(x[0]) if x[0].replace('.', '').replace('-', '').isdigit() else 0):
        try:
            alpha = float(key)
        except ValueError:
            continue

        if isinstance(metrics, dict):
            qy = metrics.get('Q_Y_mean', 0)
            qy_unobs = metrics.get('Q_Y_unobserved_mean', qy)  # Fallback to Q_Y if unobserved not available

            alphas.append(alpha)
            qy_means.append(qy)
            qy_unobs_means.append(qy_unobs)

    return alphas, qy_means, qy_unobs_means


# ============================================================
# Comparison Logic
# ============================================================

def compare_results(
    alphas_bigamp: List[float], qy_bigamp: List[float], qy_unobs_bigamp: List[float],
    alphas_agd: List[float], qy_agd: List[float], qy_unobs_agd: List[float],
    tolerance: float = DEFAULT_TOLERANCE
) -> Dict:
    """
    Compare Q_Y and Q_Y_unobserved between two algorithms.

    Returns comparison statistics and PASS/FAIL status.
    """
    # Find common alphas (within small tolerance)
    common_alphas = []
    bigamp_idx = []
    agd_idx = []

    for i, a_b in enumerate(alphas_bigamp):
        for j, a_a in enumerate(alphas_agd):
            if abs(a_b - a_a) < 0.001:  # Alpha matching tolerance
                common_alphas.append(a_b)
                bigamp_idx.append(i)
                agd_idx.append(j)
                break

    if len(common_alphas) == 0:
        return {
            'status': 'ERROR',
            'message': 'No common alpha values found between the two results',
            'common_alphas': 0
        }

    # Extract values for common alphas
    qy_b = np.array([qy_bigamp[i] for i in bigamp_idx])
    qy_a = np.array([qy_agd[i] for i in agd_idx])
    qy_unobs_b = np.array([qy_unobs_bigamp[i] for i in bigamp_idx])
    qy_unobs_a = np.array([qy_unobs_agd[i] for i in agd_idx])

    # Compute differences
    qy_diff = np.abs(qy_b - qy_a)
    qy_unobs_diff = np.abs(qy_unobs_b - qy_unobs_a)

    # Statistics
    stats = {
        'common_alphas': len(common_alphas),
        'alpha_range': (min(common_alphas), max(common_alphas)),

        'Q_Y': {
            'max_diff': float(np.max(qy_diff)),
            'mean_diff': float(np.mean(qy_diff)),
            'std_diff': float(np.std(qy_diff)),
            'max_diff_alpha': common_alphas[int(np.argmax(qy_diff))],
        },

        'Q_Y_unobserved': {
            'max_diff': float(np.max(qy_unobs_diff)),
            'mean_diff': float(np.mean(qy_unobs_diff)),
            'std_diff': float(np.std(qy_unobs_diff)),
            'max_diff_alpha': common_alphas[int(np.argmax(qy_unobs_diff))],
        },

        'tolerance': tolerance,
        'data': {
            'alphas': common_alphas,
            'qy_bigamp': qy_b.tolist(),
            'qy_agd': qy_a.tolist(),
            'qy_unobs_bigamp': qy_unobs_b.tolist(),
            'qy_unobs_agd': qy_unobs_a.tolist(),
        }
    }

    # Determine PASS/FAIL
    qy_pass = stats['Q_Y']['max_diff'] < tolerance
    qy_unobs_pass = stats['Q_Y_unobserved']['max_diff'] < tolerance

    if qy_pass and qy_unobs_pass:
        stats['status'] = 'PASS'
        stats['message'] = f'Both Q_Y and Q_Y_unobserved within {tolerance*100:.1f}% tolerance'
    else:
        stats['status'] = 'FAIL'
        failures = []
        if not qy_pass:
            failures.append(f"Q_Y max diff {stats['Q_Y']['max_diff']:.4f} > {tolerance}")
        if not qy_unobs_pass:
            failures.append(f"Q_Y_unobs max diff {stats['Q_Y_unobserved']['max_diff']:.4f} > {tolerance}")
        stats['message'] = 'FAILED: ' + ', '.join(failures)

    return stats


# ============================================================
# Visualization
# ============================================================

def plot_comparison(stats: Dict, save_path: Path):
    """Generate comparison plots"""
    data = stats['data']
    alphas = np.array(data['alphas'])
    qy_b = np.array(data['qy_bigamp'])
    qy_a = np.array(data['qy_agd'])
    qy_unobs_b = np.array(data['qy_unobs_bigamp'])
    qy_unobs_a = np.array(data['qy_unobs_agd'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ============================================================
    # Plot 1: Q_Y comparison
    # ============================================================
    ax1 = axes[0, 0]
    ax1.plot(alphas, qy_b, 'o-', linewidth=2, markersize=5, color='#d62728',
             label='BiG-AMP', alpha=0.8)
    ax1.plot(alphas, qy_a, 's--', linewidth=2, markersize=5, color='#1f77b4',
             label='AGD', alpha=0.8)
    ax1.set_xlabel(r'$\tilde{\alpha}$', fontsize=12)
    ax1.set_ylabel(r'$Q_Y$', fontsize=12)
    ax1.set_title('Q_Y: BiG-AMP vs AGD', fontsize=13, fontweight='bold')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # ============================================================
    # Plot 2: Q_Y_unobserved comparison
    # ============================================================
    ax2 = axes[0, 1]
    ax2.plot(alphas, qy_unobs_b, 'o-', linewidth=2, markersize=5, color='#d62728',
             label='BiG-AMP', alpha=0.8)
    ax2.plot(alphas, qy_unobs_a, 's--', linewidth=2, markersize=5, color='#1f77b4',
             label='AGD', alpha=0.8)
    ax2.set_xlabel(r'$\tilde{\alpha}$', fontsize=12)
    ax2.set_ylabel(r'$Q_Y^{unobserved}$', fontsize=12)
    ax2.set_title('Q_Y_unobserved: BiG-AMP vs AGD', fontsize=13, fontweight='bold')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    # ============================================================
    # Plot 3: Q_Y difference
    # ============================================================
    ax3 = axes[1, 0]
    qy_diff = np.abs(qy_b - qy_a)
    ax3.bar(alphas, qy_diff, width=0.08, color='#ff7f0e', alpha=0.7)
    ax3.axhline(y=stats['tolerance'], color='r', linestyle='--', linewidth=2,
                label=f'Tolerance ({stats["tolerance"]*100:.0f}%)')
    ax3.set_xlabel(r'$\tilde{\alpha}$', fontsize=12)
    ax3.set_ylabel('|Q_Y difference|', fontsize=12)
    ax3.set_title(f'Q_Y Difference (max={stats["Q_Y"]["max_diff"]:.4f})', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)

    # ============================================================
    # Plot 4: Q_Y_unobserved difference
    # ============================================================
    ax4 = axes[1, 1]
    qy_unobs_diff = np.abs(qy_unobs_b - qy_unobs_a)
    ax4.bar(alphas, qy_unobs_diff, width=0.08, color='#2ca02c', alpha=0.7)
    ax4.axhline(y=stats['tolerance'], color='r', linestyle='--', linewidth=2,
                label=f'Tolerance ({stats["tolerance"]*100:.0f}%)')
    ax4.set_xlabel(r'$\tilde{\alpha}$', fontsize=12)
    ax4.set_ylabel('|Q_Y_unobs difference|', fontsize=12)
    ax4.set_title(f'Q_Y_unobserved Difference (max={stats["Q_Y_unobserved"]["max_diff"]:.4f})',
                  fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)

    # Title with status
    status_color = 'green' if stats['status'] == 'PASS' else 'red'
    fig.suptitle(f'AGD vs BiG-AMP Comparison - {stats["status"]}',
                 fontsize=16, fontweight='bold', color=status_color, y=1.02)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved: {save_path}")
    plt.close(fig)


def print_report(stats: Dict):
    """Print comparison report to console"""
    print("\n" + "=" * 70)
    print("AGD vs BiG-AMP COMPARISON REPORT")
    print("=" * 70)

    status_symbol = "PASS" if stats['status'] == 'PASS' else "FAIL"
    print(f"\nStatus: {status_symbol}")
    print(f"Message: {stats['message']}")
    print(f"\nCommon alpha points: {stats['common_alphas']}")
    print(f"Alpha range: [{stats['alpha_range'][0]:.2f}, {stats['alpha_range'][1]:.2f}]")

    print(f"\n--- Q_Y Statistics ---")
    print(f"  Max difference: {stats['Q_Y']['max_diff']:.6f} (at alpha={stats['Q_Y']['max_diff_alpha']:.2f})")
    print(f"  Mean difference: {stats['Q_Y']['mean_diff']:.6f}")
    print(f"  Std difference: {stats['Q_Y']['std_diff']:.6f}")

    print(f"\n--- Q_Y_unobserved Statistics ---")
    print(f"  Max difference: {stats['Q_Y_unobserved']['max_diff']:.6f} (at alpha={stats['Q_Y_unobserved']['max_diff_alpha']:.2f})")
    print(f"  Mean difference: {stats['Q_Y_unobserved']['mean_diff']:.6f}")
    print(f"  Std difference: {stats['Q_Y_unobserved']['std_diff']:.6f}")

    print(f"\nTolerance: {stats['tolerance']*100:.1f}%")
    print("=" * 70)


# ============================================================
# Mode: from_json
# ============================================================

def compare_from_json(bigamp_json: str, agd_json: str, tolerance: float, output_dir: Path):
    """Compare results from two JSON files"""
    print(f"Loading BiG-AMP results from: {bigamp_json}")
    bigamp_data = load_json_results(bigamp_json)
    alphas_b, qy_b, qy_unobs_b = extract_qy_from_results(bigamp_data)

    print(f"Loading AGD results from: {agd_json}")
    agd_data = load_json_results(agd_json)
    alphas_a, qy_a, qy_unobs_a = extract_qy_from_results(agd_data)

    print(f"BiG-AMP: {len(alphas_b)} alpha points")
    print(f"AGD: {len(alphas_a)} alpha points")

    # Compare
    stats = compare_results(
        alphas_b, qy_b, qy_unobs_b,
        alphas_a, qy_a, qy_unobs_a,
        tolerance=tolerance
    )

    # Report
    print_report(stats)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save stats JSON
    stats_path = output_dir / 'comparison_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved: {stats_path}")

    # Generate plot
    if 'data' in stats:
        plot_path = output_dir / 'comparison_plot.png'
        plot_comparison(stats, plot_path)

    return stats


# ============================================================
# Mode: run_both (placeholder - actual implementation would import from scripts)
# ============================================================

def run_both_and_compare(
    n1: int, n2: int, m: int,
    alpha_start: float, alpha_stop: float, alpha_step: float,
    bigamp_steps: int, agd_epochs: int,
    samples: int, tolerance: float, output_dir: Path
):
    """
    Run both algorithms with same parameters and compare.

    Note: This is a placeholder. Full implementation would import
    training functions from bigamp_optimized.py and Main_multi_alpha.py.
    """
    print("=" * 70)
    print("RUN BOTH MODE")
    print("=" * 70)
    print(f"Matrix: N1={n1}, N2={n2}, M={m}")
    print(f"Alpha range: [{alpha_start}, {alpha_stop}], step={alpha_step}")
    print(f"BiG-AMP steps: {bigamp_steps}")
    print(f"AGD epochs: {agd_epochs}")
    print(f"Samples: {samples}")
    print("=" * 70)

    print("\nNote: run_both mode requires importing training functions.")
    print("For now, please run the algorithms separately and use from_json mode:")
    print()
    print("  1. Run BiG-AMP:")
    print(f"     python _legacy/bigamp_optimized.py --n1 {n1} --m {m} --steps {bigamp_steps}")
    print()
    print("  2. Run AGD:")
    print(f"     python _legacy/Main_multi_alpha.py")
    print(f"     (Configure N1={n1}, N2={n2}, M={m}, EPOCHS_PER_ALPHA={agd_epochs})")
    print()
    print("  3. Compare:")
    print("     python _legacy/compare_agd_bigamp.py --mode from_json \\")
    print("         --bigamp-json path/to/bigamp_results.json \\")
    print("         --agd-json path/to/agd_results.json")

    return None


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compare AGD and BiG-AMP algorithm results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare from JSON files
  python compare_agd_bigamp.py --mode from_json \\
      --bigamp-json Result/400_400_50/bigamp_results_steps5000.json \\
      --agd-json Result/400_400_50/agd_results_epoch50000.json

  # Run both algorithms (placeholder)
  python compare_agd_bigamp.py --mode run_both --n1 200 --m 50
        """
    )

    parser.add_argument('--mode', type=str, choices=['from_json', 'run_both'],
                        default='from_json', help='Comparison mode')

    # from_json mode arguments
    parser.add_argument('--bigamp-json', type=str, help='Path to BiG-AMP results JSON')
    parser.add_argument('--agd-json', type=str, help='Path to AGD results JSON')

    # run_both mode arguments
    parser.add_argument('--n1', type=int, default=DEFAULT_N1, help='Matrix N1 dimension')
    parser.add_argument('--n2', type=int, default=DEFAULT_N2, help='Matrix N2 dimension')
    parser.add_argument('--m', type=int, default=DEFAULT_M, help='Latent dimension M')
    parser.add_argument('--alpha-start', type=float, default=DEFAULT_ALPHA_START)
    parser.add_argument('--alpha-stop', type=float, default=DEFAULT_ALPHA_STOP)
    parser.add_argument('--alpha-step', type=float, default=DEFAULT_ALPHA_STEP)
    parser.add_argument('--bigamp-steps', type=int, default=DEFAULT_BIGAMP_STEPS)
    parser.add_argument('--agd-epochs', type=int, default=DEFAULT_AGD_EPOCHS)
    parser.add_argument('--samples', type=int, default=DEFAULT_SAMPLES)

    # Common arguments
    parser.add_argument('--tolerance', type=float, default=DEFAULT_TOLERANCE,
                        help='Maximum allowed difference (default: 0.05 = 5%%)')
    parser.add_argument('--output-dir', type=str, default=str(RESULT_DIR),
                        help='Output directory for results')

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    if args.mode == 'from_json':
        if not args.bigamp_json or not args.agd_json:
            parser.error("--mode from_json requires --bigamp-json and --agd-json")

        compare_from_json(
            bigamp_json=args.bigamp_json,
            agd_json=args.agd_json,
            tolerance=args.tolerance,
            output_dir=output_dir
        )

    elif args.mode == 'run_both':
        run_both_and_compare(
            n1=args.n1, n2=args.n2, m=args.m,
            alpha_start=args.alpha_start,
            alpha_stop=args.alpha_stop,
            alpha_step=args.alpha_step,
            bigamp_steps=args.bigamp_steps,
            agd_epochs=args.agd_epochs,
            samples=args.samples,
            tolerance=args.tolerance,
            output_dir=output_dir
        )


if __name__ == "__main__":
    main()
