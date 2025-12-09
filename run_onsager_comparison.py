#!/usr/bin/env python3
"""
Onsager Fix Comparison Script (Fixed Version)

Uses subprocess to ensure each version runs with fresh module imports.

Runs three versions of the algorithm:
1. No Onsager (baseline)
2. FIX #1 only (Onsager × damping)
3. FIX #1 + FIX #2 (Onsager × damping + Bayesian update)
"""

import sys
import os
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt

ALGO_FILE = Path("/home/sucia/Sparse-Matrix/smf/modules/algorithms/bigamp_spreading_parallel.py")
OUTPUT_ROOT = Path("/home/sucia/Sparse-Matrix")

# Single experiment runner script (will be created temporarily)
RUNNER_SCRIPT = OUTPUT_ROOT / "_run_single_experiment.py"

RUNNER_CODE = '''#!/usr/bin/env python3
"""Single experiment runner - runs in isolated process"""
import sys
sys.path.insert(0, "/home/sucia/Sparse-Matrix")

from smf.core.config import Config, MatrixConfig, AlphaConfig, TrainingConfig, AlgorithmConfig, SpreadingConfig, ExecutionConfig
from smf.runner import run_experiment

config = Config(
    matrix=MatrixConfig(N1=200, N2=200, M=50),
    alpha=AlphaConfig(start=0.0, stop=4.0, step=0.1),
    training=TrainingConfig(
        max_steps=5000,
        samples_per_alpha=4,
        seed=42,
        resample_mask=True
    ),
    algorithm=AlgorithmConfig(
        convergence_threshold=1.0e-06,
        damping=0.5,
        early_stop=False,
        learning_rate=0.01,
        noise_var=1.0e-10,
        use_compile=True
    ),
    spreading=SpreadingConfig(
        f_distribution='rademacher',
        seed=12345,
        teacher_type='standard'
    ),
    execution=ExecutionConfig(
        include_qy_plot=True,
        include_summary_plot=True,
        metrics_to_compute=['Q_Y', 'Q_W', 'Q_X', 'Q_W_prime', 'Q_X_prime', 'Gen_Error'],
        plots=[]
    ),
    algorithm_key='bigamp_spreading_parallel',
    graph_key='random',
    teacher_key='standard'
)

result = run_experiment(config)
print(f"RESULT_PATH:{result['result_path']}")
'''


def restore_original():
    """Restore original file from git"""
    subprocess.run(["git", "restore", str(ALGO_FILE)], check=True, cwd=OUTPUT_ROOT)
    print("[OK] Restored original (no Onsager) version")


def apply_fix1_only(content: str) -> str:
    """Apply FIX #1: Onsager × damping in both functions"""

    # FIX #1 for bigamp_spreading_parallel_step
    # Insert Onsager before residual computation
    old_block1 = '''    )  # (A, C_max)

    # ===== Compute residuals and beliefs ====='''

    new_block1 = '''    )  # (A, C_max)

    # ===== FIX #1: Onsager correction with damping =====
    if prev_s is not None:
        onsager_term = prev_s * V * damping
        Z_hat = Z_hat - onsager_term
        Z_hat = Z_hat * alpha_mask.float()

    # ===== Compute residuals and beliefs ====='''

    content = content.replace(old_block1, new_block1)

    # FIX #1 for bigamp_step_disjoint_union
    old_block2 = '''    V = V * alpha_mask_exp.float() + 1e-10

    # ===== 5. Residuals ====='''

    new_block2 = '''    V = V * alpha_mask_exp.float() + 1e-10

    # ===== FIX #1: Onsager correction with damping =====
    if prev_s is not None:
        onsager_term = prev_s * V * damping
        Z_hat = Z_hat - onsager_term
        Z_hat = Z_hat * alpha_mask_exp.float()

    # ===== 5. Residuals ====='''

    content = content.replace(old_block2, new_block2)

    # Remove duplicate mask application (bug fix)
    content = content.replace(
        '    s_values = s_values * alpha_mask_exp.float()\n\n    s_values = s_values * alpha_mask_exp.float()',
        '    s_values = s_values * alpha_mask_exp.float()'
    )

    return content


def apply_fix2(content: str) -> str:
    """Apply FIX #2: Bayesian update formula"""

    # FIX #2 for W in bigamp_spreading_parallel_step
    content = content.replace(
        '    W_hat_new = W_hat + W_var_new * r_W  # CRITICAL FIX: incremental update (was missing + W_hat)',
        '    W_hat_new = W_var_new * (r_W + W_hat * tau_W)  # FIX #2: Bayesian posterior update'
    )

    # FIX #2 for X in bigamp_spreading_parallel_step
    content = content.replace(
        '    X_hat_new = X_hat + X_var_new * r_X  # CRITICAL FIX: incremental update (was missing + X_hat)',
        '    X_hat_new = X_var_new * (r_X + X_hat * tau_X)  # FIX #2: Bayesian posterior update'
    )

    # FIX #2 for W in bigamp_step_disjoint_union
    content = content.replace(
        '    W_hat_new = W_flat + W_var_new * r_W  # CRITICAL FIX: incremental update (was missing + W_flat)',
        '    W_hat_new = W_var_new * (r_W + W_flat * tau_W)  # FIX #2: Bayesian posterior update'
    )

    # FIX #2 for X in bigamp_step_disjoint_union
    content = content.replace(
        '    X_hat_new = X_flat + X_var_new * r_X  # CRITICAL FIX: incremental update (was missing + X_flat)',
        '    X_hat_new = X_var_new * (r_X + X_flat * tau_X)  # FIX #2: Bayesian posterior update'
    )

    return content


def run_single_experiment(version_name: str) -> str:
    """Run a single experiment in a subprocess and return result path"""
    print(f"\n{'='*60}")
    print(f"Running: {version_name}")
    print(f"{'='*60}")

    # Write runner script
    RUNNER_SCRIPT.write_text(RUNNER_CODE)

    start = time.time()
    result = subprocess.run(
        [sys.executable, str(RUNNER_SCRIPT)],
        capture_output=True,
        text=True,
        cwd=OUTPUT_ROOT
    )
    elapsed = time.time() - start

    # Parse result path from output
    result_path = None
    for line in result.stdout.split('\n'):
        if line.startswith('RESULT_PATH:'):
            result_path = line.split(':', 1)[1].strip()
            break

    if result.returncode != 0:
        print(f"[ERROR] Experiment failed!")
        print(result.stderr)
        return None

    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Result path: {result_path}")

    return result_path


def load_metrics(result_dir):
    """Load metrics.json and extract alpha-keyed data."""
    json_path = Path(result_dir) / "metrics.json"
    if not json_path.exists():
        return None

    with open(json_path, "r") as f:
        raw = json.load(f)

    # Data might be under 'results' key or at top level
    data = raw.get('results', raw)

    # Filter out non-numeric keys (like 'config', 'metadata', etc.)
    filtered_data = {}
    for k, v in data.items():
        try:
            float(k)
            filtered_data[k] = v
        except ValueError:
            continue

    return filtered_data


def plot_three_versions(results: dict, output_path: Path):
    """Plot Q_Y and Q_W for all three versions"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['blue', 'orange', 'green']
    labels = ['No Onsager', 'FIX #1 (Onsager×damping)', 'FIX #1+2 (Bayesian)']

    for idx, (version_name, result_path) in enumerate(results.items()):
        if result_path is None:
            print(f"[WARN] No result for {version_name}")
            continue

        data = load_metrics(result_path)
        if data is None:
            print(f"[WARN] No metrics for {version_name}")
            continue

        alphas = sorted([float(k) for k in data.keys()])
        q_y_means = [data[str(a)].get('Q_Y_mean', 0) for a in alphas]
        q_w_means = [data[str(a)].get('Q_W_mean', 0) for a in alphas]

        axes[0].plot(alphas, q_y_means, marker='o', markersize=2,
                     color=colors[idx], label=labels[idx], linewidth=1.5)
        axes[1].plot(alphas, q_w_means, marker='o', markersize=2,
                     color=colors[idx], label=labels[idx], linewidth=1.5)

    axes[0].set_xlabel('Alpha (α)', fontsize=12)
    axes[0].set_ylabel('Q_Y (Reconstruction Quality)', fontsize=12)
    axes[0].set_title('Q_Y vs Alpha', fontsize=14)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    axes[1].set_xlabel('Alpha (α)', fontsize=12)
    axes[1].set_ylabel('Q_W (Factor Overlap)', fontsize=12)
    axes[1].set_title('Q_W vs Alpha', fontsize=14)
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    plt.suptitle('Onsager Correction Comparison (200×200, M=50, S=4, Steps=5000)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Comparison plot saved to: {output_path}")


def verify_modification(content: str, version: str):
    """Verify that the modification was applied"""
    if version == "fix1_only":
        if "FIX #1: Onsager correction with damping" in content:
            print("[OK] FIX #1 verified in file")
            return True
        else:
            print("[ERROR] FIX #1 NOT found in file!")
            return False
    elif version == "fix1_fix2":
        has_fix1 = "FIX #1: Onsager correction with damping" in content
        has_fix2 = "FIX #2: Bayesian posterior update" in content
        if has_fix1 and has_fix2:
            print("[OK] FIX #1 + FIX #2 verified in file")
            return True
        else:
            print(f"[ERROR] FIX #1: {has_fix1}, FIX #2: {has_fix2}")
            return False
    return True


def main():
    results = {}
    original_content = ALGO_FILE.read_text()

    try:
        # === Version 1: No Onsager (baseline) ===
        restore_original()
        result1 = run_single_experiment("no_onsager")
        results["no_onsager"] = result1

        # === Version 2: FIX #1 only ===
        print("\n[Applying FIX #1: Onsager × damping]")
        fix1_content = apply_fix1_only(original_content)
        ALGO_FILE.write_text(fix1_content)
        verify_modification(fix1_content, "fix1_only")
        result2 = run_single_experiment("fix1_only")
        results["fix1_only"] = result2

        # === Version 3: FIX #1 + FIX #2 ===
        print("\n[Applying FIX #1 + FIX #2: Onsager × damping + Bayesian update]")
        fix1_fix2_content = apply_fix2(fix1_content)
        ALGO_FILE.write_text(fix1_fix2_content)
        verify_modification(fix1_fix2_content, "fix1_fix2")
        result3 = run_single_experiment("fix1_fix2")
        results["fix1_fix2"] = result3

    finally:
        # Restore original
        restore_original()
        # Clean up runner script
        if RUNNER_SCRIPT.exists():
            RUNNER_SCRIPT.unlink()

    # Generate comparison plot
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_path = OUTPUT_ROOT / f"onsager_comparison_{timestamp}.png"
    plot_three_versions(results, output_path)

    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)
    print(f"\nResults:")
    for name, path in results.items():
        print(f"  {name}: {path}")
    print(f"\nComparison plot: {output_path}")


if __name__ == "__main__":
    main()
