#!/usr/bin/env python3
"""
Post-processing script for Replica Analysis results.

Features:
1. Generates Comparison Plot:
   - Teacher-Student Overlap (Q_W_mean)
   - Student-Student Overlap (Q_W_replica_mean)
   vs Alpha
2. Cleans up redundant X plots if N1=N2 (symmetric).
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from smf.modules.outputs.plotting import create_gif
from smf.core.config import Config
from smf.modules.registry import get_teacher
from smf.modules.metrics.overlap import compute_physical_overlap
import re


def recreate_teacher(config_dict):
    """Recreate teacher from config."""
    try:
        matrix_cfg = config_dict.get('matrix', {})
        # Flatten if needed
        if not matrix_cfg and 'N1' in config_dict:
             matrix_cfg = config_dict
             
        N1 = matrix_cfg.get('N1', 200)
        N2 = matrix_cfg.get('N2', 200)
        M = matrix_cfg.get('M', 50)
        
        training_cfg = config_dict.get('training', {})
        seed = training_cfg.get('seed', 42)
        
        teacher_key = config_dict.get('teacher_key', 'standard')
        
        # Get class
        teacher_cls = get_teacher(teacher_key).cls
        teacher = teacher_cls()
        
        # Create
        device = torch.device('cpu')
        W_t, X_t, _ = teacher.create_with_Y(N1, N2, M, device, seed)
        return W_t
    except Exception as e:
        print(f"Error recreating teacher: {e}")
        return None

def compute_physical_curves(result_dir):
    """Compute Physical Overlap curves from raw data."""
    raw_dir = result_dir / "raw_data"
    if not raw_dir.exists():
        return None, None, None
        
    config_path = result_dir / "metrics.json"
    if not config_path.exists():
        return None, None, None
        
    with open(config_path) as f:
        data = json.load(f)
        config_dict = data.get('config', {})
        
    W_t = recreate_teacher(config_dict)
    if W_t is None:
        return None, None, None
        
    files = sorted(list(raw_dir.glob("alpha_*.npz")), 
                   key=lambda p: float(re.search(r"alpha_(\d+\.\d+)", p.name).group(1)))
                   
    alphas = []
    ts_means = []
    ss_means = []
    
    for p in files:
        alpha = float(re.search(r"alpha_(\d+\.\d+)", p.name).group(1))
        alphas.append(alpha)
        
        try:
            with np.load(p) as data:
                W_s_np = data['W_student']
            
            W_s = torch.from_numpy(W_s_np).float()
            S = W_s.shape[0]
            
            # TS Overlap
            ts_vals = []
            for s in range(S):
                val = compute_physical_overlap(W_s[s], W_t, absolute=True)
                ts_vals.append(val)
            ts_means.append(np.mean(ts_vals))
            
            # SS Overlap
            ss_vals = []
            if S > 1:
                for i in range(S):
                    for j in range(i+1, S):
                         val = compute_physical_overlap(W_s[i], W_s[j], absolute=True)
                         ss_vals.append(val)
                ss_means.append(np.mean(ss_vals))
            else:
                ss_means.append(1.0)
                
        except Exception as e:
            # print(f"Error computing physical for alpha {alpha}: {e}")
            pass
            
    return alphas, ts_means, ss_means


def process_experiment(result_dir: Path):
    """Process a single result directory."""
    print(f"\nProcessing: {result_dir.name}")
    
    metrics_path = result_dir / "metrics.json"
    if not metrics_path.exists():
        print("  Skipping: metrics.json not found")
        return

    try:
        with open(metrics_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Error loading metrics: {e}")
        return

    config = data.get('config', {})
    results = data.get('results', {})
    
    alphas_gram = sorted([float(k) for k in results.keys()])
    q_ts_gram = []
    q_ss_gram = []
    
    for alpha in alphas_gram:
        metrics = results[str(alpha)]
        if 'interaction_matrix_W' in metrics:
            mat = np.array(metrics['interaction_matrix_W'])
            if mat.shape[1] > 1:
                q_ts_gram.append(np.mean(mat[0, 1:]))
            else:
                q_ts_gram.append(0.0)
            if mat.shape[0] > 1:
                student_block = mat[1:, 1:]
                S = student_block.shape[0]
                if S > 1:
                    mask = ~np.eye(S, dtype=bool)
                    q_ss_gram.append(np.mean(student_block[mask]))
                else:
                     q_ss_gram.append(1.0)
            else:
                q_ss_gram.append(0.0)
        else:
            q_ts_gram.append(0.0)
            q_ss_gram.append(0.0)
            
    # Plot Gram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(alphas_gram, q_ts_gram, 'b-', label='Teacher-Student (Q_TS)', linewidth=2)
    ax.plot(alphas_gram, q_ss_gram, 'r--', label='Student-Student (Q_SS)', linewidth=2)
    ax.set_xlabel('Alpha (Measurement Ratio)')
    ax.set_ylabel('Overlap (Gram Normalized)')
    ax.set_title(f'Overlap Evolution (Gram)\n{result_dir.name}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(-0.1, 1.1)
    plt.savefig(result_dir / "overlap_evolution_gram.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Generated: overlap_evolution_gram.png")
    
    # 2. PHYSICAL Overlap
    alphas_phys, q_ts_phys, q_ss_phys = compute_physical_curves(result_dir)
    if alphas_phys:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(alphas_phys, q_ts_phys, 'b-', label='Teacher-Student (Q_TS)', linewidth=2)
        ax.plot(alphas_phys, q_ss_phys, 'r--', label='Student-Student (Q_SS)', linewidth=2)
        ax.set_xlabel('Alpha (Measurement Ratio)')
        ax.set_ylabel('Overlap (Physical / Dot Product)')
        ax.set_title(f'Overlap Evolution (Physical)\n{result_dir.name}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(-0.1, 1.1)
        plt.savefig(result_dir / "overlap_evolution_physical.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("  Generated: overlap_evolution_physical.png")
    else:
        print("  Skipping Physical: Could not compute from raw data")
    
    # 3. Q_Y Plot (from metrics.json if available)
    q_y_values = []
    q_y_unobs_values = []
    has_qy = False
    has_qy_unobs = False
    
    for alpha in alphas_gram:
        metrics = results[str(alpha)]
        if 'Q_Y_mean' in metrics:
            q_y_values.append(metrics['Q_Y_mean'])
            has_qy = True
        else:
            q_y_values.append(None)
            
        if 'Q_Y_unobserved_mean' in metrics:
            q_y_unobs_values.append(metrics['Q_Y_unobserved_mean'])
            has_qy_unobs = True
        else:
            q_y_unobs_values.append(None)
    
    if has_qy or has_qy_unobs:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if has_qy:
            # Filter out None values
            valid_alphas = [a for a, v in zip(alphas_gram, q_y_values) if v is not None]
            valid_qy = [v for v in q_y_values if v is not None]
            ax.plot(valid_alphas, valid_qy, 'g-', label='Q_Y (Cosine Sim)', linewidth=2, marker='o', markersize=3)
        
        if has_qy_unobs:
            valid_alphas_u = [a for a, v in zip(alphas_gram, q_y_unobs_values) if v is not None]
            valid_qy_u = [v for v in q_y_unobs_values if v is not None]
            ax.plot(valid_alphas_u, valid_qy_u, 'm--', label='Q_Y_unobserved', linewidth=2, marker='s', markersize=3)
        
        ax.set_xlabel('Alpha (Measurement Ratio)')
        ax.set_ylabel('Q_Y Overlap')
        ax.set_title(f'Q_Y Evolution\n{result_dir.name}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(-0.1, 1.1)
        plt.savefig(result_dir / "qy_evolution.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("  Generated: qy_evolution.png")
    else:
        print("  Skipping Q_Y: No Q_Y data in metrics.json")
            
    # Clean up X plots if Symmetric
    config = data.get('config', {})
    matrix_cfg = config.get('matrix', {})
    # Handle case where matrix might be top-level or in config
    if not matrix_cfg:
        matrix_cfg = config  # Fallback
        
    N1 = matrix_cfg.get('N1', 0)
    N2 = matrix_cfg.get('N2', 0)
    
    if N1 == N2 and N1 > 0:
        print("  Matrix is symmetric (N1=N2). Validating cleanup of X plots...")
        # Check if X plots exist
        plots_dir = result_dir / "plots"
        if plots_dir.exists():
            x_gif = plots_dir / "animation_X.gif"
            if x_gif.exists():
                print("  Note: animation_X.gif exists (could be removed)")


def main():
    root_dir = Path("smf/Replica_results")
    if not root_dir.exists():
        print("No Replica_results directory found.")
        return
        
    # Find all subdirectories
    dirs = [d for d in root_dir.iterdir() if d.is_dir()]
    dirs.sort()
    
    for d in dirs:
        process_experiment(d)


if __name__ == "__main__":
    main()
