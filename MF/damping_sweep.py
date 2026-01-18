#!/usr/bin/env python3
"""
Damping Parameter Sweep - Fixed metrics extraction.
"""
import subprocess
import re
import time
import torch
from pathlib import Path
import numpy as np

DAMPING_VALUES = [round(0.1 + i * 0.05, 2) for i in range(9)]
CONFIG_PATH = Path("/home/sucia/Sparse-Matrix/smf/config.yaml")

def update_damping(damping_value: float):
    with open(CONFIG_PATH, 'r') as f:
        content = f.read()
    content = re.sub(
        r'damping: [0-9.]+.*$',
        f'damping: {damping_value}  # SWEEP',
        content,
        flags=re.MULTILINE
    )
    with open(CONFIG_PATH, 'w') as f:
        f.write(content)

def run_experiment():
    start_time = time.time()
    result = subprocess.run(
        ['smf'],
        cwd='/home/sucia/Sparse-Matrix',
        capture_output=True,
        text=True
    )
    elapsed = time.time() - start_time
    
    output = result.stdout + result.stderr
    match = re.search(r'Done! Saved to: (smf/Replica_results/alpha_scan/[^\s]+)', output)
    result_path = None
    if match:
        result_path = Path("/home/sucia/Sparse-Matrix") / match.group(1)
    
    return result_path, elapsed

def analyze_results(result_path: Path):
    """Extract variance from results.pt - FIXED"""
    results_file = result_path / "results.pt"
    if not results_file.exists():
        return None
    
    try:
        data = torch.load(results_file, weights_only=False)
        metrics = data.get('metrics', {})
        
        stds = []
        means = []
        
        for alpha_str, alpha_metrics in metrics.items():
            if isinstance(alpha_metrics, dict):
                # Use physical_overlap_Y_std as main metric
                if 'physical_overlap_Y_std' in alpha_metrics:
                    stds.append(float(alpha_metrics['physical_overlap_Y_std']))
                if 'physical_overlap_Y_mean' in alpha_metrics:
                    means.append(float(alpha_metrics['physical_overlap_Y_mean']))
        
        if stds:
            return {
                'mean_std': float(np.mean(stds)),
                'max_std': float(np.max(stds)),
                'mean_overlap': float(np.mean(means)) if means else None
            }
    except Exception as e:
        print(f"  Error: {e}")
    
    return None

def main():
    print("=" * 70)
    print("Damping Sweep (Fixed)")
    print(f"Testing: {DAMPING_VALUES}")
    print("=" * 70)
    
    results = []
    
    for i, damping in enumerate(DAMPING_VALUES):
        print(f"\n[{i+1}/{len(DAMPING_VALUES)}] damping = {damping:.2f}")
        
        update_damping(damping)
        result_path, elapsed = run_experiment()
        
        metrics = None
        if result_path and result_path.exists():
            metrics = analyze_results(result_path)
        
        results.append({
            'damping': damping,
            'elapsed': elapsed,
            'metrics': metrics
        })
        
        if metrics:
            print(f"  ✓ {elapsed:.1f}s | mean_std={metrics['mean_std']:.4f} | max_std={metrics['max_std']:.4f}")
        else:
            print(f"  ✗ {elapsed:.1f}s | metrics failed")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Damping':<10} {'Time':<8} {'Mean Std':<12} {'Max Std':<12} {'Score':<10}")
    print("-" * 52)
    
    best = None
    best_score = float('inf')
    
    for r in results:
        d = r['damping']
        t = r['elapsed']
        m = r['metrics']
        
        if m:
            ms, xs = m['mean_std'], m['max_std']
            score = ms + 0.2 * xs
            print(f"{d:<10.2f} {t:<8.1f} {ms:<12.4f} {xs:<12.4f} {score:<10.4f}")
            if score < best_score:
                best_score = score
                best = d
        else:
            print(f"{d:<10.2f} {t:<8.1f} {'N/A':<12} {'N/A':<12}")
    
    print("-" * 52)
    if best:
        print(f"\n🏆 BEST DAMPING: {best:.2f} (score: {best_score:.4f})")
    
    import json
    with open('/home/sucia/Sparse-Matrix/smf/damping_sweep_final.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

if __name__ == '__main__':
    main()
