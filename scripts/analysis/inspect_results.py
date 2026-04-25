
import torch
import sys
import os

base_dir = os.environ.get("MF_REPLICA_RESULTS", "src/matrix_factorization/Replica_results")
path = f"{base_dir}/alpha_scan/20251225_1848_bigamp_spreading_standard_200x200_M50/results.pt"
try:
    data = torch.load(path)
    print("Keys:", data.keys())
    if 'metrics' in data:
        metrics = data['metrics']
        # Check alpha 2.0 (last one)
        keys = list(metrics.keys())
        last_key = keys[-1]
        print(f"\nMetrics for Alpha={last_key}:")
        m = metrics[last_key]
        for k in ['Q_Y_mean', 'Q_Y_observed_mean', 'Q_W_mean', 'Q_X_mean']:
            if k in m:
                print(f"{k}: {m[k]}")
            else:
                print(f"{k}: MISSING")
except Exception as e:
    print(f"Error: {e}")
