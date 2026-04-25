
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

REPLICA_RESULTS_DIR = os.environ.get(
    "MF_REPLICA_RESULTS",
    "src/matrix_factorization/Replica_results",
)

def compare(dir1, dir2):
    p1 = f"{REPLICA_RESULTS_DIR}/alpha_scan/{dir1}/results.pt"
    p2 = f"{REPLICA_RESULTS_DIR}/alpha_scan/{dir2}/results.pt"
    
    print(f"Loading Run 1: {dir1}")
    try:
        r1 = torch.load(p1, map_location='cpu')
    except Exception as e:
        print(f"Failed to load Run 1: {e}")
        return

    print(f"Loading Run 2: {dir2}")
    try:
        r2 = torch.load(p2, map_location='cpu')
    except Exception as e:
        print(f"Failed to load Run 2: {e}")
        return
    
    # Extract metrics
    # ExperimentResult.results is a Dict[float, SingleRunResult]
    print(f"Run 1 Type: {type(r1)}")
    if isinstance(r1, dict):
        # Maybe it's the internal __dict__ or custom dict save?
        # Check keys
        print(f"Run 1 Keys: {list(r1.keys())}")
        # If it's a dict, adjust access
        alphas = sorted(list(r1['results'].keys()))
    else:
        alphas = sorted(list(r1.results.keys()))
    
    print("\n=== Comparison: Q_W_mean (Overlap) ===")
    print(f"{'Alpha':<10} | {'Gaussian':<10} | {'Rademacher':<10} | {'Diff':<10}")
    print("-" * 50)
    
    
    
    diffs = []

    def get_metrics_map(r):
        # Handle dict format from result.save()
        if isinstance(r, dict):
            if 'metrics' in r:
                return r['metrics']
            elif 'results' in r: # maybe unified format
                return {k: v['metrics'] for k, v in r['results'].items()}
        # Handle object
        if hasattr(r, 'results'):
             return {k: v.metrics for k, v in r.results.items()}
        return {}

    metrics1 = get_metrics_map(r1)
    metrics2 = get_metrics_map(r2)
    
    # Get alphas from metrics keys
    alphas = sorted(list(metrics1.keys()), key=lambda x: float(x))
    
    for alpha in alphas:
        if alpha in metrics2:
            m1_map = metrics1[alpha]
            m2_map = metrics2[alpha]
            
            # Key might be Q_W_mean or physical_overlap_W_mean
            key = 'Q_W_mean'
            if 'physical_overlap_W_mean' in m1_map:
                key = 'physical_overlap_W_mean'
            
            val1 = m1_map.get(key, 0.0)
            val2 = m2_map.get(key, 0.0)

            diff = val2 - val1
            diffs.append(abs(diff))
            
            # Print only significant points or spaced out
            if abs(diff) > 0.02 or (alphas.index(alpha) % 5 == 0):
                print(f"{float(alpha):<10.2f} | {val1:<10.4f} | {val2:<10.4f} | {diff:<10.4f}")
                
    avg_diff = sum(diffs) / len(diffs) if diffs else 0
    print("-" * 50)
    print(f"Average Absolute Difference: {avg_diff:.6f}")
    
    if avg_diff > 0.01:
        print("\n✅ VERIFIED: Changing teacher distribution CHANGED the result.")
    else:
        print("\n❌ PROBLEM: Results are still effectively identical.")

if __name__ == "__main__":
    compare(
        "20260108_0554_bigamp_spreading_standard_200x200_M50_general",
        "20260108_0603_bigamp_spreading_standard_200x200_M50_general"
    )
