
import torch
import sys

def inspect_results(path):
    try:
        data = torch.load(path, map_location='cpu')
        print(f"Loaded keys: {data.keys()}")
        
        # Depending on structure, print relevant metrics
        # Usually 'results' is a list of dicts or a dict of lists


        if 'metrics' in data:
            metrics = data['metrics']
            # metrics is dict of {alpha_str: {metric_name: value}}
            if isinstance(metrics, dict):
                first_alpha = list(metrics.keys())[0] # e.g. '0.0'
                print(f"Inspecting Alpha {first_alpha}:")
                alpha_metrics = metrics[first_alpha]
                
                # Check metrics inside first alpha
                for key in alpha_metrics:
                    if 'physical' in key or 'overlap' in key or 'Q_' in key:
                         val = alpha_metrics[key]
                         print(f"  {key}: {val}")


        if 'results' in data:
            results = data['results']
            # Accessing scalar metrics if available
            # It seems results might be a list of trials, or aggregated
            
            # Printing first few entries
            if isinstance(results, list):
                print(f"Results is a list of len {len(results)}")
                sample = results[0]  # First alpha or first trial
                print(f"Sample keys: {sample.keys()}")
                
                # Check for physical overlap
                for key in sample:
                    if 'physical' in key and 'overlap' in key:
                        print(f"{key}: {sample[key]}")
                        
            elif isinstance(results, dict):
                 print(f"Results is a dict with keys: {results.keys()}")
                 for key in results:
                     if 'physical' in key or 'overlap' in key:
                         print(f"{key}: {results[key]}")
        
    except Exception as e:
        print(f"Error loading: {e}")

if __name__ == "__main__":
    inspect_results("/home/sucia/Sparse-Matrix/smf/Replica_results/hysteresis_scan/20260109_0955_hysteresis_placeholder/Warm_Start_m1.0/results.pt")
