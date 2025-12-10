import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, "/home/sucia/Sparse-Matrix")

from smf.modules.outputs.plotting import plot_comparison

def load_metrics(result_dir):
    json_path = result_dir / "metrics.json"
    if not json_path.exists():
        print(f"Missing: {json_path}")
        return None
        
    with open(json_path, "r") as f:
        data = json.load(f)
        
    if "results" in data:
        data = data["results"]
        
    # Filter out non-numeric keys
    filtered_data = {}
    for k, v in data.items():
        try:
            float(k)
            filtered_data[k] = v
        except ValueError:
            continue
    return filtered_data

def main():
    root = Path("/home/sucia/Sparse-Matrix/smf/results")
    
    # 1. Baseline (0230 - likely pre-fix or intermediate)
    baseline_dir = root / "bigamp_spreading_parallel_random_200x200_M50_1209_0230"
    
    # 2. Final (0347 - Fixed, Rademacher, 5000 steps)
    final_dir = root / "bigamp_spreading_parallel_random_200x200_M50_1209_0347"
    
    print(f"Loading Baseline: {baseline_dir}")
    baseline_data = load_metrics(baseline_dir)
    
    print(f"Loading Final: {final_dir}")
    final_data = load_metrics(final_dir)
    
    if baseline_data and final_data:
        print("Generating Comparison Plot...")
        plot_comparison(
            results_list=[baseline_data, final_data],
            labels=["Baseline (02:30)", "Final (Fixed)"],
            output_path=final_dir / "comparison_manual.png",
            metric="Q_Y_mean"
        )
        print(f"Comparison plot saved to {final_dir / 'comparison_manual.png'}")
    else:
        print("Failed to load data.")

if __name__ == "__main__":
    main()
