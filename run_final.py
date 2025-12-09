import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, "/home/sucia/Sparse-Matrix")

from smf.core.config import Config, MatrixConfig, AlphaConfig, TrainingConfig, AlgorithmConfig, SpreadingConfig, ExecutionConfig
from smf.runner import run_experiment
from smf.modules.outputs.plotting import plot_comparison

def load_metrics(result_dir):
    """Load metrics.json and filter for alpha keys only."""
    json_path = result_dir / "metrics.json"
    if not json_path.exists():
        return None
        
    with open(json_path, "r") as f:
        data = json.load(f)
        
    # Filter out non-numeric keys (like 'config', 'metadata', etc.)
    filtered_data = {}
    for k, v in data.items():
        try:
            float(k)
            filtered_data[k] = v
        except ValueError:
            continue
            
    return filtered_data

def main():
    # 1. Define Config exactly as requested
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
    
    print("Starting Final Experiment with User Parameters...")
    print(f"Matrix: 200x200, M=50, S=4, Alpha: 0.0-4.0")
    
    result = run_experiment(config)
    new_path = result['result_path']
    print(f"Final Result Saved to: {new_path}")
    
    # 2. Find Baseline (Use the most recent *successful* run that is NOT the current one)
    # We want to compare against the 'broken' or 'different' version if possible, 
    # but simplest is just to compare against the immediately preceding valid run.
    results_dir = Path("/home/sucia/Sparse-Matrix/smf/results")
    candidates = sorted([d for d in results_dir.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime)
    
    old_path = None
    old_data = None
    
    # Iterate backwards, skipping the current new_path
    for cand in reversed(candidates):
        if cand == new_path:
            continue
            
        metrics = load_metrics(cand)
        if metrics and len(metrics) > 0:
            # Found a valid one
            old_path = cand
            old_data = metrics
            break
            
    if old_path and old_data:
        print(f"Found Comparison Baseline: {old_path}")
        new_data = load_metrics(new_path)
        
        print("Generating Comparison Plot...")
        try:
            plot_comparison(
                results_list=[old_data, new_data],
                labels=["Previous", "Current (Req Params)"],
                output_path=new_path / "comparison_final.png",
                metric="Q_Y_mean"
            )
            print(f"Comparison plot saved to {new_path / 'comparison_final.png'}")
        except Exception as e:
            print(f"Plotting failed: {e}")
            import traceback
            traceback.print_exc()
        
    else:
        print("No valid previous result found for comparison.")

if __name__ == "__main__":
    main()
