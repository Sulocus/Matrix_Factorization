import sys
from pathlib import Path
import yaml
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, "/home/sucia/Sparse-Matrix")

from smf.core.config import Config, MatrixConfig, AlphaConfig, TrainingConfig, AlgorithmConfig, SpreadingConfig, ExecutionConfig
from smf.runner import run_experiment
from smf.modules.outputs.plotting import plot_comparison

def load_result(result_dir):
    import json
    with open(result_dir / "metrics.json", "r") as f:
        return json.load(f)

def reconstruct_config(yaml_path):
    with open(yaml_path, 'r') as f:
        d = yaml.safe_load(f)
    
    # Reconstruct Config object manually or via mapping
    # Assuming d structure matches Config fields
    matrix = MatrixConfig(**d.get('matrix', {}))
    alpha = AlphaConfig(**d.get('alpha', {}))
    training = TrainingConfig(**d.get('training', {}))
    algorithm = AlgorithmConfig(**d.get('algorithm', {}))
    # algorithm config might need specific handling if flattened in yaml
    
    spreading = SpreadingConfig(**d.get('spreading', {}))
    execution = ExecutionConfig(**d.get('execution', {}))

    return Config(
        matrix=matrix,
        alpha=alpha,
        training=training,
        spreading=spreading,
        execution=execution,
        algorithm=algorithm,
        algorithm_key=d.get('algorithm_key', 'bigamp'),
        graph_key=d.get('graph_key', 'random'),
        teacher_key=d.get('teacher_key', 'standard')
    )

def main():
    # 1. Config Path (Fixed based on user request)
    config_path = Path("/home/sucia/Sparse-Matrix/smf/results/bigamp_spreading_parallel_random_200x200_M50_1209_0230/config.yaml")
    
    print(f"Loading config from {config_path}...")
    config = reconstruct_config(config_path)
    # Ensure SpreadingConfig is set if needed (it might be missing in default reconstruct)
    # The yaml has 'spreading' section
    config.spreading.f_distribution = 'rademacher' # User stacktrace said rademacher
    
    print("Running New Experiment (Fixed Code)...")
    new_result = run_experiment(config)
    new_path = new_result['result_path']
    print(f"New Result Saved to: {new_path}")
    
    # 2. Find Old Result (The one that failed might not have results.json, try to find a previous one)
    results_dir = Path("/home/sucia/Sparse-Matrix/smf/results")
    candidates = sorted([d for d in results_dir.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime)
    
    # Filter for similar config (same M, N1) if possible, or just take the previous one
    old_path = None
    for cand in reversed(candidates[:-1]): # Skip the one we just created
        if (cand / "metrics.json").exists() and "bigamp_spreading_parallel" in cand.name:
             # Basic check if it's relevant
             old_path = cand
             break
    
    if old_path:
        print(f"Found Comparison Baseline: {old_path}")
        old_data = load_result(old_path)
        new_data = load_result(new_path)
        
        # Compare
        print("Generating Comparison Plot...")
        plot_comparison(
            results_list=[old_data, new_data],
            labels=["Old (Baseline)", "New (Fixed)"],
            output_path=new_path / "comparison.png",
            metric="Q_Y_mean"
        )
        print(f"Comparison plot saved to {new_path / 'comparison.png'}")
        
    else:
        print("No valid previous result found for comparison.")

if __name__ == "__main__":
    main()
