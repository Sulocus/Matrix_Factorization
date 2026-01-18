import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import matplotlib.cm as cm
import numpy as np

def main():
    # File Path
    data_path = Path("smf/results/debug_damping.pt")
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Run an adaptive damping experiment first to generate data.")
        sys.exit(1)

    # Load Data
    print(f"Loading data from {data_path}...")
    try:
        data = torch.load(data_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)
        
    alpha_values = data.get('alpha_values')
    steps = data.get('steps')
    history = data.get('damping_history') # Shape (Steps, B)
    timestamp = data.get('timestamp', 'Unknown Time')

    if history is None or alpha_values is None:
        print("Error: Invalid data format (missing keys).")
        sys.exit(1)
        
    # Convert to numpy
    steps_np = steps.numpy()
    history_np = history.numpy()
    
    num_alphas = len(alpha_values)
    print(f"Found {num_alphas} alphas, {len(steps_np)} steps.")
    print(f"Alphas: {alpha_values}")

    # Plotting
    plt.figure(figsize=(12, 7))
    plt.title(f"Adaptive Damping Evolution (Run: {timestamp})", fontsize=14)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Damping Value", fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Use colormap
    colors = cm.viridis(np.linspace(0, 1, num_alphas))
    
    for i in range(num_alphas):
        alpha = alpha_values[i]
        damp_curve = history_np[:, i]
        plt.plot(steps_np, damp_curve, label=f"α={alpha:.2f}", color=colors[i], linewidth=1.5, alpha=0.8)
        
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Alpha")
    plt.tight_layout()
    
    # Save
    out_path = data_path.with_suffix('.png')
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to: {out_path}")
    print("Done.")

if __name__ == "__main__":
    main()
