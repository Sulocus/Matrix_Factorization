
import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path.cwd()))

from MF.core.experiment.config import ExperimentConfig, MatrixParams, TrainingParams, SpreadingConfig
from MF.modules.algorithms.bigamp_spreading import BiGAMPSpreading

def debug_graph_structure():
    print("--- Debugging Graph Structure ---")
    
    # 1. Setup Config
    N1 = 200
    N2 = 200
    M = 50
    # Use config that enables general graph
    config = ExperimentConfig(
        matrix=MatrixParams(N1=N1, N2=N2, M=M),
        training=TrainingParams(samples_per_alpha=1, max_steps=1),
        algorithm_key="bigamp_spreading",
        scan=None, # Not needed for this test
        spreading=SpreadingConfig(
            f_distribution="rademacher",
            allow_intra_connection=True, # ALERT: Testing this flag!
            onsager_correction=False,
        ),
        experiment_name="debug_graph"
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 2. Instantiate Algorithm
    algo = BiGAMPSpreading(config, device)
    
    # 3. Create dummy teacher matrices
    W_teacher = torch.randn(N1, M, device=device)
    X_teacher = torch.randn(M, N2, device=device)
    alpha_values = [2.0] # High alpha to get many edges
    
    # 4. Generate Spreading Data
    print("Generating spreading data...")
    spreading_data = algo.create_spreading_data(
        W_teacher, X_teacher, alpha_values, S=1, base_seed=42
    )
    
    # 5. Inspect Graph
    if hasattr(spreading_data, 'supergraph') and hasattr(spreading_data.supergraph, 'edge_type'):
        sg = spreading_data.supergraph
        edge_type = sg.edge_type[0] # Sample 0
        
        counts = torch.bincount(edge_type.long(), minlength=3)
        total = counts.sum().item()
        
        print("\n[Edge Distribution]")
        print(f"Total Edges: {total}")
        print(f"W-W (Type 0): {counts[0].item()} ({counts[0].item()/total*100:.1f}%)")
        print(f"W-X (Type 1): {counts[1].item()} ({counts[1].item()/total*100:.1f}%)")
        print(f"X-X (Type 2): {counts[2].item()} ({counts[2].item()/total*100:.1f}%)")
        
        if counts[0] > 0 or counts[2] > 0:
            print("\n✅ SUCCESS: Intra-connections detected!")
        else:
            print("\n❌ FAILURE: Only W-X connections found (Bipartite structure)!")
            
        # Verify Node Indices
        a = sg.a_idx[0]
        b = sg.b_idx[0]
        max_idx = max(a.max().item(), b.max().item())
        print(f"\nMax Node Index: {max_idx} (Expected < {N1+N2})")
        
    else:
        print("\n❌ FAILURE: `edge_type` not found or incorrect data structure.")
        print(f"Data type: {type(spreading_data)}")

if __name__ == "__main__":
    debug_graph_structure()
