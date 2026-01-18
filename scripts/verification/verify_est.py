
import math
from MF.core.parallel.memory_estimator_general import estimate_general_edge_count
from MF.modules.graphs.supergraph_general import create_supergraph_general
import torch

def verify_memory_estimation():
    N1 = 1000
    N2 = 1000
    M = 100
    alpha = 1.0
    
    print(f"Parameters: N1={N1}, N2={N2}, M={M}, alpha={alpha}")
    
    # 1. Current Estimator Output
    current_est = estimate_general_edge_count(alpha, N1, N2, M)
    print(f"\nCurrent Estimator (estimate_general_edge_count): {current_est:,} edges")
    
    # 2. Analyze the components of current estimator
    bipartite_edges_wrong = int(math.ceil(alpha * N1 * N2 / M))
    dense_floor = N1 * N2
    print(f"  - Wrong Formula (alpha*N1*N2/M): {bipartite_edges_wrong:,}")
    print(f"  - Dense Floor (N1*N2): {dense_floor:,}")
    
    # 3. True Graph Edge Count (from SuperGraph logic)
    # C = floor(alpha * M * N1)
    true_C = int(math.floor(alpha * M * N1))
    print(f"\nTrue Graph Count (alpha * M * N1): {true_C:,} edges")
    
    # 4. Total Possible Edges in General Graph
    N_total = N1 + N2
    total_possible = N_total * (N_total - 1) // 2
    print(f"Total Possible Edges ((N1+N2) choose 2): {total_possible:,}")
    
    # 5. Overestimation Factor
    factor = current_est / true_C
    print(f"\nOverestimation Factor: {factor:.2f}x")
    
    if factor > 5.0:
        print("CONCLUSION: Memory estimation is grossly conservative.")
    else:
        print("CONCLUSION: Memory estimation is reasonable.")

if __name__ == "__main__":
    verify_memory_estimation()
