
import torch
import numpy as np
from smf.modules.graphs.supergraph import create_supergraph
from smf.modules.graphs.supergraph_general import create_supergraph_general
from smf.modules.parallel.memory_estimator import MemoryEstimator, EstimationParams

def test_counts():
    N1 = 200
    N2 = 200
    M = 50
    alpha_max = 3.0
    alpha_values = np.linspace(0, 3.0, 41).tolist()
    S = 50
    base_seed = 12345
    device = torch.device('cpu') # Use CPU for size check

    print(f"Params: N={N1}, M={M}, S={S}, AlphaMax={alpha_max}")

    print("\n--- Bipartite (SuperGraph) ---")
    sg_bip = create_supergraph(N1, N2, M, alpha_values, S, base_seed, device)
    print(f"C_max: {sg_bip.C_max}")
    print(f"i_idx shape: {sg_bip.i_idx.shape}")
    print(f"Total Edges (S*C): {sg_bip.i_idx.numel()}")
    
    # Check what Estimator thinks
    est_bip = MemoryEstimator()
    params_bip = EstimationParams(
        N1=N1, N2=N2, M=M, S=S, alpha_max=alpha_max,
        algorithm_key='bigamp_spreading',
        batch_size=41,
        use_compile=False, use_bf16=False,
        f_distribution='gaussian', adaptive_damping=False
    )
    # Hack: Inject Bipartite C_max into estimator via patched class or just check logic?
    # Estimator logic is hardcoded. Let's see what logic gives.
    import math
    est_c_max = max(1, int(math.ceil(alpha_max * N1 * N2 / M)))
    print(f"Estimator Formula (Old/Reverted): {est_c_max}")
    est_c_max_new = max(1, int(math.ceil(alpha_max * M * N1)))
    print(f"Estimator Formula (New/General): {est_c_max_new}")


    print("\n--- General (SuperGraphGeneral) ---")
    sg_gen = create_supergraph_general(N1, N2, M, alpha_values, S, base_seed, device)
    print(f"C_max: {sg_gen.C_max}")
    print(f"a_idx shape: {sg_gen.a_idx.shape}")
    print(f"Total Edges (S*C): {sg_gen.a_idx.numel()}")

if __name__ == "__main__":
    test_counts()
