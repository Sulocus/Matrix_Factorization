import numpy as np
import matplotlib.pyplot as plt
import time
from big_amp_repro.apps.matrix_completion import MatrixCompletion
from big_amp_repro.utils.metrics import compute_nmse, is_success

def run_synthetic_ptc():
    """
    Reproduces Figure 2 (Phase Transitions) for Matrix Completion.
    Varies sampling ratio (delta = |Omega| / (M*L)) and rank ratio (rho = N(M+L-N) / |Omega|).
    """
    print("Starting Synthetic Phase Transition Experiment (Fig 2)...")
    
    # Dimensions
    M, L = 100, 100
    
    # Grid for Phase Transition
    # delta: sampling ratio |Omega| / (M*L)
    # rho: rank ratio N(M+L-N) / |Omega|
    deltas = np.linspace(0.1, 0.9, 10)
    rhos = np.linspace(0.1, 0.9, 10)
    
    results = np.zeros((len(rhos), len(deltas)))
    
    for j, delta in enumerate(deltas):
        num_obs = int(delta * M * L)
        for i, rho in enumerate(rhos):
            # Calculate rank N such that rho = N(M+L-N) / num_obs
            # N^2 - (M+L)N + rho*num_obs = 0
            # N = [(M+L) - sqrt((M+L)^2 - 4*rho*num_obs)] / 2
            a_coeff = 1
            b_coeff = -(M + L)
            c_coeff = rho * num_obs
            discriminant = b_coeff**2 - 4 * a_coeff * c_coeff
            if discriminant < 0:
                results[i, j] = -10 # Failure
                continue
            
            N = int(0.5 * (-b_coeff - np.sqrt(discriminant)))
            if N < 1: N = 1
            
            print(f"Testing delta={delta:.2f}, rho={rho:.2f} (Rank N={N}, Obs={num_obs})")
            
            # Generate True Matrices
            A_true = np.random.randn(M, N)
            X_true = np.random.randn(N, L)
            Y_true = A_true @ X_true
            
            # Generate Mask
            mask = np.zeros(M * L, dtype=bool)
            mask[:num_obs] = True
            np.random.shuffle(mask)
            mask = mask.reshape(M, L)
            
            # Observations (Noise-free for PTC)
            Y_obs = Y_true.copy()
            Y_obs[~mask] = 0
            
            # Run BiG-AMP
            mc = MatrixCompletion(rank=N, nit=200, tol=1e-6, use_em=True, verbose=False)
            mc.fit(Y_obs, mask=mask)
            
            # Evaluate
            nmse = mc.get_metrics(Y_true, mask_test=~mask)['nmse']
            results[i, j] = nmse
            
            success = is_success(nmse, threshold=-60)
            print(f"  NMSE: {nmse:.2f} dB - {'SUCCESS' if success else 'FAILURE'}")

    # Plotting
    plt.figure(figsize=(8, 6))
    # Success is NMSE < -60dB
    success_grid = (results < -60).astype(float)
    
    plt.imshow(success_grid, extent=[deltas[0], deltas[-1], rhos[0], rhos[-1]], 
               origin='lower', aspect='auto', cmap='gray')
    plt.colorbar(label='Success (1) / Failure (0)')
    plt.xlabel('Sampling Ratio ($\delta = |\Omega|/ML$)')
    plt.ylabel('Rank Ratio ($\\rho = N(M+L-N)/|\Omega|$)')
    plt.title('BiG-AMP Phase Transition (Matrix Completion)')
    
    # Theoretical Boundary (Section VI-A)
    # For large M, L, the boundary is rho = 1
    plt.axhline(y=1.0, color='r', linestyle='--', label='Theoretical Boundary')
    
    plt.savefig('synthetic_ptc.png')
    print("Phase Transition plot saved to synthetic_ptc.png")
    plt.close()

if __name__ == "__main__":
    run_synthetic_ptc()
