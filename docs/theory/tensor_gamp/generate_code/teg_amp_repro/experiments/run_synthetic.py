import numpy as np
import matplotlib.pyplot as plt
import os
import time
from ..solvers import TeGAMPSolver, AltMinTR
from ..data import generate_tr_data
from ..utils.tensor_math import nmse
from ..utils.priors import GaussianPrior
from ..utils.channels import AWGNChannel

def run_synthetic_experiment(trials=10, save_path='results'):
    """
    Reproduces Figure 2: Recovery vs Sampling Rate for TR-rank [2,2,2] and [2,3,3].
    Note: Default trials reduced to 10 for faster execution in reproduction, 
    paper uses 50.
    """
    print("Starting Synthetic Phase Transition Experiment (Figure 2)...")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    shape = (6, 7, 8)
    rank_configs = [(2, 2, 2), (2, 3, 3)]
    sampling_rates = np.linspace(0.1, 0.9, 9)
    
    results = {str(rank): {'teg_amp': [], 'altmin': []} for rank in rank_configs}

    for ranks in rank_configs:
        print(f"\nTesting TR-ranks: {ranks}")
        for sr in sampling_rates:
            teg_errors = []
            altmin_errors = []
            
            print(f"  Sampling Rate: {sr:.1f}", end=" ", flush=True)
            
            for t in range(trials):
                # Generate data
                data = generate_tr_data(shape, ranks, sampling_rate=sr, snr_db=100, seed=t)
                u_true = data['u_true']
                y = data['y']
                mask = data['mask']
                
                # 1. TeG-AMP Solver
                prior = GaussianPrior(mean=0.0, var=1.0)
                channel = AWGNChannel(noise_var=1e-6, estimate_noise_var=True)
                solver = TeGAMPSolver(
                    shape=shape, 
                    ranks=ranks, 
                    prior=prior, 
                    channel=channel, 
                    max_iter=100, 
                    damping_beta=0.5
                )
                
                u_hat_teg, _ = solver.solve(y, mask, ground_truth=u_true)
                teg_errors.append(nmse(u_true, u_hat_teg))
                
                # 2. AltMin (ALS) Baseline
                altmin_solver = AltMinTR(shape=shape, ranks=ranks, max_iter=50)
                u_hat_alt, _ = altmin_solver.solve(y, mask, ground_truth=u_true)
                altmin_errors.append(nmse(u_true, u_hat_alt))
                
                print(".", end="", flush=True)
            
            avg_teg = np.median(teg_errors)
            avg_altmin = np.median(altmin_errors)
            results[str(ranks)]['teg_amp'].append(avg_teg)
            results[str(ranks)]['altmin'].append(avg_altmin)
            print(f" Done. TeG-AMP NMSE: {avg_teg:.2e}, AltMin NMSE: {avg_altmin:.2e}")

    # Plotting
    plt.figure(figsize=(10, 6))
    colors = ['b', 'r']
    markers = ['o', 's']
    
    for i, ranks in enumerate(rank_configs):
        label_rank = f"TR-rank {ranks}"
        plt.semilogy(sampling_rates, results[str(ranks)]['teg_amp'], 
                     label=f"TeG-AMP ({label_rank})", 
                     color=colors[i], marker=markers[i], linestyle='-')
        plt.semilogy(sampling_rates, results[str(ranks)]['altmin'], 
                     label=f"AltMin ({label_rank})", 
                     color=colors[i], marker=markers[i], linestyle='--')

    plt.xlabel('Sampling Rate')
    plt.ylabel('Median NMSE')
    plt.title('Synthetic Tensor Recovery (Figure 2 Reproduction)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    plot_file = os.path.join(save_path, 'synthetic_phase_transition.png')
    plt.savefig(plot_file)
    print(f"\nExperiment complete. Plot saved to {plot_file}")
    
    return results

if __name__ == "__main__":
    run_synthetic_experiment(trials=5)
