import numpy as np
import matplotlib.pyplot as plt
import os
from ..solvers import TeGAMPSolver, TeSAMPSolver, AltMinTR
from ..data import generate_cp_data
from ..utils.priors import GaussianPrior
from ..utils.channels import AWGNChannel
from ..utils.tensor_math import nmse

def run_snr_experiment(trials=5, save_path='results'):
    """
    Reproduces SNR robustness comparisons.
    Compares TeG-AMP and TeS-AMP on CP-rank tensors (where both are applicable).
    This relates to Experiment 4 (CP-rank recovery) and general SNR robustness.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    snr_levels = [10, 20, 30, 40, 50, 60]
    shape = (8, 8, 8)
    rank = 2
    sampling_rate = 0.5

    results = {
        'TeG-AMP': [],
        'TeS-AMP': [],
        'AltMin': []
    }

    print(f"Starting SNR robustness experiment (Trials: {trials})...")

    for snr in snr_levels:
        print(f"  Testing SNR: {snr} dB")
        trial_nmse_teg = []
        trial_nmse_tes = []
        trial_nmse_alt = []

        for t in range(trials):
            # Generate CP data
            data = generate_cp_data(shape, rank, sampling_rate=sampling_rate, snr_db=snr, seed=42+t+int(snr))
            y, mask, u_true = data['y'], data['mask'], data['u_true']

            # 1. TeS-AMP (CP specific)
            prior_tes = GaussianPrior(mean=0.0, var=1.0)
            channel_tes = AWGNChannel(noise_var=data['noise_var'], estimate_noise_var=True)
            solver_tes = TeSAMPSolver(shape, rank, prior_tes, channel_tes, max_iter=100, damping_beta=0.5)
            u_tes, _ = solver_tes.solve(y, mask)
            trial_nmse_tes.append(nmse(u_true, u_tes))

            # 2. TeG-AMP (TR ranks [rank, rank, rank] for CP-like structure)
            # Note: A CP tensor of rank R can be represented as a TR tensor of ranks [R, R, R]
            prior_teg = GaussianPrior(mean=0.0, var=1.0)
            channel_teg = AWGNChannel(noise_var=data['noise_var'], estimate_noise_var=True)
            solver_teg = TeGAMPSolver(shape, [rank, rank, rank], prior_teg, channel_teg, max_iter=100, damping_beta=0.5)
            u_teg, _ = solver_teg.solve(y, mask)
            trial_nmse_teg.append(nmse(u_true, u_teg))

            # 3. AltMin (TR)
            solver_alt = AltMinTR(shape, [rank, rank, rank], max_iter=30)
            u_alt, _ = solver_alt.solve(y, mask)
            trial_nmse_alt.append(nmse(u_true, u_alt))

        results['TeG-AMP'].append(np.median(trial_nmse_teg))
        results['TeS-AMP'].append(np.median(trial_nmse_tes))
        results['AltMin'].append(np.median(trial_nmse_alt))

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.semilogy(snr_levels, results['TeG-AMP'], 'o-', label='TeG-AMP (TR-rank)')
    plt.semilogy(snr_levels, results['TeS-AMP'], 's--', label='TeS-AMP (CP-rank)')
    plt.semilogy(snr_levels, results['AltMin'], 'd-.', label='AltMin (TR)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Median NMSE')
    plt.title(f'SNR Robustness on CP-rank Tensors (Sampling Rate: {sampling_rate})')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    plot_file = os.path.join(save_path, 'snr_robustness.png')
    plt.savefig(plot_file)
    print(f"SNR experiment plot saved to {plot_file}")
    
    return results

if __name__ == "__main__":
    run_snr_experiment(trials=3)
