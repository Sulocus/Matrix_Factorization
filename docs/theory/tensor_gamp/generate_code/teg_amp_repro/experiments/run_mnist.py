import os
import numpy as np
import matplotlib.pyplot as plt
from ..data.mnist_loader import prepare_mnist_experiment
from ..solvers.teg_amp import TeGAMPSolver
from ..solvers.altmin_baseline import AltMinTR
from ..utils.priors import GaussianPrior
from ..utils.channels import AWGNChannel
from ..utils.tensor_math import nmse

def run_mnist_experiment(sampling_rate=0.4, tr_ranks=(14, 14, 6), save_path='results'):
    """
    Reproduces Figure 3: MNIST Tensor Completion.
    Stacks 6 MNIST digits into a 28x28x6 tensor and recovers it from 40% samples.
    """
    print(f"--- Starting MNIST Experiment (Sampling Rate: {sampling_rate}) ---")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 1. Prepare Data
    u_true, y, mask = prepare_mnist_experiment(sampling_rate=sampling_rate, seed=42)
    shape = u_true.shape
    
    # 2. Run TeG-AMP
    print("Running TeG-AMP...")
    prior = GaussianPrior(mean=0.0, var=1.0)
    channel = AWGNChannel(noise_var=1e-6, estimate_noise_var=True)
    
    teg_solver = TeGAMPSolver(
        shape=shape,
        ranks=tr_ranks,
        prior=prior,
        channel=channel,
        max_iter=100,
        tol=1e-5,
        damping_beta=0.5
    )
    
    u_teg, teg_history = teg_solver.solve(y, mask, ground_truth=u_true)
    teg_nmse = nmse(u_true, u_teg)
    print(f"TeG-AMP NMSE: {teg_nmse:.4e}")

    # 3. Run AltMin (Baseline)
    print("Running AltMin (ALS) Baseline...")
    altmin_solver = AltMinTR(
        shape=shape,
        ranks=tr_ranks,
        max_iter=30,
        tol=1e-5
    )
    u_altmin, altmin_history = altmin_solver.solve(y, mask, ground_truth=u_true)
    altmin_nmse = nmse(u_true, u_altmin)
    print(f"AltMin NMSE: {altmin_nmse:.4e}")

    # 4. Visualization
    fig, axes = plt.subplots(3, shape[2], figsize=(15, 8))
    
    for i in range(shape[2]):
        # Original
        axes[0, i].imshow(u_true[:, :, i], cmap='gray')
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title("Original")
        
        # Masked (Observations)
        obs_viz = np.copy(y[:, :, i])
        obs_viz[mask[:, :, i] == 0] = np.nan
        axes[1, i].imshow(obs_viz, cmap='gray')
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title(f"Observed ({int(sampling_rate*100)}%)")
        
        # TeG-AMP Reconstruction
        axes[2, i].imshow(u_teg[:, :, i], cmap='gray')
        axes[2, i].axis('off')
        if i == 0: axes[2, i].set_title(f"TeG-AMP (NMSE: {teg_nmse:.2e})")

    plt.tight_layout()
    plot_file = os.path.join(save_path, 'mnist_reconstruction.png')
    plt.savefig(plot_file)
    print(f"Reconstruction plot saved to {plot_file}")
    
    # Convergence Plot
    plt.figure(figsize=(8, 5))
    plt.semilogy([h['nmse'] for h in teg_history], label='TeG-AMP NMSE')
    plt.semilogy([h['nmse'] for h in altmin_history], label='AltMin NMSE')
    plt.xlabel('Iteration')
    plt.ylabel('NMSE')
    plt.title('MNIST Completion Convergence')
    plt.legend()
    plt.grid(True)
    conv_file = os.path.join(save_path, 'mnist_convergence.png')
    plt.savefig(conv_file)
    print(f"Convergence plot saved to {conv_file}")

    return {
        'teg_nmse': teg_nmse,
        'altmin_nmse': altmin_nmse,
        'u_teg': u_teg,
        'u_altmin': u_altmin
    }

if __name__ == "__main__":
    run_mnist_experiment()
