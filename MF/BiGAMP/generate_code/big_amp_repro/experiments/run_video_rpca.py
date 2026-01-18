import numpy as np
import matplotlib.pyplot as plt
from big_amp_repro.apps.robust_pca import RobustPCA
from big_amp_repro.utils.metrics import compute_nmse

def generate_synthetic_video(height=128, width=160, num_frames=100, rank=1, sparsity=0.05):
    """
    Generates a synthetic video sequence with a static background and moving objects.
    """
    print(f"Generating synthetic video: {height}x{width}x{num_frames}...")
    
    # Background: Low-rank matrix (Rank 1)
    # Each frame is a slightly different version of the same background or exactly the same
    bg_base = np.random.rand(height, width)
    L_true = np.tile(bg_base.flatten(), (num_frames, 1)).T # (M*L) x T
    
    # Foreground: Sparse moving objects
    S_true = np.zeros((height * width, num_frames))
    for t in range(num_frames):
        # Create a moving square
        obj_size = 10
        r = int((height - obj_size) * (0.5 + 0.4 * np.sin(2 * np.pi * t / num_frames)))
        c = int((width - obj_size) * (0.5 + 0.4 * np.cos(2 * np.pi * t / num_frames)))
        
        frame_s = np.zeros((height, width))
        frame_s[r:r+obj_size, c:c+obj_size] = 1.0
        S_true[:, t] = frame_s.flatten()
        
    # Observations
    Y = L_true + S_true + 0.01 * np.random.randn(*(L_true.shape))
    
    return Y, L_true, S_true, (height, width)

def run_video_rpca_experiment():
    """
    Reproduces the Video RPCA experiment (Fig 11).
    Separates a video into background (low-rank) and foreground (sparse).
    """
    print("--- Starting Video RPCA Experiment ---")
    
    # Parameters
    height, width = 64, 80 # Reduced size for faster execution in reproduction
    num_frames = 50
    rank = 2
    
    # Generate data
    Y, L_true, S_true, dims = generate_synthetic_video(height, width, num_frames, rank=1)
    
    # Initialize Robust PCA
    # Note: RobustPCA app uses the augmented formulation Y = [A I][X; S]^T
    # In our case, Y is (M*L) x T. 
    # The app expects Y to be M x L, but here M=pixels, L=frames.
    rpca = RobustPCA(
        rank=rank, 
        nit=50, 
        tol=1e-4, 
        use_em=True, 
        sparsity_rate=0.05,
        verbose=True
    )
    
    # Fit model
    print("Fitting Robust PCA model...")
    rpca.fit(Y)
    
    # Extract components
    L_hat = rpca.get_low_rank()
    S_hat = rpca.get_sparse()
    
    # Metrics
    nmse_l = compute_nmse(L_true, L_hat)
    nmse_s = compute_nmse(S_true, S_hat)
    
    print(f"Background NMSE: {nmse_l:.2f} dB")
    print(f"Foreground NMSE: {nmse_s:.2f} dB")
    
    # Visualization
    frame_idx = num_frames // 2
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # True components
    axes[0, 0].imshow(Y[:, frame_idx].reshape(dims), cmap='gray')
    axes[0, 0].set_title(f"Original Frame {frame_idx}")
    
    axes[0, 1].imshow(L_true[:, frame_idx].reshape(dims), cmap='gray')
    axes[0, 1].set_title("True Background")
    
    axes[0, 2].imshow(S_true[:, frame_idx].reshape(dims), cmap='gray')
    axes[0, 2].set_title("True Foreground")
    
    # Recovered components
    axes[1, 0].imshow(Y[:, frame_idx].reshape(dims), cmap='gray')
    axes[1, 0].set_title("Observation")
    
    axes[1, 1].imshow(L_hat[:, frame_idx].reshape(dims), cmap='gray')
    axes[1, 1].set_title("Recovered Background")
    
    axes[1, 2].imshow(S_hat[:, frame_idx].reshape(dims), cmap='gray')
    axes[1, 2].set_title("Recovered Foreground")
    
    for ax in axes.flatten():
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig("video_rpca_results.png")
    print("Results saved to video_rpca_results.png")
    plt.close()

if __name__ == "__main__":
    run_video_rpca_experiment()
