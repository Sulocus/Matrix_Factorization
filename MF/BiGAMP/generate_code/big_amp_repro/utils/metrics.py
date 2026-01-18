import numpy as np

def compute_nmse(y_true, y_pred, mask=None):
    """
    Compute Normalized Mean Squared Error in dB.
    NMSE = 10 * log10( ||y_true - y_pred||_F^2 / ||y_true||_F^2 )
    
    Args:
        y_true: Ground truth matrix.
        y_pred: Estimated matrix.
        mask: Optional boolean mask for observed entries.
    """
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    
    mse = np.sum((y_true - y_pred)**2)
    norm_sq = np.sum(y_true**2)
    
    if norm_sq == 0:
        return 0.0
        
    nmse = mse / norm_sq
    return 10 * np.log10(nmse + 1e-20)

def compute_nmae(y_true, y_pred, mask=None):
    """
    Compute Normalized Mean Absolute Error.
    NMAE = sum(|y_true - y_pred|) / sum(|y_true|)
    
    Args:
        y_true: Ground truth matrix.
        y_pred: Estimated matrix.
        mask: Optional boolean mask for observed entries.
    """
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
    mae = np.sum(np.abs(y_true - y_pred))
    norm_abs = np.sum(np.abs(y_true))
    
    if norm_abs == 0:
        return 0.0
        
    return mae / norm_abs

def compute_snr(signal, noise):
    """
    Compute Signal-to-Noise Ratio in dB.
    """
    p_signal = np.mean(signal**2)
    p_noise = np.mean(noise**2)
    if p_noise == 0:
        return np.inf
    return 10 * np.log10(p_signal / p_noise)

def is_success(nmse, threshold=-60):
    """
    Check if recovery is successful based on NMSE threshold.
    """
    return nmse < threshold

def get_phase_transition_stats(results_grid, threshold=-60):
    """
    Convert a grid of NMSE results into a success/failure grid.
    """
    return (results_grid < threshold).astype(float)
