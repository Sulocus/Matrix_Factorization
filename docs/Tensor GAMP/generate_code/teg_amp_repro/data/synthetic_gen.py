import numpy as np
from ..models.tensor_ring import TensorRingModel
from ..models.cp_model import CPModel

def generate_tr_data(shape, ranks, sampling_rate=1.0, snr_db=None, seed=None):
    """
    Generates a synthetic Tensor Ring (TR) tensor, applies a sampling mask, and adds AWGN.
    
    Args:
        shape (tuple): Dimensions of the tensor.
        ranks (list): TR-ranks.
        sampling_rate (float): Fraction of observed entries (0.0 to 1.0).
        snr_db (float, optional): Signal-to-Noise Ratio in dB. If None, no noise is added.
        seed (int, optional): Random seed for reproducibility.
        
    Returns:
        dict: Contains 'u_true' (ground truth), 'y' (noisy masked observations), 
              'mask' (binary mask), and 'noise_var' (actual noise variance).
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 1. Generate ground truth TR cores and contract
    tr_model = TensorRingModel(shape, ranks)
    # Standard normal initialization for cores as per common synthetic benchmarks
    tr_model.initialize_cores(method='gaussian', scale=1.0)
    u_true = tr_model.get_full_tensor()
    
    # 2. Generate sampling mask
    mask = (np.random.rand(*shape) < sampling_rate).astype(float)
    
    # 3. Add noise based on SNR
    y = u_true.copy()
    noise_var = 0.0
    if snr_db is not None:
        # P_signal = E[u^2]
        signal_pwr = np.mean(u_true**2)
        # SNR = P_signal / P_noise => P_noise = P_signal / 10^(SNR_dB/10)
        noise_var = signal_pwr / (10**(snr_db / 10.0))
        noise = np.random.normal(0, np.sqrt(noise_var), shape)
        y += noise
    else:
        # If no SNR provided, we can still have a tiny epsilon noise or pure noiseless
        noise_var = 0.0
        
    y_masked = y * mask
    
    return {
        'u_true': u_true,
        'y': y_masked,
        'mask': mask,
        'noise_var': noise_var
    }

def generate_cp_data(shape, rank, sampling_rate=1.0, snr_db=None, seed=None):
    """
    Generates a synthetic Canonical Polyadic (CP) tensor, applies a sampling mask, and adds AWGN.
    
    Args:
        shape (tuple): Dimensions of the tensor.
        rank (int): CP-rank.
        sampling_rate (float): Fraction of observed entries (0.0 to 1.0).
        snr_db (float, optional): Signal-to-Noise Ratio in dB.
        seed (int, optional): Random seed.
        
    Returns:
        dict: Contains 'u_true', 'y', 'mask', and 'noise_var'.
    """
    if seed is not None:
        np.random.seed(seed)
        
    # 1. Generate ground truth CP factors and contract
    cp_model = CPModel(shape, rank)
    cp_model.initialize_factors(method='gaussian', scale=1.0)
    u_true = cp_model.get_full_tensor()
    
    # 2. Generate sampling mask
    mask = (np.random.rand(*shape) < sampling_rate).astype(float)
    
    # 3. Add noise
    y = u_true.copy()
    noise_var = 0.0
    if snr_db is not None:
        signal_pwr = np.mean(u_true**2)
        noise_var = signal_pwr / (10**(snr_db / 10.0))
        noise = np.random.normal(0, np.sqrt(noise_var), shape)
        y += noise
        
    y_masked = y * mask
    
    return {
        'u_true': u_true,
        'y': y_masked,
        'mask': mask,
        'noise_var': noise_var
    }
