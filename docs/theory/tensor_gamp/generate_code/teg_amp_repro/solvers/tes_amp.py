import numpy as np
import opt_einsum as oe
from .base_solver import BaseAMPSolver
from ..utils.damping import calculate_kl_gaussian
from ..utils.tensor_math import cp_contract

class TeSAMPSolver(BaseAMPSolver):
    """
    Tensor Simplified AMP (TeS-AMP) for CP-rank tensors (Algorithm 2).
    Optimized for tensors represented as a sum of rank-1 components.
    """
    def __init__(self, shape, rank, prior, channel, max_iter=100, tol=1e-6, damping_beta=1.0):
        super().__init__(max_iter, tol, damping_beta)
        self.shape = shape
        self.rank = rank
        self.prior = prior
        self.channel = channel
        self.d = len(shape)
        
        # State variables
        self.Z_hat = []  # List of Ni x R factor matrices
        self.Z_var = []  # List of Ni x R variance matrices
        self.s_hat = None  # Residual mean tensor
        self.s_var = None  # Residual variance tensor
        
        # Previous state for damping/rejection
        self.prev_Z_hat = []
        self.prev_Z_var = []
        self.prev_s_hat = None
        self.prev_s_var = None

    def _initialize(self, y, mask, **kwargs):
        self.y = y
        self.mask = mask
        
        # Initialize factor means and variances
        # Use small random values for means to break symmetry
        self.Z_hat = [np.random.normal(0, 0.1, (self.shape[i], self.rank)) for i in range(self.d)]
        self.Z_var = [np.ones((self.shape[i], self.rank)) for i in range(self.d)]
        
        # Initialize residuals
        self.s_hat = np.zeros(self.shape)
        self.s_var = np.zeros(self.shape)
        
        # Initialize noise variance if adaptive
        if hasattr(self.channel, 'estimate_noise_var') and self.channel.estimate_noise_var:
            if np.any(mask):
                self.channel.noise_var = np.var(y[mask])
            else:
                self.channel.noise_var = 1.0

    def _get_reconstruction(self):
        return cp_contract(self.Z_hat)

    def _backup_state(self):
        self.prev_Z_hat = [z.copy() for z in self.Z_hat]
        self.prev_Z_var = [v.copy() for v in self.Z_var]
        self.prev_s_hat = self.s_hat.copy() if self.s_hat is not None else None
        self.prev_s_var = self.s_var.copy() if self.s_var is not None else None

    def _restore_state(self):
        self.Z_hat = [z.copy() for z in self.prev_Z_hat]
        self.Z_var = [v.copy() for v in self.prev_Z_var]
        self.s_hat = self.prev_s_hat.copy()
        self.s_var = self.prev_s_var.copy()

    def _step(self, t):
        # --- 1. Forward Pass ---
        # Compute noiseless tensor estimate p_hat
        p_hat = cp_contract(self.Z_hat)
        
        # Compute variance p_var (Exact for CP)
        # p_var_x = sum_r [ prod_i (Z_var_ir + Z_hat_ir^2) - prod_i Z_hat_ir^2 ]
        M2 = [self.Z_var[i] + self.Z_hat[i]**2 for i in range(self.d)]
        p_var = cp_contract(M2) - cp_contract([z**2 for z in self.Z_hat])
        p_var = np.maximum(p_var, 1e-12)
        
        # --- 2. Channel Update (Residuals) ---
        u_hat, u_var = self.channel.estimate(p_hat, p_var, self.y, self.mask)
        
        new_s_hat = (u_hat - p_hat) / p_var
        new_s_var = (1.0 - u_var / p_var) / p_var
        new_s_var = np.maximum(new_s_var, 1e-12)
        
        # Apply damping to residuals
        self.s_hat = self.damping.apply(new_s_hat, self.s_hat)
        self.s_var = self.damping.apply(new_s_var, self.s_var)
        
        # --- 3. Backward Pass (Factor Updates) ---
        new_Z_hat = []
        new_Z_var = []
        
        letters = 'abcdefghijklmnopqrstuvwxyz'
        s_indices = letters[:self.d]
        factor_indices = [f"{letters[j]}R" for j in range(self.d)]
        
        for i in range(self.d):
            # Compute r_var and r_hat for factor i
            # r_var_inv = sum_{x \ i} s_var_x * prod_{j!=i} Z_hat_jr^2
            input_indices_var = [s_indices] + [factor_indices[j] for j in range(self.d) if j != i]
            output_indices = factor_indices[i]
            
            other_factors_sq = [self.Z_hat[j]**2 for j in range(self.d) if j != i]
            r_var_inv = oe.contract(f"{','.join(input_indices_var)}->{output_indices}", 
                                    self.s_var, *other_factors_sq)
            r_var = 1.0 / np.maximum(r_var_inv, 1e-12)
            
            # r_hat = Z_hat_i + r_var * sum_{x \ i} s_hat_x * prod_{j!=i} Z_hat_jr
            input_indices_hat = [s_indices] + [factor_indices[j] for j in range(self.d) if j != i]
            other_factors = [self.Z_hat[j] for j in range(self.d) if j != i]
            r_hat_term = oe.contract(f"{','.join(input_indices_hat)}->{output_indices}", 
                                     self.s_hat, *other_factors)
            r_hat = self.Z_hat[i] + r_var * r_hat_term
            
            # Prior update (Denoising)
            z_hat_i, z_var_i = self.prior.estimate(r_hat, r_var)
            
            # Apply damping to factors
            z_hat_i = self.damping.apply(z_hat_i, self.Z_hat[i])
            z_var_i = self.damping.apply(z_var_i, self.Z_var[i])
            
            new_Z_hat.append(z_hat_i)
            new_Z_var.append(z_var_i)
            
        # Update state
        self.Z_hat = new_Z_hat
        self.Z_var = new_Z_var
        
        # --- 4. Cost Calculation J(t) ---
        kl_div = 0
        for i in range(self.d):
            # Assuming GaussianPrior(0, 1) for KL calculation
            # In a more general case, we'd pull these from self.prior
            kl_div += calculate_kl_gaussian(self.Z_hat[i], self.Z_var[i], 0.0, 1.0)
            
        log_like = self.channel.compute_log_likelihood(self.y, self.mask, u_hat, u_var)
        cost = kl_div - log_like
        
        # Update noise variance if adaptive
        if hasattr(self.channel, 'estimate_noise_var') and self.channel.estimate_noise_var:
            self.channel.update_noise_var(self.y, self.mask, u_hat, u_var)
            
        return cost
