import numpy as np
import opt_einsum as oe
from .base_solver import BaseAMPSolver
from ..utils.damping import calculate_kl_gaussian
from ..utils.tensor_math import tr_contract

class TeGAMPSolver(BaseAMPSolver):
    """
    Tensor Generalized AMP (TeG-AMP) for Tensor Ring (TR) decomposition.
    Implements Algorithm 1 from the paper.
    """
    def __init__(self, shape, ranks, prior, channel, max_iter=100, tol=1e-6, damping_beta=0.5):
        super().__init__(max_iter, tol, damping_beta)
        self.shape = shape
        self.ranks = ranks
        self.d = len(shape)
        self.prior = prior
        self.channel = channel
        
        # State variables
        self.Z_hat = []  # Means of cores
        self.Z_var = []  # Variances of cores
        self.s_hat = None  # Residual mean
        self.s_var = None  # Residual variance
        
        # Backup for damping
        self.Z_hat_prev = []
        self.Z_var_prev = []
        self.s_hat_prev_val = None
        self.s_var_prev_val = None

    def _initialize(self, y, mask, **kwargs):
        """Initialize solver state."""
        self.Z_hat = []
        self.Z_var = []
        for i in range(self.d):
            r_in = self.ranks[i]
            r_out = self.ranks[(i + 1) % self.d]
            # Initialization: N(0, 0.1) for means, 1.0 for variances
            self.Z_hat.append(np.random.normal(0, 0.1, (self.shape[i], r_in, r_out)))
            self.Z_var.append(np.ones((self.shape[i], r_in, r_out)))
            
        self.s_hat = np.zeros(self.shape)
        self.s_var = np.zeros(self.shape)
        
        # Initial noise variance estimation if requested
        if hasattr(self.channel, 'update_noise_var') and self.channel.estimate_noise_var:
            u_hat = tr_contract(self.Z_hat)
            u_var = np.ones(self.shape) # Heuristic initial variance
            self.channel.update_noise_var(y, mask, u_hat, u_var)

    def _step(self, t):
        """Perform one TeG-AMP iteration."""
        # --- Forward Pass ---
        # 1. Compute p_var (Eq 25)
        # nu_p_x = sum_{k} [ prod_i (Z_hat_i^2 + Z_var_i) - prod_i Z_hat_i^2 ]
        M2 = [self.Z_hat[i]**2 + self.Z_var[i] for i in range(self.d)]
        H2 = [self.Z_hat[i]**2 for i in range(self.d)]
        
        p_var = tr_contract(M2) - tr_contract(H2)
        p_var = np.maximum(p_var, 1e-12)
        
        # 2. Compute p_hat (Eq 26)
        # p_hat = tr_contract(Z_hat) - p_var * s_hat_prev
        p_hat = tr_contract(self.Z_hat) - p_var * self.s_hat
        
        # 3. Channel update (Eq 17, 18)
        u_hat, u_var = self.channel.estimate(p_hat, p_var, self.y, self.mask)
        
        # 4. Residuals (Eq 19, 20)
        new_s_var = (1.0 - u_var / p_var) / p_var
        new_s_hat = (u_hat - p_hat) / p_var
        
        # --- Backward Pass ---
        # 5. Update latent observations and priors (Eq 27, 28, 21, 22)
        new_Z_hat = []
        new_Z_var = []
        
        for i in range(self.d):
            # Indices for s_hat: 0, 1, ..., d-1
            # Indices for Z_hat_j: j, d+j, d+(j+1)%d
            # Output indices for core i: i, d+i, d+(i+1)%d
            
            s_indices = list(range(self.d))
            z_indices_list = []
            for j in range(self.d):
                if j == i:
                    z_indices_list.append(None) # Placeholder
                else:
                    z_indices_list.append([j, self.d + j, self.d + (j + 1) % self.d])
            
            out_indices = [i, self.d + i, self.d + (i + 1) % self.d]
            
            # Compute G_i (for r_hat)
            operands = [new_s_hat, s_indices]
            for j in range(self.d):
                if j != i:
                    operands.extend([self.Z_hat[j], z_indices_list[j]])
            operands.append(out_indices)
            G_i = oe.contract(*operands)
            
            # Compute Q_i (for r_var)
            operands_var = [new_s_var, s_indices]
            for j in range(self.d):
                if j != i:
                    operands_var.extend([self.Z_hat[j]**2, z_indices_list[j]])
            operands_var.append(out_indices)
            Q_i = oe.contract(*operands_var)
            
            # r_var = 1 / Q_i
            r_var = 1.0 / np.maximum(Q_i, 1e-12)
            # r_hat = Z_hat_i + r_var * G_i
            r_hat = self.Z_hat[i] + r_var * G_i
            
            # Prior update
            z_hat_next, z_var_next = self.prior.estimate(r_hat, r_var)
            new_Z_hat.append(z_hat_next)
            new_Z_var.append(z_var_next)
            
        # --- Damping ---
        beta = self.damping.beta
        self.s_hat = beta * new_s_hat + (1 - beta) * self.s_hat
        self.s_var = beta * new_s_var + (1 - beta) * self.s_var
        for i in range(self.d):
            self.Z_hat[i] = beta * new_Z_hat[i] + (1 - beta) * self.Z_hat[i]
            self.Z_var[i] = beta * new_Z_var[i] + (1 - beta) * self.Z_var[i]
            
        # Update noise variance if requested
        if hasattr(self.channel, 'update_noise_var') and self.channel.estimate_noise_var:
            self.channel.update_noise_var(self.y, self.mask, u_hat, u_var)
            
        # --- Cost Calculation ---
        # J(t) = sum KL(q_i || p_i) - E[log p(y|u)]
        kl_sum = 0
        mu_p = getattr(self.prior, 'mean', 0.0)
        var_p = getattr(self.prior, 'var', 1.0)
        for i in range(self.d):
            kl_sum += calculate_kl_gaussian(self.Z_hat[i], self.Z_var[i], mu_p, var_p)
            
        log_like = self.channel.compute_log_likelihood(self.y, self.mask, u_hat, u_var)
        cost = kl_sum - log_like
        
        return cost

    def _get_reconstruction(self):
        """Return the current tensor estimate."""
        return tr_contract(self.Z_hat)

    def _backup_state(self):
        """Backup current state for potential rejection."""
        self.Z_hat_prev = [z.copy() for z in self.Z_hat]
        self.Z_var_prev = [v.copy() for v in self.Z_var]
        self.s_hat_prev_val = self.s_hat.copy()
        self.s_var_prev_val = self.s_var.copy()

    def _restore_state(self):
        """Restore state from backup."""
        self.Z_hat = [z.copy() for z in self.Z_hat_prev]
        self.Z_var = [v.copy() for v in self.Z_var_prev]
        self.s_hat = self.s_hat_prev_val.copy()
        self.s_var = self.s_var_prev_val.copy()
