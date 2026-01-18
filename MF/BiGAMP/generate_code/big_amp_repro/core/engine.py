import numpy as np
from .base_solver import BaseBiGAMP

class BiGAMPEngine(BaseBiGAMP):
    """
    Implementation of Scalar Variance BiG-AMP (Table IV).
    Provides O(N|Omega|) complexity for bilinear inference.
    """

    def __init__(self, est_a, est_x, est_z, params=None):
        """
        Initialize BiG-AMP Engine.
        
        Args:
            est_a: Estimator for matrix A (M x N)
            est_x: Estimator for matrix X (N x L)
            est_z: Estimator for likelihood Z = AX (M x L)
            params: Dictionary of hyperparameters
        """
        super().__init__(params)
        self.est_a = est_a
        self.est_x = est_x
        self.est_z = est_z
        
        # Default parameters for damping and EM
        self.damping = self.params.get('damping', 1.0)
        self.min_damping = self.params.get('min_damping', 0.1)
        self.max_damping = 1.0
        
    def solve(self, Y, mask=None):
        """
        Run BiG-AMP iterations.
        
        Args:
            Y: Observation matrix (M x L)
            mask: Binary mask (M x L), 1 if observed, 0 otherwise.
        """
        M, L = Y.shape
        N = self.params.get('rank', 10)
        
        if mask is None:
            mask = np.ones_like(Y)
        
        # Initialization
        # a_hat (M x N), x_hat (N x L)
        a_hat = self.params.get('a_init', np.random.randn(M, N) * 0.1)
        x_hat = self.params.get('x_init', np.random.randn(N, L) * 0.1)
        
        # Scalar variances
        nu_a = self.params.get('nu_a_init', 1.0)
        nu_x = self.params.get('nu_x_init', 1.0)
        
        # s_hat (M x L)
        s_hat = np.zeros((M, L))
        nu_s = 1.0
        
        # History for convergence and damping
        history = {'nmse': [], 'cost': []}
        
        prev_a_hat = a_hat.copy()
        prev_x_hat = x_hat.copy()
        
        for t in range(self.nit):
            # --- Step 1: Z-updates (R1-R4) ---
            # nu_p = (1/ML) * sum_{m,l} sum_n (|a_mn|^2 nu_x + nu_a |x_nl|^2 + nu_a nu_x)
            # For scalar variance, this simplifies:
            sum_a_sq = np.sum(a_hat**2)
            sum_x_sq = np.sum(x_hat**2)
            nu_p = (1.0 / (M * L)) * (sum_a_sq * nu_x + nu_a * sum_x_sq + M * L * N * nu_a * nu_x)
            
            # p_hat = A_hat * X_hat - nu_p * s_hat_prev
            p_hat = a_hat @ x_hat - nu_p * s_hat
            
            # --- Step 2: Residual updates (R5-R8) ---
            # Call Z-Estimator (Likelihood)
            z_hat, nu_z = self.est_z.estimate(p_hat, nu_p, Y, mask)
            
            # nu_s = (1 - nu_z / nu_p) / nu_p
            # s_hat = (z_hat - p_hat) / nu_p
            nu_s = (1.0 - np.mean(nu_z) / nu_p) / nu_p
            s_hat_new = (z_hat - p_hat) / nu_p
            
            # Apply damping to s_hat
            s_hat = (1 - self.damping) * s_hat + self.damping * s_hat_new
            
            # --- Step 3: R/Q-updates (R9-R12) ---
            # nu_r = 1 / ( (1/N) * sum_m |a_mn|^2 * nu_s )
            # For scalar variance: nu_r = 1 / ( (1/N) * ||A||_F^2 * nu_s / L ) ? 
            # Actually Table IV: nu_r = 1 / ( (1/L) * sum_m,l |a_mn|^2 * nu_s )
            # nu_r = 1 / ( (1/L) * sum_a_sq * nu_s )
            nu_r = 1.0 / ( (1.0 / L) * sum_a_sq * nu_s )
            # r_hat = x_hat + nu_r * (A_hat.T @ s_hat)
            r_hat = x_hat + nu_r * (a_hat.T @ s_hat)
            
            # nu_q = 1 / ( (1/M) * sum_l |x_nl|^2 * nu_s )
            nu_q = 1.0 / ( (1.0 / M) * sum_x_sq * nu_s )
            # q_hat = a_hat + nu_q * (s_hat @ X_hat.T)
            q_hat = a_hat + nu_q * (s_hat @ x_hat.T)
            
            # --- Step 4: A/X-updates (R13-R16) ---
            # Call X-Estimator (Prior)
            x_hat_new, nu_x_new = self.est_x.estimate(r_hat, nu_r)
            # Call A-Estimator (Prior)
            a_hat_new, nu_a_new = self.est_a.estimate(q_hat, nu_q)
            
            # Apply damping
            a_hat = (1 - self.damping) * prev_a_hat + self.damping * a_hat_new
            x_hat = (1 - self.damping) * prev_x_hat + self.damping * x_hat_new
            
            # Update scalar variances (averaging)
            nu_a = np.mean(nu_a_new)
            nu_x = np.mean(nu_x_new)
            
            # Convergence check
            diff = np.linalg.norm(a_hat @ x_hat - prev_a_hat @ prev_x_hat, 'fro') / np.linalg.norm(a_hat @ x_hat, 'fro')
            if self.verbose and t % 10 == 0:
                print(f"Iteration {t}: diff = {diff:.6e}")
            
            if diff < self.tol:
                break
                
            prev_a_hat = a_hat.copy()
            prev_x_hat = x_hat.copy()
            
        return {
            'a_hat': a_hat,
            'x_hat': x_hat,
            'z_hat': z_hat,
            'nu_a': nu_a,
            'nu_x': nu_x,
            'nit': t
        }
