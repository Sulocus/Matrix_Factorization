import numpy as np
from ..core.engine import BiGAMPEngine
from ..estimators.gaussian import GaussianEstimator, PIAWGNEstimator
from ..estimators.bernoulli_bg import BernoulliGaussianEstimator
from ..utils.em_tuner import EMTuner
from ..utils.rank_logic import RankManager
from ..utils.metrics import compute_nmse

class RPCAEstimatorA:
    """Custom estimator for augmented matrix A = [A_low_rank, I]"""
    def __init__(self, M, N, var_a=1.0):
        self.M = M
        self.N = N
        self.var_prior = var_a  # For EMTuner compatibility (refers to A_low_rank)
        self.I = np.eye(M)
        
    def estimate(self, q_hat, nu_q):
        M, TotalN = q_hat.shape
        a_hat = np.zeros_like(q_hat)
        
        # Part 1: Low-rank basis A (M x N)
        v_a = self.var_prior
        denom = v_a + nu_q
        a_hat[:, :self.N] = (q_hat[:, :self.N] * v_a) / denom
        var_a_post = (v_a * nu_q) / denom
        
        # Part 2: Fixed Identity (M x M)
        a_hat[:, self.N:] = self.I
        var_i_post = 0.0 # Fixed
        
        # Average variance across all elements of A_aug
        avg_var = (self.N * var_a_post + self.M * var_i_post) / TotalN
        return a_hat, avg_var

class RPCAEstimatorX:
    """Custom estimator for augmented matrix X = [X_low_rank; S_sparse]"""
    def __init__(self, L, N, M, sparsity_rate=0.1, var_x=1.0, var_s=1.0):
        self.L = L
        self.N = N
        self.M = M
        self.var_prior = var_x # For EMTuner compatibility (refers to X_low_rank)
        self.sparsity_rate = sparsity_rate
        self.var_s = var_s
        self.bg_est = BernoulliGaussianEstimator(sparsity_rate, var_s)
        
    def estimate(self, r_hat, nu_r):
        TotalN, L = r_hat.shape
        x_hat = np.zeros_like(r_hat)
        
        # Part 1: Low-rank coefficients X (N x L)
        v_x = self.var_prior
        denom = v_x + nu_r
        x_hat[:self.N, :] = (r_hat[:self.N, :] * v_x) / denom
        var_x_post = (v_x * nu_r) / denom
        
        # Part 2: Sparse outliers S (M x L)
        s_hat_part, var_s_post = self.bg_est.estimate(r_hat[self.N:, :], nu_r)
        x_hat[self.N:, :] = s_hat_part
        
        # Average variance across all elements of X_aug
        avg_var = (self.N * var_x_post + self.M * var_s_post) / TotalN
        return x_hat, avg_var

class RobustPCA:
    """
    Robust PCA implementation using BiG-AMP.
    Decomposes Y = L + S + W, where L is low-rank, S is sparse, and W is noise.
    Uses the augmented matrix formulation (Eq 127):
    Y = [A, I] * [X; S] + W
    """
    def __init__(self, rank=10, nit=100, tol=1e-4, use_em=True, 
                 sparsity_rate=0.1, var_s=1.0, verbose=False):
        self.rank = rank
        self.nit = nit
        self.tol = tol
        self.use_em = use_em
        self.sparsity_rate = sparsity_rate
        self.var_s = var_s
        self.verbose = verbose
        
        self.a_hat = None
        self.x_hat = None
        self.s_hat = None
        self.l_hat = None

    def fit(self, Y, mask=None):
        M, L = Y.shape
        N = self.rank
        
        # Initialize estimators
        est_a = RPCAEstimatorA(M, N, var_a=1.0)
        est_x = RPCAEstimatorX(L, N, M, self.sparsity_rate, var_x=1.0, var_s=self.var_s)
        est_z = PIAWGNEstimator(Y, mask, var_noise=1e-3)
        
        engine_params = {
            'rank': N + M,
            'nit': self.nit,
            'tol': self.tol,
            'verbose': self.verbose
        }
        
        engine = BiGAMPEngine(est_a, est_x, est_z, engine_params)
        
        # Outer loop for EM updates
        max_em_iter = 10 if self.use_em else 1
        last_res = None
        
        for em_iter in range(max_em_iter):
            res = engine.solve(Y, mask)
            last_res = res
            
            if not self.use_em:
                break
                
            # Update parameters
            # Note: Standard EMTuner might need adjustment for augmented structure
            # We'll do manual updates for the specific parts
            
            # 1. Update noise variance
            est_z.var_noise = EMTuner.update_noise_var(est_z, res['z_hat'], res['var_z'], Y, mask)
            
            # 2. Update low-rank variances
            # Extract low-rank parts
            a_low = res['a_hat'][:, :N]
            x_low = res['x_hat'][:N, :]
            # We don't have individual posterior variances for A and X easily from the scalar engine
            # but we can use the empirical variance of the means as an estimate
            est_a.var_prior = np.mean(a_low**2)
            est_x.var_prior = np.mean(x_low**2)
            
            # 3. Update sparse parameters
            s_part = res['x_hat'][N:, :]
            est_x.bg_est.sparsity_rate = np.mean(est_x.bg_est.pi_hat)
            est_x.bg_est.var_prior = np.sum(est_x.bg_est.pi_hat * (s_part**2)) / (np.sum(est_x.bg_est.pi_hat) + 1e-10)
            
            if self.verbose:
                print(f"EM Iter {em_iter}: noise_var={est_z.var_noise:.4e}, lambda={est_x.bg_est.sparsity_rate:.4f}")

        # Extract final results
        a_aug = last_res['a_hat']
        x_aug = last_res['x_hat']
        
        self.a_hat = a_aug[:, :N]
        self.x_hat = x_aug[:N, :]
        self.s_hat = x_aug[N:, :]
        self.l_hat = self.a_hat @ self.x_hat
        
        return last_res

    def get_low_rank(self):
        return self.l_hat
    
    def get_sparse(self):
        return self.s_hat

    def get_metrics(self, L_true, S_true):
        nmse_l = compute_nmse(L_true, self.l_hat)
        nmse_s = compute_nmse(S_true, self.s_hat)
        return {'nmse_l': nmse_l, 'nmse_s': nmse_s}
