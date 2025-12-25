import numpy as np
from ..core.engine import BiGAMPEngine
from ..estimators.gaussian import GaussianEstimator, PIAWGNEstimator
from ..estimators.mixture import PIAWLNEstimator
from ..utils.em_tuner import EMTuner
from ..utils.rank_logic import RankManager
from ..utils.metrics import compute_nmse, compute_nmae

class MatrixCompletion:
    """
    Matrix Completion wrapper for BiG-AMP.
    Implements Section VI of the BiG-AMP paper.
    """
    def __init__(self, rank=10, nit=100, tol=1e-5, verbose=False, 
                 use_em=True, use_rank_contraction=False, noise_type='gaussian'):
        self.rank = rank
        self.nit = nit
        self.tol = tol
        self.verbose = verbose
        self.use_em = use_em
        self.use_rank_contraction = use_rank_contraction
        self.noise_type = noise_type
        
        self.engine = None
        self.est_a = None
        self.est_x = None
        self.est_z = None
        self.results = None

    def fit(self, Y, mask=None, var_noise=1e-3, var_a=1.0, var_x=1.0):
        """
        Fits the BiG-AMP model to the observed entries of Y.
        
        Parameters:
        Y: (M, L) observation matrix.
        mask: (M, L) boolean mask (True for observed).
        var_noise: Initial noise variance.
        var_a: Initial variance for A.
        var_x: Initial variance for X.
        """
        M, L = Y.shape
        if mask is None:
            mask = np.ones_like(Y, dtype=bool)
            
        # Initialize Estimators
        self.est_a = GaussianEstimator(mean_prior=0.0, var_prior=var_a)
        self.est_x = GaussianEstimator(mean_prior=0.0, var_prior=var_x)
        
        if self.noise_type == 'gaussian':
            self.est_z = PIAWGNEstimator(Y, mask, var_noise=var_noise)
        elif self.noise_type == 'laplacian':
            # For Laplacian, we use scale parameter b. var = 2*b^2.
            scale = np.sqrt(var_noise / 2.0)
            self.est_z = PIAWLNEstimator(Y, mask, scale=scale)
        else:
            raise ValueError(f"Unknown noise_type: {self.noise_type}")

        # Initialize Engine
        params = {
            'rank': self.rank,
            'nit': self.nit,
            'tol': self.tol,
            'verbose': self.verbose
        }
        self.engine = BiGAMPEngine(self.est_a, self.est_x, self.est_z, params)

        # Main Loop with EM and Rank Contraction
        # We might run multiple outer loops if EM is enabled
        max_outer = 20 if self.use_em else 1
        
        for outer in range(max_outer):
            if self.verbose:
                print(f"--- Outer Iteration {outer+1}/{max_outer} ---")
            
            # Run BiG-AMP
            self.results = self.engine.solve(Y, mask)
            
            if not self.use_em and not self.use_rank_contraction:
                break
                
            # EM Tuning
            if self.use_em:
                EMTuner.tune_all(self.results, self.est_a, self.est_x, self.est_z, Y, mask)
                if self.verbose:
                    print(f"  EM Updated: var_a={self.est_a.var_prior:.4e}, "
                          f"var_x={self.est_x.var_prior:.4e}, "
                          f"var_noise={getattr(self.est_z, 'var_noise', getattr(self.est_z, 'scale', 0)):.4e}")

            # Rank Contraction
            if self.use_rank_contraction and outer % 2 == 0:
                a_hat = self.results['a_hat']
                x_hat = self.results['x_hat']
                a_new, x_new, new_rank = RankManager.contract_rank(a_hat, x_hat)
                if new_rank < self.rank:
                    if self.verbose:
                        print(f"  Rank Contracted: {self.rank} -> {new_rank}")
                    self.rank = new_rank
                    self.engine.params['rank'] = new_rank
                    # Re-initialize engine state with truncated values
                    # Note: In a real implementation, we'd update the engine's internal state
                    # Here we just update the starting point for the next solve
                    self.engine.params['a_hat_init'] = a_new
                    self.engine.params['x_hat_init'] = x_new

            # Check for convergence of the outer loop (simplified)
            if outer > 0:
                # Could check if parameters stopped changing
                pass

        return self.results

    def predict(self):
        """
        Returns the reconstructed matrix A * X.
        """
        if self.results is None:
            raise ValueError("Model must be fitted before prediction.")
        return self.results['a_hat'] @ self.results['x_hat']

    def get_metrics(self, Y_true, mask_test=None):
        """
        Computes NMSE and NMAE against ground truth.
        """
        Y_pred = self.predict()
        nmse = compute_nmse(Y_true, Y_pred, mask_test)
        nmae = compute_nmae(Y_true, Y_pred, mask_test)
        return {'nmse': nmse, 'nmae': nmae}
