import numpy as np
from ..core.engine import BiGAMPEngine
from ..estimators.gaussian import GaussianEstimator, PIAWGNEstimator
from ..estimators.bernoulli_bg import BernoulliGaussianEstimator
from ..utils.em_tuner import EMTuner
from ..utils.metrics import compute_nmse

class DictLearning:
    """
    Application wrapper for Dictionary Learning (DL) using BiG-AMP.
    Solves Y = AX + W where X is sparse.
    """
    def __init__(self, rank, nit=100, tol=1e-5, use_em=True, verbose=False):
        """
        Parameters:
            rank (int): Number of dictionary atoms (inner dimension N).
            nit (int): Maximum number of iterations.
            tol (float): Convergence tolerance.
            use_em (bool): Whether to use EM for hyperparameter tuning.
            verbose (bool): Print progress.
        """
        self.rank = rank
        self.nit = nit
        self.tol = tol
        self.use_em = use_em
        self.verbose = verbose
        self.a_hat = None
        self.x_hat = None
        self.z_hat = None
        self.est_a = None
        self.est_x = None
        self.est_z = None

    def fit(self, Y, mask=None, sparsity_rate=0.1, var_a=1.0, var_x=1.0, var_noise=1e-3):
        """
        Fit the Dictionary Learning model.
        
        Parameters:
            Y (np.ndarray): Observation matrix (M x L).
            mask (np.ndarray, optional): Sampling mask.
            sparsity_rate (float): Initial sparsity rate for X.
            var_a (float): Initial variance for dictionary A.
            var_x (float): Initial variance for sparse coefficients X.
            var_noise (float): Initial noise variance.
        """
        M, L = Y.shape
        N = self.rank

        # Initialize Estimators
        # A is typically Gaussian (dictionary atoms)
        self.est_a = GaussianEstimator(mean_prior=0.0, var_prior=var_a)
        # X is sparse (Bernoulli-Gaussian)
        self.est_x = BernoulliGaussianEstimator(sparsity_rate=sparsity_rate, var_prior=var_x)
        # Z is the likelihood (PIAWGN)
        self.est_z = PIAWGNEstimator(Y, mask=mask, var_noise=var_noise)

        # Initialize Engine
        params = {
            'rank': N,
            'nit': self.nit,
            'tol': self.tol,
            'verbose': self.verbose
        }
        engine = BiGAMPEngine(self.est_a, self.est_x, self.est_z, params)

        if not self.use_em:
            # Standard BiG-AMP run
            results = engine.solve(Y, mask)
            self.a_hat = results['a_hat']
            self.x_hat = results['x_hat']
            self.z_hat = results['z_hat']
        else:
            # EM-BiG-AMP: Run in blocks of iterations and update parameters
            # Or integrate EM into the engine loop if the engine supports it.
            # Here we implement an outer EM loop for clarity.
            em_iters = 10
            inner_nit = self.nit // em_iters
            
            curr_Y = Y
            curr_mask = mask
            
            for i in range(em_iters):
                engine.params['nit'] = inner_nit
                results = engine.solve(curr_Y, curr_mask)
                
                self.a_hat = results['a_hat']
                self.x_hat = results['x_hat']
                self.z_hat = results['z_hat']
                
                # Update hyperparameters via EM
                EMTuner.tune_all(
                    results, 
                    self.est_a, 
                    self.est_x, 
                    self.est_z, 
                    curr_Y, 
                    curr_mask
                )
                
                if self.verbose:
                    nmse = compute_nmse(curr_Y, self.a_hat @ self.x_hat, curr_mask)
                    print(f"EM Iter {i+1}/{em_iters} - NMSE: {nmse:.2f} dB, Noise Var: {self.est_z.var_noise:.2e}")

        return self

    def predict(self):
        """Return the reconstructed matrix A * X."""
        if self.a_hat is None or self.x_hat is None:
            raise ValueError("Model not fitted yet.")
        return self.a_hat @ self.x_hat

    def get_metrics(self, A_true=None, X_true=None, Y_true=None):
        """
        Calculate performance metrics.
        Note: DL has permutation and scale invariance for A and X.
        """
        metrics = {}
        if Y_true is not None:
            metrics['nmse_y'] = compute_nmse(Y_true, self.predict())
        
        # For A and X, we'd ideally need to handle permutation/scale
        # but for simple validation we can check NMSE if they are aligned.
        if A_true is not None and X_true is not None:
            # This is a naive check; real DL evaluation often uses correlation
            metrics['nmse_a'] = compute_nmse(A_true, self.a_hat)
            metrics['nmse_x'] = compute_nmse(X_true, self.x_hat)
            
        return metrics
