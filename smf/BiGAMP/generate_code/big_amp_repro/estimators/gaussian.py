import numpy as np
from abc import ABC, abstractmethod

class BaseEstimator(ABC):
    """
    Abstract base class for BiG-AMP estimators.
    Each estimator must implement the estimate method to compute posterior 
    mean and variance given the input mean and variance from the BiG-AMP loop.
    """
    @abstractmethod
    def estimate(self, mean_in, var_in):
        """
        Compute posterior mean and variance.
        
        Args:
            mean_in (np.ndarray): Input mean (e.g., p_hat, q_hat, r_hat).
            var_in (float or np.ndarray): Input variance (e.g., nu_p, nu_q, nu_r).
            
        Returns:
            mean_out (np.ndarray): Posterior mean.
            var_out (float or np.ndarray): Posterior variance.
        """
        pass

class GaussianEstimator(BaseEstimator):
    """
    Gaussian Prior Estimator.
    Assumes a prior p(x) = N(x; mean_prior, var_prior).
    """
    def __init__(self, mean_prior=0.0, var_prior=1.0):
        self.mean_prior = mean_prior
        self.var_prior = var_prior

    def estimate(self, mean_in, var_in):
        # Posterior variance: 1 / (1/var_prior + 1/var_in)
        # var_out = (var_prior * var_in) / (var_prior + var_in)
        var_out = (self.var_prior * var_in) / (self.var_prior + var_in)
        
        # Posterior mean: var_out * (mean_prior/var_prior + mean_in/var_in)
        # mean_out = (mean_in * var_prior + mean_prior * var_in) / (var_prior + var_in)
        mean_out = (mean_in * self.var_prior + self.mean_prior * var_in) / (self.var_prior + var_in)
        
        return mean_out, var_out

class PIAWGNEstimator(BaseEstimator):
    """
    Possibly Incomplete Additive White Gaussian Noise (PIAWGN) Likelihood Estimator.
    Used for the Z-update in Matrix Completion and other problems.
    
    For (m,l) in Omega: p(y|z) = N(y; z, var_noise)
    For (m,l) not in Omega: p(y|z) = 1 (Uniform/Incomplete)
    """
    def __init__(self, Y, mask=None, var_noise=1e-3):
        self.Y = Y
        self.mask = mask
        self.var_noise = var_noise

    def estimate(self, p_hat, var_p):
        """
        p_hat: Input mean from BiG-AMP (M x L matrix)
        var_p: Input variance (scalar for Table IV)
        """
        # Initialize outputs with the "unobserved" case (z_hat = p_hat, var_z = var_p)
        z_hat = np.copy(p_hat)
        
        if np.isscalar(var_p):
            var_z = np.full(p_hat.shape, var_p)
        else:
            var_z = np.copy(var_p)
            
        # Compute the "observed" case values
        # var_z_obs = 1 / (1/var_p + 1/var_noise)
        # z_hat_obs = var_z_obs * (p_hat/var_p + Y/var_noise)
        
        var_z_obs = (var_p * self.var_noise) / (var_p + self.var_noise)
        z_hat_obs = (p_hat * self.var_noise + self.Y * var_p) / (var_p + self.var_noise)
        
        if self.mask is not None:
            # Apply observed values only where mask is True
            z_hat[self.mask] = z_hat_obs[self.mask]
            var_z[self.mask] = var_z_obs if np.isscalar(var_z_obs) else var_z_obs[self.mask]
        else:
            # All entries are observed
            z_hat = z_hat_obs
            var_z = var_z_obs
            
        return z_hat, var_z
