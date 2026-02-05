import numpy as np
from abc import ABC, abstractmethod

class BaseChannel(ABC):
    """
    Abstract base class for likelihood channels in AMP.
    """
    @abstractmethod
    def estimate(self, p_hat, p_var, y, mask):
        """
        Compute posterior mean and variance of the noiseless output u.
        
        Args:
            p_hat: Mean of the message from the linear part to the channel.
            p_var: Variance of the message from the linear part to the channel.
            y: Observed data.
            mask: Binary mask (1 for observed, 0 for missing).
            
        Returns:
            u_hat: Posterior mean E[u | y, p_hat, p_var].
            u_var: Posterior variance Var[u | y, p_hat, p_var].
        """
        pass

class AWGNChannel(BaseChannel):
    """
    Additive White Gaussian Noise (AWGN) channel for tensor completion and denoising.
    Model: y = u + w, where w ~ N(0, noise_var).
    """
    def __init__(self, noise_var=1e-3, estimate_noise_var=False):
        """
        Args:
            noise_var: Initial noise variance.
            estimate_noise_var: Whether to adaptively update noise variance.
        """
        self.noise_var = max(noise_var, 1e-12)
        self.estimate_noise_var = estimate_noise_var

    def estimate(self, p_hat, p_var, y, mask):
        """
        Posterior mean and variance for AWGN channel with mask.
        
        For observed indices (mask == 1):
            u_hat = (y * p_var + p_hat * noise_var) / (p_var + noise_var)
            u_var = (p_var * noise_var) / (p_var + noise_var)
        For unobserved indices (mask == 0):
            u_hat = p_hat
            u_var = p_var
        """
        u_hat = np.copy(p_hat)
        u_var = np.copy(p_var)
        
        # Observed indices
        obs = (mask == 1)
        if np.any(obs):
            # Add epsilon for numerical stability
            denom = p_var[obs] + self.noise_var + 1e-12
            u_hat[obs] = (y[obs] * p_var[obs] + p_hat[obs] * self.noise_var) / denom
            u_var[obs] = (p_var[obs] * self.noise_var) / denom
            
        return u_hat, u_var

    def update_noise_var(self, y, mask, u_hat, u_var):
        """
        Estimate noise variance adaptively (Eq 165 in the paper).
        noise_var = (1/|Omega|) * sum_{x in Omega} (|y_x - u_hat_x|^2 + u_var_x)
        
        Args:
            y: Observed data.
            mask: Binary mask.
            u_hat: Current posterior mean.
            u_var: Current posterior variance.
            
        Returns:
            Updated noise variance.
        """
        if not self.estimate_noise_var:
            return self.noise_var
            
        obs = (mask == 1)
        num_obs = np.sum(obs)
        if num_obs > 0:
            new_var = np.sum((y[obs] - u_hat[obs])**2 + u_var[obs]) / num_obs
            self.noise_var = max(new_var, 1e-12)
            
        return self.noise_var

    def compute_log_likelihood(self, y, mask, u_hat, u_var):
        """
        Computes the log-likelihood part of the cost function J(t).
        E[log p(y|u)] = -0.5 * sum_{x in Omega} (log(2*pi*noise_var) + (|y_x - u_hat_x|^2 + u_var_x)/noise_var)
        """
        obs = (mask == 1)
        num_obs = np.sum(obs)
        if num_obs == 0:
            return 0.0
            
        term1 = num_obs * np.log(2 * np.pi * self.noise_var)
        term2 = np.sum((y[obs] - u_hat[obs])**2 + u_var[obs]) / self.noise_var
        
        return -0.5 * (term1 + term2)
