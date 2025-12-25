import numpy as np
from .gaussian import BaseEstimator

class BernoulliGaussianEstimator(BaseEstimator):
    """
    Bernoulli-Gaussian Estimator for sparse priors.
    p(x) = (1 - lambda) * delta(x) + lambda * N(x; 0, var_prior)
    """
    def __init__(self, sparsity_rate=0.1, var_prior=1.0):
        """
        Args:
            sparsity_rate (float): Probability of being non-zero (lambda).
            var_prior (float): Variance of the Gaussian component.
        """
        self.sparsity_rate = sparsity_rate
        self.var_prior = var_prior
        self.pi_hat = None # Store for EM updates

    def estimate(self, mean_in, var_in):
        """
        Compute posterior mean and variance for BG prior.
        
        Args:
            mean_in (np.ndarray): Input mean (r_hat or q_hat).
            var_in (float): Input variance (nu_r or nu_q).
            
        Returns:
            mean_out (np.ndarray): Posterior mean.
            var_out (float): Posterior variance (averaged).
        """
        # Avoid division by zero or negative variances
        var_in = np.maximum(var_in, 1e-12)
        
        # Posterior mean and variance of the Gaussian component
        mu_post = (mean_in * self.var_prior) / (self.var_prior + var_in)
        tau_post = (self.var_prior * var_in) / (self.var_prior + var_in)
        
        # Log-ratio for numerical stability
        # ratio = (1-lambda)/lambda * N(r; 0, var_in) / N(r; 0, var_in + var_prior)
        # log_ratio = log((1-lambda)/lambda) + 0.5*log((var_in + var_prior)/var_in) - 0.5 * r^2 * var_prior / (var_in * (var_in + var_prior))
        
        log_ratio = (np.log(1.0 - self.sparsity_rate + 1e-12) - np.log(self.sparsity_rate + 1e-12) +
                     0.5 * np.log(var_in + self.var_prior) - 0.5 * np.log(var_in) -
                     0.5 * (mean_in**2) * self.var_prior / (var_in * (var_in + self.var_prior)))
        
        # pi_hat = 1 / (1 + exp(log_ratio))
        # Use clip to avoid overflow in exp
        self.pi_hat = 1.0 / (1.0 + np.exp(np.clip(log_ratio, -50, 50)))
        
        mean_out = self.pi_hat * mu_post
        
        # Variance: E[x^2|r] - (E[x|r])^2
        # E[x^2|r] = pi_hat * (mu_post^2 + tau_post)
        var_elements = self.pi_hat * (mu_post**2 + tau_post) - (mean_out**2)
        
        # Scalar variance (average over all elements)
        var_out = np.mean(var_elements)
        
        return mean_out, var_out
