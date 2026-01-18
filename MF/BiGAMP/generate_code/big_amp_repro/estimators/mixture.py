import numpy as np
from scipy.special import erfcx
from .gaussian import BaseEstimator

class GaussianMixtureEstimator(BaseEstimator):
    """
    Gaussian Mixture Model (GMM) Estimator for BiG-AMP.
    p(x) = sum_k weights[k] * N(x; means[k], vars[k])
    """
    def __init__(self, weights, means, vars_prior):
        self.weights = np.array(weights)
        self.means = np.array(means)
        self.vars_prior = np.array(vars_prior)
        
    def estimate(self, mean_in, var_in):
        """
        Computes the posterior mean and average variance for a GMM prior.
        """
        K = len(self.weights)
        
        # Reshape for broadcasting
        # mean_in is (M, L), var_in is scalar
        w = self.weights.reshape(K, 1, 1)
        m = self.means.reshape(K, 1, 1)
        v = self.vars_prior.reshape(K, 1, 1)
        
        total_vars = v + var_in
        diff = mean_in - m
        
        # Log-pdf (ignoring constant 2*pi)
        log_pdf = -0.5 * (np.log(total_vars) + (diff**2) / total_vars)
        log_weighted_pdf = np.log(w + 1e-300) + log_pdf
        
        # Log-sum-exp for numerical stability
        max_log = np.max(log_weighted_pdf, axis=0)
        weighted_pdf = np.exp(log_weighted_pdf - max_log)
        sum_weighted_pdf = np.sum(weighted_pdf, axis=0)
        
        # Posterior weights
        post_weights = weighted_pdf / (sum_weighted_pdf + 1e-300)
        
        # Component posterior means and variances
        post_vars_k = (var_in * v) / total_vars
        post_means_k = post_vars_k * (mean_in / var_in + m / (v + 1e-300))
        
        # Overall posterior mean
        mean_out = np.sum(post_weights * post_means_k, axis=0)
        
        # Overall posterior variance (Law of Total Variance)
        # Var(X) = E[Var(X|K)] + Var(E[X|K])
        term1 = np.sum(post_weights * (post_vars_k + post_means_k**2), axis=0)
        var_out_elementwise = term1 - mean_out**2
        
        return mean_out, np.mean(var_out_elementwise)

class LaplacianEstimator(BaseEstimator):
    """
    Laplacian Prior Estimator (MMSE).
    p(x) = (1/2b) * exp(-|x-mu|/b)
    """
    def __init__(self, scale=1.0, mean=0.0):
        self.scale = scale
        self.mean = mean

    def estimate(self, mean_in, var_in):
        """
        Computes the posterior mean for a Laplacian prior using erfcx for stability.
        """
        y = mean_in - self.mean
        sigma = np.sqrt(var_in)
        b = self.scale
        
        # Use erfcx for numerical stability: erfcx(x) = exp(x^2) * erfc(x)
        z1 = (sigma / b - y / sigma) / np.sqrt(2)
        z2 = (sigma / b + y / sigma) / np.sqrt(2)
        
        e1 = erfcx(z1)
        e2 = erfcx(z2)
        
        denom = e1 + e2
        denom = np.where(denom < 1e-150, 1e-150, denom)
        
        ratio = (e1 - e2) / denom
        mean_out = y + (var_in / b) * ratio + self.mean
        
        # For variance, we use var_in as a stable approximation in the scalar-variance case.
        # Exact MMSE variance for Laplacian is complex and often leads to instability 
        # in BiG-AMP if not handled with extreme care.
        return mean_out, var_in

class PIAWLNEstimator(BaseEstimator):
    """
    Possibly Incomplete Additive White Laplacian Noise Likelihood.
    p(y|z) = (1/2b) * exp(-|y-z|/b)
    Used in Matrix Completion with outliers (e.g., MovieLens).
    """
    def __init__(self, Y, mask=None, scale=1.0):
        self.Y = Y
        self.mask = mask if mask is not None else np.ones_like(Y, dtype=bool)
        self.scale = scale

    def estimate(self, p_hat, var_p):
        """
        Likelihood update for Laplacian noise.
        """
        sigma = np.sqrt(var_p)
        b = self.scale
        
        # Only compute for masked entries
        y = self.Y - p_hat
        
        z1 = (sigma / b - y / sigma) / np.sqrt(2)
        z2 = (sigma / b + y / sigma) / np.sqrt(2)
        
        e1 = erfcx(z1)
        e2 = erfcx(z2)
        
        denom = e1 + e2
        denom = np.where(denom < 1e-150, 1e-150, denom)
        ratio = (e1 - e2) / denom
        
        # Posterior mean of Z
        z_hat_masked = p_hat + (var_p / b) * ratio
        
        # Apply mask: if not observed, z_hat = p_hat
        z_hat = np.where(self.mask, z_hat_masked, p_hat)
        
        # Return scalar variance
        return z_hat, var_p
