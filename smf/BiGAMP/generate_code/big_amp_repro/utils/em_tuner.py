import numpy as np

class EMTuner:
    """
    Expectation-Maximization (EM) parameter tuning for BiG-AMP.
    
    This utility implements the update rules for hyperparameters such as 
    noise variance, prior variances, and sparsity rates as described in 
    Section V-A of the BiG-AMP paper.
    """

    @staticmethod
    def update_noise_var(estimator_z, z_hat, var_z, Y, mask=None):
        """
        Update noise variance for PIAWGN likelihood.
        Formula: nu_w = (1/|Omega|) * sum((Y - z_hat)^2 + var_z) over Omega.
        
        Parameters:
            estimator_z: The PIAWGN estimator instance.
            z_hat: Posterior mean of Z (M x L).
            var_z: Posterior variance of Z (scalar or M x L).
            Y: Observation matrix (M x L).
            mask: Binary mask for observed entries.
        """
        if not hasattr(estimator_z, 'var_noise'):
            return None

        if mask is None:
            mask = np.ones_like(Y, dtype=bool)
        
        num_obs = np.sum(mask)
        if num_obs == 0:
            return estimator_z.var_noise
            
        # Compute squared error on observed entries
        diff_sq = (Y - z_hat)**2
        
        # Handle scalar vs matrix variance
        if np.isscalar(var_z):
            # Sum of (y - z_hat)^2 + var_z over mask
            total_sum = np.sum(diff_sq[mask]) + (num_obs * var_z)
        else:
            total_sum = np.sum(diff_sq[mask] + var_z[mask])
            
        new_var_noise = total_sum / num_obs
        
        # Avoid numerical collapse to zero
        new_var_noise = max(new_var_noise, 1e-10)
        
        estimator_z.var_noise = float(new_var_noise)
        return estimator_z.var_noise

    @staticmethod
    def update_prior_vars(estimator, hat_val, var_val):
        """
        Update prior variance for Gaussian or BG priors.
        Formula: nu = (1/TotalElements) * sum(hat_val^2 + var_val)
        
        Parameters:
            estimator: The A or X estimator instance.
            hat_val: Posterior mean (M x N or N x L).
            var_val: Posterior variance (scalar).
        """
        if not hasattr(estimator, 'var_prior'):
            return None

        n_elements = hat_val.size
        # sum(E[x^2]) = sum(E[x]^2 + Var(x))
        new_var = np.sum(hat_val**2 + var_val) / n_elements
        
        # Avoid numerical collapse
        new_var = max(new_var, 1e-10)
        
        estimator.var_prior = float(new_var)
        return new_var

    @staticmethod
    def update_sparsity_rate(estimator):
        """
        Update sparsity rate (lambda) for Bernoulli-Gaussian priors.
        Formula: lambda = (1/TotalElements) * sum(pi_hat)
        
        Parameters:
            estimator: The BernoulliGaussianEstimator instance.
        """
        if hasattr(estimator, 'pi_hat') and estimator.pi_hat is not None:
            new_lambda = np.mean(estimator.pi_hat)
            # Constrain lambda to reasonable range
            new_lambda = np.clip(new_lambda, 1e-4, 1.0 - 1e-4)
            estimator.sparsity_rate = float(new_lambda)
            return new_lambda
        return None

    @classmethod
    def tune_all(cls, engine_state, est_a, est_x, est_z, Y, mask=None):
        """
        Unified method to tune all parameters based on current engine state.
        """
        # Update noise variance
        cls.update_noise_var(est_z, engine_state['z_hat'], engine_state['var_z'], Y, mask)
        
        # Update prior variances
        cls.update_prior_vars(est_a, engine_state['a_hat'], engine_state['var_a'])
        cls.update_prior_vars(est_x, engine_state['x_hat'], engine_state['var_x'])
        
        # Update sparsity rates if applicable
        cls.update_sparsity_rate(est_a)
        cls.update_sparsity_rate(est_x)
