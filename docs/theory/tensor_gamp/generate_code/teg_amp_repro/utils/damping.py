import numpy as np

class AdaptiveDamping:
    """
    Implements the adaptive damping logic for AMP solvers as described in Appendix G.
    
    The damping factor beta is adjusted based on the cost function J(t).
    If J(t) increases, the step is rejected, beta is decreased, and the state is restored.
    If J(t) decreases or stays the same, the step is accepted and beta is slightly increased.
    """
    def __init__(self, initial_beta=1.0, min_beta=0.01, max_beta=1.0, 
                 decrease_factor=0.5, increase_factor=1.1):
        self.beta = initial_beta
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.decrease_factor = decrease_factor
        self.increase_factor = increase_factor
        self.prev_cost = float('inf')
        
    def update(self, current_cost):
        """
        Updates the damping factor based on the current cost.
        
        Returns:
            bool: True if the step is accepted, False otherwise.
        """
        if current_cost > self.prev_cost:
            # Reject step
            self.beta = max(self.min_beta, self.beta * self.decrease_factor)
            return False
        else:
            # Accept step
            self.prev_cost = current_cost
            self.beta = min(self.max_beta, self.beta * self.increase_factor)
            return True

    def apply(self, current_val, prev_val):
        """
        Applies damping to a value: val = beta * current + (1 - beta) * prev
        """
        return self.beta * current_val + (1.0 - self.beta) * prev_val

    def reset(self, initial_beta=None):
        if initial_beta is not None:
            self.beta = initial_beta
        self.prev_cost = float('inf')

def calculate_kl_gaussian(mu_q, var_q, mu_p, var_p):
    """
    Calculates KL(q||p) for two Gaussian distributions.
    q = N(mu_q, var_q), p = N(mu_p, var_p)
    """
    # KL(N0||N1) = 0.5 * (log(var1/var0) + (var0 + (mu0-mu1)^2)/var1 - 1)
    # Adding epsilon for stability
    eps = 1e-12
    var_q = np.maximum(var_q, eps)
    var_p = np.maximum(var_p, eps)
    
    kl = 0.5 * (np.log(var_p / var_q) + (var_q + (mu_q - mu_p)**2) / var_p - 1.0)
    return np.sum(kl)
