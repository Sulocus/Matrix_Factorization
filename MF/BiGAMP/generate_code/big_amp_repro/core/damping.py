import numpy as np

class AdaptiveDamping:
    """
    Implements adaptive damping and Bethe Free Energy approximation J(t) 
    to ensure convergence in BiG-AMP as described in Section IV-C.
    """
    def __init__(self, initial_beta=0.5, beta_min=0.01, beta_max=1.0, 
                 beta_inc=1.1, beta_dec=0.5, threshold=1e-4):
        self.beta = initial_beta
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_inc = beta_inc
        self.beta_dec = beta_dec
        self.threshold = threshold
        self.prev_cost = float('inf')

    def compute_cost(self, z_hat, p_hat, var_p, var_z, y, mask=None):
        """
        Computes a simplified Bethe Free Energy approximation J(t).
        
        According to the plan:
        For AWGN, J(t) simplifies to a quadratic penalty on (z_hat - p_hat) 
        and log-variance terms.
        
        J(t) ≈ Σ [ (z_hat - p_hat)^2 / var_p + log(var_p) ]
        """
        # Ensure var_p is not zero to avoid division by zero
        safe_var_p = np.maximum(var_p, 1e-12)
        
        if mask is not None:
            # Only consider observed entries for the cost if applicable, 
            # but BiG-AMP cost usually involves the whole Z matrix.
            # However, the residual term (z_hat - p_hat) is defined for all.
            cost_matrix = (z_hat - p_hat)**2 / safe_var_p + np.log(safe_var_p)
            cost = np.sum(cost_matrix)
        else:
            cost_matrix = (z_hat - p_hat)**2 / safe_var_p + np.log(safe_var_p)
            cost = np.sum(cost_matrix)
            
        return cost

    def update_step_size(self, current_cost):
        """
        Updates the damping factor beta based on the cost evolution.
        Returns True if the step is accepted, False otherwise.
        """
        if current_cost <= self.prev_cost + self.threshold:
            # Step accepted: increase beta (less damping)
            self.beta = min(self.beta * self.beta_inc, self.beta_max)
            self.prev_cost = current_cost
            return True
        else:
            # Step rejected: decrease beta (more damping)
            self.beta = max(self.beta * self.beta_dec, self.beta_min)
            # Note: In a full implementation, we would also revert the state.
            # For this reproduction, we update beta and allow the engine to decide.
            return False

    def apply_damping(self, current_val, prev_val):
        """
        Applies the current damping factor to a variable update.
        """
        return self.beta * current_val + (1 - self.beta) * prev_val
