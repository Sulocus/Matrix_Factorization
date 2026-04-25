import numpy as np
from teg_amp_repro.utils.tensor_math import cp_contract

class CPModel:
    """
    Canonical Polyadic (CP) decomposition model.
    A tensor U is represented as a sum of R rank-1 tensors:
    U = sum_{r=1}^R a_r^{(1)} o a_r^{(2)} o ... o a_r^{(d)}
    where o denotes the outer product.
    In matrix form, U is represented by factor matrices A^{(1)}, ..., A^{(d)}
    where A^{(i)} is of size N_i x R.
    """
    def __init__(self, shape, rank):
        """
        Initialize CP model.
        :param shape: tuple of dimensions (N1, N2, ..., Nd)
        :param rank: CP rank (scalar R)
        """
        self.shape = tuple(shape)
        self.rank = rank
        self.factors = []  # List of N_i x R matrices

    def initialize_factors(self, method='gaussian', scale=1.0):
        """
        Initialize factor matrices.
        :param method: 'gaussian' or 'zeros'
        :param scale: standard deviation for gaussian initialization
        """
        self.factors = []
        for dim in self.shape:
            if method == 'gaussian':
                factor = np.random.normal(0, scale, (dim, self.rank))
            elif method == 'zeros':
                factor = np.zeros((dim, self.rank))
            else:
                raise ValueError(f"Unknown initialization method: {method}")
            self.factors.append(factor)

    def set_factors(self, factors):
        """
        Manually set factor matrices.
        :param factors: list of numpy arrays
        """
        self.factors = [np.array(f, copy=True) for f in factors]

    def get_full_tensor(self):
        """
        Reconstruct the full tensor from factor matrices.
        :return: np.ndarray of shape self.shape
        """
        if not self.factors:
            raise ValueError("Factors not initialized.")
        return cp_contract(self.factors)

    def get_element(self, indices):
        """
        Compute a single element of the CP tensor.
        u_{i1, i2, ..., id} = sum_{r=1}^R prod_{k=1}^d A^{(k)}_{ik, r}
        :param indices: tuple of indices
        :return: scalar value
        """
        if len(indices) != len(self.shape):
            raise ValueError(f"Indices length {len(indices)} does not match tensor order {len(self.shape)}")
        
        # Start with a vector of ones of size R
        res = np.ones(self.rank)
        for k, idx in enumerate(indices):
            res *= self.factors[k][idx, :]
        return np.sum(res)

    @property
    def num_params(self):
        """
        Total number of parameters in the CP decomposition.
        """
        return sum(f.size for f in self.factors)

    def __repr__(self):
        return f"CPModel(shape={self.shape}, rank={self.rank})"
