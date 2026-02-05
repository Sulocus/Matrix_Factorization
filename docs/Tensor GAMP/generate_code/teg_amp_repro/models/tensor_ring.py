import numpy as np
from teg_amp_repro.utils.tensor_math import tr_contract, tr_get_slice

class TensorRingModel:
    """
    Tensor Ring (TR) model implementation.
    Handles TR cores, contraction to full tensor, and element-wise access.
    
    The TR decomposition of a d-th order tensor U of shape (N1, ..., Nd) 
    is represented by d cores Z_i of shape (r_i, N_i, r_{i+1}), where r_{d+1} = r_1.
    The element U(x1, ..., xd) is given by:
    U(x1, ..., xd) = Trace( Z_1(:, x1, :) * Z_2(:, x2, :) * ... * Z_d(:, xd, :) )
    """
    
    def __init__(self, shape, ranks):
        """
        Initialize the TR model.
        
        Args:
            shape (tuple): The dimensions of the target tensor (N1, N2, ..., Nd).
            ranks (tuple): The TR ranks (r1, r2, ..., rd). 
                           Note that r_{d+1} is implicitly equal to r1.
        """
        self.shape = shape
        self.ranks = list(ranks)
        self.d = len(shape)
        if len(self.ranks) != self.d:
            raise ValueError(f"Number of ranks ({len(self.ranks)}) must match number of dimensions ({self.d})")
        
        self.cores = None

    def initialize_cores(self, method='gaussian', scale=1.0):
        """
        Initialize TR cores using a specified method.
        
        Args:
            method (str): 'gaussian' or 'zeros'.
            scale (float): Standard deviation for Gaussian initialization.
        """
        self.cores = []
        for i in range(self.d):
            r_in = self.ranks[i]
            r_out = self.ranks[(i + 1) % self.d]
            n_i = self.shape[i]
            
            if method == 'gaussian':
                core = np.random.normal(0, scale, (r_in, n_i, r_out))
            elif method == 'zeros':
                core = np.zeros((r_in, n_i, r_out))
            else:
                raise ValueError(f"Unknown initialization method: {method}")
            
            self.cores.append(core)

    def set_cores(self, cores):
        """
        Set the TR cores manually.
        
        Args:
            cores (list of np.ndarray): List of d cores.
        """
        if len(cores) != self.d:
            raise ValueError(f"Expected {self.d} cores, got {len(cores)}")
        self.cores = [np.array(c) for c in cores]

    def get_full_tensor(self):
        """
        Contract all cores to produce the full tensor.
        
        Returns:
            np.ndarray: The reconstructed full tensor.
        """
        if self.cores is None:
            raise ValueError("Cores are not initialized.")
        return tr_contract(self.cores)

    def get_element(self, indices):
        """
        Compute a single element of the TR tensor without forming the full tensor.
        
        Args:
            indices (tuple): Indices (x1, x2, ..., xd).
            
        Returns:
            float: The value of the tensor at the specified indices.
        """
        if self.cores is None:
            raise ValueError("Cores are not initialized.")
        return tr_get_slice(self.cores, indices)

    def get_core_slice(self, core_idx, slice_idx):
        """
        Get a slice of a specific core: Z_i(:, slice_idx, :).
        
        Args:
            core_idx (int): Index of the core (0 to d-1).
            slice_idx (int): Index along the N_i dimension.
            
        Returns:
            np.ndarray: A matrix of shape (r_i, r_{i+1}).
        """
        if self.cores is None:
            raise ValueError("Cores are not initialized.")
        return self.cores[core_idx][:, slice_idx, :]

    @property
    def num_params(self):
        """
        Total number of parameters in the TR decomposition.
        """
        if self.cores is None:
            return 0
        return sum(core.size for core in self.cores)

    def __repr__(self):
        return f"TensorRingModel(shape={self.shape}, ranks={self.ranks})"
