"""
Data Factory for Experiment Framework.

Creates all data needed for experiments:
1. Teacher matrices (W_teacher, X_teacher, Y_teacher)
2. Observation masks (for dense algorithms)
3. SpreadingData (for spreading algorithms)

Separates data creation from algorithm computation.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List, Any, TYPE_CHECKING
import torch
import math

if TYPE_CHECKING:
    from .config import ExperimentConfig


@dataclass
class ExperimentData:
    """
    Container for all experiment data.
    
    This is what gets passed to algorithm.run_single().
    """
    # Teacher matrices
    W_teacher: torch.Tensor  # (N1, M)
    X_teacher: torch.Tensor  # (M, N2)
    Y_teacher: torch.Tensor  # (N1, N2)
    
    # For dense algorithms (AGD, BiGAMP)
    masks: Optional[torch.Tensor] = None  # (num_alphas, N1, N2) or (N1, N2)
    
    # For spreading algorithms
    spreading_data: Optional[Any] = None  # SpreadingDataParallel
    
    # Parameters
    alpha_values: Optional[List[float]] = None
    device: Optional[torch.device] = None
    
    @property
    def N1(self) -> int:
        return self.W_teacher.shape[0]
    
    @property
    def N2(self) -> int:
        return self.X_teacher.shape[1]
    
    @property
    def M(self) -> int:
        return self.W_teacher.shape[1]


class DataFactory:
    """
    Factory for creating experiment data.
    
    Centralizes all data creation logic, separating it from algorithms.
    
    Usage:
        factory = DataFactory(device)
        data = factory.create(config, alpha_values=[0.5, 1.0, 1.5])
    """
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def create(
        self,
        config: 'ExperimentConfig',
        alpha_values: Optional[List[float]] = None,
    ) -> ExperimentData:
        """
        Create all data needed for an experiment.
        
        Args:
            config: Experiment configuration
            alpha_values: Override alpha values (else use config.alpha_values)
            
        Returns:
            ExperimentData with all required tensors
        """
        alpha_values = alpha_values or config.alpha_values
        
        # Get teacher initialization distribution
        init_distribution = "gaussian"  # default
        if hasattr(config, 'teacher') and config.teacher:
            init_distribution = config.teacher.init_distribution
        
        # Create teacher
        W_teacher, X_teacher, Y_teacher = self.create_teacher(
            N1=config.matrix.N1,
            N2=config.matrix.N2,
            M=config.matrix.M,
            teacher_key=getattr(config, 'teacher_key', 'standard'),
            seed=config.seeds.teacher_seed,
            init_distribution=init_distribution,
        )
        
        # Create algorithm-specific data
        if getattr(config, 'is_spreading_algorithm', 'spreading' in config.algorithm_key):
            spreading_data = self.create_spreading_data(
                config=config,
                W_teacher=W_teacher,
                X_teacher=X_teacher,
                alpha_values=alpha_values,
            )
            return ExperimentData(
                W_teacher=W_teacher,
                X_teacher=X_teacher,
                Y_teacher=Y_teacher,
                spreading_data=spreading_data,
                alpha_values=alpha_values,
                device=self.device,
            )
        else:
            masks = self.create_masks(
                N1=config.matrix.N1,
                N2=config.matrix.N2,
                M=config.matrix.M,
                alpha_values=alpha_values,
                seed=config.seeds.base_seed,
            )
            return ExperimentData(
                W_teacher=W_teacher,
                X_teacher=X_teacher,
                Y_teacher=Y_teacher,
                masks=masks,
                alpha_values=alpha_values,
                device=self.device,
            )
    
    def create_teacher(
        self,
        N1: int,
        N2: int,
        M: int,
        teacher_key: str = "standard",
        seed: int = 12345,
        init_distribution: str = "gaussian",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create teacher matrices.
        
        Args:
            N1, N2, M: Matrix dimensions
            teacher_key: "standard" or "orthogonal"
            seed: Random seed
            init_distribution: "gaussian" or "rademacher" (for standard teacher only)
            
        Returns:
            (W_teacher, X_teacher, Y_teacher)
        """
        torch.manual_seed(seed)
        scale = 1.0 / math.sqrt(M)
        
        if teacher_key == "orthogonal":
            # Orthogonal teacher using SVD
            W_raw = torch.randn(N1, M, device=self.device)
            X_raw = torch.randn(M, N2, device=self.device)
            
            # Make orthonormal
            U, _, _ = torch.linalg.svd(W_raw, full_matrices=False)
            W_teacher = U[:, :M] * math.sqrt(M)  # Scale for correct variance
            
            _, _, Vh = torch.linalg.svd(X_raw, full_matrices=False)
            X_teacher = Vh[:M, :] * math.sqrt(M)
        else:
            # Standard teacher with configurable distribution
            if init_distribution == "rademacher":
                # Rademacher: ±1 with equal probability, scaled by 1/√M
                W_teacher = (2 * torch.randint(0, 2, (N1, M), device=self.device, dtype=torch.float32) - 1) * scale
                X_teacher = (2 * torch.randint(0, 2, (M, N2), device=self.device, dtype=torch.float32) - 1) * scale
            else:
                # Gaussian: N(0, 1/√M)
                W_teacher = torch.randn(N1, M, device=self.device) * scale
                X_teacher = torch.randn(M, N2, device=self.device) * scale
        
        # Compute Y = (1/√M) * W @ X
        Y_teacher = scale * torch.matmul(W_teacher, X_teacher)
        
        return W_teacher, X_teacher, Y_teacher
    
    def create_masks(
        self,
        N1: int,
        N2: int,
        M: int,
        alpha_values: List[float],
        seed: int = 42,
    ) -> torch.Tensor:
        """
        Create observation masks for dense algorithms.
        
        Uses the standard definition: num_observed = alpha * M * N1
        This means alpha represents the average degree per row divided by M.
        
        Args:
            N1, N2: Matrix dimensions
            M: Hidden dimension (used for alpha scaling)
            alpha_values: Observation density parameters
            seed: Random seed
            
        Returns:
            masks: (num_alphas, N1, N2) boolean tensor
        """
        num_alphas = len(alpha_values)
        masks = torch.zeros(num_alphas, N1, N2, dtype=torch.bool, device=self.device)
        
        for a, alpha in enumerate(alpha_values):
            # Use alpha-specific seed (consistent with reference implementation)
            alpha_seed = seed + int(alpha * 1000)
            torch.manual_seed(alpha_seed)
            
            # Number of observed entries: alpha * M * N1
            # This gives average degree per row = alpha * M
            num_observed = int(alpha * M * N1)
            num_observed = min(num_observed, N1 * N2)  # Cap at total elements
            
            # Random permutation for observation positions
            perm = torch.randperm(N1 * N2, device=self.device)[:num_observed]
            mask_flat = torch.zeros(N1 * N2, dtype=torch.bool, device=self.device)
            mask_flat[perm] = True
            masks[a] = mask_flat.reshape(N1, N2)
        
        return masks
    
    def create_spreading_data(
        self,
        config: 'ExperimentConfig',
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
        alpha_values: List[float],
    ) -> Any:
        """
        Create SpreadingDataParallel for spreading algorithms.
        
        This separates data creation from the algorithm, allowing:
        - Reuse of same SpreadingData across step scans
        - Control of graph structure via seeds
        
        Args:
            config: Experiment configuration
            W_teacher, X_teacher: Teacher matrices
            alpha_values: List of alpha values
            
        Returns:
            SpreadingDataParallel instance
        """
        # Import here to avoid circular imports
        from ...modules.graphs.supergraph import create_supergraph
        from ...modules.algorithms.bigamp.spreading import (
            generate_F_super,
            compute_Y_super,
            SpreadingDataParallel,
        )
        
        N1, M = W_teacher.shape
        _, N2 = X_teacher.shape
        S = config.training.samples_per_alpha
        
        # Create SuperGraph (graph structure)
        supergraph = create_supergraph(
            N1=N1,
            N2=N2,
            M=M,
            alpha_values=alpha_values,
            S=S,
            base_seed=config.seeds.base_seed,
            device=self.device,
            num_workers=config.training.num_workers,
        )
        
        # Generate F (spreading coefficients)
        f_distribution = config.spreading.f_distribution if config.spreading else "rademacher"
        F_super = generate_F_super(
            supergraph=supergraph,
            M=M,
            base_seed=config.seeds.spreading_seed,
            device=self.device,
            f_distribution=f_distribution,
        )
        
        # Compute Y (observations)
        Y_super = compute_Y_super(
            W_teacher=W_teacher,
            X_teacher=X_teacher,
            supergraph=supergraph,
            F_super=F_super,
        )
        
        return SpreadingDataParallel(
            supergraph=supergraph,
            F_super=F_super,
            Y_super=Y_super,
            M=M,
            alpha_values=torch.tensor(alpha_values, device=self.device),
            W_teacher=W_teacher,
            X_teacher=X_teacher,
            f_distribution=f_distribution,
        )
    
    def create_spreading_data_for_steps_scan(
        self,
        config: 'ExperimentConfig',
        alpha: float,
    ) -> Any:
        """
        Create SpreadingData for step scanning (single alpha, reusable).
        
        This creates data that can be reused across multiple step counts.
        
        Args:
            config: Experiment configuration
            alpha: Fixed alpha value for step scan
            
        Returns:
            SpreadingDataParallel with single alpha
        """
        W_teacher, X_teacher, _ = self.create_teacher(
            N1=config.matrix.N1,
            N2=config.matrix.N2,
            M=config.matrix.M,
            teacher_key=getattr(config, 'teacher_key', 'standard'),
            seed=config.seeds.teacher_seed,
        )
        
        return self.create_spreading_data(
            config=config,
            W_teacher=W_teacher,
            X_teacher=X_teacher,
            alpha_values=[alpha],  # Single alpha for step scan
        )
