"""
Unified Teacher Generator module.

Supports all teacher initialization types:
1. standard: Gaussian N(0, 1) initialization
2. orthogonal: QR-decomposed orthonormal columns/rows
3. scaled_variance: Adjustable variance N(0, k/√M)
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Literal, TYPE_CHECKING
import torch
import math

from ..registry import register_teacher
from .base import TeacherBase

if TYPE_CHECKING:
    from ..graphs.supergraph import SuperGraphData

# Type alias
TeacherType = Literal["standard", "orthogonal", "scaled_variance"]


# ============================================================================
# Spreading Data Structures
# ============================================================================

@dataclass
class SpreadingData:
    """
    Sparse storage for random spreading coefficients.
    """
    i_idx: torch.Tensor
    j_idx: torch.Tensor
    F: torch.Tensor
    Y_values: torch.Tensor
    seed: int
    M: int

    @property
    def num_edges(self) -> int:
        return len(self.i_idx)

    @property
    def device(self) -> torch.device:
        return self.F.device

    def to(self, device: torch.device) -> "SpreadingData":
        return SpreadingData(
            i_idx=self.i_idx.to(device),
            j_idx=self.j_idx.to(device),
            F=self.F.to(device),
            Y_values=self.Y_values.to(device),
            seed=self.seed,
            M=self.M,
        )

    def clone(self) -> "SpreadingData":
        return SpreadingData(
            i_idx=self.i_idx.clone(),
            j_idx=self.j_idx.clone(),
            F=self.F.clone(),
            Y_values=self.Y_values.clone(),
            seed=self.seed,
            M=self.M,
        )


@dataclass
class SpreadingDataParallel:
    """Parallel spreading data for Super-Graph strategy."""
    supergraph: 'SuperGraphData'
    F_super: torch.Tensor
    Y_super: torch.Tensor
    M: int
    alpha_values: torch.Tensor
    W_teacher: torch.Tensor
    X_teacher: torch.Tensor

    @property
    def S(self) -> int:
        return self.F_super.shape[0]

    @property
    def A(self) -> int:
        return len(self.alpha_values)

    @property
    def C_max(self) -> int:
        return self.F_super.shape[1]

    @property
    def device(self) -> torch.device:
        return self.F_super.device

    def get_F(self, sample_idx: int) -> torch.Tensor:
        return self.F_super[sample_idx]

    def get_Y_masked(self, sample_idx: int, alpha_idx: int) -> torch.Tensor:
        C_k = self.supergraph.get_active_edges(alpha_idx)
        return self.Y_super[sample_idx, :C_k]

    def to(self, device: torch.device) -> 'SpreadingDataParallel':
        return SpreadingDataParallel(
            supergraph=self.supergraph.to(device),
            F_super=self.F_super.to(device),
            Y_super=self.Y_super.to(device),
            M=self.M,
            alpha_values=self.alpha_values.to(device),
            W_teacher=self.W_teacher.to(device),
            X_teacher=self.X_teacher.to(device),
        )


# ============================================================================
# Helper Functions
# ============================================================================

def generate_spreading_coefficients(
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    M: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate deterministic spreading coefficients F."""
    C = len(i_idx)
    if C == 0:
        return torch.empty(0, M, device=device, dtype=torch.float32)

    combined_seed = seed ^ 0x5DEECE66D
    gen = torch.Generator(device=device)
    gen.manual_seed(combined_seed)
    F = torch.randn(C, M, device=device, dtype=torch.float32, generator=gen)
    return F


def compute_sparse_Y(
    W: torch.Tensor,
    X: torch.Tensor,
    F: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
) -> torch.Tensor:
    """Compute Y values at observed positions with spreading."""
    C = len(i_idx)
    if C == 0:
        return torch.empty(0, device=W.device, dtype=W.dtype)

    M = W.shape[1]
    alpha_scale = 1.0 / math.sqrt(M)

    W_selected = W[i_idx, :]
    X_selected = X[:, j_idx].T
    Y_values = alpha_scale * (F * W_selected * X_selected).sum(dim=1)
    return Y_values


def compute_sparse_Y_batched(
    W: torch.Tensor,
    X: torch.Tensor,
    F: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
) -> torch.Tensor:
    """Compute Y values for batched W and X."""
    S = W.shape[0]
    C = len(i_idx)
    if C == 0:
        return torch.empty(S, 0, device=W.device, dtype=W.dtype)

    M = W.shape[2]
    alpha_scale = 1.0 / math.sqrt(M)

    W_selected = W[:, i_idx, :]
    X_selected = X[:, :, j_idx].transpose(1, 2)
    F_expanded = F.unsqueeze(0)

    Y_values = alpha_scale * (F_expanded * W_selected * X_selected).sum(dim=2)
    return Y_values


# ============================================================================
# Teacher Initialization Strategies
# ============================================================================

def create_standard(
    N1: int, N2: int, M: int, device: torch.device, seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standard Gaussian N(0, 1) initialization."""
    torch.manual_seed(seed)
    W = torch.randn((N1, M), device=device, dtype=torch.float32)
    X = torch.randn((M, N2), device=device, dtype=torch.float32)
    return W, X


def create_orthogonal(
    N1: int, N2: int, M: int, device: torch.device, seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Orthogonal teacher using QR decomposition.
    
    Properties:
    - W^T W = I_M (orthonormal columns)
    - X X^T = I_M (orthonormal rows)
    - Scaled to match expected Frobenius norm of standard teacher
    """
    torch.manual_seed(seed)
    
    # Generate random matrices
    W_raw = torch.randn(N1, M, device=device, dtype=torch.float32)
    X_raw = torch.randn(M, N2, device=device, dtype=torch.float32)
    
    # QR decomposition for W (thin QR: N1 x M -> Q: N1 x M)
    W_ortho, _ = torch.linalg.qr(W_raw, mode='reduced')
    
    # QR decomposition for X^T, then transpose back
    X_ortho_T, _ = torch.linalg.qr(X_raw.T, mode='reduced')
    X_ortho = X_ortho_T.T
    
    # Scale to match expected Frobenius norm of standard teacher
    # Standard: E[||W||_F^2] = N1 * M * 1 = N1*M, but per element var=1
    # Orthogonal: ||W_ortho||_F^2 = M (since orthonormal columns)
    # Scale factor: sqrt(N1/M) to get ||W||_F^2 ≈ N1
    W_true = W_ortho * math.sqrt(N1 / M)
    X_true = X_ortho * math.sqrt(N2 / M)
    
    return W_true, X_true


def create_scaled_variance(
    N1: int, N2: int, M: int, device: torch.device, seed: int,
    variance_scale: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled variance Gaussian initialization.
    
    Elements are sampled from N(0, variance_scale / sqrt(M)).
    - variance_scale = 1.0: standard initialization
    - variance_scale = 2.0: doubled variance
    """
    torch.manual_seed(seed)
    
    scale = variance_scale / math.sqrt(M)
    W = torch.randn((N1, M), device=device, dtype=torch.float32) * scale
    X = torch.randn((M, N2), device=device, dtype=torch.float32) * scale
    
    return W, X


# Strategy dictionary
TEACHER_CREATORS = {
    'standard': create_standard,
    'orthogonal': create_orthogonal,
    'scaled_variance': create_scaled_variance,
}


# ============================================================================
# Main Teacher Class
# ============================================================================

@register_teacher(
    key="generator",
    name="Unified Teacher Generator",
    description="Unified teacher supporting Standard, Orthogonal, and Scaled Variance",
    default_params={"type": "standard", "variance_scale": 1.0},
)
class TeacherGenerator(TeacherBase):
    """
    Unified Teacher Generator.

    Supports:
    1. standard: Gaussian N(0, 1) initialization
    2. orthogonal: QR-decomposed orthonormal columns/rows  
    3. scaled_variance: Adjustable variance N(0, k/√M)
    
    Also provides spreading data creation for spreading models.
    """

    def __init__(
        self, 
        type: TeacherType = "standard",
        variance_scale: float = 1.0,
        spreading_seed: int = 12345,
    ):
        """
        Initialize teacher generator.
        
        Args:
            type: Teacher type ('standard', 'orthogonal', 'scaled_variance')
            variance_scale: Variance multiplier (only for scaled_variance)
            spreading_seed: Random seed for spreading coefficients F
        """
        self.type = type
        self.variance_scale = variance_scale
        self.spreading_seed = spreading_seed
        
        if type not in TEACHER_CREATORS:
            raise ValueError(
                f"Unknown teacher type: {type}. "
                f"Available: {list(TEACHER_CREATORS.keys())}"
            )

    def create(
        self,
        N1: int,
        N2: int,
        M: int,
        device: torch.device,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create teacher matrices W, X based on configured type."""
        if self.type == 'scaled_variance':
            return create_scaled_variance(N1, N2, M, device, seed, self.variance_scale)
        else:
            creator = TEACHER_CREATORS[self.type]
            return creator(N1, N2, M, device, seed)

    def create_with_spreading(
        self,
        N1: int,
        N2: int,
        M: int,
        i_idx: torch.Tensor,
        j_idx: torch.Tensor,
        device: torch.device,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor, SpreadingData]:
        """Create W, X and SpreadingData."""
        W, X = self.create(N1, N2, M, device, seed)

        i_idx = i_idx.to(device)
        j_idx = j_idx.to(device)

        F = generate_spreading_coefficients(
            i_idx, j_idx, M, self.spreading_seed, device
        )
        Y_values = compute_sparse_Y(W, X, F, i_idx, j_idx)

        spreading_data = SpreadingData(
            i_idx=i_idx,
            j_idx=j_idx,
            F=F,
            Y_values=Y_values,
            seed=self.spreading_seed,
            M=M,
        )
        return W, X, spreading_data

    def regenerate_spreading_data(
        self,
        W: torch.Tensor,
        X: torch.Tensor,
        i_idx: torch.Tensor,
        j_idx: torch.Tensor,
    ) -> SpreadingData:
        """Regenerate SpreadingData for new edges."""
        device = W.device
        M = W.shape[1]

        i_idx = i_idx.to(device)
        j_idx = j_idx.to(device)

        F = generate_spreading_coefficients(
            i_idx, j_idx, M, self.spreading_seed, device
        )
        Y_values = compute_sparse_Y(W, X, F, i_idx, j_idx)

        return SpreadingData(
            i_idx=i_idx,
            j_idx=j_idx,
            F=F,
            Y_values=Y_values,
            seed=self.spreading_seed,
            M=M,
        )
    
    def get_description(self) -> str:
        """Get human-readable description of this teacher."""
        if self.type == 'standard':
            return "Standard Gaussian N(0, 1)"
        elif self.type == 'orthogonal':
            return "Orthogonal (QR decomposition)"
        elif self.type == 'scaled_variance':
            return f"Scaled Gaussian N(0, {self.variance_scale}/√M)"
        else:
            return f"Unknown type: {self.type}"
