"""
Combined teacher generator for flexible configuration.

Supports mixing multiple teacher features:
- orthogonal: QR decomposition orthogonalization
- scaled_variance: Adjustable variance scaling
- standard: Default Gaussian

This module enables LLM-driven configuration to combine teacher properties
based on natural language descriptions.
"""

import re
from typing import Tuple, Dict, Any
import torch

from ..registry import register_teacher
from .base import TeacherBase
from .standard import StandardTeacher
from .scaled_variance import ScaledVarianceTeacher
from .orthogonal import OrthogonalTeacher


@register_teacher(
    key="combined",
    name="Combined Teacher",
    description="Flexible teacher combining orthogonal and scaling features",
    default_params={
        "orthogonal": False,
        "scale": 1.0,
    },
)
class CombinedTeacher(TeacherBase):
    """
    Flexible teacher that combines multiple features.

    Supports any combination of:
    - Orthogonalization (QR decomposition)
    - Variance scaling

    Example configurations:
    - {"orthogonal": True} -> Pure orthogonal teacher
    - {"scale": 2.0} -> Scaled variance teacher
    - {"orthogonal": True, "scale": 1.5} -> Orthogonal + scaled

    This is the recommended teacher for LLM-driven configuration
    as it can handle mixed requirements.
    """

    def __init__(
        self,
        orthogonal: bool = False,
        scale: float = 1.0,
    ):
        """
        Initialize combined teacher.

        Args:
            orthogonal: Whether to use QR orthogonalization
            scale: Variance scaling factor (1.0 = no scaling)
        """
        self.orthogonal = orthogonal
        self.scale = scale

    def create(
        self,
        N1: int,
        N2: int,
        M: int,
        device: torch.device,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create teacher matrices with combined features.

        Args:
            N1: Number of rows in W
            N2: Number of columns in X
            M: Latent dimension
            device: Torch device
            seed: Random seed

        Returns:
            W_true, X_true: Teacher matrices
        """
        # Choose base teacher
        if self.orthogonal:
            W, X = OrthogonalTeacher().create(N1, N2, M, device, seed)
        else:
            W, X = StandardTeacher().create(N1, N2, M, device, seed)

        # Apply scaling if needed
        if self.scale != 1.0:
            W = W * self.scale
            X = X * self.scale

        return W, X

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CombinedTeacher":
        """
        Create teacher from configuration dictionary.

        Args:
            config: Dictionary with keys:
                - orthogonal: bool (default False)
                - scale: float (default 1.0)
                - type: str (optional, for compatibility)

        Returns:
            CombinedTeacher instance
        """
        orthogonal = config.get("orthogonal", False)
        scale = config.get("scale", 1.0)

        # Handle type-based configuration
        teacher_type = config.get("type", "").lower()
        if "orthogonal" in teacher_type:
            orthogonal = True
        if "scaled" in teacher_type:
            scale = config.get("scale", 2.0)

        return cls(orthogonal=orthogonal, scale=scale)

    @classmethod
    def from_natural_language(cls, description: str) -> "CombinedTeacher":
        """
        Parse natural language description to create teacher.

        Supports:
        - "orthogonal teacher"
        - "scaled teacher with scale 2.0"
        - "orthogonal + scaled teacher"
        - Chinese: "正交教师", "缩放教师"

        Args:
            description: Natural language description

        Returns:
            CombinedTeacher instance
        """
        desc_lower = description.lower()
        orthogonal = False
        scale = 1.0

        # Check for orthogonal
        if "正交" in description or "orthogonal" in desc_lower:
            orthogonal = True

        # Check for scaling
        if "缩放" in description or "scale" in desc_lower:
            # Try to extract scale value
            # Match patterns like: "2倍", "scale=2.0", "scale: 2", "2x"
            patterns = [
                r"(\d+\.?\d*)\s*倍",
                r"scale\s*[=:]\s*(\d+\.?\d*)",
                r"(\d+\.?\d*)\s*x\b",
                r"variance\s*[=:]\s*(\d+\.?\d*)",
            ]
            for pattern in patterns:
                match = re.search(pattern, desc_lower)
                if match:
                    scale = float(match.group(1))
                    break
            else:
                # Default scale if scaling mentioned but no value given
                scale = 2.0

        return cls(orthogonal=orthogonal, scale=scale)

    def __repr__(self) -> str:
        features = []
        if self.orthogonal:
            features.append("orthogonal")
        if self.scale != 1.0:
            features.append(f"scale={self.scale}")
        if not features:
            features.append("standard")
        return f"CombinedTeacher({', '.join(features)})"
