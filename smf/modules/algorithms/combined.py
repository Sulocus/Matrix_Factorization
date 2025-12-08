"""
Combined algorithm selector for flexible configuration.

Supports selecting and configuring algorithms:
- bigamp: BiG-AMP (fast, recommended for large matrices)
- agd: AGD (Alternating Gradient Descent, more robust)

This module enables LLM-driven configuration to select algorithms
based on natural language descriptions.
"""

import re
from typing import Dict, Any, Optional
import torch

from ..registry import register_algorithm
from .base import AlgorithmBase
from .bigamp import BiGAMPAlgorithm
from .agd import AGDAlgorithm
from ...core.config import Config


# Default parameters for each algorithm
ALGORITHM_DEFAULTS = {
    "bigamp": {
        "max_steps": 1000,
        "damping": 0.5,
        "noise_var": 1e-10,
        "samples_per_alpha": 10,
    },
    "agd": {
        "max_epochs": 20000,
        "learning_rate": 0.01,
        "samples_per_alpha": 5,
    },
}


@register_algorithm(
    key="combined",
    name="Combined Algorithm",
    description="Flexible algorithm selection (bigamp or agd)",
    default_params={
        "algorithm": "bigamp",
        "max_steps": 1000,
        "damping": 0.5,
    },
)
class CombinedAlgorithm:
    """
    Flexible algorithm selector.

    Provides a unified interface for selecting between BiG-AMP and AGD,
    with configurable parameters for each.

    Example configurations:
    - {"algorithm": "bigamp", "max_steps": 5000} -> BiG-AMP with 5000 steps
    - {"algorithm": "agd", "max_epochs": 20000} -> AGD with 20000 epochs
    - {"algorithm": "auto"} -> Auto-select based on problem size

    This is the recommended approach for LLM-driven algorithm selection.
    """

    def __init__(
        self,
        algorithm: str = "bigamp",
        max_steps: int = 1000,
        max_epochs: int = 20000,
        damping: float = 0.5,
        learning_rate: float = 0.01,
        noise_var: float = 1e-10,
        samples_per_alpha: int = 10,
        use_compile: bool = True,
    ):
        """
        Initialize combined algorithm selector.

        Args:
            algorithm: Algorithm name ("bigamp", "agd", or "auto")
            max_steps: Max steps for BiG-AMP
            max_epochs: Max epochs for AGD
            damping: Damping for BiG-AMP
            learning_rate: Learning rate for AGD
            noise_var: Noise variance for BiG-AMP
            samples_per_alpha: Number of trials per alpha
            use_compile: Whether to use torch.compile for BiG-AMP
        """
        self.algorithm = algorithm.lower()
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        self.damping = damping
        self.learning_rate = learning_rate
        self.noise_var = noise_var
        self.samples_per_alpha = samples_per_alpha
        self.use_compile = use_compile

    def get_algorithm_class(self, N1: int = None, N2: int = None, M: int = None):
        """
        Get the appropriate algorithm class.

        Args:
            N1, N2, M: Problem dimensions (used for auto-selection)

        Returns:
            Algorithm class (BiGAMPAlgorithm or AGDAlgorithm)
        """
        algo = self.algorithm

        # Auto-select based on problem size
        if algo == "auto":
            if N1 is not None and N2 is not None:
                # BiG-AMP is better for large matrices
                if N1 * N2 > 1000 * 1000:
                    algo = "bigamp"
                else:
                    algo = "agd"
            else:
                algo = "bigamp"  # Default to BiG-AMP

        if algo == "bigamp":
            return BiGAMPAlgorithm
        elif algo == "agd":
            return AGDAlgorithm
        else:
            raise ValueError(f"Unknown algorithm: {algo}")

    def get_params(self) -> Dict[str, Any]:
        """Get algorithm-specific parameters."""
        if self.algorithm in ["bigamp", "auto"]:
            return {
                "max_steps": self.max_steps,
                "damping": self.damping,
                "noise_var": self.noise_var,
                "samples_per_alpha": self.samples_per_alpha,
                "use_compile": self.use_compile,
            }
        else:
            return {
                "max_epochs": self.max_epochs,
                "learning_rate": self.learning_rate,
                "samples_per_alpha": self.samples_per_alpha,
            }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CombinedAlgorithm":
        """
        Create algorithm selector from configuration dictionary.

        Args:
            config: Dictionary with algorithm parameters

        Returns:
            CombinedAlgorithm instance
        """
        return cls(
            algorithm=config.get("algorithm", "bigamp"),
            max_steps=config.get("max_steps", 1000),
            max_epochs=config.get("max_epochs", 20000),
            damping=config.get("damping", 0.5),
            learning_rate=config.get("learning_rate", 0.01),
            noise_var=config.get("noise_var", 1e-10),
            samples_per_alpha=config.get("samples_per_alpha", 10),
            use_compile=config.get("use_compile", True),
        )

    @classmethod
    def from_natural_language(cls, description: str) -> "CombinedAlgorithm":
        """
        Parse natural language description to configure algorithm.

        Supports:
        - "use bigamp with 5000 steps"
        - "agd algorithm"
        - "fast algorithm" -> bigamp
        - "accurate algorithm" -> agd
        - Chinese: "使用BiG-AMP", "梯度下降"

        Args:
            description: Natural language description

        Returns:
            CombinedAlgorithm instance
        """
        desc_lower = description.lower()
        algorithm = "bigamp"
        max_steps = 1000
        max_epochs = 20000
        damping = 0.5

        # Check for algorithm selection
        if any(kw in desc_lower for kw in ["agd", "gradient", "梯度下降"]):
            algorithm = "agd"
        elif any(kw in desc_lower for kw in ["bigamp", "big-amp", "amp", "message passing"]):
            algorithm = "bigamp"
        elif "fast" in desc_lower or "快速" in description:
            algorithm = "bigamp"
        elif "accurate" in desc_lower or "精确" in description:
            algorithm = "agd"
        elif "auto" in desc_lower or "自动" in description:
            algorithm = "auto"

        # Extract step count
        step_match = re.search(r"(\d+)\s*(?:steps?|迭代)", desc_lower)
        if step_match:
            max_steps = int(step_match.group(1))

        # Extract epoch count
        epoch_match = re.search(r"(\d+)\s*(?:epochs?|轮)", desc_lower)
        if epoch_match:
            max_epochs = int(epoch_match.group(1))

        # Extract damping
        damping_match = re.search(r"damping\s*[=:]\s*(\d+\.?\d*)", desc_lower)
        if damping_match:
            damping = float(damping_match.group(1))

        return cls(
            algorithm=algorithm,
            max_steps=max_steps,
            max_epochs=max_epochs,
            damping=damping,
        )

    def __repr__(self) -> str:
        if self.algorithm in ["bigamp", "auto"]:
            return f"CombinedAlgorithm({self.algorithm}, steps={self.max_steps}, damping={self.damping})"
        else:
            return f"CombinedAlgorithm({self.algorithm}, epochs={self.max_epochs})"
