"""
Parameter Space for configuration sweeps.

Supports:
- Grid search over parameter combinations
- Random sampling from parameter space
- LLM-generated parameter ranges

Example:
    space = ParameterSpace()
    space.add_range("damping", 0.3, 0.7, 0.1)
    space.add_choice("graph_key", ["random", "biregular"])

    for config in space.generate_grid():
        # config = {"damping": 0.3, "graph_key": "random"}
        run_experiment(merge_config(base_config, config))
"""

from dataclasses import dataclass, field
from typing import List, Any, Dict, Optional
from itertools import product
import numpy as np
import random


@dataclass
class ParameterSpec:
    """Specification for a single parameter."""
    name: str
    values: List[Any]
    description: str = ""

    def sample(self) -> Any:
        """Sample a random value from this parameter."""
        return random.choice(self.values)


@dataclass
class ParameterSpace:
    """
    Parameter search space for configuration sweeps.

    Supports both grid search (all combinations) and random sampling.
    """
    params: List[ParameterSpec] = field(default_factory=list)
    name: str = "sweep"

    def add_range(
        self,
        name: str,
        start: float,
        stop: float,
        step: float,
        description: str = "",
    ) -> "ParameterSpace":
        """
        Add a continuous parameter range (discretized).

        Args:
            name: Parameter name (e.g., "damping")
            start: Start value (inclusive)
            stop: Stop value (inclusive)
            step: Step size
            description: Optional description

        Returns:
            self for chaining
        """
        values = list(np.arange(start, stop + step / 2, step))
        # Round to avoid floating point artifacts
        values = [round(v, 6) for v in values]
        self.params.append(ParameterSpec(name, values, description))
        return self

    def add_choice(
        self,
        name: str,
        choices: List[Any],
        description: str = "",
    ) -> "ParameterSpace":
        """
        Add a discrete choice parameter.

        Args:
            name: Parameter name (e.g., "graph_key")
            choices: List of possible values
            description: Optional description

        Returns:
            self for chaining
        """
        self.params.append(ParameterSpec(name, choices, description))
        return self

    def add_log_range(
        self,
        name: str,
        start: float,
        stop: float,
        num_points: int,
        description: str = "",
    ) -> "ParameterSpace":
        """
        Add a logarithmically-spaced parameter range.

        Useful for learning rates, noise variances, etc.

        Args:
            name: Parameter name
            start: Start value (must be > 0)
            stop: Stop value
            num_points: Number of points
            description: Optional description

        Returns:
            self for chaining
        """
        values = list(np.logspace(np.log10(start), np.log10(stop), num_points))
        self.params.append(ParameterSpec(name, values, description))
        return self

    def generate_grid(self) -> List[Dict[str, Any]]:
        """
        Generate all combinations (grid search).

        Returns:
            List of config dictionaries
        """
        if not self.params:
            return [{}]

        names = [p.name for p in self.params]
        value_lists = [p.values for p in self.params]

        configs = []
        for combo in product(*value_lists):
            configs.append(dict(zip(names, combo)))
        return configs

    def sample_random(self, n: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Randomly sample n configurations.

        Args:
            n: Number of configurations to sample
            seed: Optional random seed

        Returns:
            List of config dictionaries
        """
        if seed is not None:
            random.seed(seed)

        configs = []
        for _ in range(n):
            config = {p.name: p.sample() for p in self.params}
            configs.append(config)
        return configs

    def get_total_combinations(self) -> int:
        """Get total number of grid search combinations."""
        if not self.params:
            return 1
        total = 1
        for p in self.params:
            total *= len(p.values)
        return total

    def describe(self) -> str:
        """Get human-readable description of the parameter space."""
        lines = [f"ParameterSpace: {self.name}"]
        lines.append(f"Total combinations: {self.get_total_combinations()}")
        lines.append("")
        for p in self.params:
            values_str = str(p.values) if len(p.values) <= 5 else f"[{p.values[0]}, ..., {p.values[-1]}] ({len(p.values)} values)"
            lines.append(f"  {p.name}: {values_str}")
            if p.description:
                lines.append(f"    {p.description}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "name": self.name,
            "params": [
                {
                    "name": p.name,
                    "values": p.values,
                    "description": p.description,
                }
                for p in self.params
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParameterSpace":
        """Create from dictionary (e.g., from LLM output)."""
        space = cls(name=data.get("name", "sweep"))
        for p in data.get("params", []):
            space.params.append(ParameterSpec(
                name=p["name"],
                values=p["values"],
                description=p.get("description", ""),
            ))
        return space


def merge_config(base_config: Any, overrides: Dict[str, Any]) -> Any:
    """
    Merge override parameters into a base config.

    Supports nested paths like "algorithm.damping".

    Args:
        base_config: Base Config object
        overrides: Dictionary of overrides

    Returns:
        New config with overrides applied
    """
    import copy

    # Deep copy to avoid modifying original
    config = copy.deepcopy(base_config)

    for key, value in overrides.items():
        if "." in key:
            # Nested path
            parts = key.split(".")
            obj = config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        else:
            # Top-level or direct attribute
            if hasattr(config, key):
                setattr(config, key, value)
            elif hasattr(config.algorithm, key):
                setattr(config.algorithm, key, value)
            elif hasattr(config.matrix, key):
                setattr(config.matrix, key, value)
            elif hasattr(config.training, key):
                setattr(config.training, key, value)
            else:
                # Just set it at top level
                setattr(config, key, value)

    return config


# Convenience builders
def quick_damping_sweep(start: float = 0.3, stop: float = 0.7, step: float = 0.1) -> ParameterSpace:
    """Create a quick damping parameter sweep."""
    return ParameterSpace(name="damping_sweep").add_range("damping", start, stop, step)


def quick_noise_sweep(start: float = 1e-12, stop: float = 1e-8, num_points: int = 5) -> ParameterSpace:
    """Create a quick noise variance sweep (log scale)."""
    return ParameterSpace(name="noise_sweep").add_log_range("noise_var", start, stop, num_points)


def quick_size_sweep(sizes: List[int] = None) -> ParameterSpace:
    """Create a quick matrix size sweep."""
    if sizes is None:
        sizes = [100, 200, 500, 1000]
    space = ParameterSpace(name="size_sweep")
    space.add_choice("N1", sizes)
    space.add_choice("N2", sizes)
    return space
