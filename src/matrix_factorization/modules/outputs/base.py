"""
Base classes for output handlers.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any

from ...core.config import Config


class OutputBase(ABC):
    """Base class for output handlers."""

    def __init__(self, config: Config, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def save(self, results: Dict[str, Any], **kwargs) -> Path:
        """Save results and return path."""
        pass
