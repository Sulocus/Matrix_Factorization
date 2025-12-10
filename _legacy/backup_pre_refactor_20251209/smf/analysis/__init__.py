"""
Analysis tools for SMF experiments.

Includes:
- Experiment comparison
- Publication figure generation
"""

from .compare import ExperimentComparer
from .publication import PublicationFigure, FigureStyle, export_for_publication

__all__ = [
    'ExperimentComparer',
    'PublicationFigure',
    'FigureStyle',
    'export_for_publication',
]
