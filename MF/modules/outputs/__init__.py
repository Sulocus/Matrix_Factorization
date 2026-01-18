"""
Output handling module.

Available outputs:
- plotting: Result visualization
- storage: Data persistence
- combined: Flexible output configuration
- publication_style: Publication-quality figure settings
"""

from .plotting import ResultPlotter
from .storage import ResultStorage
from .combined import CombinedOutput
from .publication_style import (
    apply_publication_style,
    PUB_CONFIG,
    ERROR_CONFIG,
    StyleCycler,
    plot_with_error,
    auto_legend,
    get_figure_size,
    create_figure,
)

__all__ = [
    'ResultPlotter',
    'ResultStorage',
    'CombinedOutput',
    # Publication style exports
    'apply_publication_style',
    'PUB_CONFIG',
    'ERROR_CONFIG',
    'StyleCycler',
    'plot_with_error',
    'auto_legend',
    'get_figure_size',
    'create_figure',
]
