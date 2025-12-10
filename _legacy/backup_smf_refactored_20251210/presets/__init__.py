"""
Preset management for SMF experiments.

Presets are saved configurations that can be quickly loaded and reused.
"""

from .manager import PresetManager, list_presets, load_preset, save_preset

__all__ = ['PresetManager', 'list_presets', 'load_preset', 'save_preset']
