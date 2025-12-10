"""
Simple Configuration Handler for NiceGUI.
"""

from nicegui import ui
from typing import Any, get_origin, get_args, Literal
from dataclasses import fields, is_dataclass
from smf.core.config import Config, UI_LABELS


class ConfigHandlerNiceGUI:
    """Simple configuration panel generator."""

    def __init__(self, config: Config, language: str = 'en'):
        self.config = config
        self.language = language
        self.labels = UI_LABELS.get(language, UI_LABELS['en'])

    def render_config_panel(self):
        """Render config panel."""
        
        ui.label('⚙️ Config').classes('text-lg font-bold')
        
        # Matrix section
        with ui.expansion('Matrix', value=True).classes('w-full'):
            with ui.row().classes('w-full flex-wrap gap-2'):
                ui.number('N1', value=float(self.config.matrix.N1)).props('dense').classes('w-24').on_value_change(
                    lambda e: setattr(self.config.matrix, 'N1', int(e.value)) if e.value else None)
                ui.number('N2', value=float(self.config.matrix.N2)).props('dense').classes('w-24').on_value_change(
                    lambda e: setattr(self.config.matrix, 'N2', int(e.value)) if e.value else None)
                ui.number('M', value=float(self.config.matrix.M)).props('dense').classes('w-24').on_value_change(
                    lambda e: setattr(self.config.matrix, 'M', int(e.value)) if e.value else None)
        
        # Alpha section
        with ui.expansion('Alpha', value=True).classes('w-full'):
            with ui.row().classes('w-full flex-wrap gap-2'):
                ui.number('start', value=self.config.alpha.start, step=0.1).props('dense').classes('w-24').bind_value(self.config.alpha, 'start')
                ui.number('stop', value=self.config.alpha.stop, step=0.1).props('dense').classes('w-24').bind_value(self.config.alpha, 'stop')
                ui.number('step', value=self.config.alpha.step, step=0.05).props('dense').classes('w-24').bind_value(self.config.alpha, 'step')
        
        # Algorithm section
        with ui.expansion('Algorithm', value=True).classes('w-full'):
            ui.select(
                ['standard', 'spreading', 'spreading_parallel'],
                label='Mode',
                value=self.config.algorithm.mode
            ).classes('w-full').bind_value(self.config.algorithm, 'mode')
            
            with ui.row().classes('w-full flex-wrap gap-2'):
                ui.number('damping', value=self.config.algorithm.damping, step=0.05).props('dense').classes('w-28').bind_value(self.config.algorithm, 'damping')
                ui.checkbox('early_stop', value=self.config.algorithm.early_stop).bind_value(self.config.algorithm, 'early_stop')
                ui.checkbox('use_compile', value=self.config.algorithm.use_compile).bind_value(self.config.algorithm, 'use_compile')
        
        # Training section
        with ui.expansion('Training', value=True).classes('w-full'):
            with ui.row().classes('w-full flex-wrap gap-2'):
                ui.number('max_steps', value=float(self.config.training.max_steps)).props('dense').classes('w-28').on_value_change(
                    lambda e: setattr(self.config.training, 'max_steps', int(e.value)) if e.value else None)
                ui.number('samples', value=float(self.config.training.samples_per_alpha)).props('dense').classes('w-24').on_value_change(
                    lambda e: setattr(self.config.training, 'samples_per_alpha', int(e.value)) if e.value else None)
                
            with ui.row().classes('items-center gap-2'):
                ui.label('Device:')
                ui.radio(['cpu', 'cuda'], value=self.config.training.device).props('inline').bind_value(self.config.training, 'device')
