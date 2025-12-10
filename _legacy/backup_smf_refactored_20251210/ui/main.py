"""
SMF Experiment Platform - NiceGUI (Minimal Version)
Simplified for debugging.
"""

import os
from typing import Dict, List, Any, Optional

from nicegui import ui
from smf.core.config import Config
from smf.ui.execution_nicegui import ExecutionManagerNiceGUI, TerminalSimulator
from smf.ui.plots import plot_phase_transition


# Global state
class AppState:
    def __init__(self):
        self.config = Config()
        self.execution_manager = ExecutionManagerNiceGUI()
        self.results = {}

_state = None

def get_state():
    global _state
    if _state is None:
        _state = AppState()
    return _state


@ui.page('/')
def main_page():
    state = get_state()
    cfg = state.config
    
    # Simple header
    ui.label('🔬 SMF Platform').classes('text-2xl font-bold')
    ui.separator()
    
    # Two column layout
    with ui.row().classes('w-full gap-4'):
        
        # Left: Config
        with ui.card().classes('w-80'):
            ui.label('Configuration').classes('text-lg font-bold')
            
            # Matrix
            with ui.row().classes('gap-2'):
                ui.number('N1', value=float(cfg.matrix.N1)).classes('w-20').on_value_change(
                    lambda e: setattr(cfg.matrix, 'N1', int(e.value)) if e.value else None)
                ui.number('N2', value=float(cfg.matrix.N2)).classes('w-20').on_value_change(
                    lambda e: setattr(cfg.matrix, 'N2', int(e.value)) if e.value else None)
                ui.number('M', value=float(cfg.matrix.M)).classes('w-20').on_value_change(
                    lambda e: setattr(cfg.matrix, 'M', int(e.value)) if e.value else None)
            
            # Alpha
            with ui.row().classes('gap-2'):
                a_start = ui.number('α start', value=cfg.alpha.start).classes('w-20')
                a_stop = ui.number('α stop', value=cfg.alpha.stop).classes('w-20')
                a_step = ui.number('α step', value=cfg.alpha.step).classes('w-20')
            a_start.bind_value(cfg.alpha, 'start')
            a_stop.bind_value(cfg.alpha, 'stop')
            a_step.bind_value(cfg.alpha, 'step')
            
            # Algorithm mode
            ui.select(
                options=['standard', 'spreading', 'spreading_parallel'],
                label='Algorithm Mode',
                value=cfg.algorithm.mode
            ).classes('w-full').bind_value(cfg.algorithm, 'mode')
            
            # Training
            ui.number('Max Steps', value=float(cfg.training.max_steps)).classes('w-full').on_value_change(
                lambda e: setattr(cfg.training, 'max_steps', int(e.value)) if e.value else None)
            
            ui.separator()
            
            # Run button
            run_btn = ui.button('🚀 RUN', color='primary').classes('w-full')
            status = ui.label('Ready').classes('text-gray-500')
        
        # Right: Output
        with ui.column().classes('flex-1'):
            with ui.tabs().classes('w-full') as tabs:
                tab_term = ui.tab('Terminal')
                tab_plot = ui.tab('Plots')
            
            with ui.tab_panels(tabs, value=tab_term).classes('w-full'):
                with ui.tab_panel(tab_term):
                    term_container = ui.column().classes('w-full')
                    terminal = TerminalSimulator(term_container)
                
                with ui.tab_panel(tab_plot):
                    plot_container = ui.column().classes('w-full')
                    ui.label('Run experiment to see plots').classes('text-gray-400')
    
    # Run logic
    def on_complete(result, error):
        if error:
            status.text = f'Error: {error}'
            ui.notify(f'Error: {error}', type='negative')
        elif result:
            status.text = 'Done!'
            ui.notify('Experiment completed!', type='positive')
            # Show plot
            plot_container.clear()
            with plot_container:
                with ui.matplotlib(figsize=(8, 5)).figure as fig:
                    plot_phase_transition(result, fig=fig)
        run_btn.enable()
    
    def start_run():
        run_btn.disable()
        status.text = 'Running...'
        state.execution_manager.start(cfg, terminal, on_complete=on_complete)
    
    run_btn.on_click(start_run)


# Entry point - native=False to prevent opening browser automatically
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title='SMF Platform',
        port=8524,
        show=False,      # Don't auto-open browser
        reload=False,    # No hot reload
        native=False,    # Not native mode
    )
