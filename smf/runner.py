"""
Runner Glue Code.

This module acts as the bridge between the high-level UI/CLI and the low-level
execution engine (ExperimentRunner). It sets up the ProgressBridge to ensure
UI updates are event-driven and robust.
"""

from typing import Optional, Dict, Any
from pathlib import Path

from smf.core.experiment.config import ExperimentConfig
from smf.core.experiment.runner import ExperimentRunner
from smf.core.experiment.result import ExperimentResult
from smf.core.progress import ProgressBridge

def run_experiment(
    config: ExperimentConfig,
    save: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run an experiment with full UI integration.
    
    Args:
        config: Experiment configuration
        save: Whether to save results to disk automatically
        verbose: Whether to print status messages
        
    Returns:
        Dict containing 'result_path' and raw 'results'
    """
    # 1. Setup Progress Bridge (The Glue)
    # This connects the Runner's events to the Dynamic Capsule UI
    bridge = ProgressBridge(use_rich=True)
    
    # 2. Setup Runner
    runner = ExperimentRunner(verbose=False) # Delegate verbosity to Bridge
    
    # 3. Execute with Observer
    try:
        result = runner.run(config, observer=bridge.on_event)
        
        # 4. Save results
        result_path = None
        if save:
            # Generate default path based on experiment name
            from smf.modules.outputs.storage import ResultStorage
            storage = ResultStorage(config)
            result_path = storage.save(result)
            
            
            # Auto-generate plots if requested
            plots_config = getattr(config, 'execution', None)
            if plots_config and getattr(plots_config, 'include_summary_plot', True):
                try:
                    from smf.modules.outputs.plotting import ResultPlotter
                    plotter = ResultPlotter(config, result_path.parent)
                    # Convert result to dict format for plotting
                    results_dict = {}
                    if hasattr(result, 'results'):
                        for val, single in result.results.items():
                            results_dict[val] = single.metrics if hasattr(single, 'metrics') else {}
                    plotter.plot_summary(results_dict)
                except Exception as plot_err:
                    import logging
                    logging.warning(f"Auto-plot failed: {plot_err}")
        
        return {
            'status': 'success',
            'result_path': result_path,
            'results': result # Return raw object for in-memory use
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'status': 'failed',
            'error': str(e)
        }
