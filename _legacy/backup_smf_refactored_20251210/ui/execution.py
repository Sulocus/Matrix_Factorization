"""
Threaded Execution Manager for SMF - v4.0 (Fixed Progress).
Simple time-based progress estimation with proper Streamlit integration.
"""

import threading
import queue
import time
import sys
import os
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from smf.core.config import Config
from smf.core.runner import run_experiment


@dataclass 
class ProgressState:
    """Simple progress state for Streamlit UI."""
    is_running: bool = False
    start_time: float = 0.0
    estimated_total: float = 60.0
    message: str = "Ready"
    
    @property
    def elapsed(self) -> float:
        if self.start_time <= 0:
            return 0.0
        return time.time() - self.start_time
    
    @property
    def progress(self) -> float:
        if self.estimated_total <= 0:
            return 0.0
        return min(0.99, self.elapsed / self.estimated_total) if self.is_running else 1.0 if self.elapsed > 0 else 0.0


def clean_terminal_logs(text: str) -> str:
    """Clean ANSI codes for plain text display."""
    # Remove ANSI escape sequences
    text = re.sub(r'\x1b\[[0-9;]*[mKA]', '', text)
    # Handle carriage returns
    lines = []
    for line in text.split('\n'):
        if '\r' in line:
            line = line.split('\r')[-1]
        lines.append(line)
    return '\n'.join(lines)


class ExperimentThread(threading.Thread):
    """Thread for running experiments."""
    
    def __init__(self, config: Config, run_name: str, log_queue: queue.Queue):
        super().__init__()
        self.config = config
        self.run_name = run_name
        self.log_queue = log_queue
        self.result = None
        self.error = None
        self.daemon = True
        self.start_time = 0.0
        self.is_complete = False

    def run(self):
        """Run experiment with log capture."""
        self.start_time = time.time()
        
        # Capture stdout/stderr
        class LogWriter:
            def __init__(self, q):
                self.q = q
                self.encoding = 'utf-8'
            def write(self, msg):
                if msg:
                    self.q.put(msg)
            def flush(self):
                pass
            def isatty(self):
                return True
        
        # Enable UI mode to disable Rich Live (avoids scrolling issues)
        os.environ["SMF_UI_MODE"] = "1"
        os.environ["FORCE_COLOR"] = "1"
        os.environ["COLUMNS"] = "120"
        
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = LogWriter(self.log_queue)
        sys.stderr = LogWriter(self.log_queue)
        
        try:
            self.log_queue.put(f"\n{'='*60}\n")
            self.log_queue.put(f"[{self.run_name}] Starting experiment...\n")
            self.log_queue.put(f"{'='*60}\n\n")
            
            self.result = run_experiment(self.config)
            self.result['name'] = self.run_name
            
            elapsed = time.time() - self.start_time
            self.log_queue.put(f"\n{'='*60}\n")
            self.log_queue.put(f"[{self.run_name}] Complete in {elapsed:.1f}s\n")
            self.log_queue.put(f"{'='*60}\n")
            
        except Exception as e:
            import traceback
            self.error = str(e)
            self.log_queue.put(f"\n[ERROR] {e}\n{traceback.format_exc()}")
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            self.is_complete = True


class ExecutionManager:
    """Execution manager for Streamlit."""
    
    def __init__(self):
        self.current_thread: Optional[ExperimentThread] = None
        self.log_queue = queue.Queue()
        self.log_history: List[str] = []
        self._estimated_time = 60.0
        
    def start_run(self, config: Config, name: str):
        """Start a new run."""
        if self.is_running():
            return
        
        # Reset state
        self.log_queue = queue.Queue()
        self.log_history = []
        
        # Estimate time: ~0.01s per step per alpha for GPU
        n_alphas = len(config.alpha_values)
        n_steps = config.training.max_steps
        self._estimated_time = max(30, n_alphas * n_steps * 0.002)
        
        self.current_thread = ExperimentThread(config, name, self.log_queue)
        self.current_thread.start()
        
    def is_running(self) -> bool:
        return self.current_thread is not None and self.current_thread.is_alive()
    
    def get_progress(self) -> ProgressState:
        """Get current progress state."""
        if self.current_thread is None:
            return ProgressState()
        
        is_running = self.current_thread.is_alive()
        start_time = self.current_thread.start_time
        
        if self.current_thread.is_complete:
            msg = "Complete!"
        elif is_running:
            elapsed = time.time() - start_time if start_time > 0 else 0
            msg = f"Running... ({elapsed:.0f}s)"
        else:
            msg = "Ready"
        
        return ProgressState(
            is_running=is_running,
            start_time=start_time,
            estimated_total=self._estimated_time,
            message=msg
        )
    
    def get_logs(self) -> str:
        """Get cleaned logs."""
        self._poll_queue()
        return clean_terminal_logs("".join(self.log_history))
    
    def get_raw_logs(self) -> str:
        """Get raw logs with ANSI codes."""
        self._poll_queue()
        return "".join(self.log_history)
    
    def _poll_queue(self):
        """Drain log queue."""
        while True:
            try:
                msg = self.log_queue.get_nowait()
                self.log_history.append(msg)
            except queue.Empty:
                break
    
    def get_latest_result(self) -> Optional[Dict[str, Any]]:
        if self.current_thread and self.current_thread.is_complete:
            return self.current_thread.result
        return None
    
    def get_error(self) -> Optional[str]:
        if self.current_thread and self.current_thread.is_complete:
            return self.current_thread.error
        return None
