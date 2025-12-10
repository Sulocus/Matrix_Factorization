"""
Execution Manager for NiceGUI (v2).

Features:
- Threaded experiment execution
- Terminal output simulation with ANSI code cleaning
- Auto-scroll to bottom
- Log queue for real-time updates
"""

import threading
import queue
import time
import sys
import re
from typing import Optional, Dict, Any, List

from nicegui import ui
from smf.core.config import Config
from smf.core.runner import run_experiment


def clean_terminal_logs(text: str) -> str:
    """
    Clean ANSI control codes and handle cursor movements.
    Converts Rich/tqdm output to clean text for display.
    """
    # 1. Split into tokens: Text or Cursor Up (\x1b[nA)
    tokens = re.split(r'(\x1b\[\d+A)', text)
    lines = []
    
    for token in tokens:
        if not token:
            continue
        if token.startswith('\x1b[') and token.endswith('A'):
            try:
                n = int(token[2:-1])
                for _ in range(n):
                    if lines:
                        lines.pop()
            except ValueError:
                pass
        else:
            # Strip ANSI color/style codes
            token = re.sub(r'\x1b\[[0-9;]*[mK]', '', token)
            # Strip other escape sequences
            token = re.sub(r'\x1b\[[0-9;]*[HJfsu]', '', token)
            
            sublines = token.split('\n')
            for i, subline in enumerate(sublines):
                if '\r' in subline:
                    subline = subline.split('\r')[-1]
                if i == 0:
                    if lines:
                        lines[-1] += subline
                    else:
                        lines.append(subline)
                else:
                    lines.append(subline)
    
    return "\n".join(lines)


class TerminalSimulator:
    """
    A NiceGUI element that behaves like a terminal.
    Interprets basic Cursor Up and CR codes to update content in-place.
    """
    
    def __init__(self, container):
        self.container = container
        self.raw_buffer = ""
        self._element_id = None
        
        with self.container:
            # Pre-formatted text, dark theme terminal style
            # sanitize=False required for NiceGUI 3.x to allow raw HTML
            self.element = ui.html('', sanitize=False).classes(
                'font-mono text-sm bg-gray-900 text-green-400 p-4 rounded-lg '
                'h-96 overflow-auto whitespace-pre'
            )
            self._element_id = self.element.id

    def update_content(self, text_chunk: str):
        """Parse chunk and update display incrementally."""
        self.raw_buffer += text_chunk
        cleaned = clean_terminal_logs(self.raw_buffer)
        self.element.content = f'<pre style="margin:0;white-space:pre-wrap;">{self._escape_html(cleaned)}</pre>'
        self._scroll_to_bottom()

    def set_content(self, full_text: str):
        """Set the full text content (replaces buffer)."""
        self.raw_buffer = full_text
        cleaned = clean_terminal_logs(full_text)
        self.element.content = f'<pre style="margin:0;white-space:pre-wrap;">{self._escape_html(cleaned)}</pre>'
        self._scroll_to_bottom()

    def clear(self):
        """Clear the terminal."""
        self.raw_buffer = ""
        self.element.content = ""

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;'))

    def _scroll_to_bottom(self):
        """Scroll the terminal to the bottom."""
        if self._element_id:
            ui.run_javascript(f'''
                const el = document.getElementById("c{self._element_id}");
                if (el) el.scrollTop = el.scrollHeight;
            ''')


class ExperimentThread(threading.Thread):
    """Thread for running experiments in background."""
    
    def __init__(self, config: Config, log_queue: queue.Queue):
        super().__init__()
        self.config = config
        self.log_queue = log_queue
        self.result = None
        self.error = None
        self._stop_event = threading.Event()
        self.daemon = True

    def run(self):
        """Run the experiment, capturing stdout to the queue."""
        
        class QueueWriter:
            def __init__(self, q):
                self.q = q
                self.encoding = 'utf-8'
                
            def write(self, msg):
                if msg:
                    self.q.put(msg)
                    
            def flush(self):
                pass
                
            def isatty(self):
                # Force Rich/tqdm to treat this as a terminal
                return True
        
        # Force environment variables for Rich
        import os
        os.environ["FORCE_COLOR"] = "1"
        os.environ["COLUMNS"] = "80"

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = QueueWriter(self.log_queue)
        sys.stderr = QueueWriter(self.log_queue)
        
        try:
            self.log_queue.put("Starting execution...\n")
            self.result = run_experiment(self.config)
            self.log_queue.put("\n✓ Finished successfully.\n")
        except Exception as e:
            import traceback
            self.error = str(e)
            trace = traceback.format_exc()
            self.log_queue.put(f"\n✗ Error:\n{trace}\n")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def stop(self):
        self._stop_event.set()


class ExecutionManagerNiceGUI:
    """Manages experiment execution and terminal output for NiceGUI."""
    
    def __init__(self):
        self.thread: Optional[ExperimentThread] = None
        self.log_queue = queue.Queue()
        self.log_history: List[str] = []
        self.terminal: Optional[TerminalSimulator] = None
        self.timer = None
        self._on_complete_callback = None

    def start(self, config: Config, terminal_element: TerminalSimulator, on_complete=None):
        """Start a new experiment run."""
        if self.thread and self.thread.is_alive():
            return False
        
        # Reset state
        self.log_queue = queue.Queue()
        self.log_history = []
        self.terminal = terminal_element
        self.terminal.clear()
        self._on_complete_callback = on_complete
        
        # Start thread
        self.thread = ExperimentThread(config, self.log_queue)
        self.thread.start()
        
        # Start polling timer
        self.timer = ui.timer(0.1, self._poll_logs)
        
        return True

    def _poll_logs(self):
        """Poll the log queue and update terminal."""
        new_data = False
        
        while not self.log_queue.empty():
            try:
                msg = self.log_queue.get_nowait()
                self.log_history.append(msg)
                new_data = True
            except queue.Empty:
                break
        
        if new_data and self.terminal:
            full_text = "".join(self.log_history)
            self.terminal.set_content(full_text)
        
        # Check for completion
        if self.thread and not self.thread.is_alive() and self.log_queue.empty():
            self.timer.deactivate()
            if self._on_complete_callback:
                self._on_complete_callback(self.thread.result, self.thread.error)

    def is_running(self) -> bool:
        """Check if an experiment is currently running."""
        return self.thread is not None and self.thread.is_alive()

    def get_result(self) -> Optional[Dict[str, Any]]:
        """Get the result if the thread has completed."""
        if self.thread and not self.thread.is_alive():
            return self.thread.result
        return None

    def get_error(self) -> Optional[str]:
        """Get any error that occurred."""
        if self.thread and not self.thread.is_alive():
            return self.thread.error
        return None
