
import os
import sys
import time
from smf.core.config import Config
from smf.ui.execution import ExecutionManager

def reproduce_logs():
    print("Starting reproduction script...")
    
    # 1. Setup Config
    config = Config()
    config.matrix.N1 = 100
    config.matrix.N2 = 100
    config.matrix.M = 20
    config.alpha.start = 0.5
    config.alpha.stop = 0.5
    config.alpha.step = 0.1
    config.training.max_steps = 10
    config.training.samples_per_alpha = 1
    config.training.device = "cpu"
    config.algorithm.mode = "spreading_parallel"
    
    # 2. Setup Execution Manager
    manager = ExecutionManager()
    manager.start_run(config, "debug_run")
    
    # 3. Poll logs and save raw output
    raw_log_file = "raw_log_capture.txt"
    with open(raw_log_file, "w") as f:
        f.write("--- RAW LOG CAPTURE START ---\n")
        
    print(f"Capturing logs to {raw_log_file}...")
    
    while manager.is_running():
        # Access the raw queue directly to avoid cleaner
        try:
            while not manager.log_queue.empty():
                msg = manager.log_queue.get_nowait()
                with open(raw_log_file, "a") as f:
                    # Write repr to see exact chars
                    f.write(repr(msg) + "\n")
        except Exception:
            pass
        time.sleep(0.1)
        
    print("Run finished. Logs captured.")

if __name__ == "__main__":
    # Force rich to think it's in a terminal?
    # os.environ["FORCE_COLOR"] = "1" 
    # os.environ["COLUMNS"] = "80"
    reproduce_logs()
