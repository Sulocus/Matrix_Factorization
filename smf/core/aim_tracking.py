"""
Experiment tracking with Aim integration.

Provides:
- Git hash tracking for reproducibility
- Aim-based experiment logging with Web UI
- Unified result storage structure
"""

import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Lazy import Aim to avoid hard dependency
_aim_available = None


def _check_aim():
    """Check if Aim is available."""
    global _aim_available
    if _aim_available is None:
        try:
            import aim
            _aim_available = True
        except ImportError:
            _aim_available = False
    return _aim_available


@dataclass
class GitInfo:
    """Git repository information."""
    commit_hash: str
    branch: str
    is_dirty: bool
    commit_message: str = ""

    @classmethod
    def from_repo(cls, repo_path: Path = None) -> "GitInfo":
        """Get git info from current repository."""
        try:
            cwd = str(repo_path) if repo_path else None

            # Get commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, cwd=cwd
            )
            commit_hash = result.stdout.strip()[:8] if result.returncode == 0 else "unknown"

            # Get branch name
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, cwd=cwd
            )
            branch = result.stdout.strip() if result.returncode == 0 else "unknown"

            # Check if dirty
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, cwd=cwd
            )
            is_dirty = bool(result.stdout.strip()) if result.returncode == 0 else True

            # Get commit message
            result = subprocess.run(
                ["git", "log", "-1", "--format=%s"],
                capture_output=True, text=True, cwd=cwd
            )
            commit_message = result.stdout.strip() if result.returncode == 0 else ""

            return cls(
                commit_hash=commit_hash,
                branch=branch,
                is_dirty=is_dirty,
                commit_message=commit_message
            )
        except Exception:
            return cls(
                commit_hash="unknown",
                branch="unknown",
                is_dirty=True,
                commit_message=""
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "commit_hash": self.commit_hash,
            "branch": self.branch,
            "is_dirty": self.is_dirty,
            "commit_message": self.commit_message,
        }


@dataclass
class Experiment:
    """
    Experiment tracking wrapper.

    Supports both Aim (when available) and fallback JSON logging.

    Usage:
        exp = Experiment(name="bigamp_sweep")
        exp.set_params(N1=200, N2=200, M=50, algorithm="bigamp")

        for alpha, q_y in results:
            exp.track(q_y, name="Q_Y", context={"alpha": alpha})

        exp.track_image(plot_path, name="phase_diagram")
        exp.close()

    View results:
        aim up  # Opens http://127.0.0.1:43800
    """

    name: str
    repo_path: Optional[Path] = None
    git_info: GitInfo = field(default_factory=GitInfo.from_repo)
    start_time: datetime = field(default_factory=datetime.now)
    _aim_run: Any = field(default=None, repr=False)
    _params: Dict[str, Any] = field(default_factory=dict, repr=False)
    _metrics: Dict[str, list] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Initialize Aim run if available."""
        if _check_aim():
            from aim import Run
            self._aim_run = Run(
                experiment=self.name,
                repo=str(self.repo_path) if self.repo_path else None,
            )
            # Track git info
            self._aim_run["git"] = self.git_info.to_dict()
            self._aim_run["start_time"] = self.start_time.isoformat()

    def set_params(self, **params):
        """Set experiment hyperparameters."""
        self._params.update(params)
        if self._aim_run is not None:
            self._aim_run["hparams"] = self._params

    def track(self, value: float, name: str, step: int = None, context: Dict = None):
        """
        Track a scalar metric.

        Args:
            value: Metric value
            name: Metric name (e.g., "Q_Y", "loss")
            step: Optional step number
            context: Optional context dict (e.g., {"alpha": 1.5})
        """
        if self._aim_run is not None:
            self._aim_run.track(value, name=name, step=step, context=context or {})

        # Fallback logging
        key = f"{name}_{context}" if context else name
        if key not in self._metrics:
            self._metrics[key] = []
        self._metrics[key].append({"step": step, "value": value, "context": context})

    def track_image(self, image_path: str, name: str, context: Dict = None):
        """Track an image (e.g., plot)."""
        if self._aim_run is not None:
            from aim import Image
            self._aim_run.track(Image(image_path), name=name, context=context or {})

    def track_figure(self, fig, name: str, context: Dict = None):
        """Track a matplotlib figure."""
        if self._aim_run is not None:
            from aim import Figure
            self._aim_run.track(Figure(fig), name=name, context=context or {})

    def generate_run_id(self) -> str:
        """Generate unique run ID."""
        algo = self._params.get("algorithm", "exp")
        n1 = self._params.get("N1", 0)
        n2 = self._params.get("N2", 0)
        m = self._params.get("M", 0)
        timestamp = self.start_time.strftime("%Y%m%d_%H%M")
        git_hash = self.git_info.commit_hash

        return f"{algo}_{n1}x{n2}_M{m}_{timestamp}_{git_hash}"

    def save_json(self, base_dir: Path) -> Path:
        """
        Save experiment to JSON (fallback when Aim not available).

        Creates:
            base_dir/runs/{run_id}/
                ├── config.json
                ├── metrics.json
                └── git_info.json
        """
        run_id = self.generate_run_id()
        run_dir = base_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(run_dir / "config.json", "w") as f:
            json.dump(self._params, f, indent=2)

        # Save metrics
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(self._metrics, f, indent=2)

        # Save git info
        with open(run_dir / "git_info.json", "w") as f:
            json.dump(self.git_info.to_dict(), f, indent=2)

        return run_dir

    def close(self):
        """Close the experiment run."""
        if self._aim_run is not None:
            self._aim_run.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def install_aim():
    """Helper to install Aim."""
    print("Installing Aim...")
    subprocess.run(["pip", "install", "aim"], check=True)
    print("Aim installed. Run 'aim up' to start the Web UI.")


def launch_aim_ui(port: int = 43800):
    """Launch Aim Web UI."""
    if not _check_aim():
        print("Aim not installed. Run: pip install aim")
        return

    print(f"Starting Aim UI at http://127.0.0.1:{port}")
    subprocess.run(["aim", "up", "--port", str(port)])


# Convenience functions
def quick_experiment(name: str, **params) -> Experiment:
    """Create an experiment with parameters in one call."""
    exp = Experiment(name=name)
    exp.set_params(**params)
    return exp
