"""
LLM Conversation Logger - records LLM conversations and execution status.

Tracks:
- User input to LLM
- Raw LLM response
- Parsed configuration
- Execution status (success/partial/failed)
"""
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List


@dataclass
class LLMLogEntry:
    """Single log entry."""
    timestamp: str
    session_id: str
    stage: str  # "request" | "response" | "parsed" | "execution"

    # Request stage
    user_input: Optional[str] = None

    # Response stage
    raw_response: Optional[str] = None
    parse_success: Optional[bool] = None
    parse_error: Optional[str] = None

    # Parsed stage
    experiment_type: Optional[str] = None
    config_summary: Optional[Dict[str, Any]] = None
    comparison_steps: Optional[List[Dict]] = None

    # Execution stage
    execution_status: Optional[str] = None  # "success" | "partial" | "failed"
    steps_completed: Optional[int] = None
    steps_total: Optional[int] = None
    error_message: Optional[str] = None
    result_paths: Optional[List[str]] = None


class LLMLogger:
    """LLM conversation log manager."""

    LOG_DIR = Path("smf/logs/llm")

    def __init__(self, session_id: str = None):
        """
        Initialize logger.

        Args:
            session_id: Session identifier (auto-generated if None)
        """
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_file = self.LOG_DIR / f"session_{self.session_id}.jsonl"
        self.entries: List[LLMLogEntry] = []

    def log_request(self, user_input: str):
        """
        Log user request.

        Args:
            user_input: User's natural language input
        """
        entry = LLMLogEntry(
            timestamp=datetime.now().isoformat(),
            session_id=self.session_id,
            stage="request",
            user_input=user_input,
        )
        self._append(entry)

    def log_response(self, raw_response: str, parse_success: bool,
                     parse_error: str = None):
        """
        Log LLM response.

        Args:
            raw_response: Raw response text from LLM
            parse_success: Whether parsing succeeded
            parse_error: Error message if parsing failed
        """
        # Truncate very long responses
        truncated = raw_response[:3000] if raw_response else ""
        if raw_response and len(raw_response) > 3000:
            truncated += f"\n... [truncated, total {len(raw_response)} chars]"

        entry = LLMLogEntry(
            timestamp=datetime.now().isoformat(),
            session_id=self.session_id,
            stage="response",
            raw_response=truncated,
            parse_success=parse_success,
            parse_error=parse_error,
        )
        self._append(entry)

    def log_parsed(self, experiment_type: str, config_summary: Dict,
                   comparison_steps: List[Dict] = None):
        """
        Log parsed configuration.

        Args:
            experiment_type: "standard" or "comparison"
            config_summary: Summary of configuration parameters
            comparison_steps: List of comparison steps (for comparison experiments)
        """
        entry = LLMLogEntry(
            timestamp=datetime.now().isoformat(),
            session_id=self.session_id,
            stage="parsed",
            experiment_type=experiment_type,
            config_summary=config_summary,
            comparison_steps=comparison_steps,
        )
        self._append(entry)

    def log_execution(self, status: str, steps_completed: int = 0,
                      steps_total: int = 1, error_message: str = None,
                      result_paths: List[str] = None):
        """
        Log execution status.

        Args:
            status: "success", "partial", or "failed"
            steps_completed: Number of steps completed
            steps_total: Total number of steps
            error_message: Error message if failed
            result_paths: Paths to result files
        """
        entry = LLMLogEntry(
            timestamp=datetime.now().isoformat(),
            session_id=self.session_id,
            stage="execution",
            execution_status=status,
            steps_completed=steps_completed,
            steps_total=steps_total,
            error_message=error_message,
            result_paths=result_paths,
        )
        self._append(entry)

    def _append(self, entry: LLMLogEntry):
        """Append log entry to file."""
        self.entries.append(entry)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + '\n')

    @classmethod
    def list_sessions(cls, limit: int = 10) -> List[Path]:
        """
        List recent session log files.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of log file paths (newest first)
        """
        if not cls.LOG_DIR.exists():
            return []
        files = sorted(cls.LOG_DIR.glob("session_*.jsonl"), reverse=True)
        return files[:limit]

    @classmethod
    def load_session(cls, session_id: str) -> List[LLMLogEntry]:
        """
        Load entries from a session.

        Args:
            session_id: Session identifier

        Returns:
            List of log entries
        """
        log_file = cls.LOG_DIR / f"session_{session_id}.jsonl"
        if not log_file.exists():
            return []
        entries = []
        with open(log_file, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    entries.append(LLMLogEntry(**data))
        return entries

    @classmethod
    def format_session_summary(cls, session_id: str) -> str:
        """
        Format a session summary for display.

        Args:
            session_id: Session identifier

        Returns:
            Formatted summary string
        """
        entries = cls.load_session(session_id)
        if not entries:
            return f"Session {session_id}: No entries found"

        lines = [f"Session: {session_id}"]
        lines.append("=" * 50)

        for entry in entries:
            timestamp = entry.timestamp.split('T')[1].split('.')[0]  # HH:MM:SS

            if entry.stage == "request":
                lines.append(f"\n[{timestamp}] USER INPUT:")
                lines.append(f"  {entry.user_input}")

            elif entry.stage == "response":
                status = "OK" if entry.parse_success else "FAILED"
                lines.append(f"\n[{timestamp}] LLM RESPONSE: {status}")
                if entry.parse_error:
                    lines.append(f"  Error: {entry.parse_error}")
                if entry.raw_response:
                    # Show first few lines
                    preview = entry.raw_response[:500]
                    if len(entry.raw_response) > 500:
                        preview += "..."
                    lines.append(f"  Response preview:")
                    for line in preview.split('\n')[:10]:
                        lines.append(f"    {line}")

            elif entry.stage == "parsed":
                lines.append(f"\n[{timestamp}] PARSED CONFIG:")
                lines.append(f"  Type: {entry.experiment_type}")
                if entry.config_summary:
                    for k, v in entry.config_summary.items():
                        lines.append(f"  {k}: {v}")
                if entry.comparison_steps:
                    lines.append(f"  Comparison steps: {len(entry.comparison_steps)}")
                    for i, step in enumerate(entry.comparison_steps):
                        lines.append(f"    [{i+1}] {step.get('label', 'unnamed')}")

            elif entry.stage == "execution":
                lines.append(f"\n[{timestamp}] EXECUTION:")
                lines.append(f"  Status: {entry.execution_status}")
                lines.append(f"  Progress: {entry.steps_completed}/{entry.steps_total}")
                if entry.error_message:
                    lines.append(f"  Error: {entry.error_message}")
                if entry.result_paths:
                    lines.append(f"  Results:")
                    for p in entry.result_paths:
                        lines.append(f"    - {p}")

        return '\n'.join(lines)


# Global logger instance (created per session)
_current_logger: Optional[LLMLogger] = None


def get_logger(session_id: str = None) -> LLMLogger:
    """
    Get or create logger for current session.

    Args:
        session_id: Optional session ID (creates new if None)

    Returns:
        LLMLogger instance
    """
    global _current_logger
    if _current_logger is None or (session_id and _current_logger.session_id != session_id):
        _current_logger = LLMLogger(session_id)
    return _current_logger


def reset_logger():
    """Reset global logger (for testing)."""
    global _current_logger
    _current_logger = None
