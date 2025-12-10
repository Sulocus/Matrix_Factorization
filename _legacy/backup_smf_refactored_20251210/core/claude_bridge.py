"""
Claude Bridge - Dynamic module generation via Claude Code.

When the LLM (e.g., Gemini in wizard mode) detects a request that requires
functionality not yet implemented, it can call Claude Code to:
1. Evaluate the task complexity
2. Recommend whether to generate immediately or defer to formal modification
3. Generate the module if approved
4. Provide a copyable prompt for manual Claude Code session if deferred

Example workflow:
    User: "I want to test different initialization variances for W"
    LLM: Detects this needs new functionality
    LLM: Calls Claude Bridge to evaluate
    Claude: Returns {complexity: "low", recommendation: "generate"}
    LLM: Asks user for confirmation
    User: Confirms
    Claude: Generates the module
    LLM: Reloads modules and continues
"""

import subprocess
import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class Complexity(Enum):
    """Task complexity levels."""
    LOW = "low"        # <50 lines, single file, safe to generate immediately
    MEDIUM = "medium"  # 50-150 lines, 2-3 files, can generate but test carefully
    HIGH = "high"      # >150 lines, many files, recommend formal modification


@dataclass
class EvaluationResult:
    """Result of Claude Code task evaluation."""
    complexity: Complexity
    estimated_lines: int
    files_to_modify: List[str]
    files_to_create: List[str]
    risk_factors: List[str]
    recommendation: str  # "generate" or "exit"
    detailed_task: str
    prompt_for_claude: str  # Copyable prompt for manual session

    def to_dict(self) -> Dict[str, Any]:
        return {
            "complexity": self.complexity.value,
            "estimated_lines": self.estimated_lines,
            "files_to_modify": self.files_to_modify,
            "files_to_create": self.files_to_create,
            "risk_factors": self.risk_factors,
            "recommendation": self.recommendation,
            "detailed_task": self.detailed_task,
        }


class ClaudeBridge:
    """
    Bridge to call Claude Code for dynamic module generation.

    Usage:
        bridge = ClaudeBridge(project_root)

        # Evaluate task
        result = bridge.evaluate_task(user_request, context)

        if result.recommendation == "generate":
            success = bridge.generate_module(result.detailed_task)
            if success:
                reload_modules()
        else:
            # Show copyable prompt
            print(result.prompt_for_claude)
    """

    def __init__(self, project_root: Path, timeout: int = 120):
        """
        Initialize Claude Bridge.

        Args:
            project_root: Path to the project root
            timeout: Timeout in seconds for Claude Code calls
        """
        self.project_root = Path(project_root)
        self.timeout = timeout

    def evaluate_task(
        self,
        user_request: str,
        context: Dict[str, Any],
    ) -> EvaluationResult:
        """
        Call Claude Code to evaluate the task complexity.

        Args:
            user_request: User's original request
            context: Current SMF context (available modules, config, etc.)

        Returns:
            EvaluationResult with recommendation
        """
        evaluation_prompt = self._build_evaluation_prompt(user_request, context)

        try:
            result = subprocess.run(
                ["claude", "--print", "-p", evaluation_prompt],
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
                timeout=self.timeout,
            )

            if result.returncode != 0:
                return self._default_evaluation(user_request, context, error=result.stderr)

            # Parse response
            data = self._parse_json_response(result.stdout)

            # Generate copyable prompt
            prompt_for_claude = self._generate_detailed_prompt(
                user_request,
                data.get("detailed_task", user_request),
                context,
            )

            return EvaluationResult(
                complexity=Complexity(data.get("complexity", "high")),
                estimated_lines=data.get("estimated_lines", 100),
                files_to_modify=data.get("files_to_modify", []),
                files_to_create=data.get("files_to_create", []),
                risk_factors=data.get("risk_factors", []),
                recommendation=data.get("recommendation", "exit"),
                detailed_task=data.get("detailed_task", user_request),
                prompt_for_claude=prompt_for_claude,
            )

        except subprocess.TimeoutExpired:
            return self._default_evaluation(user_request, context, error="Timeout")
        except FileNotFoundError:
            return self._default_evaluation(user_request, context, error="Claude Code not found")
        except Exception as e:
            return self._default_evaluation(user_request, context, error=str(e))

    def generate_module(self, task_description: str) -> bool:
        """
        Call Claude Code to generate the module.

        Args:
            task_description: Detailed task description

        Returns:
            True if successful
        """
        generate_prompt = f"""
{task_description}

Important requirements:
- Follow SMF module patterns (see smf/modules/ for examples)
- Use @register_algorithm, @register_graph, or @register_teacher decorators
- Keep code minimal and focused
- Do NOT modify unrelated files
- Add type hints
- Test basic imports work
"""

        try:
            result = subprocess.run(
                ["claude", "--yes", "-p", generate_prompt],
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
                timeout=300,  # 5 minutes max for generation
            )

            return result.returncode == 0

        except Exception:
            return False

    def _build_evaluation_prompt(self, user_request: str, context: Dict[str, Any]) -> str:
        """Build the evaluation prompt for Claude Code."""
        return f'''You are evaluating a task for the SMF (Sparse Matrix Factorization) framework.

## User Request
{user_request}

## Current Context
Available algorithms: {context.get('algorithms', [])}
Available graphs: {context.get('graphs', [])}
Available teachers: {context.get('teachers', [])}
Current config: {json.dumps(context.get('config', {}), indent=2, default=str)}

## Task
Analyze this request and return a JSON evaluation. Be conservative - if unsure, recommend "exit".

Return ONLY valid JSON (no markdown code blocks):
{{
    "complexity": "low|medium|high",
    "estimated_lines": <number>,
    "files_to_modify": ["path1", "path2"],
    "files_to_create": ["path1"],
    "risk_factors": ["reason1", "reason2"],
    "recommendation": "generate|exit",
    "detailed_task": "Detailed description for implementation"
}}

Rules:
- "low" complexity: <50 lines, single file, no breaking changes, safe patterns
- "medium" complexity: 50-150 lines, 2-3 files, minor risk
- "high" complexity: >150 lines, many files, or architectural changes

Recommend "generate" only for low/medium complexity with clear implementation path.
Recommend "exit" for high complexity, unclear requirements, or risky changes.
'''

    def _generate_detailed_prompt(
        self,
        user_request: str,
        detailed_task: str,
        context: Dict[str, Any],
    ) -> str:
        """Generate a copyable prompt for manual Claude Code session."""
        return f'''# SMF New Feature Request

## User's Original Request
{user_request}

## Detailed Task Description
{detailed_task}

## Current Context
- Project path: {self.project_root}
- Framework: smf/
- Available algorithms: {context.get('algorithms', [])}
- Available graphs: {context.get('graphs', [])}

## Implementation Requirements
1. Follow smf/modules/ patterns
2. Use appropriate @register_* decorator to register the module
3. Keep code minimal and focused
4. Add type hints
5. Test that imports work after creation

## Example Module Pattern
```python
from ..registry import register_algorithm
from .base import AlgorithmBase

@register_algorithm(
    key="my_algorithm",
    name="My Algorithm",
    description="Description here",
    default_params={{'param1': 1.0}},
)
class MyAlgorithm(AlgorithmBase):
    def __init__(self, config, device):
        super().__init__(config, device)
        # Initialize from config

    def train_single_alpha(self, W_t, X_t, Y_t, mask, alpha, seed):
        # Implementation
        pass
```

Please implement this feature.
'''

    def _default_evaluation(
        self,
        user_request: str,
        context: Dict[str, Any],
        error: str = "",
    ) -> EvaluationResult:
        """Return a safe default evaluation when Claude Code fails."""
        prompt = self._generate_detailed_prompt(user_request, user_request, context)
        return EvaluationResult(
            complexity=Complexity.HIGH,
            estimated_lines=0,
            files_to_modify=[],
            files_to_create=[],
            risk_factors=[f"Could not evaluate: {error}"],
            recommendation="exit",
            detailed_task=user_request,
            prompt_for_claude=prompt,
        )

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from Claude response."""
        # Try to find JSON in the response
        # First, try the whole text
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the text
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Try to find JSON with nested objects
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return {}

    def check_available(self) -> bool:
        """Check if Claude Code is available."""
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False


def reload_module(module_path: Path) -> bool:
    """
    Dynamically reload a module after it's been modified.

    Args:
        module_path: Path to the module file

    Returns:
        True if successful
    """
    import importlib
    import sys

    try:
        # Convert path to module name
        # e.g., smf/modules/algorithms/new_algo.py -> smf.modules.algorithms.new_algo
        relative = module_path.relative_to(Path.cwd())
        module_name = str(relative).replace("/", ".").replace(".py", "")

        if module_name in sys.modules:
            # Reload existing module
            importlib.reload(sys.modules[module_name])
        else:
            # Import new module
            importlib.import_module(module_name)

        return True

    except Exception as e:
        print(f"Failed to reload module: {e}")
        return False


def reload_all_modules() -> bool:
    """Reload the entire SMF modules registry."""
    import importlib
    import sys

    try:
        # Reload registry first
        if "smf.modules.registry" in sys.modules:
            importlib.reload(sys.modules["smf.modules.registry"])

        # Reload modules package
        if "smf.modules" in sys.modules:
            importlib.reload(sys.modules["smf.modules"])

        return True

    except Exception as e:
        print(f"Failed to reload modules: {e}")
        return False
