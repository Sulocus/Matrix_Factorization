"""Base runtime intervention types.

This file establishes the hook interface without changing any training logic.
Concrete interventions should operate only through AlgorithmStateView.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from matrix_factorization.core.contracts import (
    AlgorithmSpec,
    AlgorithmStateView,
    InterventionSpec,
    get_intervention_specs,
)


class HookPoint(str, Enum):
    BEFORE_INITIALIZE = "before_initialize"
    AFTER_INITIALIZE = "after_initialize"
    BEFORE_STEP = "before_step"
    AFTER_STEP = "after_step"
    ON_PLATEAU = "on_plateau"
    AFTER_BATCH = "after_batch"
    AFTER_RUN = "after_run"


@dataclass
class RuntimeHookContext:
    algorithm_key: str
    hook: HookPoint
    config: Any = None
    alpha: Optional[float] = None
    step_index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class InterventionBase:
    """No-formula-change base class for future state-modifying hooks."""

    spec_key: str = ""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = dict(config or {})

    @property
    def spec(self) -> InterventionSpec:
        specs = get_intervention_specs()
        if self.spec_key not in specs:
            raise KeyError(f"Unknown InterventionSpec: {self.spec_key}")
        return specs[self.spec_key]

    def validate_compatible(self, algorithm_spec: AlgorithmSpec) -> None:
        spec = self.spec
        if algorithm_spec.key not in spec.compatible_algorithms:
            raise ValueError(
                f"intervention '{spec.key}' is not compatible with algorithm "
                f"'{algorithm_spec.key}'"
            )
        missing_state = sorted(set(spec.requires_state) - set(algorithm_spec.state_capabilities))
        if missing_state:
            raise ValueError(
                f"intervention '{spec.key}' requires state {missing_state}, "
                f"but algorithm '{algorithm_spec.key}' does not expose it"
            )

    def apply(
        self,
        state: AlgorithmStateView,
        context: RuntimeHookContext,
    ) -> AlgorithmStateView:
        """Return a possibly modified state.

        The base implementation is intentionally no-op so Phase 5 can register
        hook positions before moving legacy behavior.
        """
        return state


class NoOpIntervention(InterventionBase):
    spec_key = "cold_start"


class ColdStartIntervention(InterventionBase):
    spec_key = "cold_start"


class WarmStartIntervention(InterventionBase):
    spec_key = "warm_start"


class AdaptiveRestartIntervention(InterventionBase):
    spec_key = "adaptive_restart"


class MetropolisKickIntervention(InterventionBase):
    spec_key = "metropolis_kick"


_INTERVENTIONS = {
    "cold_start": ColdStartIntervention,
    "warm_start": WarmStartIntervention,
    "adaptive_restart": AdaptiveRestartIntervention,
    "metropolis_kick": MetropolisKickIntervention,
}


def build_intervention(key: str, config: Optional[Dict[str, Any]] = None) -> InterventionBase:
    if key not in _INTERVENTIONS:
        raise KeyError(f"Unknown intervention implementation: {key}")
    return _INTERVENTIONS[key](config=config)
