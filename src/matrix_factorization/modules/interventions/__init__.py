"""Runtime intervention hook skeletons.

These classes are structural placeholders for the hard-interface migration.
They do not change current algorithm behavior; existing warm start and restart
logic remains inside the algorithms until it is migrated deliberately.
"""

from .base import (
    HookPoint,
    RuntimeHookContext,
    InterventionBase,
    NoOpIntervention,
    ColdStartIntervention,
    WarmStartIntervention,
    AdaptiveRestartIntervention,
    MetropolisKickIntervention,
    build_intervention,
)
from .executor import RuntimeExtensionExecutor, RuntimeExtensionReport

__all__ = [
    "HookPoint",
    "RuntimeHookContext",
    "InterventionBase",
    "NoOpIntervention",
    "ColdStartIntervention",
    "WarmStartIntervention",
    "AdaptiveRestartIntervention",
    "MetropolisKickIntervention",
    "build_intervention",
    "RuntimeExtensionExecutor",
    "RuntimeExtensionReport",
]
