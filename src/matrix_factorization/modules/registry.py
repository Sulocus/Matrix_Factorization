"""
Module registration system.

Provides decorators for registering modules and functions for retrieving them.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Type

if TYPE_CHECKING:
    from matrix_factorization.core.contracts import AlgorithmSpec
    from matrix_factorization.core.contracts import GraphSpec
    from matrix_factorization.core.contracts import MetricSpec
    from matrix_factorization.core.contracts import OutputSpec
    from matrix_factorization.core.contracts import TeacherSpec


@dataclass
class ModuleInfo:
    """Information about a registered module."""
    key: str
    name: str
    description: str
    cls: Type
    default_params: Dict[str, Any]
    algorithm_spec: Optional["AlgorithmSpec"] = None
    graph_spec: Optional["GraphSpec"] = None
    teacher_spec: Optional["TeacherSpec"] = None
    metric_spec: Optional["MetricSpec"] = None
    output_spec: Optional["OutputSpec"] = None


# Registry dictionaries
_algorithms: Dict[str, ModuleInfo] = {}
_graphs: Dict[str, ModuleInfo] = {}
_teachers: Dict[str, ModuleInfo] = {}
_metrics: Dict[str, ModuleInfo] = {}
_outputs: Dict[str, ModuleInfo] = {}


def _register(registry: Dict[str, ModuleInfo], key: str, name: str,
              description: str, default_params: Dict[str, Any] = None,
              algorithm_spec: Optional["AlgorithmSpec"] = None,
              graph_spec: Optional["GraphSpec"] = None,
              teacher_spec: Optional["TeacherSpec"] = None,
              metric_spec: Optional["MetricSpec"] = None,
              output_spec: Optional["OutputSpec"] = None):
    """Generic registration decorator factory."""
    def decorator(cls: Type) -> Type:
        if key in registry:
            raise KeyError(f"Duplicate registry key '{key}'")
        registry[key] = ModuleInfo(
            key=key,
            name=name,
            description=description,
            cls=cls,
            default_params=default_params or {},
            algorithm_spec=algorithm_spec,
            graph_spec=graph_spec,
            teacher_spec=teacher_spec,
            metric_spec=metric_spec,
            output_spec=output_spec,
        )
        return cls
    return decorator


# Registration decorators
def register_algorithm(key: str, name: str, description: str = "",
                       default_params: Dict[str, Any] = None):
    """Register a training algorithm."""
    spec = _resolve_algorithm_spec_for_registration(key)
    return _register(_algorithms, key, name, description, default_params, spec)


def _resolve_algorithm_spec_for_registration(key: str) -> "AlgorithmSpec":
    """Return the required AlgorithmSpec for a registered algorithm.

    Algorithm registration is a hard-interface gate: adding a new algorithm
    class without adding a contract should fail during import, before the
    algorithm can enter the main registry.
    """
    from matrix_factorization.core.contracts import get_algorithm_specs

    specs = get_algorithm_specs()
    if key not in specs:
        available = ", ".join(sorted(specs))
        raise ValueError(
            f"Algorithm '{key}' cannot be registered without an AlgorithmSpec. "
            f"Known specs: {available}"
        )
    return specs[key]


def register_graph(key: str, name: str, description: str = "",
                   default_params: Dict[str, Any] = None):
    """Register a graph generation method."""
    spec = _resolve_graph_spec_for_registration(key)
    return _register(_graphs, key, name, description, default_params, graph_spec=spec)


def _resolve_graph_spec_for_registration(key: str) -> "GraphSpec":
    """Return the required GraphSpec for a registered graph generator."""
    from matrix_factorization.core.contracts import get_graph_specs

    specs = get_graph_specs()
    if key not in specs:
        available = ", ".join(sorted(specs))
        raise ValueError(
            f"Graph '{key}' cannot be registered without a GraphSpec. "
            f"Known specs: {available}"
        )
    return specs[key]


def register_teacher(key: str, name: str, description: str = "",
                     default_params: Dict[str, Any] = None):
    """Register a teacher model initializer."""
    spec = _resolve_teacher_spec_for_registration(key)
    return _register(_teachers, key, name, description, default_params, teacher_spec=spec)


def _resolve_teacher_spec_for_registration(key: str) -> "TeacherSpec":
    """Return the required TeacherSpec for a registered teacher initializer."""
    from matrix_factorization.core.contracts import get_teacher_specs

    specs = get_teacher_specs()
    if key not in specs:
        available = ", ".join(sorted(specs))
        raise ValueError(
            f"Teacher '{key}' cannot be registered without a TeacherSpec. "
            f"Known specs: {available}"
        )
    return specs[key]


def register_metric(key: str, name: str, description: str = "",
                    default_params: Dict[str, Any] = None):
    """Register an evaluation metric."""
    spec = _resolve_metric_spec_for_registration(key)
    return _register(_metrics, key, name, description, default_params, metric_spec=spec)


def _resolve_metric_spec_for_registration(key: str) -> "MetricSpec":
    """Return the required MetricSpec for a registered metric handler."""
    from matrix_factorization.core.contracts import get_metric_specs

    specs = get_metric_specs()
    if key not in specs:
        available = ", ".join(sorted(specs))
        raise ValueError(
            f"Metric '{key}' cannot be registered without a MetricSpec. "
            f"Known specs: {available}"
        )
    return specs[key]


def register_output(key: str, name: str, description: str = "",
                    default_params: Dict[str, Any] = None):
    """Register an output handler."""
    spec = _resolve_output_spec_for_registration(key)
    return _register(_outputs, key, name, description, default_params, output_spec=spec)


def _resolve_output_spec_for_registration(key: str) -> "OutputSpec":
    """Return the required OutputSpec for a registered output handler."""
    from matrix_factorization.core.contracts import get_output_specs

    specs = get_output_specs()
    if key not in specs:
        available = ", ".join(sorted(specs))
        raise ValueError(
            f"Output '{key}' cannot be registered without an OutputSpec. "
            f"Known specs: {available}"
        )
    return specs[key]


# Getter functions
def _get(registry: Dict[str, ModuleInfo], key: str, category: str) -> ModuleInfo:
    """Get a module by key."""
    if key not in registry:
        available = ", ".join(registry.keys())
        raise KeyError(f"Unknown {category} '{key}'. Available: {available}")
    return registry[key]


def get_algorithm(key: str) -> ModuleInfo:
    """Get an algorithm by key."""
    return _get(_algorithms, key, "algorithm")


def get_algorithm_spec(key: str) -> "AlgorithmSpec":
    """Get the hard-interface contract for an algorithm."""
    from matrix_factorization.core.contracts import get_algorithm_specs

    if key in _algorithms and _algorithms[key].algorithm_spec is not None:
        return _algorithms[key].algorithm_spec

    specs = get_algorithm_specs()
    if key not in specs:
        available = ", ".join(specs.keys())
        raise KeyError(f"Unknown algorithm spec '{key}'. Available: {available}")
    return specs[key]


def get_graph(key: str) -> ModuleInfo:
    """Get a graph generator by key."""
    return _get(_graphs, key, "graph")


def get_graph_spec(key: str) -> "GraphSpec":
    """Get the hard-interface contract for a graph generator."""
    from matrix_factorization.core.contracts import get_graph_specs

    if key in _graphs and _graphs[key].graph_spec is not None:
        return _graphs[key].graph_spec

    specs = get_graph_specs()
    if key not in specs:
        available = ", ".join(specs.keys())
        raise KeyError(f"Unknown graph spec '{key}'. Available: {available}")
    return specs[key]


def get_teacher(key: str) -> ModuleInfo:
    """Get a teacher initializer by key."""
    return _get(_teachers, key, "teacher")


def get_teacher_spec(key: str) -> "TeacherSpec":
    """Get the hard-interface contract for a teacher initializer."""
    from matrix_factorization.core.contracts import get_teacher_specs

    if key in _teachers and _teachers[key].teacher_spec is not None:
        return _teachers[key].teacher_spec

    specs = get_teacher_specs()
    if key not in specs:
        available = ", ".join(specs.keys())
        raise KeyError(f"Unknown teacher spec '{key}'. Available: {available}")
    return specs[key]


def get_metric(key: str) -> ModuleInfo:
    """Get a metric by key."""
    return _get(_metrics, key, "metric")


def get_metric_spec(key: str) -> "MetricSpec":
    """Get the hard-interface contract for a metric handler."""
    from matrix_factorization.core.contracts import get_metric_specs

    if key in _metrics and _metrics[key].metric_spec is not None:
        return _metrics[key].metric_spec

    specs = get_metric_specs()
    if key not in specs:
        available = ", ".join(specs.keys())
        raise KeyError(f"Unknown metric spec '{key}'. Available: {available}")
    return specs[key]


def get_output(key: str) -> ModuleInfo:
    """Get an output handler by key."""
    return _get(_outputs, key, "output")


def get_output_spec(key: str) -> "OutputSpec":
    """Get the hard-interface contract for an output handler."""
    from matrix_factorization.core.contracts import get_output_specs

    if key in _outputs and _outputs[key].output_spec is not None:
        return _outputs[key].output_spec

    specs = get_output_specs()
    if key not in specs:
        available = ", ".join(specs.keys())
        raise KeyError(f"Unknown output spec '{key}'. Available: {available}")
    return specs[key]


# List functions
def _list(registry: Dict[str, ModuleInfo]) -> List[ModuleInfo]:
    """List all modules in a registry."""
    return list(registry.values())


def list_algorithms() -> List[ModuleInfo]:
    """List all registered algorithms."""
    return _list(_algorithms)


def list_algorithm_specs() -> List["AlgorithmSpec"]:
    """List all hard-interface algorithm contracts."""
    from matrix_factorization.core.contracts import get_algorithm_specs

    return list(get_algorithm_specs().values())


def validate_algorithm_registry_contracts() -> List[str]:
    """Return contract errors for registered algorithms.

    This is intentionally side-effect free so tests and preflight code can use
    it as a hard gate.
    """
    from matrix_factorization.core.contracts import get_algorithm_specs

    specs = get_algorithm_specs()
    errors: List[str] = []
    for key, info in _algorithms.items():
        if key not in specs:
            errors.append(f"registered algorithm '{key}' has no AlgorithmSpec")
            continue
        if info.algorithm_spec is None:
            errors.append(f"registered algorithm '{key}' has no bound AlgorithmSpec")
        elif info.algorithm_spec.key != key:
            errors.append(
                f"registered algorithm '{key}' is bound to mismatched "
                f"AlgorithmSpec '{info.algorithm_spec.key}'"
            )
    for key, spec in specs.items():
        if spec.status == "active" and key not in _algorithms:
            errors.append(f"active AlgorithmSpec '{key}' is not registered")
    return errors


def validate_output_registry_contracts() -> List[str]:
    """Return contract errors for registered output handlers."""
    from matrix_factorization.core.contracts import get_output_specs

    specs = get_output_specs()
    errors: List[str] = []
    for key, info in _outputs.items():
        if key not in specs:
            errors.append(f"registered output '{key}' has no OutputSpec")
            continue
        if info.output_spec is None:
            errors.append(f"registered output '{key}' has no bound OutputSpec")
        elif info.output_spec.key != key:
            errors.append(
                f"registered output '{key}' is bound to mismatched "
                f"OutputSpec '{info.output_spec.key}'"
            )
    return errors


def validate_metric_registry_contracts() -> List[str]:
    """Return contract errors for registered metric handlers."""
    from matrix_factorization.core.contracts import get_metric_specs

    specs = get_metric_specs()
    errors: List[str] = []
    for key, info in _metrics.items():
        if key not in specs:
            errors.append(f"registered metric '{key}' has no MetricSpec")
            continue
        if info.metric_spec is None:
            errors.append(f"registered metric '{key}' has no bound MetricSpec")
        elif info.metric_spec.key != key:
            errors.append(
                f"registered metric '{key}' is bound to mismatched "
                f"MetricSpec '{info.metric_spec.key}'"
            )
    return errors


def validate_teacher_registry_contracts() -> List[str]:
    """Return contract errors for registered teacher initializers."""
    from matrix_factorization.core.contracts import get_teacher_specs

    specs = get_teacher_specs()
    errors: List[str] = []
    for key, info in _teachers.items():
        if key not in specs:
            errors.append(f"registered teacher '{key}' has no TeacherSpec")
            continue
        if info.teacher_spec is None:
            errors.append(f"registered teacher '{key}' has no bound TeacherSpec")
        elif info.teacher_spec.key != key:
            errors.append(
                f"registered teacher '{key}' is bound to mismatched "
                f"TeacherSpec '{info.teacher_spec.key}'"
            )
    return errors


def validate_graph_registry_contracts() -> List[str]:
    """Return contract errors for registered graph generators."""
    from matrix_factorization.core.contracts import get_graph_specs

    specs = get_graph_specs()
    errors: List[str] = []
    for key, info in _graphs.items():
        if key not in specs:
            errors.append(f"registered graph '{key}' has no GraphSpec")
            continue
        if info.graph_spec is None:
            errors.append(f"registered graph '{key}' has no bound GraphSpec")
        elif info.graph_spec.key != key:
            errors.append(
                f"registered graph '{key}' is bound to mismatched "
                f"GraphSpec '{info.graph_spec.key}'"
            )
    return errors


def list_graphs() -> List[ModuleInfo]:
    """List all registered graph generators."""
    return _list(_graphs)


def list_teachers() -> List[ModuleInfo]:
    """List all registered teacher initializers."""
    return _list(_teachers)


def list_metrics() -> List[ModuleInfo]:
    """List all registered metrics."""
    return _list(_metrics)


def list_outputs() -> List[ModuleInfo]:
    """List all registered output handlers."""
    return _list(_outputs)


def print_registry_summary():
    """Print a summary of all registered modules."""
    print("\n=== SMF Module Registry ===\n")

    categories = [
        ("Algorithms", _algorithms),
        ("Graphs", _graphs),
        ("Teachers", _teachers),
        ("Metrics", _metrics),
        ("Outputs", _outputs),
    ]

    for name, registry in categories:
        print(f"{name}:")
        if not registry:
            print("  (none registered)")
        else:
            for key, info in registry.items():
                print(f"  [{key}] {info.name}")
                if info.description:
                    print(f"        {info.description}")
        print()
