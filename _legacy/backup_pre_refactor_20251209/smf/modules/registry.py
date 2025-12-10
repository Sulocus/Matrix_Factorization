"""
Module registration system.

Provides decorators for registering modules and functions for retrieving them.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type


@dataclass
class ModuleInfo:
    """Information about a registered module."""
    key: str
    name: str
    description: str
    cls: Type
    default_params: Dict[str, Any]


# Registry dictionaries
_algorithms: Dict[str, ModuleInfo] = {}
_graphs: Dict[str, ModuleInfo] = {}
_teachers: Dict[str, ModuleInfo] = {}
_metrics: Dict[str, ModuleInfo] = {}
_outputs: Dict[str, ModuleInfo] = {}


def _register(registry: Dict[str, ModuleInfo], key: str, name: str,
              description: str, default_params: Dict[str, Any] = None):
    """Generic registration decorator factory."""
    def decorator(cls: Type) -> Type:
        registry[key] = ModuleInfo(
            key=key,
            name=name,
            description=description,
            cls=cls,
            default_params=default_params or {},
        )
        return cls
    return decorator


# Registration decorators
def register_algorithm(key: str, name: str, description: str = "",
                       default_params: Dict[str, Any] = None):
    """Register a training algorithm."""
    return _register(_algorithms, key, name, description, default_params)


def register_graph(key: str, name: str, description: str = "",
                   default_params: Dict[str, Any] = None):
    """Register a graph generation method."""
    return _register(_graphs, key, name, description, default_params)


def register_teacher(key: str, name: str, description: str = "",
                     default_params: Dict[str, Any] = None):
    """Register a teacher model initializer."""
    return _register(_teachers, key, name, description, default_params)


def register_metric(key: str, name: str, description: str = "",
                    default_params: Dict[str, Any] = None):
    """Register an evaluation metric."""
    return _register(_metrics, key, name, description, default_params)


def register_output(key: str, name: str, description: str = "",
                    default_params: Dict[str, Any] = None):
    """Register an output handler."""
    return _register(_outputs, key, name, description, default_params)


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


def get_graph(key: str) -> ModuleInfo:
    """Get a graph generator by key."""
    return _get(_graphs, key, "graph")


def get_teacher(key: str) -> ModuleInfo:
    """Get a teacher initializer by key."""
    return _get(_teachers, key, "teacher")


def get_metric(key: str) -> ModuleInfo:
    """Get a metric by key."""
    return _get(_metrics, key, "metric")


def get_output(key: str) -> ModuleInfo:
    """Get an output handler by key."""
    return _get(_outputs, key, "output")


# List functions
def _list(registry: Dict[str, ModuleInfo]) -> List[ModuleInfo]:
    """List all modules in a registry."""
    return list(registry.values())


def list_algorithms() -> List[ModuleInfo]:
    """List all registered algorithms."""
    return _list(_algorithms)


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
