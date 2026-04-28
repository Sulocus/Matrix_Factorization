"""Shared distribution-name helpers."""

from __future__ import annotations

from typing import Any


F_DISTRIBUTION_ISING = "ising"
F_DISTRIBUTION_GAUSSIAN = "gaussian"
F_DISTRIBUTION_CHOICES = (F_DISTRIBUTION_ISING, F_DISTRIBUTION_GAUSSIAN)


def normalize_f_distribution(value: Any, *, default: str = F_DISTRIBUTION_ISING) -> str:
    """Return the canonical spreading-F distribution name.

    Canonical names are ``ising`` and ``gaussian``.  The historical
    ``rademacher`` spelling remains accepted as an input alias for ``ising`` so
    old configs and checkpoints do not fail at load time.
    """
    if value is None:
        value = default

    if isinstance(value, bool):
        key = str(value).lower()
    elif isinstance(value, int):
        key = str(value)
    else:
        key = str(value).strip().lower()

    aliases = {
        "1": F_DISTRIBUTION_ISING,
        F_DISTRIBUTION_ISING: F_DISTRIBUTION_ISING,
        "rademacher": F_DISTRIBUTION_ISING,
        "binary": F_DISTRIBUTION_ISING,
        "pm1": F_DISTRIBUTION_ISING,
        "+/-1": F_DISTRIBUTION_ISING,
        "2": F_DISTRIBUTION_GAUSSIAN,
        F_DISTRIBUTION_GAUSSIAN: F_DISTRIBUTION_GAUSSIAN,
        "normal": F_DISTRIBUTION_GAUSSIAN,
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(
            "spreading.f_distribution must be one of "
            f"{list(F_DISTRIBUTION_CHOICES)}; legacy alias 'rademacher' is accepted. "
            f"Got {value!r}."
        ) from exc


def is_ising_f_distribution(value: Any) -> bool:
    """Whether a spreading-F distribution has exact +/-1 Ising entries."""
    return normalize_f_distribution(value) == F_DISTRIBUTION_ISING


def f_distribution_bytes_per_element(value: Any) -> int:
    """Storage bytes per F element under the canonical distribution."""
    return 1 if is_ising_f_distribution(value) else 4
