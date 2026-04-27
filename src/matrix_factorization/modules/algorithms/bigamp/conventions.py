"""Shared BiG-AMP numerical conventions."""

from __future__ import annotations

from typing import Optional

import torch


DAMPING_SEMANTICS = "beta_new_weight"


def blend_new_old(new: torch.Tensor, old: torch.Tensor, beta) -> torch.Tensor:
    """Blend states using BiG-AMP reference damping semantics.

    beta=1 fully accepts the new state; beta=0 freezes the old state.
    """
    return beta * new + (1.0 - beta) * old


def damped_state(
    new: torch.Tensor,
    old: Optional[torch.Tensor],
    beta,
) -> torch.Tensor:
    """Return the damped state, treating a missing old state as first iteration."""
    if old is None:
        return new
    return blend_new_old(new, old, beta)


def gaussian_posterior_update(
    current: torch.Tensor,
    residual: torch.Tensor,
    precision: torch.Tensor,
    prior_precision: float,
    prior_variance: float,
    gain_correction: Optional[torch.Tensor] = None,
    min_variance: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-mean Gaussian posterior update from BiG-AMP input messages.

    `precision` is the diagonal observation precision tau. `gain_correction`
    is the BiG-AMP self-reaction term from the other variable variance. With
    no correction this reduces to new_var * (tau * current + residual), which
    is equivalent to current + new_var * (residual - prior_precision * current).
    """
    precision = precision.clamp(min=1e-10)
    posterior_var = torch.clamp(
        1.0 / (float(prior_precision) + precision),
        min=min_variance,
        max=float(prior_variance),
    )
    if gain_correction is None:
        effective_precision = precision
    else:
        effective_precision = torch.clamp(precision - gain_correction, min=0.0)
    posterior_mean = posterior_var * (effective_precision * current + residual)
    return posterior_mean, posterior_var
