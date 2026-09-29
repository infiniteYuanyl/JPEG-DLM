from __future__ import annotations

import math
from typing import Callable, Optional

import torch


def x_to_v(x: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Velocity induced by the clean-embedding prediction ``x`` at time ``t``."""
    return (x - z_t) / (1.0 - t[:, None, None])


def solver_grid(
    steps: int,
    mean: float = -1.5,
    std: float = 0.8,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    quantile: bool = False,
) -> torch.Tensor:
    """Build the time grid: endpoints 0 and 1 with sorted logit-normal inner times.

    The inner times are random by default. With ``quantile=True`` they are the
    fixed quantiles sigmoid(mean + std * Phi^-1(i / steps)) for i = 1, ...,
    steps - 1, computed in float64 without drawing random numbers.
    """
    steps = int(steps)
    if steps < 1:
        raise ValueError("steps must be at least 1")
    lo = torch.tensor([0.0], device=device, dtype=dtype)
    hi = torch.tensor([1.0], device=device, dtype=dtype)
    if quantile:
        probs = torch.arange(1, steps, dtype=torch.float64) / float(steps)
        normal = math.sqrt(2.0) * torch.erfinv(2.0 * probs - 1.0)
        inner = torch.sigmoid(float(mean) + float(std) * normal)
        return torch.cat([lo, inner.to(device=device, dtype=dtype), hi], dim=0)
    raw = torch.randn((steps - 1,), device=device, dtype=dtype)
    inner = torch.sort(torch.sigmoid(raw * float(std) + float(mean))).values
    return torch.cat([lo, inner, hi], dim=0)


def euler_rollout(
    model_predict_fn: Callable[..., torch.Tensor],
    z: torch.Tensor,
    grid: torch.Tensor,
    *,
    sc_scale: torch.Tensor,
) -> torch.Tensor:
    """Integrate the flow ODE along ``grid`` with Euler steps.

    ``model_predict_fn(z, t, self_cond=x_prev, sc_scale=sc_scale, decode=False)``
    predicts the clean embedding; each prediction is the self-conditioning
    input of the next step.
    """
    grid = grid.to(device=z.device, dtype=z.dtype)
    x_prev = None
    for index in range(grid.numel() - 1):
        t_scalar = grid[index]
        dt = grid[index + 1] - t_scalar
        t = t_scalar.expand(z.size(0))
        x = model_predict_fn(z, t, self_cond=x_prev, sc_scale=sc_scale, decode=False)
        v = x_to_v(x, z, t)
        z = z + dt * v
        x_prev = x
    return z
