from __future__ import annotations

import math
from typing import Callable, Optional

import torch


def _broadcast_time(t: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    t = t.to(device=reference.device, dtype=reference.dtype)
    while t.ndim < reference.ndim:
        t = t.unsqueeze(-1)
    return t


def x_to_v(
    x: torch.Tensor,
    z_t: torch.Tensor,
    t: torch.Tensor,
    t_eps: float = 0.05,
) -> torch.Tensor:
    """Convert an x-prediction into the induced velocity."""
    if x.shape != z_t.shape:
        raise ValueError("x and z_t must have matching shapes")
    denom = (1.0 - _broadcast_time(t, z_t)).clamp_min(float(t_eps))
    return (x - z_t) / denom


def solver_grid(
    steps: int,
    mean: float = -1.5,
    std: float = 0.8,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    quantile: bool = False,
) -> torch.Tensor:
    """Build the ODE grid: endpoints 0 and 1 with sorted logit-normal inner times.

    The inner times are a random draw unless ``quantile`` is set. Then they are
    the fixed quantiles sigmoid(mean + std * Phi^-1(i / steps)), i = 1, ...,
    steps - 1, computed in float64 and draw no random numbers.
    """
    steps = int(steps)
    if steps < 1:
        raise ValueError("steps must be at least 1")
    lo = torch.tensor([0.0], device=device, dtype=dtype)
    hi = torch.tensor([1.0], device=device, dtype=dtype)
    inner_count = steps - 1
    if inner_count == 0:
        return torch.cat([lo, hi], dim=0)
    if quantile:
        probs = torch.arange(1, steps, dtype=torch.float64) / float(steps)
        normal = math.sqrt(2.0) * torch.erfinv(2.0 * probs - 1.0)
        inner = torch.sigmoid(float(mean) + float(std) * normal)
        return torch.cat([lo, inner.to(device=device, dtype=dtype), hi], dim=0)
    raw = torch.randn((inner_count,), device=device, dtype=dtype)
    inner = torch.sort(torch.sigmoid(raw * float(std) + float(mean))).values
    return torch.cat([lo, inner, hi], dim=0)


def euler_rollout(
    model_predict_fn: Callable[..., torch.Tensor],
    z: torch.Tensor,
    grid: torch.Tensor,
    *,
    sc_scale: Optional[torch.Tensor] = None,
    t_eps: float = 0.05,
) -> torch.Tensor:
    """Euler rollout for JPEG-DLM x-prediction samplers.

    ``model_predict_fn`` is called as
    ``fn(z, t, self_cond=x_prev, sc_scale=sc_scale, decode=False)``.
    """
    if grid.ndim != 1 or grid.numel() < 2:
        raise ValueError("grid must be a 1D tensor with at least two points")
    grid = grid.to(device=z.device, dtype=z.dtype)
    x_prev = None
    for index in range(grid.numel() - 1):
        t_scalar = grid[index]
        dt = grid[index + 1] - t_scalar
        t = t_scalar.expand(z.size(0))
        x = model_predict_fn(
            z,
            t,
            self_cond=x_prev,
            sc_scale=sc_scale,
            decode=False,
        )
        v = x_to_v(x, z, t, t_eps=t_eps)
        z = z + dt * v
        x_prev = x
    return z
