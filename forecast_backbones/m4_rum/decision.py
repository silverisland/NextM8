"""Independent quantile interpolation and short-term point selection for Rum."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor


def quantile_cols(levels: Sequence[float]) -> list[str]:
    """Round-trip probability names prevent collisions for custom close levels."""
    return [f"q_{float(level)}" for level in levels]


def _interpolate(q: Tensor, levels: Sequence[float], taus: Tensor) -> Tensor:
    # Keep mixed-precision predictions on their device, but compute in FP32 or above.
    q = q.to(torch.float64 if q.dtype == torch.float64 else torch.float32)
    levels = torch.as_tensor(levels, dtype=q.dtype, device=q.device)
    taus = taus.to(device=q.device, dtype=q.dtype)
    lo = (torch.searchsorted(levels, taus, right=True) - 1).clamp(0, len(levels) - 2)
    weight = ((taus - levels[lo]) / (levels[lo + 1] - levels[lo])).clamp(0, 1)
    return q[..., lo] * (1 - weight) + q[..., lo + 1] * weight


def quantile_grid(q: Tensor, levels: Sequence[float], step: float = 0.005) -> Tensor:
    """Evaluate ordered quantiles at midpoint probabilities, with flat tails."""
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    taus = torch.arange(step / 2, 1.0, step, dtype=dtype, device=q.device)
    return _interpolate(q, levels, taus)


def level_value(q: Tensor, levels: Sequence[float], tau: float) -> Tensor:
    """Interpolate a single probability, extending the endpoint values outside."""
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    taus = torch.tensor([tau], dtype=dtype, device=q.device)
    return _interpolate(q, levels, taus).squeeze(-1)


def p_star(
    q: Tensor,
    levels: Sequence[float],
    capacity: Tensor,
    floor_ratio: float = 0.2,
    step: float = 0.005,
) -> Tensor:
    """Minimize expected squared error weighted by max(target, floor) ** -2.

    ``q`` has shape [batch, horizon, quantile] and is already ordered. Capacity
    is positive, has shape [batch] or [batch, 1], and uses the same units as q.
    Scaling every weight by floor ** 2 leaves the weighted mean unchanged and
    avoids overflow from reciprocal squared powers at small capacities.
    """
    y = quantile_grid(q, levels, step)
    floor = capacity.to(device=y.device, dtype=y.dtype).reshape(-1, 1, 1) * floor_ratio
    weights = (floor / torch.maximum(y, floor)).square()
    return (y * weights).sum(-1) / weights.sum(-1)


def decide(
    q_raw: Tensor,
    levels: Sequence[float],
    capacity: Tensor,
    point_cfg: Mapping,
) -> dict[str, Tensor]:
    """Sort, optionally clip, then apply global or step-specific decisions.

    Physical clipping is opt-in; raw training quantiles are never modified.
    """
    if q_raw.ndim != 3 or q_raw.shape[-1] != len(levels):
        raise ValueError("quantiles must have shape [batch, horizon, len(levels)]")
    if capacity.shape not in ((q_raw.shape[0],), (q_raw.shape[0], 1)):
        raise ValueError("capacity must have shape [batch] or [batch, 1]")
    quantiles = q_raw.sort(dim=-1).values
    if point_cfg.get("clip_negative", False):
        quantiles = quantiles.clamp_min(0)
    if point_cfg.get("capacity_cap", False):
        quantiles = torch.minimum(quantiles, capacity.to(quantiles).reshape(-1, 1, 1))
    median = level_value(quantiles, levels, 0.5)
    optimal = p_star(
        quantiles,
        levels,
        capacity,
        floor_ratio=point_cfg.get("floor_ratio", 0.2),
        step=point_cfg.get("tau_grid_step", 0.005),
    )
    method = point_cfg.get("default", "median")
    rules = point_cfg.get("by_step", [])
    choices = {"median": median, "p_star": optimal}
    if method == "p_star_l1" or any(rule["method"] == "p_star_l1" for rule in rules):
        grid = quantile_grid(quantiles, levels, point_cfg.get("tau_grid_step", 0.005))
        floor = capacity.to(grid).reshape(-1, 1, 1) * point_cfg.get("floor_ratio", 0.2)
        weights = floor / torch.maximum(grid, floor)
        cdf = weights.cumsum(-1)
        index = (cdf >= cdf[..., -1:] * 0.5).to(torch.int64).argmax(-1, keepdim=True)
        choices["p_star_l1"] = grid.gather(-1, index).squeeze(-1)
    if method not in choices:
        raise ValueError(f"Unsupported point method: {method!r}")
    point = choices[method]
    if rules:
        point = point.clone()
        for rule in rules:
            start, end = rule["steps"]
            point[:, start:end] = choices[rule["method"]][:, start:end]
    return {
        "quantiles_cal": quantiles,
        "median": median,
        "p_star": optimal,
        "point": point,
    }
