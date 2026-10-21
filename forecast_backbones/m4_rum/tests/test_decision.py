"""Numerical contracts for Rum's independent probabilistic decision layer."""

from fractions import Fraction

import numpy as np
import pytest
import torch

from forecast_backbones.m4_rum.decision import decide, level_value, quantile_grid


def _reference(q, levels, capacity, floor_ratio, step):
    # NumPy's scalar interpolation gives an independent reference implementation.
    taus = np.arange(step / 2, 1, step)
    result = np.empty(q.shape[:2])
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            values = np.interp(taus, levels, np.sort(q[b, h]))
            weights = np.maximum(values, floor_ratio * capacity[b]) ** -2
            result[b, h] = np.average(values, weights=weights)
    return result


def test_hand_computed_weighted_point_and_crossing_repair():
    # The four grid values are [0, 2, 6, 8], with floor = 2.
    q = torch.tensor([[[8.0, 0.0]]], dtype=torch.float64)
    out = decide(q, [0.25, 0.75], torch.tensor([10.0]), {"tau_grid_step": 0.25})
    expected = (2 * Fraction(1, 4) + 6 * Fraction(1, 36) + 8 * Fraction(1, 64)) / (
        Fraction(1, 4) + Fraction(1, 4) + Fraction(1, 36) + Fraction(1, 64)
    )
    torch.testing.assert_close(out["quantiles_cal"], q.flip(-1))
    assert out["median"].item() == 4
    assert out["p_star"].item() == pytest.approx(float(expected))
    assert out["point"] is out["median"]
    assert out["p_star"].item() < out["median"].item()
    assert q.tolist() == [[[8, 0]]]  # Sorting must not modify raw network outputs.


@pytest.mark.parametrize("step", [0.005, 0.07, 0.3, 1.0])
def test_matches_independent_reference_with_nonuniform_levels(step):
    levels = [0.05, 0.15, 0.4, 0.6, 0.95]
    q = np.random.default_rng(42).uniform(-15, 200, size=(2, 7, len(levels)))
    capacity = np.array([10.0, 250.0])
    out = decide(
        torch.from_numpy(q),
        levels,
        torch.from_numpy(capacity),
        {"default": "p_star", "floor_ratio": 0.3, "tau_grid_step": step},
    )
    np.testing.assert_allclose(
        out["p_star"].numpy(), _reference(q, levels, capacity, 0.3, step), rtol=1e-12
    )
    assert out["point"] is out["p_star"]


def test_interpolation_has_flat_tails_and_nonuniform_median():
    q = torch.tensor([[[2.0, 10.0]]], dtype=torch.float64)
    levels = [0.3, 0.7]
    torch.testing.assert_close(
        quantile_grid(q, levels, step=0.2),
        torch.tensor([[[2.0, 2.0, 6.0, 10.0, 10.0]]], dtype=torch.float64),
    )
    assert level_value(q, levels, 0.0).item() == 2
    assert level_value(q, levels, 1.0).item() == 10
    assert level_value(q, [0.1, 0.9], 0.5).item() == 6
    assert level_value(q, [0.1, 0.6], 0.5).item() == pytest.approx(8.4)


def test_zero_and_negative_quantiles_are_not_physically_clipped():
    q = torch.tensor([[[0.0, 0.0], [-8.0, -2.0], [20.0, 30.0]]])
    out = decide(q, [0.25, 0.75], torch.tensor([[10.0]]), {"default": "p_star"})
    assert out["point"][0, 0].item() == 0
    assert out["point"][0, 1].item() == pytest.approx(-5)
    assert out["point"][0, 2].item() > 10
    torch.testing.assert_close(out["quantiles_cal"], q)


def test_power_and_capacity_scale_together():
    single = torch.tensor([[[0.01, 0.2, 0.8], [-0.1, 0.3, 1.4]]])
    q = torch.cat([single, single * 1000], dim=0)
    capacities = torch.tensor([[1.0], [1000.0]])
    out = decide(q, [0.1, 0.5, 0.9], capacities, {"default": "p_star"})
    for value in out.values():
        torch.testing.assert_close(value[1], value[0] * 1000)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_small_capacities_and_low_precision_weights_remain_finite(dtype):
    q = torch.tensor([[[0.0001, 0.0004, 0.001]]], dtype=dtype)
    capacity = torch.tensor([0.001], dtype=dtype)
    out = decide(q, [0.1, 0.5, 0.9], capacity, {"default": "p_star"})
    assert out["point"].dtype == torch.float32
    assert all(value.device == q.device for value in out.values())
    assert all(torch.isfinite(value).all() for value in out.values())
    expected = _reference(
        q.float().numpy(), [0.1, 0.5, 0.9], capacity.float().numpy(), 0.2, 0.005
    )
    np.testing.assert_allclose(out["point"].numpy(), expected, rtol=1e-6)


def test_invalid_tensor_shapes_fail_before_broadcasting():
    with pytest.raises(ValueError, match="quantiles must"):
        decide(torch.zeros(2, 3), [0.1, 0.9], torch.ones(2), {})
    with pytest.raises(ValueError, match="capacity must"):
        decide(torch.zeros(2, 3, 2), [0.1, 0.9], torch.ones(2, 3), {})
