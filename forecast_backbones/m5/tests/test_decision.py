import numpy as np
import pytest

from forecast_backbones.m5.decision import physics_clip


def _q(v, T=4, K=3):
    return np.full((1, T, K), float(v))


def test_no_clear_sky_ceiling_clips_only_at_capacity():
    q = _q(120.0); q[0, 1] = 30.0
    pc = np.array([[0.0, 0.0, 50.0, np.nan]])                       # 夜间 / 夜间 / 白天 / 缺测
    out = physics_clip(q, np.zeros((1, 4), bool), pc, np.array([[100.0]]), clear_sky_cap_ratio=None)
    assert np.allclose(out[0, :, 0], [100.0, 30.0, 100.0, 100.0])   # 只钳到容量；p_clear=0 的夜间步不被钳成 0


def test_no_ceiling_and_no_capacity_cap_leaves_values():
    out = physics_clip(_q(120.0), np.zeros((1, 4), bool), np.zeros((1, 4)), np.array([[100.0]]),
                       clear_sky_cap_ratio=None, capacity_cap=False)
    assert np.allclose(out, 120.0)


def test_none_ratio_still_clips_negative_and_applies_night_mask():
    night = np.array([[False, True, False, False]])
    out = physics_clip(_q(-5.0), night, np.zeros((1, 4)), np.array([[100.0]]), clear_sky_cap_ratio=None)
    assert np.allclose(out, 0.0)
    out2 = physics_clip(_q(40.0), night, np.zeros((1, 4)), np.array([[100.0]]), clear_sky_cap_ratio=None)
    assert np.allclose(out2[0, :, 0], [40.0, 0.0, 40.0, 40.0])


def test_numeric_ratio_keeps_m8_behaviour():
    pc = np.array([[0.0, 40.0, 200.0, np.nan]])
    out = physics_clip(_q(120.0), np.zeros((1, 4), bool), pc, np.array([[100.0]]), clear_sky_cap_ratio=1.05)
    assert np.allclose(out[0, :, 0], [0.0, 42.0, 100.0, 100.0])      # night_ceiling 默认 True：p_clear=0 → 0
    out_nc = physics_clip(_q(120.0), np.zeros((1, 4), bool), pc, np.array([[100.0]]), clear_sky_cap_ratio=1.05,
                          night_ceiling=False)
    assert np.allclose(out_nc[0, :, 0], [100.0, 42.0, 100.0, 100.0])


# ---- O0：p* 网格尾部线性外推；T3：下尾 / 上尾离散度缩放 ----
from forecast_backbones.m5.decision import MODEL_LEVELS, quantile_grid, p_star, p_star_l1, scale_spread, level_value


def _lin_q(scale=100.0):
    """q_k = scale·τ_k：分位函数本身是直线，线性外推应在全网格上精确还原 scale·τ。"""
    return np.asarray(MODEL_LEVELS, float)[None, None, :] * scale


def test_linear_tail_extends_the_quantile_line_flat_tail_does_not():
    q = _lin_q()
    taus = np.arange(0.0025, 1.0, 0.005)
    lin, flat = quantile_grid(q, MODEL_LEVELS, tail="linear")[0, 0], quantile_grid(q, MODEL_LEVELS)[0, 0]
    assert np.allclose(lin, 100 * taus)
    assert np.allclose(flat[taus < 0.1], 10.0) and np.allclose(flat[taus > 0.9], 90.0)        # 默认仍是平延（逐位不变）
    assert np.array_equal(flat, quantile_grid(q, MODEL_LEVELS, tail="flat")[0, 0])


def test_linear_tail_lowers_p_star_and_is_clipped_to_zero_and_cap():
    q = _lin_q()
    cap = np.array([[100.0]])
    flat, lin = p_star(q, MODEL_LEVELS, cap), p_star(q, MODEL_LEVELS, cap, tail="linear")
    assert lin[0, 0] < flat[0, 0]                                  # 1/y² 权重压在低端：低端外推下去 → p* 变小
    steep = np.asarray([0.0, 1, 2, 3, 50, 97, 98, 99, 100.0])[None, None, :]     # 外推会越过 0 与 cap
    g = quantile_grid(steep, MODEL_LEVELS, tail="linear")
    assert g.min() < 0 and g.max() > 100                           # 网格本身不钳
    for f in (p_star, p_star_l1):
        v = f(steep, MODEL_LEVELS, cap, tail="linear")[0, 0]
        assert 0.0 <= v <= 100.0
    with pytest.raises(ValueError, match="tail"):
        quantile_grid(q, MODEL_LEVELS, tail="cubic")


def test_scale_spread_moves_only_the_requested_side_around_the_median():
    q = np.asarray([10.0, 20, 30, 40, 50, 60, 70, 80, 90])[None, None, :]
    assert np.array_equal(scale_spread(q, MODEL_LEVELS), q)                          # 都不给 = 原样
    lo = scale_spread(q, MODEL_LEVELS, lo=2.0)[0, 0]
    assert np.allclose(lo, [-30, -10, 10, 30, 50, 60, 70, 80, 90])                   # 中位数以下离中位数的距离 ×2；钳位交给 physics_clip
    hi = scale_spread(q, MODEL_LEVELS, hi=0.5)[0, 0]
    assert np.allclose(hi, [10, 20, 30, 40, 50, 55, 60, 65, 70])
    assert np.allclose(level_value(lo[None, None], MODEL_LEVELS, 0.5), 50.0)
    with pytest.raises(ValueError, match="> 0"):
        scale_spread(q, MODEL_LEVELS, lo=0.0)
