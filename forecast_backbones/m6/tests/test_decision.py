import numpy as np
import pytest
from forecast_backbones.m6.decision import (MODEL_LEVELS, CAL_LEVELS, quantile_grid, level_value,
                                            p_star, p_star_l1, global_shrink_factor, enforce_monotone,
                                            physics_clip, select_point)

C = 100.0
W2 = lambda y: 1.0 / np.maximum(y, 0.2 * C) ** 2
W1 = lambda y: 1.0 / np.maximum(y, 0.2 * C)

def test_quantile_grid_flat_tails_and_shape():
    q = np.array([[10.0, 20.0, 30.0]]); lv = [0.25, 0.5, 0.75]
    g = quantile_grid(q, lv, step=0.25)                   # taus = .125,.375,.625,.875
    assert g.shape == (1, 4)
    assert np.allclose(g[0], [10.0, 15.0, 25.0, 30.0])

def test_degenerate_quantiles_give_truth():
    q = np.full((1, 9), 42.0)
    assert np.isclose(p_star(q, MODEL_LEVELS, C)[0], 42.0)
    assert np.isclose(p_star_l1(q, MODEL_LEVELS, C)[0], 42.0)
    assert np.isclose(level_value(q, MODEL_LEVELS, 0.5)[0], 42.0)

def test_p_star_matches_two_point_closed_form():
    # 双峰：云态 20 / 晴态 80 各半 → p* = Σ y w / Σ w = 23.5（spec §0 的数）
    q = np.array([[20.0] * 5 + [80.0] * 5]); lv = np.linspace(0.05, 0.95, 10)
    got = p_star(q, lv, C, step=0.001)[0]
    exp = (20 * W2(20) + 80 * W2(80)) / (W2(20) + W2(80))   # = 23.5
    # 0.45~0.55 之间 10% 的概率质量被线性插值成 20→80 的过渡带，把 p* 抬高约 0.7；容差 1.0
    assert abs(got - exp) < 1.0

def test_p_star_l1_is_weighted_median():
    q = np.array([[20.0] * 5 + [80.0] * 5]); lv = np.linspace(0.05, 0.95, 10)
    got = p_star_l1(q, lv, C, step=0.001)[0]
    # w1(20)=1/20, w1(80)=1/80 → 低峰权重 0.05·0.5=0.025, 高峰 0.00625 → 中位数落在低峰
    assert got < 30.0

def test_p_star_below_median_under_symmetric_spread():
    med = 60.0; q = med * np.linspace(0.6, 1.4, 9)[None, :]
    assert p_star(q, MODEL_LEVELS, C)[0] < med
    assert p_star_l1(q, MODEL_LEVELS, C)[0] < med

def test_global_shrink_factor_closed_form():
    rng = np.random.default_rng(0)
    p = rng.uniform(10, 90, 500); y = p * (1 + rng.normal(0, 0.3, 500)); y = np.clip(y, 0, None)
    f = global_shrink_factor(y, p, C)
    d = np.maximum(y, 0.2 * C)
    assert np.isclose(f, np.sum(y * p / d**2) / np.sum(p**2 / d**2))
    assert f < 1.0

def test_global_shrink_factor_drops_nan_cap_points():
    y = np.array([10.0, 20.0, 30.0]); p = np.array([9.0, 19.0, 29.0]); cap = np.array([100.0, np.nan, 100.0])
    got = global_shrink_factor(y, p, cap)
    clean = global_shrink_factor(np.array([10.0, 30.0]), np.array([9.0, 29.0]), 100.0)
    assert np.isclose(got, clean)

def test_global_shrink_factor_all_nan_cap_raises():
    y = np.array([10.0, 20.0]); p = np.array([9.0, 19.0]); cap = np.array([np.nan, np.nan])
    with pytest.raises(ValueError):
        global_shrink_factor(y, p, cap)

def test_global_shrink_factor_per_row_cap_matches_manual_aggregation():
    # 逐点 cap 数组（Tasks 9/11 用法：cap[:, None] 广播到每行不同容量）应等价于逐行标量 cap 手工累加
    y = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    p = np.array([[9.0, 21.0, 28.0], [39.0, 52.0, 63.0]])
    cap_per_row = np.array([100.0, 150.0])
    cap_full = np.broadcast_to(cap_per_row[:, None], y.shape)
    got = global_shrink_factor(y, p, cap_full)
    num = 0.0; den = 0.0
    for i in range(2):
        d = np.maximum(y[i], 0.2 * cap_per_row[i])
        num += np.sum(y[i] * p[i] / d ** 2)
        den += np.sum(p[i] ** 2 / d ** 2)
    assert np.isclose(got, num / den)

def test_p_star_accepts_per_row_cap_array():
    # cap[:, None] 广播到 [B,T,K]（Tasks 9/11 用法）应等价于逐行用标量 cap 分别调用
    B, T, K = 2, 4, 9
    base = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])
    q = np.tile(base[None, None, :], (B, T, 1))
    cap = np.array([100.0, 200.0])
    got = p_star(q, MODEL_LEVELS, cap[:, None])
    assert got.shape == (B, T)
    row0 = p_star(q[0:1], MODEL_LEVELS, cap[0])
    row1 = p_star(q[1:2], MODEL_LEVELS, cap[1])
    assert np.allclose(got[0], row0[0])
    assert np.allclose(got[1], row1[0])

def test_enforce_monotone_and_physics_clip():
    q = np.array([[[30.0, 10.0, 20.0]]])                   # [B=1,T=1,K=3]
    assert enforce_monotone(q).tolist() == [[[10.0, 20.0, 30.0]]]
    night = np.array([[True]]); pc = np.array([[50.0]]); cap = np.array([[100.0]])
    assert physics_clip(q, night, pc, cap).tolist() == [[[0.0, 0.0, 0.0]]]
    day = np.array([[False]])
    assert physics_clip(np.array([[[70.0, 80.0, 200.0]]]), day, pc, cap).tolist() == [[[52.5, 52.5, 52.5]]]

def test_physics_clip_handles_nonfinite_p_clear():
    # p_clear 缺测（NaN）的白天步退化为只用装机容量钳位，不产生 NaN
    q = np.array([[[1500.0], [2000.0], [999.0]]])          # [B=1,T=3,K=1]
    night = np.array([[False, False, False]])
    p_clear = np.array([[np.nan, 500.0, 0.0]])
    cap = np.array([[1000.0]])
    out = physics_clip(q, night, p_clear, cap)
    assert not np.isnan(out).any()
    assert np.allclose(out[0, :, 0], [1000.0, 525.0, 0.0])   # clip-to-cap / clip-to-1.05·p_clear / 0

def test_physics_clip_night_ceiling_switch():
    # 默认 night_ceiling=True：p_clear=0 的步 upper=0 → 钳成 0（隐式夜间归零，m6 口径不变）
    # night_ceiling=False：p_clear<=0 的步只用 cap 钳位；NaN 步与白天步两种口径一致
    q = np.array([[[1500.0], [600.0], [999.0], [-5.0]]])          # [B=1,T=4,K=1]
    night = np.array([[False, False, False, False]])
    p_clear = np.array([[np.nan, 500.0, 0.0, 0.0]])
    cap = np.array([[1000.0]])
    a = physics_clip(q, night, p_clear, cap)
    b = physics_clip(q, night, p_clear, cap, night_ceiling=False)
    assert np.allclose(a[0, :, 0], [1000.0, 525.0, 0.0, 0.0])
    assert np.allclose(b[0, :, 0], [1000.0, 525.0, 999.0, 0.0])    # 夜间正值保留（≤cap），负值仍截到 0
    # night 掩码为 True 的步无论哪种口径都归零
    c = physics_clip(q, np.array([[False, False, True, False]]), p_clear, cap, night_ceiling=False)
    assert np.allclose(c[0, :, 0], [1000.0, 525.0, 0.0, 0.0])
    # capacity_cap=False 时夜间步仍用 cap 钳位（与非有限值的处理一致，不会无上界）
    d = physics_clip(np.array([[[5000.0]]]), np.array([[False]]), np.array([[0.0]]), np.array([[1000.0]]),
                     capacity_cap=False, night_ceiling=False)
    assert d.tolist() == [[[1000.0]]]
    # clip_negative=False：负值保留（严格裸对比）；默认截到 0
    e = physics_clip(np.array([[[-5.0], [-5.0]]]), np.array([[False, False]]), np.array([[0.0, 500.0]]), np.array([[1000.0]]),
                     night_ceiling=False, clip_negative=False)
    assert e[0, :, 0].tolist() == [-5.0, -5.0]
    assert physics_clip(np.array([[[-5.0]]]), np.array([[False]]), np.array([[500.0]]), np.array([[1000.0]])).tolist() == [[[0.0]]]

def test_select_point_routes_by_step():
    B, T, K = 1, 200, 9
    q = np.tile((60.0 * np.linspace(0.6, 1.4, K))[None, None, :], (B, T, 1))
    cap = np.array([C])
    out = select_point(q, MODEL_LEVELS, cap, default="median",
                       by_step=[{"steps": [0, 16], "method": "p_star_l1"},
                                {"steps": [55, 151], "method": "p_star"}])
    assert out.shape == (B, T)
    assert np.isclose(out[0, 20], 60.0)                    # 默认 median
    assert out[0, 5] < 60.0 and out[0, 100] < 60.0         # 两个打分窗都收缩
    assert out[0, 100] < out[0, 5]                          # L2 口径收缩比 L1 狠
    g = select_point(q, MODEL_LEVELS, cap, default="global_shrink", shrink_factor=0.8)
    assert np.isclose(g[0, 0], 48.0)
    t3 = select_point(q, MODEL_LEVELS, cap, default="quantile:0.3")
    assert np.isclose(t3[0, 0], 60.0 * 0.8)
