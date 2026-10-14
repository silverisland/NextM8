import numpy as np

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
