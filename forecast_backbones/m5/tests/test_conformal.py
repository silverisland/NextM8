"""M5 离线滚动 conformal 回放：分位公式、无未来泄漏、season 类池、收缩加权、端到端回放、predict 选行与落盘。"""
import math
import types

import numpy as np
import pandas as pd
import pytest

from forecast_backbones.m5.conformal import (Q_COLS, ReplayConfig, ReplayData, Replayer, cell_deltas, replay,
                                             summarize)
from forecast_backbones.m6.config import load_config
from forecast_backbones.m6.decision import MODEL_LEVELS

H = 192
DECISION = load_config(None)["decision"]


def _shape():
    """10:00 起报，第 t 步 = 10:00 + (t+1)·15min；白天 6~18 点正弦。"""
    hours = (10.0 + (np.arange(H) + 1) * 0.25) % 24.0
    return np.where((hours > 6) & (hours < 18), np.sin(np.pi * (hours - 6) / 12), 0.0)


def _raw(station, start, days, bias=1.0, cap=100.0, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    sh = _shape(); pc = sh * cap; f = 0.7 * pc
    rows = []
    for i in range(days):
        y = bias * f * (1 + noise * rng.standard_normal(H))
        rec = {"station": station, "timestamp_win": pd.Timestamp(start) + pd.Timedelta(days=i, hours=10), "cap": cap,
               "observe_power_future": y.astype(np.float32), "elev_deg": (sh * 60).astype(np.float32),
               "p_clear": pc.astype(np.float32), "night": sh <= 0, "var_bin": 1}
        for j, c in enumerate(Q_COLS):
            rec[c] = (f * (0.8 + 0.05 * j)).astype(np.float32)
        rows.append(rec)
    return pd.DataFrame(rows)


def test_cell_deltas_is_ceil_rank_order_statistic():
    rng = np.random.default_rng(1)
    scores = rng.standard_normal((50, 9)); codes = np.r_[np.zeros(20, int), np.ones(30, int)]
    groups = np.arange(50) // 5
    taus = np.array([0.1, 0.5, 0.9]); jidx = np.array([0, 4, 8])
    uc, d, nd = cell_deltas(scores, codes, groups, taus, jidx)
    assert list(uc) == [0, 1] and list(nd) == [4, 6]
    for g, idx in ((0, np.arange(20)), (1, np.arange(20, 50))):
        n = len(idx)
        for t, j, got in zip(taus, jidx, d[g]):
            k = min(max(math.ceil((n + 1) * t), 1), n)
            assert got == np.sort(scores[idx, j])[k - 1]


def test_lag_below_two_days_rejected():
    with pytest.raises(ValueError, match="lag_days"):
        ReplayConfig(lag_days=1)


def test_day_d_deltas_do_not_see_truth_from_day_d_minus_1_or_later():
    raw = pd.concat([_raw("A", "2025-06-01", 40, bias=1.2, seed=1), _raw("B", "2025-06-01", 40, seed=2)], ignore_index=True)
    cfg = ReplayConfig(ks=(15.0,))
    data = ReplayData(raw, cfg)
    i = int(np.flatnonzero((data.st == 0) & (data.day == data.day.min() + 30))[0])
    before, _ = Replayer(data).deltas(i)
    cut = pd.Timestamp("2025-06-01") + pd.Timedelta(days=29)          # d − 1 起（含当天）的真值全改掉
    late = raw["timestamp_win"] >= cut
    raw2 = raw.assign(observe_power_future=[np.full(H, 1e4, np.float32) if x else y
                                            for x, y in zip(late, raw["observe_power_future"])])
    after, _ = Replayer(ReplayData(raw2, cfg)).deltas(i)
    for arm in before:
        np.testing.assert_array_equal(before[arm], after[arm])


def test_season_pool_takes_other_years_but_never_own_future():
    raw = pd.concat([_raw("A", "2025-06-01", 60, seed=1), _raw("A", "2026-06-01", 60, seed=3),
                     _raw("B", "2026-06-01", 60, seed=2)], ignore_index=True)
    data = ReplayData(raw, ReplayConfig())
    rp = Replayer(data)
    i = int(np.flatnonzero((data.st == 0) & (data.day == data.day.min() + 50))[0])   # A，2025-07-21
    own, pool = rp.windows(i)
    d = data.day[i]
    assert own.any() and (data.p_day[own] <= d - 2).all() and (data.p_st[own] == 0).all()
    assert (data.p_st[pool] == 1).any()                                                 # B 的 2026 同季数据进类池
    assert not ((data.p_st[pool] == 0) & (data.p_day[pool] > d)).any()                 # A 自己 2026 的数据不进
    cal = Replayer(ReplayData(raw, ReplayConfig(pool="calendar")))
    _, pool_cal = cal.windows(i)
    assert not (data.p_st[pool_cal] == 1).any()                                         # calendar：同一绝对时间窗里 B 没数据


def test_shrink_is_day_weighted_blend_and_new_station_uses_pool():
    raw = pd.concat([_raw("A", "2025-06-01", 60, bias=1.3, seed=1)] +
                    [_raw(s, "2025-05-01", 90, bias=1.0, seed=k) for k, s in enumerate("BCD", 2)], ignore_index=True)
    k = 15.0
    cfg = ReplayConfig(ks=(k,))
    data = ReplayData(raw, cfg)
    rp = Replayer(data)
    j = list(cfg.levels).index(0.5)
    t = 80                                                                              # D+1 白天、lead 档 4
    first = int(np.flatnonzero(data.st == 0)[np.argmin(data.day[data.st == 0])])
    ds0, dg0 = rp.deltas(first)
    assert dg0["own_days"] == 0
    np.testing.assert_array_equal(ds0["shrink_k15"], ds0["pool"])                     # 新站：全用类
    i = int(np.flatnonzero((data.st == 0) & (data.day == data.day[first] + 22))[0])
    ds, dg = rp.deltas(i)
    D = dg["own_days"]
    assert D == 21 and D >= cfg.d_min()[j]
    w = D / (D + k)
    assert ds["shrink_k15"][t, j] == pytest.approx(w * ds["station"][t, j] + (1 - w) * ds["pool"][t, j])
    assert ds["station"][t, j] > ds["pool"][t, j] > 0                                  # A 偏高得多，类被 BCD 拉低


def test_replay_end_to_end_fixes_a_biased_model():
    raw = pd.concat([_raw("A", "2025-03-01", 120, bias=1.25, seed=1),                   # 一站一年里的一段
                     _raw("B", "2026-01-10", 40, bias=1.2, seed=2),                      # 另一年、短
                     _raw("C", "2025-12-01", 12, bias=1.2, seed=3)], ignore_index=True)  # 只有十几天
    daily, cov = replay(raw, ReplayConfig(ks=(5.0, 15.0)), DECISION, progress=False)
    assert set(daily["arm"]) == {"raw", "station", "pool", "shrink_k5", "shrink_k15"}
    assert daily.groupby("arm").size().nunique() == 1 and len(daily[daily["arm"] == "raw"]) == 172
    t = summarize(daily, cov)
    s = t["summary"].set_index(["phase", "arm"])
    assert s.loc[("second_half", "shrink_k15"), "gain"] > 1.0                          # 系统性偏低被校准纠回来
    assert s.loc[("all", "raw"), "gain"] == 0.0
    c = t["coverage"].set_index(["arm", "tau"])
    assert abs(c.loc[("shrink_k15", 0.5), "coverage"] - 0.5) < abs(c.loc[("raw", 0.5), "coverage"] - 0.5)
    assert set(t["cold_start"]["own_days_bucket"].astype(str)) >= {"0", "30+"}


# ------------------------------------------------------------------ predict
class _FakeBackbone:
    def __init__(self):
        self.seen = []

    def predict_batch(self, inputs, horizon, **kw):
        outs = []
        for x in inputs:
            fc = np.full(horizon, 10.0, np.float32)
            outs.append(types.SimpleNamespace(forecast=fc[None, :], quantiles=(fc[:, None] * np.linspace(0.8, 1.2, 9))[None]))
        return outs


def test_predict_station_raw_one_row_per_day_at_or_before_ten(tmp_path):
    from forecast_backbones.m5.conformal_replay import predict_station_raw, scored_rows
    from forecast_backbones.m5.transformer_models import TransformersAdaptedM5Model
    from forecast_backbones.m6.features import StationMeta

    L = 672
    rows = []
    for day in range(3):
        for hh in (8, 10, 11):
            rows.append({"timestamp_win": pd.Timestamp("2025-06-01") + pd.Timedelta(days=day, hours=hh), "station": "1001",
                         "observe_power": np.linspace(0, 50, L, dtype=np.float32),
                         "observe_power_future": np.full(H, 7.0, np.float32),
                         "general_ghi": np.full(L, 300.0, np.float32), "general_ghi_predict": np.full(H, 300.0, np.float32),
                         "cap_power_on": 80.0})
    path = tmp_path / "station=1001.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    assert list(scored_rows(pd.read_parquet(path))["timestamp_win"].dt.hour) == [10, 10, 10]
    meta = StationMeta(pd.DataFrame([{"station_id": 1001, "station": "1001", "lat": 23.1, "lon": 113.3, "alt_m": 10, "cap_mw": 90}]))
    model = TransformersAdaptedM5Model({"timesfm3": {"eager_init": False,
                                                     "past_future_cols": [["general_ghi", "general_ghi_predict"]]}})
    model._backbone = _FakeBackbone(); model._backbone_device = "cpu"
    seen = []
    fwd = model.forward
    model.forward = lambda bd, labels=None: (seen.append(set(bd)), fwd(bd))[1]
    raw = predict_station_raw(model, path, load_config(None), meta, cap_fallback=465.0, batch_size=2)
    assert len(raw) == 3 and (raw["timestamp_win"].dt.hour == 10).all() and (raw["station"] == "1001").all()
    assert all("observe_power_future" not in s for s in seen)                          # 真值不进模型
    assert (raw["cap"] == 90.0).all() and not raw["degraded"].any()                   # 容量只取站表 cap_mw，数据里的 80 不用
    r = raw.iloc[0]
    assert r["q_0.5"].shape == (H,) and np.allclose(r["observe_power_future"], 7.0)
    assert r["night"].any() and (~r["night"]).any() and r["p_clear"].max() > 0
    ReplayData(raw, ReplayConfig())                                                     # 能直接喂回放


def test_shard_paths_split_stations_without_overlap():
    from forecast_backbones.m5.conformal_replay import shard_paths
    paths = [f"station={i}.parquet" for i in range(17)]
    parts = [shard_paths(paths, i, 8) for i in range(8)]
    assert sorted(sum(parts, [])) == sorted(paths) and max(map(len, parts)) - min(map(len, parts)) <= 1
    with pytest.raises(ValueError):
        shard_paths(paths, 8, 8)
