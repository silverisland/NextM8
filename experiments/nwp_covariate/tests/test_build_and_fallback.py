import json

import numpy as np
import pandas as pd

from build_feature_store import build, feature_store_dir, write_feature_store
from fallback import apply_fallback
from ops import BuildContext, Pair


def _arr(df, col):
    return np.stack(df[col].to_numpy())


def test_build_maps_units_derives_and_writes(ds_dir):
    cfg, raw = ds_dir
    frames, rep = build(cfg, "ultra_short")
    assert set(frames) == set(raw["station"].unique())
    assert rep["rows"] == len(raw)
    g = frames["plant_guangfu1025"]
    # DS 同格式：历史 672 list，未来 192 list
    assert _arr(g, "observe_power").shape == (3, 672)
    assert _arr(g, "observe_power_future").shape == (3, 192)
    # 单位：EC 累积 -> W/m²，与 general_GHI×1.02 接近；t2m 已转 °C
    ghi_ec = _arr(g, "ghi_ec_predict"); sg = _arr(g, "ghi_solargis_predict")
    day = sg > 100
    assert np.nanmedian(ghi_ec[day] / sg[day]) == pytest_approx(1.02, 0.03)
    assert 0 < np.nanmean(_arr(g, "t2m_ec")) < 45
    # 衍生量都算出来了
    for n in ("ghi_clear", "kt_cons", "ghi_cons", "p_phys", "spread", "dhi_ec", "beam_tr_ec", "precip_ec"):
        assert n in rep["derived_ok"], (n, rep["derived_fail"])
        assert n in g.columns and n + "_predict" in g.columns
    # 物理功率代理量纲与功率一致
    assert 0.5 < np.nanmax(_arr(g, "p_phys")) / np.nanmax(_arr(g, "observe_power")) < 2.0
    # history_only 的实测列没有 _predict
    assert "ghi_obs" in g.columns and "ghi_obs_predict" not in g.columns
    # 未映射到的可选字段进 missing_raw 且 optional
    miss = {m["std"]: m for m in rep["missing_raw"]}
    assert miss["ghi_xr"]["optional"] is True
    out = feature_store_dir(cfg, "ultra_short")
    write_feature_store(frames, out, rep)
    assert (out / "station=plant_guangfu1025.parquet").exists()
    back = pd.read_parquet(out / "station=plant_guangfu1025.parquet")
    assert _arr(back, "ghi_cons_predict").shape == (3, 192)
    assert json.load(open(out / "build_report.json"))["stations"] == 3


def pytest_approx(v, tol):
    class A:
        def __eq__(self, other):
            return abs(other - v) <= tol
    return A()


def test_drop_source_ablation_triggers_fallback(ds_dir):
    cfg, raw = ds_dir
    frames, rep = build(cfg, "ultra_short", ablation={"drop_sources": ["ec"], "apply_fallback": True})
    g = frames["plant_guangfu1358"]
    assert "ghi_ec" not in rep["mapped"]
    # 链：ghi_ec <- ghi_xr(无) <- ghi_gdfs：整列由 gdfs 顶上
    fb = {r["name"]: r for r in rep["fallback"]}
    assert fb["ghi_ec"]["filled_by"].get("ghi_gdfs") == 9   # 3 站 × 3 窗，报告是全表口径
    assert np.allclose(_arr(g, "ghi_ec_predict"), _arr(g, "ghi_gdfs_predict"), equal_nan=True)
    # t2m_ec 由 t2m_gdfs 顶上
    assert fb["t2m_ec"]["filled_by"].get("t2m_gdfs") == 9


def test_mask_future_tail_small_gap_interpolates_large_gap_swaps(ds_dir):
    cfg, raw = ds_dir
    # 掩掉未来 96/192 = 50% > 30% 阈值 -> 换源；所有 NWP 源都被掩，链走到 ghi_kt_recent（晴空×近期 kt）
    frames, rep = build(cfg, "ultra_short", ablation={"mask_future_tail": 96, "apply_fallback": True})
    fb = {r["name"]: r for r in rep["fallback"]}
    assert "ghi_ec" in fb and fb["ghi_ec"]["n_dropped"] == 0
    assert fb["ghi_ec"]["filled_by"] == {"ghi_kt_recent": 9}
    g = frames["plant_guangfu1025"]
    assert np.isfinite(_arr(g, "ghi_ec_predict")).all()


def test_apply_fallback_const_expr_and_drop():
    B, C, P = 2, 8, 4
    ts = np.array([np.datetime64("2026-08-26T10:00:00")] * B)
    ctx = BuildContext(ts_win=ts, context_len=C, pred_len=P, lat=np.full(B, 23.0), lon=np.full(B, 108.0),
                       alt_m=np.zeros(B), cap=np.full(B, 100.0))
    ctx.cols["x"] = Pair(np.full((B, C), np.nan), np.full((B, P), np.nan))
    ctx.cols["skt"] = Pair(np.full((B, C), 40.0), np.array([[40.0] * P, [np.nan] * P]))
    rep = apply_fallback(ctx, "x", ["skt-8", "const:25"], {"max_missing_frac": 0.3})
    assert rep.filled_by == {"skt-8": 1, "const:25": 1}
    assert np.allclose(ctx.cols["x"].hist[0], 32.0) and np.allclose(ctx.cols["x"].hist[1], 25.0)
    ctx.cols["y"] = Pair(np.full((B, C), np.nan), np.full((B, P), np.nan))
    rep = apply_fallback(ctx, "y", ["nope", "drop", "const:1"], {"max_missing_frac": 0.3})
    assert rep.n_dropped == 2 and np.isnan(ctx.cols["y"].full()).all()


def test_nwp_lag_ablation_shifts_and_reports(ds_dir):
    cfg, raw = ds_dir
    frames, rep = build(cfg, "ultra_short", ablation={"nwp_lag_days": 1, "apply_fallback": False})
    assert any("NWP 延迟 1 天" in w for w in rep["warnings"])
    g = frames["plant_guangfu1025"]
    # 第一行没有前一天 -> 全 NaN；第二行 = 第一行整体左移 96 步
    ge = _arr(g, "ghi_gdfs")
    assert np.isnan(ge[0]).all()
    frames0, _ = build(cfg, "ultra_short")
    g0 = frames0["plant_guangfu1025"]
    full0 = np.concatenate([_arr(g0, "ghi_gdfs"), _arr(g0, "ghi_gdfs_predict")], axis=1)
    assert np.allclose(ge[1][:672 - 96 + 0], full0[0][96:672], atol=1e-3) or np.allclose(ge[1], full0[0][96:96 + 672], atol=1e-3)
