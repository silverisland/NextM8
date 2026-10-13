import numpy as np
import pytest
from forecast_backbones.m8.config import load_config
from forecast_backbones.m8.features import FeatureBuilder, cpm_token_mask
from forecast_backbones.m6.features import StationMeta


def _fb(small_cfg, **over):
    cfg = load_config({**small_cfg, **over}) if not over else load_config({**small_cfg, "train": {**small_cfg.get("train", {}), **over.get("train", {})},
                                                                           "features": {**small_cfg.get("features", {}), **over.get("features", {})}})
    return cfg, FeatureBuilder(cfg, StationMeta.from_csv(cfg["dataset"]["station_meta"]))


def test_channels_and_shapes(small_cfg, batch):
    cfg, fb = _fb(small_cfg)
    inp = fb.build(batch["batch_dict"], training=False)
    B = inp["x"].shape[0]
    assert inp["x"].shape == (B, 864, len(fb.channel_names))
    assert inp["static"].shape == (B, fb.n_static)
    assert np.isfinite(inp["x"]).all() and np.isfinite(inp["static"]).all()
    i = fb.idx
    assert (inp["x"][:, 672:, i["y_mask"]] == 1).all() and (inp["x"][:, 672:, i["is_future"]] == 1).all()
    assert (inp["x"][:, :672, i["is_future"]] == 0).all()
    assert (inp["x"][:, :672, i["y_mask"]] == 0).all()        # 推理态：历史无掩码（数据无 NaN）
    assert 0 <= inp["x"][:, :, i["k_nwp"]].max() <= 1.5
    assert inp["ctx"].night.shape == (B, 864) and inp["ctx"].night[:, :].any()
    assert np.isnan(inp["target"][:, 672:]).all()            # 没给标签 → 未来目标 NaN


def test_clear_sky_ratio_module(small_cfg, batch):
    """默认关：比值恒 1、通道不变。开：p_clear 通道与 ctx.p_clear 同乘比值，辐照比 k_nwp 不动；
    历史出力 ×3（倾斜阵列冬季 y/p_clear 可到 1.6）比值上升，×0.1 卡在下限 1。"""
    bd = batch["batch_dict"]
    cfg0, fb0 = _fb(small_cfg)
    base = fb0.build(bd, training=False)
    assert base["ctx"].cs_ratio is not None and np.allclose(base["ctx"].cs_ratio, 1.0)
    cfg1, fb1 = _fb(small_cfg, features={"derived": {"clear_sky_ratio": {"enabled": True}}})
    assert cfg1["features"]["derived"]["clear_sky_ratio"]["quantile"] == 0.98          # 其余键保留默认
    on = fb1.build(bd, training=False)
    r = on["ctx"].cs_ratio; i = fb1.idx
    assert r.shape == (len(r),) and np.isfinite(r).all() and (r >= 1.0).all() and (r <= 2.5).all()
    assert np.allclose(on["x"][:, :, i["p_clear"]], base["x"][:, :, i["p_clear"]] * r[:, None], atol=1e-6)
    assert np.allclose(on["ctx"].p_clear, base["ctx"].p_clear * r[:, None])
    assert np.allclose(on["ctx"].ghi_clear, base["ctx"].ghi_clear) and np.allclose(on["x"][:, :, i["k_nwp"]], base["x"][:, :, i["k_nwp"]])
    bd3 = dict(bd); bd3["observe_power"] = bd["observe_power"] * 3.0
    r3 = fb1.build(bd3, training=False)["ctx"].cs_ratio
    bd01 = dict(bd); bd01["observe_power"] = bd["observe_power"] * 0.1
    r01 = fb1.build(bd01, training=False)["ctx"].cs_ratio
    assert (r3 >= r - 1e-9).all() and (r3 > 1.0).any() and np.allclose(r01, 1.0)
    tr = fb1.build(bd3, training=True, y_future=batch["labels"]["observe_power_future"], rng=np.random.default_rng(0))   # CPM 掩码下也有限
    assert np.isfinite(tr["ctx"].cs_ratio).all() and np.isfinite(tr["x"]).all()


def test_cap_only_from_station_meta(small_cfg, batch):
    """容量只来自站表：数据里 cap_power_on 改成 1/10 后特征逐位不变；站表没有的站 / 缺 station_id 列直接报错。"""
    import torch
    cfg, fb = _fb(small_cfg)
    bd = batch["batch_dict"]
    ref = fb.build(bd, training=False)
    bd2 = {**bd, "cap_power_on": bd["cap_power_on"] * 0.1}
    out = fb.build(bd2, training=False)
    assert np.array_equal(ref["x"], out["x"]) and np.array_equal(ref["static"], out["static"])
    assert set(np.unique(out["ctx"].cap)) <= {500.0, 300.0}
    bd3 = {**bd, "station_id": torch.full_like(bd["station_id"], 4242)}
    with pytest.raises(ValueError, match="station_meta"):
        fb.build(bd3, training=False)
    with pytest.raises(ValueError, match="station_id"):
        fb.build({k: v for k, v in bd.items() if k != "station_id"}, training=False)


def test_inference_is_deterministic_no_augmentation(small_cfg, batch):
    cfg, fb = _fb(small_cfg)
    a = fb.build(batch["batch_dict"], training=False, rng=np.random.default_rng(1))
    b = fb.build(batch["batch_dict"], training=False, rng=np.random.default_rng(2))
    assert np.array_equal(a["x"], b["x"]) and (a["amp"] == 1).all()


def test_training_augmentations(small_cfg, batch):
    cfg, fb = _fb(small_cfg, train={"cpm": {"enabled": True, "p_row": 1.0, "min_frac": 0.2, "max_frac": 0.2, "max_runs": 2},
                                    "cov_dropout": 1.0, "amp_mod": {"p": 1.0, "lo": 0.5, "hi": 0.5}})
    y_future = batch["labels"]["observe_power_future"].numpy()
    inp = fb.build(batch["batch_dict"], training=True, y_future=y_future, rng=np.random.default_rng(0))
    i = fb.idx
    masked = inp["x"][:, :672, i["y_mask"]]
    n_tok_masked = masked.reshape(masked.shape[0], 42, 16).all(-1).sum(1)
    assert (n_tok_masked == round(0.2 * 42)).all(), n_tok_masked          # 每行恰好掩 20% 的历史 token，整 token 掩
    assert (inp["x"][:, :672, i["y"]][masked == 1] == 0).all()
    assert (inp["x"][:, :, i["cov_missing"]] == 1).all()
    for c in fb.nwp_channels:
        assert (inp["x"][:, :, c] == 0).all()
    assert (inp["amp"] == 0.5).all()
    # 标签按 amp/scale 归一；掩码处的历史真值仍在 target 里
    cap = inp["ctx"].cap
    np.testing.assert_allclose(inp["target"][:, 672:], y_future[:, :192] * 0.5 / cap[:, None], rtol=1e-5)
    assert np.isfinite(inp["target"][:, :672][masked == 1]).all()


def test_recent_peak_scale(small_cfg, batch):
    cfg, fb = _fb(small_cfg, features={"target_scale": "recent_peak"})
    inp = fb.build(batch["batch_dict"], training=False)
    assert (inp["scale"] <= inp["ctx"].cap * 1.0001).all() and (inp["scale"] >= 0.3 * inp["ctx"].cap).all()
    assert inp["x"][:, :672, fb.idx["y"]].max() <= 1.0 + 1e-5


def test_nan_history_is_masked(small_cfg, batch):
    cfg, fb = _fb(small_cfg)
    bd = dict(batch["batch_dict"])
    y = bd["observe_power"].clone(); y[:, 100:116] = float("nan"); bd["observe_power"] = y
    inp = fb.build(bd, training=False)
    assert (inp["x"][:, 100:116, fb.idx["y_mask"]] == 1).all() and (inp["x"][:, 100:116, fb.idx["y"]] == 0).all()
    assert np.isnan(inp["target"][:, 100:116]).all()         # 缺测没有真值 → 不计损


def test_cpm_mask_runs():
    rng = np.random.default_rng(0)
    m = cpm_token_mask(rng, 200, 42, p_row=1.0, min_frac=0.1, max_frac=0.3, max_runs=3)
    frac = m.mean(1)
    assert frac.min() >= 0.05 and frac.max() <= 0.35
    assert cpm_token_mask(rng, 5, 42, p_row=0.0, min_frac=0.1, max_frac=0.3, max_runs=3).sum() == 0


def test_solar_cache_matches_direct(small_cfg, batch):
    from forecast_backbones.m6.features import solar_geometry, window_timestamps
    cfg, fb = _fb(small_cfg)
    ts = window_timestamps(batch["batch_dict"]["timestamp_win"].numpy(), 672, 192, 15)
    e_c, g_c, deg = fb.solar.geometry(1001, ts[:2])
    m = fb.meta.get(1001)
    e_d, g_d = solar_geometry(ts[:2], m["lat"], m["lon"], m["alt_m"], cfg["dataset"]["tz"])
    assert not deg
    np.testing.assert_allclose(e_c, e_d, atol=1e-3); np.testing.assert_allclose(g_c, g_d, rtol=1e-4, atol=1e-2)


FO = {"future_only": [{"col": "timesfm_predict", "scale": "target"}]}


def test_future_only_channel_inference(small_cfg, batch):
    cfg, fb = _fb(small_cfg, features=FO)
    j = fb.idx["fo:timesfm_predict"]
    assert "hint_missing" in fb.idx and j not in fb.nwp_channels          # 不随 NWP dropout 一起清零
    inp = fb.build(batch["batch_dict"], training=False)
    assert (inp["x"][:, :672, j] == 0).all()                               # 历史段恒 0
    tf = batch["batch_dict"]["timesfm_predict"].numpy()[:, :192]
    np.testing.assert_allclose(inp["x"][:, 672:, j], tf / inp["ctx"].cap[:, None], rtol=1e-5)
    assert (inp["x"][:, :, fb.idx["hint_missing"]] == 0).all()
    assert (inp["base"] == 0).all()                                        # 非残差模式 base 恒 0
    assert inp["x"].shape[2] == len(fb.channel_names)


def test_future_only_amp_and_residual_base(small_cfg, batch):
    tr = {"hint_dropout": 0.0, "cov_dropout": 0.0, "amp_mod": {"p": 1.0, "lo": 0.5, "hi": 0.5}, "cpm": {"enabled": False}}
    cfg, fb = _fb(small_cfg, features={**FO, "residual_base_col": "timesfm_predict"}, train=tr)
    inp = fb.build(batch["batch_dict"], training=True, y_future=batch["labels"]["observe_power_future"].numpy(), rng=np.random.default_rng(0))
    j = fb.idx["fo:timesfm_predict"]
    tf = batch["batch_dict"]["timesfm_predict"].numpy()[:, :192]
    np.testing.assert_allclose(inp["x"][:, 672:, j], tf * 0.5 / inp["ctx"].cap[:, None], rtol=1e-5)   # 与幅度调制后的目标同尺度
    np.testing.assert_allclose(inp["base"][:, 672:], inp["x"][:, 672:, j])
    assert (inp["base"][:, :672] == 0).all()


def test_hint_dropout_zeroes_group_and_base(small_cfg, batch):
    tr = {"hint_dropout": 1.0, "cov_dropout": 0.0, "amp_mod": {"p": 0.0}, "cpm": {"enabled": False}}
    cfg, fb = _fb(small_cfg, features={**FO, "residual_base_col": "timesfm_predict"}, train=tr)
    inp = fb.build(batch["batch_dict"], training=True, rng=np.random.default_rng(0))
    assert (inp["x"][:, :, fb.idx["fo:timesfm_predict"]] == 0).all()
    assert (inp["x"][:, :, fb.idx["hint_missing"]] == 1).all()
    assert (inp["base"] == 0).all()                                        # 提示被丢弃 → 残差基准同步为 0
    # NWP 通道不受 hint_dropout 影响
    assert np.abs(inp["x"][:, :, fb.idx["pf:general_ghi"]]).sum() > 0


def test_future_only_missing_column(small_cfg, batch):
    cfg, fb = _fb(small_cfg, features={"future_only": ["nope_predict"]})
    inp = fb.build(batch["batch_dict"])
    assert (inp["x"][:, :, fb.idx["fo:nope_predict"]] == 0).all() and (inp["x"][:, :, fb.idx["hint_missing"]] == 1).all()


def test_residual_requires_target_scaled_hint(small_cfg):
    with pytest.raises(ValueError):
        load_config({**small_cfg, "features": {"future_only": [{"col": "timesfm_predict", "scale": "fixed"}], "residual_base_col": "timesfm_predict"}})
    with pytest.raises(ValueError):
        load_config({**small_cfg, "features": {"residual_base_col": "timesfm_predict"}})


def test_unknown_station_goes_to_slot_zero_and_slot_dropout(small_cfg, batch):
    cfg, fb = _fb(small_cfg)
    inp = fb.build(batch["batch_dict"], training=False)
    assert (inp["slot"] > 0).all()                                          # 没有名单：全部按站号取槽
    cfg2 = load_config({**small_cfg, "dataset": {**small_cfg["dataset"], "known_stations": [1002]}})
    fb2 = FeatureBuilder(cfg2, StationMeta.from_csv(cfg2["dataset"]["station_meta"]))
    inp2 = fb2.build(batch["batch_dict"], training=False)
    sid = batch["batch_dict"]["station_id"].numpy().astype(int)
    assert (inp2["slot"][sid == 1001] == 0).all() and (inp2["slot"][sid == 1002] > 0).all()
    cfg3, fb3 = _fb(small_cfg, train={"slot_dropout": 1.0})
    assert (fb3.build(batch["batch_dict"], training=True, rng=np.random.default_rng(0))["slot"] == 0).all()
    assert (fb3.build(batch["batch_dict"], training=False)["slot"] > 0).all()   # 推理不丢


def test_static_identity_switches(small_cfg, batch):
    """coords / log_cap / station_embedding 关掉后：静态向量前四维全 0、站槽全 0，其余维不变。"""
    cfg, fb = _fb(small_cfg)
    inp = fb.build(batch["batch_dict"], training=False)
    assert np.abs(inp["static"][:, :4]).sum() > 0
    cfg2, fb2 = _fb(small_cfg, features={"static": {"coords": False, "log_cap": False, "station_embedding": False}})
    inp2 = fb2.build(batch["batch_dict"], training=False)
    assert np.abs(inp2["static"][:, :4]).sum() == 0 and (inp2["slot"] == 0).all() and fb2.n_slots == 0
    assert np.allclose(inp2["static"][:, 4:], inp["static"][:, 4:])
    assert np.allclose(inp2["x"], inp["x"])                                      # 逐步通道（含太阳几何）不受影响


def _all_rows_batch(dataset, collate):
    import torch
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), collate_fn=collate, shuffle=False)
    return next(iter(loader))


def _tod_hours(fb, bd):
    """[B, T] 每步的本地小时数（含小数），与 FeatureBuilder 里的 window_timestamps 同口径。"""
    import pandas as pd
    from forecast_backbones.m6.features import window_timestamps
    ts = window_timestamps(np.asarray(bd["timestamp_win"]).astype(np.int64).reshape(-1), fb.L, fb.H, fb.step_min)
    t = pd.DatetimeIndex(ts.reshape(-1))
    return (t.hour + t.minute / 60.0).to_numpy().reshape(ts.shape)


def test_history_envelope_replaces_clock_geometry_for_missing_coords(tmp_path, dataset, collate):
    """features.derived.history_envelope（默认关）：缺坐标站的 p_clear / elev / 夜间掩码改由窗内 7 天历史按时刻槽的高分位包络推出，
    不再是 06:15–20:00 平顶；有坐标的站逐位不变；决策层的 p_clear 仍是"白天 = 容量"（包络只做软先验，不做硬上限）。"""
    from .conftest import write_station_meta
    meta = write_station_meta(tmp_path, drop_coords_for=(1002,))
    base_cfg = {"dataset": {"station_meta": str(meta)},
                "model": {"d_model": 32, "d_ff": 64, "head_hidden": 32, "n_heads": 2, "blocks": ["conv"], "pos_in_patch_dim": 8}}
    bd = _all_rows_batch(dataset, collate)["batch_dict"]
    cfg0 = load_config(base_cfg)
    fb0 = FeatureBuilder(cfg0, StationMeta.from_csv(cfg0["dataset"]["station_meta"]))
    off = fb0.build(bd, training=False)
    cfg1 = load_config({**base_cfg, "features": {"derived": {"history_envelope": {"enabled": True}}}})
    assert cfg1["features"]["derived"]["history_envelope"]["quantile"] == 0.9            # 其余键取默认
    fb1 = FeatureBuilder(cfg1, StationMeta.from_csv(cfg1["dataset"]["station_meta"]))
    on = fb1.build(bd, training=False)
    i = fb1.idx
    sid = np.asarray(bd["station_id"]).reshape(-1)
    deg, ok = sid == 1002, sid == 1001
    assert deg.any() and ok.any()
    assert off["ctx"].degraded[deg].all() and not off["ctx"].envelope_used.any()
    assert on["ctx"].envelope_used[deg].all() and not on["ctx"].envelope_used[ok].any()
    # 有坐标的站逐位不变
    assert np.array_equal(on["x"][ok], off["x"][ok]) and np.array_equal(on["static"][ok], off["static"][ok])
    assert np.array_equal(on["ctx"].p_clear[ok], off["ctx"].p_clear[ok]) and np.array_equal(on["ctx"].night[ok], off["ctx"].night[ok])
    # 关着：缺坐标站 = 钟表几何（06:15–20:00 内 p_clear 通道恒 1）
    hrs = _tod_hours(fb0, bd)
    clock_day = (hrs >= 6.25) & (hrs <= 20.0)
    assert np.allclose(off["x"][deg][:, :, i["p_clear"]][clock_day[deg]], 1.0)
    # 开着：日形包络。合成数据白天 7–19 点、峰值 ≈ 0.85·k_day·cap；包络在 [0.3, 1]，正午 > 0，凌晨 2 点 = 0，06:30 已是夜间（钟表几何却算白天）
    pc = on["x"][deg][:, :, i["p_clear"]]
    assert pc.max() <= 1.0 + 1e-6 and (pc.max(axis=1) > 0.3).all()
    noon = np.isclose(hrs[deg], 12.0); dawn = np.isclose(hrs[deg], 6.5); deep = np.isclose(hrs[deg], 2.0)
    assert (pc[noon] > 0.3).all() and np.allclose(pc[deep], 0.0) and np.allclose(pc[dawn], 0.0)
    nt = on["ctx"].night[deg]
    assert nt[dawn].all() and nt[deep].all() and not nt[noon].any()
    assert np.array_equal(nt, on["x"][deg][:, :, i["is_night"]].astype(bool))
    el = on["x"][deg][:, :, i["elev"]]
    assert (el[nt] < 0).all() and (el[~nt] >= 0).all() and el.max() <= 1.0 + 1e-6
    # 决策层的 p_clear：白天 = 容量，夜间 = 0（包络不进硬上限）
    pcd = on["ctx"].p_clear[deg]
    assert np.allclose(pcd[~nt], 300.0) and np.allclose(pcd[nt], 0.0)
    assert on["static"][deg][:, 4].all()                                             # 仍标记为退化站
    assert np.isfinite(on["x"]).all() and np.isfinite(on["static"]).all()


def test_history_envelope_falls_back_when_history_is_dark(tmp_path, dataset, collate):
    """整周历史无出力（全 0）或全缺测：包络峰值低于 peak_frac·cap → 该行回退到钟表几何，与关着时逐位相同。"""
    from .conftest import write_station_meta
    meta = write_station_meta(tmp_path, drop_coords_for=(1002,))
    base_cfg = {"dataset": {"station_meta": str(meta)},
                "model": {"d_model": 32, "d_ff": 64, "head_hidden": 32, "n_heads": 2, "blocks": ["conv"], "pos_in_patch_dim": 8}}
    bd = dict(_all_rows_batch(dataset, collate)["batch_dict"])
    sid = np.asarray(bd["station_id"]).reshape(-1)
    deg = sid == 1002
    import torch
    obs = bd["observe_power"].clone()
    half = np.flatnonzero(deg)[: deg.sum() // 2]
    obs[half] = 0.0                                             # 前一半退化行：整周 0
    rest = np.flatnonzero(deg)[deg.sum() // 2:]
    obs[rest] = float("nan")                                    # 后一半：整周缺测
    bd["observe_power"] = obs
    cfg0 = load_config(base_cfg)
    off = FeatureBuilder(cfg0, StationMeta.from_csv(cfg0["dataset"]["station_meta"])).build(bd, training=False)
    cfg1 = load_config({**base_cfg, "features": {"derived": {"history_envelope": {"enabled": True}}}})
    on = FeatureBuilder(cfg1, StationMeta.from_csv(cfg1["dataset"]["station_meta"])).build(bd, training=False)
    assert not on["ctx"].envelope_used[deg].any()
    assert np.array_equal(on["x"], off["x"]) and np.array_equal(on["ctx"].p_clear, off["ctx"].p_clear)


def test_history_envelope_uses_only_visible_history_and_is_finite_under_cpm(tmp_path, dataset, collate, batch):
    """训练态：CPM 掩掉的历史 token 不参与包络（否则被掩段的幅度会从 p_clear 通道漏回来）；输出全部有限。"""
    from .conftest import write_station_meta
    meta = write_station_meta(tmp_path, drop_coords_for=(1002,))
    base_cfg = {"dataset": {"station_meta": str(meta)},
                "model": {"d_model": 32, "d_ff": 64, "head_hidden": 32, "n_heads": 2, "blocks": ["conv"], "pos_in_patch_dim": 8},
                "features": {"derived": {"history_envelope": {"enabled": True}}},
                "train": {"cpm": {"enabled": True, "p_row": 1.0, "min_frac": 0.5, "max_frac": 0.5, "max_runs": 1}, "amp_mod": {"p": 0.0}}}
    cfg = load_config(base_cfg)
    fb = FeatureBuilder(cfg, StationMeta.from_csv(cfg["dataset"]["station_meta"]))
    full = _all_rows_batch(dataset, collate)
    bd, yf = full["batch_dict"], full["labels"]["observe_power_future"]
    tr = fb.build(bd, training=True, y_future=yf, rng=np.random.default_rng(0))
    assert np.isfinite(tr["x"]).all() and np.isfinite(tr["target"][:, :672][~np.isnan(tr["target"][:, :672])]).all()
    # 把可见段的功率 ×3、被掩段不动：包络只随可见段变（被掩段改了不影响）
    i = fb.idx
    sid = np.asarray(bd["station_id"]).reshape(-1); deg = sid == 1002
    rng_a, rng_b = np.random.default_rng(7), np.random.default_rng(7)
    a = fb.build(bd, training=True, y_future=yf, rng=rng_a)
    masked = a["x"][:, :672, i["y_mask"]].astype(bool)
    import torch
    obs = bd["observe_power"].clone()
    obs[torch.as_tensor(masked)] = obs[torch.as_tensor(masked)] * 10.0 + 1000.0      # 只改被掩段
    b = fb.build({**bd, "observe_power": obs}, training=True, y_future=yf, rng=rng_b)
    assert a["ctx"].envelope_used[deg].all() and b["ctx"].envelope_used[deg].all()
    assert np.array_equal(a["x"][deg][:, :, i["p_clear"]], b["x"][deg][:, :, i["p_clear"]])
    # 改可见段（×0.5）包络必须跟着变
    obs2 = bd["observe_power"].clone(); obs2[~torch.as_tensor(masked)] *= 0.5
    c = fb.build({**bd, "observe_power": obs2}, training=True, y_future=yf, rng=np.random.default_rng(7))
    assert not np.array_equal(a["x"][deg][:, :, i["p_clear"]], c["x"][deg][:, :, i["p_clear"]])
