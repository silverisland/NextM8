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
