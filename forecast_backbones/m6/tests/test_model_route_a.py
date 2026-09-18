import numpy as np
import pandas as pd
import pytest
import torch
from forecast_backbones.m6.config import load_config
from forecast_backbones.m6.transformer_models import TransformersAdaptedM6Model
from forecast_backbones.m6.decision import MODEL_LEVELS, CAL_LEVELS, p_star, p_star_l1
from forecast_backbones.m6.calibration import Calibrator, CalibrationConfig, quantile_cols
from .conftest import FakeEvaluator, make_synthetic_frame, write_station_meta

def _model(tmp_path, **over):
    cfg = {"timesfm3": {"eager_init": False}, "dataset": {"station_meta": str(write_station_meta(tmp_path))},
           "calibration": {"enabled": False}}
    for k, v in over.items():
        cfg.setdefault(k, {}).update(v)
    m = TransformersAdaptedM6Model(cfg)
    m._backbone = FakeEvaluator(); m._backbone_device = "cpu"
    return m

def _batch(df, rows):
    sub = df.iloc[rows]
    bd = {c: np.stack(sub[c].to_numpy()) for c in df.columns if c not in ("timestamp_win", "station_id", "cap_power_on", "observe_power_future")}
    bd["timestamp_win"] = (sub["timestamp_win"].astype("int64") // 10**9).to_numpy()
    bd["station_id"] = sub["station_id"].to_numpy(); bd["cap_power_on"] = sub["cap_power_on"].to_numpy()
    labels = {"observe_power_future": np.stack(sub["observe_power_future"].to_numpy())}
    return bd, labels

def test_first_local_module_is_m6_model():
    import inspect, forecast_backbones.m6.transformer_models as mod, torch.nn as nn
    first = next(o for _, o in mod.__dict__.items() if inspect.isclass(o) and o.__module__ == mod.__name__ and issubclass(o, nn.Module))
    assert first is TransformersAdaptedM6Model

def test_forward_contract_without_calibration(tmp_path):
    df = make_synthetic_frame(n_days=1, issue_every_min=360)
    m = _model(tmp_path)
    bd, labels = _batch(df, [0, 1, 2])
    out = m(bd)
    assert out.logits.shape == (3, 192) and torch.equal(out.logits, out.predictions)
    assert out.aux["quantiles"].shape == (3, 192, 9) and out.aux["quantiles_cal"].shape == (3, 192, 9)
    assert list(out.aux["levels_cal"]) == list(MODEL_LEVELS)
    night = out.aux["night"]
    assert (out.logits.numpy()[night] == 0).all()                       # 夜间归零
    day = ~night
    med = out.aux["median"].numpy()
    assert (out.logits.numpy()[day] <= med[day] + 1e-6).all()           # p* / p*_L1 不高于中位数
    assert out.loss is None

def test_forward_loss_when_labels_given(tmp_path):
    df = make_synthetic_frame(n_days=1, issue_every_min=360)
    m = _model(tmp_path); bd, labels = _batch(df, [0, 1])
    out = m(bd, labels=labels)
    assert out.loss is not None and float(out.loss) > 0
    assert float(m.compute_loss(out, labels)) == float(out.loss)

def test_by_step_routing_and_median_mode(tmp_path):
    df = make_synthetic_frame(n_days=1, issue_every_min=360)
    m = _model(tmp_path, decision={"point": {"default": "median", "by_step": []}})
    bd, _ = _batch(df, [0]); out = m(bd)
    day = ~out.aux["night"]
    assert np.allclose(out.logits.numpy()[day], out.aux["median"].numpy()[day])

def test_degraded_station_still_runs(tmp_path):
    p = write_station_meta(tmp_path, drop_coords_for=(1002,))
    cfg = {"timesfm3": {"eager_init": False}, "dataset": {"station_meta": str(p)}, "calibration": {"enabled": False}}
    m = TransformersAdaptedM6Model(cfg); m._backbone = FakeEvaluator(); m._backbone_device = "cpu"
    df = make_synthetic_frame(n_days=1, issue_every_min=360, stations=(1002,))
    bd, _ = _batch(df, [0]); out = m(bd)
    assert out.aux["degraded"].tolist() == [True] and np.isfinite(out.logits.numpy()).all()

def test_does_not_read_future_truth(tmp_path):
    df = make_synthetic_frame(n_days=1, issue_every_min=360)
    m = _model(tmp_path); bd, labels = _batch(df, [0])
    out_clean = m(bd)
    bd_poison = dict(bd)
    bd_poison["observe_power_future"] = np.full((1, 192), 1e9, np.float32)   # 若被读，logits 必爆
    fake = m._backbone
    fake.calls = []                                                          # 只看投毒这次调用
    out = m(bd_poison)
    assert out.logits.max() < 1e4
    assert np.array_equal(out.logits.numpy(), out_clean.logits.numpy())      # 未读到 → 与未投毒时完全一致
    for call in fake.calls:                                                  # 直接溯源底座收到的每个数组
        for arr in call:
            assert not np.any(np.asarray(arr) == 1e9), "泄漏：未来真值出现在了喂给底座的数组里"

def test_by_step_routing_uses_configured_methods(tmp_path):
    df = make_synthetic_frame(n_days=1, issue_every_min=360)
    m = _model(tmp_path)                                     # 默认 decision.point.by_step（不覆盖）
    bd, _ = _batch(df, [0, 1, 2]); out = m(bd)
    q_cal = out.aux["quantiles_cal"].numpy()
    levels_cal = out.aux["levels_cal"]
    cap = out.aux["cap"]
    exp_l1 = p_star_l1(q_cal, levels_cal, cap[:, None])
    exp_star = p_star(q_cal, levels_cal, cap[:, None])
    logits = out.logits.numpy()
    assert np.allclose(logits[:, :16], exp_l1[:, :16], atol=1e-4)            # [0,16) 走 p_star_l1
    assert np.allclose(logits[:, 55:151], exp_star[:, 55:151], atol=1e-4)    # [55,151) 走 p_star
    assert not np.allclose(exp_l1[:, :16], exp_star[:, :16], atol=1e-4)      # 非空验证：两法在该段确实不同

def test_loss_mask_ignores_physics_night_zero_knob(tmp_path, monkeypatch):
    import forecast_backbones.m6.transformer_models as tm
    recorded_night = []
    orig_clip = tm.physics_clip
    def spy_clip(q, night, *a, **kw):
        recorded_night.append(np.array(night, copy=True))
        return orig_clip(q, night, *a, **kw)
    monkeypatch.setattr(tm, "physics_clip", spy_clip)

    df = make_synthetic_frame(n_days=1, issue_every_min=360)
    bd, labels = _batch(df, [0, 1])
    m_true = _model(tmp_path, decision={"physics": {"night_zero": True}})
    m_false = _model(tmp_path, decision={"physics": {"night_zero": False}})
    out_true = m_true(bd, labels=labels)
    out_false = m_false(bd, labels=labels)

    # physics_clip 确实按 knob 收到了不同的 night 掩码（knob 真起作用，不是死配置）
    assert recorded_night[0].any() and not recorded_night[1].any()
    assert not np.array_equal(out_true.aux["night"], out_false.aux["night"])
    # 但损失掩码恒为真实夜间，与 knob 无关
    assert out_true.aux["night_loss"].any()                                   # 场景非空才有意义
    assert np.array_equal(out_true.aux["night_loss"], out_false.aux["night_loss"])
    assert np.array_equal(out_true.aux["night_loss"], out_true.aux["night"])  # knob=True 时二者恰好相等（对照）
    assert float(out_true.loss) == pytest.approx(float(out_false.loss))
    assert float(m_true.compute_loss(out_true, labels)) == pytest.approx(float(out_true.loss))
    assert float(m_false.compute_loss(out_false, labels)) == pytest.approx(float(out_false.loss))

def _tiny_calibration_table(n=300, seed=0):
    """calibration.fit() 期望的长表列：station_id/elev_deg/var_bin/lead/y/p_clear + q_<model_level>。"""
    rng = np.random.default_rng(seed)
    q50 = rng.uniform(30, 90, n)
    y = np.clip(q50 * (1 + rng.normal(0.0, 0.2, n)), 0, None)
    q = q50[:, None] * np.linspace(0.6, 1.4, 9)[None, :]
    df = pd.DataFrame({"station_id": 1001, "lead": rng.integers(0, 16, n), "elev_deg": 60.0,
                       "var_bin": 1, "y": y, "p_clear": 100.0})
    for c, j in zip(quantile_cols(MODEL_LEVELS), range(9)):
        df[c] = q[:, j]
    return df

def test_calibration_enabled_missing_table_warns_and_falls_back(tmp_path):
    df = make_synthetic_frame(n_days=1, issue_every_min=360)
    bd, _ = _batch(df, [0, 1])
    sm = str(write_station_meta(tmp_path))
    missing = tmp_path / "no_such_calibration.parquet"
    with pytest.warns(UserWarning, match="calibration.enabled"):
        m_on = TransformersAdaptedM6Model({"timesfm3": {"eager_init": False}, "dataset": {"station_meta": sm},
                                           "calibration": {"enabled": True, "table_path": str(missing)}})
    m_on._backbone = FakeEvaluator(); m_on._backbone_device = "cpu"
    assert m_on.calibrator is None
    m_off = TransformersAdaptedM6Model({"timesfm3": {"eager_init": False}, "dataset": {"station_meta": sm},
                                        "calibration": {"enabled": False}})
    m_off._backbone = FakeEvaluator(); m_off._backbone_device = "cpu"
    out_on = m_on(bd); out_off = m_off(bd)
    assert torch.equal(out_on.aux["quantiles_cal"], out_off.aux["quantiles_cal"])

def test_calibration_enabled_cfg_mismatch_raises_from_init(tmp_path):
    fit_cfg = CalibrationConfig(levels=CAL_LEVELS, model_levels=MODEL_LEVELS, normalization="clear_sky",
                                elev_edges=(20.0, 45.0), lead_edges=(4, 8, 16, 55, 103),
                                use_variability=True, n_min_factor=10.0)
    cal = Calibrator(fit_cfg).fit(_tiny_calibration_table())
    p = tmp_path / "cal.parquet"; cal.save(p)
    bad_levels = [0.55 if l == 0.5 else l for l in CAL_LEVELS]               # 与拟合期 levels 不一致
    cfg = {"timesfm3": {"eager_init": False}, "dataset": {"station_meta": str(write_station_meta(tmp_path))},
           "calibration": {"enabled": True, "table_path": str(p), "levels": bad_levels}}
    with pytest.raises(ValueError, match="levels"):
        TransformersAdaptedM6Model(cfg)
