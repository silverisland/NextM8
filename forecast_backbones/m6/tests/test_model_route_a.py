import numpy as np
import pandas as pd
import torch
from forecast_backbones.m6.config import load_config
from forecast_backbones.m6.transformer_models import TransformersAdaptedM6Model
from forecast_backbones.m6.decision import MODEL_LEVELS
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
    m = _model(tmp_path); bd, _ = _batch(df, [0])
    bd["observe_power_future"] = np.full((1, 192), 1e9, np.float32)      # 若被读，logits 必爆
    out = m(bd)
    assert out.logits.max() < 1e4
