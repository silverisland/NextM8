"""M5 框架契约：registry 自动发现（第一个 nn.Module 子类、Model(model_config)）+ evaluate_model 结果表。
不依赖 timesfm3：eager_init=False 后把 _backbone 换成替身。"""
import functools
import types

import numpy as np
import pandas as pd
import pytest

from pipelines.transformer_dataset import UnifiedDataset
from pipelines.model_registry import build_evaluator, import_model_class
from forecast_backbones.m5.transformer_models import TransformersAdaptedM5Model, TransformersAdaptedTimesFM3Model
from forecast_backbones.m5.transformer_trainer import evaluate_model, QUANTILE_COLS

L, H = 672, 192


class _FakeBackbone:
    """predict_batch 替身：forecast = 0.5 × context 末 H 段；quantiles = forecast × [0.6..1.4]。"""
    def predict_batch(self, inputs, horizon, **kw):
        outs = []
        for x in inputs:                                            # x: (1, L)
            fc = 0.5 * x[0, -horizon:]
            q = fc[:, None] * np.linspace(0.6, 1.4, 9)[None, :]
            outs.append(types.SimpleNamespace(forecast=fc[None, :], quantiles=q[None, :, :]))
        return outs


def _frame(n=6, stations=(1001, 1002)):
    rows = []
    for s in stations:
        for i in range(n):
            base = np.linspace(0, 100, L, dtype=np.float32) + s % 7
            rows.append({"timestamp_win": pd.Timestamp("2025-06-08") + pd.Timedelta(hours=i), "station_id": s,
                         "observe_power": base, "observe_power_future": np.full(H, 42.0, np.float32),
                         "general_ghi": base, "general_ghi_predict": np.full(H, 1.0, np.float32),
                         "general_temp": base, "general_temp_predict": np.full(H, 2.0, np.float32)})
    return pd.DataFrame(rows)


def _model():
    m = TransformersAdaptedM5Model({"timesfm3": {"eager_init": False, "past_future_cols": [["general_ghi", "general_ghi_predict"]]}})
    m._backbone = _FakeBackbone(); m._backbone_device = "cpu"
    return m


def test_registry_discovers_m5_model_and_evaluator():
    assert import_model_class("m5") is TransformersAdaptedM5Model
    assert TransformersAdaptedTimesFM3Model is TransformersAdaptedM5Model          # 旧名仍可用
    assert build_evaluator("m5") is evaluate_model


def test_evaluate_model_schema_and_values():
    df = _frame()
    ds = UnifiedDataset(df, timestamp_cols=["timestamp_win"], entity_cols=[], label_cols=["observe_power_future"],
                        for_hf_trainer=True)
    collate = functools.partial(UnifiedDataset.collate_fn, for_hf_trainer=True)
    res = evaluate_model(_model(), ds, collate, {"eval": {"batch_size": 5}})
    assert len(res) == len(df)
    assert {"timestamp_win", "station_id", "observe_power_predict", "observe_power_future", *QUANTILE_COLS} <= set(res.columns)
    assert set(res["station_id"]) == {1001, 1002}
    r = res.iloc[0]
    assert r["observe_power_predict"].shape == (H,) and r["observe_power_future"].shape == (H,)
    assert np.allclose(r["observe_power_future"], 42.0)                              # 真值只进结果表，不进模型
    ctx = df.iloc[0]["observe_power"]
    assert np.allclose(r["observe_power_predict"], 0.5 * ctx[-H:], atol=1e-5)
    assert np.allclose(r["q_0.5"], r["observe_power_predict"], atol=1e-5)
    assert np.all(r["q_0.1"] <= r["q_0.9"])
    assert res["timestamp_win"].iloc[1] - res["timestamp_win"].iloc[0] == pd.Timedelta(hours=1)


def test_evaluate_model_without_labels_and_without_station():
    df = _frame(n=2).drop(columns=["station_id"])
    ds = UnifiedDataset(df, timestamp_cols=["timestamp_win"], entity_cols=[], label_cols=["observe_power_future"],
                        for_hf_trainer=False)                                      # 非 HF 形态：batch 里没有 labels
    collate = functools.partial(UnifiedDataset.collate_fn, for_hf_trainer=False)
    res = evaluate_model(_model(), ds, collate, {})
    assert "station_id" not in res.columns and len(res) == 4                           # 2 站 × 2 行
    assert np.isnan(res.iloc[0]["observe_power_future"]).all()
