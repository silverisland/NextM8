"""M7 框架契约：registry 自动发现（第一个 nn.Module 子类、Model(model_config)）+ evaluate_model 结果表。
默认不依赖 timestral：eager_init=False 后用 _set_backbone 挂替身；装了 timestral 时再用随机小模型跑一遍真实前向。"""
import functools
import types

import numpy as np
import pandas as pd
import pytest
import torch

from pipelines.transformer_dataset import UnifiedDataset
from pipelines.model_registry import build_evaluator, import_model_class
from forecast_backbones.m7.transformer_models import TransformersAdaptedM7Model, TransformersAdaptedTimestralModel
from forecast_backbones.m7.transformer_trainer import evaluate_model, QUANTILE_COLS

L, H = 672, 192
LEVELS21 = (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
            0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)


class _FakeForecaster:
    """TimestralForecaster 替身：中位数 = 0.5 × context 末 H 段；第 k 档分位 = 中位数 × (0.5 + level)，
    因此 q_0.5 == 中位数；输出形状与真模型一致（quantiles [b,1,H,21]）。记录每次调用的输入形状。"""
    quantile_levels = LEVELS21
    supports_known_covariates = True

    def __init__(self):
        self.calls = []

    def forecast(self, context, prediction_length, past_covariates=None, known_covariates=None,
                 symmetric_averaging=True, clip_nonnegative=True):
        self.calls.append({"context": np.shape(context),
                           "known": None if known_covariates is None else np.asarray(known_covariates).copy(),
                           "past": None if past_covariates is None else np.shape(past_covariates),
                           "sym": symmetric_averaging, "clip": clip_nonnegative})
        x = torch.as_tensor(context)
        med = 0.5 * x[..., -prediction_length:]                                       # [b,1,H]
        scale = torch.tensor(LEVELS21).view(1, 1, 1, -1) + 0.5
        return types.SimpleNamespace(quantiles=med.unsqueeze(-1) * scale, quantile_levels=LEVELS21)


def _frame(n=6, stations=(1001, 1002)):
    rows = []
    for s in stations:
        for i in range(n):
            base = np.linspace(0, 100, L, dtype=np.float32) + s % 7
            rows.append({"timestamp_win": pd.Timestamp("2025-06-08") + pd.Timedelta(hours=i), "station_id": s,
                         "observe_power": base, "observe_power_future": np.full(H, 42.0, np.float32),
                         "general_ghi": np.full(L, 3.0, np.float32), "general_ghi_predict": np.full(H, 1.0, np.float32),
                         "general_temp": np.full(L, 20.0, np.float32), "general_temp_predict": np.full(H, 25.0, np.float32)})
    return pd.DataFrame(rows)


def _model(**cfg):
    m = TransformersAdaptedM7Model({"timestral": {"eager_init": False, "per_core_batch_size": 4, **cfg}})
    m._set_backbone(_FakeForecaster(), "cpu")
    return m


def test_registry_discovers_m7_model_and_evaluator():
    assert import_model_class("m7") is TransformersAdaptedM7Model
    assert TransformersAdaptedTimestralModel is TransformersAdaptedM7Model
    assert build_evaluator("m7") is evaluate_model


def test_backbone_not_registered_as_submodule_and_rolling_attrs():
    m = _model()
    assert len(m.state_dict()) == 0                          # 同 M5：外部 .to()/.train() 不碰 backbone
    assert (m.context_length, m.prediction_length) == (L, H)  # pipelines/rolling_wrapper_inference 读这两个名字


def test_evaluate_model_schema_and_values():
    df = _frame()
    ds = UnifiedDataset(df, timestamp_cols=["timestamp_win"], entity_cols=[], label_cols=["observe_power_future"],
                        for_hf_trainer=True)
    collate = functools.partial(UnifiedDataset.collate_fn, for_hf_trainer=True)
    m = _model()
    res = evaluate_model(m, ds, collate, {"eval": {"batch_size": 5}})
    assert len(res) == len(df)
    assert {"timestamp_win", "station_id", "observe_power_predict", "observe_power_future", *QUANTILE_COLS} <= set(res.columns)
    r = res.iloc[0]
    assert r["observe_power_predict"].shape == (H,) and r["observe_power_future"].shape == (H,)
    assert np.allclose(r["observe_power_future"], 42.0)
    ctx = df.iloc[0]["observe_power"]
    assert np.allclose(r["observe_power_predict"], 0.5 * ctx[-H:], atol=1e-4)
    assert np.allclose(r["q_0.5"], r["observe_power_predict"], atol=1e-4)
    assert np.all(r["q_0.1"] <= r["q_0.9"])
    calls = m._backbone.calls
    assert all(c["context"][0] <= 4 for c in calls)                              # per_core_batch_size 切块生效
    assert all(c["context"][1:] == (1, L) for c in calls)                        # 单目标 [b,1,L]
    assert all(c["sym"] and c["clip"] and c["past"] is None for c in calls)


def test_known_covariates_layout():
    """默认两对协变量：历史 L + 未来 H 拼接 → known [b, 2, L+H]；绝不读 observe_power_future。"""
    m = _model()
    b = 3
    bd = {"observe_power": np.ones((b, L), np.float32),
          "general_ghi": np.full((b, L), 3.0, np.float32), "general_ghi_predict": np.full((b, H), 1.0, np.float32),
          "general_temp": np.full((b, L), 20.0, np.float32), "general_temp_predict": np.full((b, H), 25.0, np.float32),
          "observe_power_future": np.full((b, H), np.nan, np.float32)}
    m(bd)
    k = m._backbone.calls[0]["known"]
    assert k.shape == (b, 2, L + H)
    assert np.all(k[:, 0, :L] == 3.0) and np.all(k[:, 0, L:] == 1.0)
    assert np.all(k[:, 1, :L] == 20.0) and np.all(k[:, 1, L:] == 25.0)


def test_missing_covariate_pair_skipped_and_past_only():
    m = _model(past_future_cols=[["general_ghi", "general_ghi_predict"], ["nope", "nope_predict"]],
               past_only_cols=["general_temp"])
    bd = {"observe_power": np.ones((2, L), np.float32), "general_ghi": np.ones((2, L), np.float32),
          "general_ghi_predict": np.ones((2, H), np.float32), "general_temp": np.ones((2, L), np.float32)}
    m(bd)
    c = m._backbone.calls[0]
    assert c["known"].shape == (2, 1, L + H) and c["past"] == (2, 1, L)
    assert ("nope", "nope_predict") in m._warned_pairs


def test_predict_quantile_and_make_positive():
    bd = {"observe_power": -np.ones((2, L), np.float32)}
    out = _model(past_future_cols=[], predict_quantile=0.9, make_positive=False)(bd)
    assert np.allclose(out.logits.numpy(), 0.5 * -1 * 1.4)
    assert out.aux["quantiles"].shape == (2, H, 9)
    out = _model(past_future_cols=[])(bd)
    assert float(out.logits.min()) == 0.0 and float(out.aux["quantiles"].min()) == 0.0


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        _model(predict_quantile=0.15)                 # 不在默认 9 档里
    with pytest.raises(ValueError):
        _model(quantile_levels=[0.12])                # 不在模型 21 档里
    with pytest.raises(ValueError):
        TransformersAdaptedM7Model({"timestral": {"eager_init": False, "nan_policy": "bogus"}})
    fake = _FakeForecaster()
    fake.supports_known_covariates = False
    m = TransformersAdaptedM7Model({"timestral": {"eager_init": False}})
    with pytest.raises(ValueError):
        m._set_backbone(fake, "cpu")


@pytest.mark.parametrize("policy", ["interp", "zero", "native"])
def test_nan_policies(policy):
    x = np.linspace(1, 2, L, dtype=np.float32)[None].repeat(2, 0)
    x[0, -3:] = np.nan
    x[1, :] = np.nan
    m = _model(nan_policy=policy)
    arr = m._col({"observe_power": x}, "observe_power")
    assert np.all(arr[1] == 0.0)                                                 # 整行缺失一律置 0
    assert np.isnan(arr[0]).any() == (policy == "native")


def _tiny_forecaster():
    timestral = pytest.importorskip("timestral")
    torch.manual_seed(0)
    cfg = timestral.TimestralConfig(patch_size=32, d_model=32, num_heads=2, mlp_hidden_dim=64,
                                    block_schedule=("time", "time_variate", "time"),
                                    quantile_knots=LEVELS21, block_size=8, context_length=2048)
    return timestral.TimestralForecaster(timestral.TimestralModel(cfg).eval())


@pytest.mark.parametrize("policy", ["interp", "native"])
def test_real_timestral_forward_tiny(policy):
    fc = _tiny_forecaster()
    assert fc.supports_known_covariates
    m = TransformersAdaptedM7Model({"timestral": {"eager_init": False, "per_core_batch_size": 2, "nan_policy": policy}})
    m._set_backbone(fc, "cpu")
    rng = np.random.default_rng(0)
    x = np.abs(rng.standard_normal((3, L))).astype(np.float32)
    x[0, 10:20] = np.nan
    bd = {"observe_power": x,
          "general_ghi": np.abs(rng.standard_normal((3, L))).astype(np.float32),
          "general_ghi_predict": np.abs(rng.standard_normal((3, H))).astype(np.float32),
          "general_temp": rng.standard_normal((3, L)).astype(np.float32),
          "general_temp_predict": rng.standard_normal((3, H)).astype(np.float32)}
    out = m(bd)
    assert out.logits.shape == (3, H) and out.aux["quantiles"].shape == (3, H, 9)
    assert torch.isfinite(out.logits).all() and float(out.logits.min()) >= 0.0
    q = out.aux["quantiles"]
    assert torch.all(q[..., 1:] >= q[..., :-1] - 1e-5)                           # 分位单调
    assert torch.allclose(out.logits, q[..., 4])                                 # 中位数 == q_0.5


def test_backbone_loaded_once_in_init_not_in_forward(monkeypatch):
    """同 M5：eager_init=True 时权重在 __init__ 里加载一次，之后 forward / .to() / .eval() 都不再加载。"""
    import sys
    loads = []

    class _Stub:
        @staticmethod
        def from_pretrained(path, device="cpu"):
            loads.append((path, device))
            return _FakeForecaster()

    monkeypatch.setitem(sys.modules, "timestral", types.SimpleNamespace(TimestralForecaster=_Stub))
    m = TransformersAdaptedM7Model({"timestral": {"checkpoint": "/ckpt", "device": "cpu", "past_future_cols": []}})
    assert loads == [("/ckpt", "cpu")]
    m.to("cpu").eval()
    bd = {"observe_power": np.ones((2, L), np.float32)}
    m(bd)
    m(bd)
    assert len(loads) == 1


def test_forward_without_backbone_raises():
    m = TransformersAdaptedM7Model({"timestral": {"eager_init": False}})
    with pytest.raises(RuntimeError):
        m({"observe_power": np.ones((1, L), np.float32)})
