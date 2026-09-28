"""M7 框架契约：registry 自动发现（第一个 nn.Module 子类、Model(model_config)）+ evaluate_model 结果表。
默认不依赖 granite-tsfm：eager_init=False 后用 _set_backbone 挂替身；装了 granite-tsfm 时再用随机小模型跑一遍真实前向。"""
import functools
import types

import numpy as np
import pandas as pd
import pytest
import torch

from pipelines.transformer_dataset import UnifiedDataset
from pipelines.model_registry import build_evaluator, import_model_class
from forecast_backbones.m7.transformer_models import TransformersAdaptedM7Model, TransformersAdaptedPatchTSTFMModel
from forecast_backbones.m7.transformer_trainer import evaluate_model, QUANTILE_COLS

L, H = 672, 192
LEVELS99 = [round(i / 100, 2) for i in range(1, 100)]


class _FakeBackbone:
    """PatchTSTFMForPrediction 替身：点预测 = 0.5 × context 末 H 段；第 k 档分位 = 点预测 × (0.5 + level)，
    因此 q_0.5 == 点预测；输出形状与真模型一致（[b,H,1] / [b,99,H,1]）。"""
    config = types.SimpleNamespace(quantile_levels=LEVELS99)

    def __init__(self):
        self.calls = []

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, past_values, prediction_length, return_dict=True):
        self.calls.append(tuple(past_values.shape))
        fc = 0.5 * past_values[:, -prediction_length:, :]                               # [b,H,1]
        scale = torch.tensor(LEVELS99).view(1, -1, 1, 1) + 0.5
        return types.SimpleNamespace(prediction_outputs=fc, quantile_outputs=fc.unsqueeze(1) * scale)


def _frame(n=6, stations=(1001, 1002)):
    rows = []
    for s in stations:
        for i in range(n):
            base = np.linspace(0, 100, L, dtype=np.float32) + s % 7
            rows.append({"timestamp_win": pd.Timestamp("2025-06-08") + pd.Timedelta(hours=i), "station_id": s,
                         "observe_power": base, "observe_power_future": np.full(H, 42.0, np.float32),
                         "general_ghi_predict": np.full(H, 1.0, np.float32)})
    return pd.DataFrame(rows)


def _model(**cfg):
    m = TransformersAdaptedM7Model({"patchtst_fm": {"eager_init": False, "per_core_batch_size": 4, **cfg}})
    m._set_backbone(_FakeBackbone(), "cpu")
    return m


def test_registry_discovers_m7_model_and_evaluator():
    assert import_model_class("m7") is TransformersAdaptedM7Model
    assert TransformersAdaptedPatchTSTFMModel is TransformersAdaptedM7Model
    assert build_evaluator("m7") is evaluate_model


def test_backbone_not_registered_as_submodule():
    m = _model()
    assert len(m.state_dict()) == 0                          # 同 M5：外部 .to()/.train() 不碰 backbone


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
    assert all(shape[0] <= 4 for shape in m._backbone.calls)                      # per_core_batch_size 切块生效
    assert all(shape[1:] == (L, 1) for shape in m._backbone.calls)                # 只喂单变量历史功率


def test_predict_quantile_and_make_positive():
    bd = {"observe_power": -np.ones((2, L), np.float32)}
    out = _model(predict_quantile=0.9, make_positive=False)(bd)
    assert np.allclose(out.logits.numpy(), 0.5 * -1 * 1.4)
    assert out.aux["quantiles"].shape == (2, H, 9)
    out = _model()(bd)
    assert float(out.logits.min()) == 0.0 and float(out.aux["quantiles"].min()) == 0.0


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        _model(predict_quantile=0.15)
    with pytest.raises(ValueError):
        _model(quantile_levels=[0.105])
    with pytest.raises(ValueError):
        TransformersAdaptedM7Model({"patchtst_fm": {"eager_init": False, "nan_policy": "bogus"}})


@pytest.mark.parametrize("policy", ["interp", "zero", "native"])
def test_nan_policies(policy):
    x = np.linspace(1, 2, L, dtype=np.float32)[None].repeat(2, 0)
    x[0, -3:] = np.nan
    x[1, :] = np.nan
    m = _model(nan_policy=policy)
    arr = m._col({"observe_power": x}, "observe_power")
    assert np.all(arr[1] == 0.0)                                                 # 整行缺失一律置 0
    assert np.isnan(arr[0]).any() == (policy == "native")


def test_real_patchtst_fm_forward_tiny():
    tsfm = pytest.importorskip("tsfm_public")
    from tsfm_public.models.patchtst_fm import PatchTSTFMConfig
    torch.manual_seed(0)
    cfg = PatchTSTFMConfig(context_length=1024, d_patch=16, patch_stride=8, patch_loss_windowing="hamming",
                           d_model=32, n_head=2, n_layer=2, block_type="conformer", conv_kernel_size=3)
    m = TransformersAdaptedM7Model({"patchtst_fm": {"eager_init": False, "per_core_batch_size": 2}})
    m._set_backbone(tsfm.PatchTSTFMForPrediction(cfg), "cpu")
    x = np.abs(np.random.randn(3, L)).astype(np.float32)
    x[0, 10:20] = np.nan
    out = m({"observe_power": x})
    assert out.logits.shape == (3, H) and out.aux["quantiles"].shape == (3, H, 9)
    assert torch.isfinite(out.logits).all() and float(out.logits.min()) >= 0.0
