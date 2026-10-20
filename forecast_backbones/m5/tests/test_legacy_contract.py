"""legacy 模式（线上 nanwang_online / eval_pretrain / experiments 用的零样本配置，没有 dataset.station_meta）
与重构前的 M5 逐位相同：同输入 → 同 logits / aux、同样的底座调用参数、同样的实例属性。"""
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from forecast_backbones.m5.transformer_models import TransformersAdaptedM5Model, TransformersAdaptedTimesFM3Model
from . import legacy_m5_forward as old
from .conftest import write_station_meta

L, H = 672, 192
M5 = Path(__file__).resolve().parents[1]


class _RecordingBackbone:
    """predict_batch 替身：记录全部输入与关键字参数；forecast = 0.5×context 末 H 段 + 0.01×第一协变量末 H 段；
    quantiles = forecast × [0.6..1.4]；return_quantiles=False 时 quantiles=None。"""

    def __init__(self):
        self.calls = []

    def predict_batch(self, inputs, horizon, **kw):
        rec = {k: ([np.array(a, copy=True) for a in v] if isinstance(v, list) else v) for k, v in kw.items()}
        self.calls.append(([np.array(x, copy=True) for x in inputs], rec))
        pf = kw.get("past_future_covariates")
        outs = []
        for i, x in enumerate(inputs):
            fc = 0.5 * x[0, -horizon:] + (0.01 * pf[i][0, -horizon:] if pf else 0.0)
            q = fc[:, None] * np.linspace(0.6, 1.4, 9)[None, :] if kw.get("return_quantiles", True) else None
            outs.append(types.SimpleNamespace(forecast=fc[None, :], quantiles=None if q is None else q[None]))
        return iter(outs)


def _batch(B=3, nan=False, drop=()):
    rng = np.random.default_rng(0)
    t = lambda n: torch.as_tensor(rng.uniform(0, 100, (B, n)), dtype=torch.float32)
    bd = {"timestamp_win": torch.arange(B) * 3600, "observe_power": t(L),
          "general_ghi": t(L), "general_ghi_predict": t(H), "general_temp": t(L), "general_temp_predict": t(H)}
    if nan:
        bd["observe_power"][0, 10:20] = float("nan"); bd["general_ghi"][1, :] = float("nan")
    for c in drop:
        bd.pop(c)
    return bd


def _pair(cfg):
    cfg = {**cfg, "timesfm3": {**(cfg.get("timesfm3") or {}), "eager_init": False}}
    new, ref = TransformersAdaptedM5Model(cfg), old.TransformersAdaptedM5Model(cfg)
    for m in (new, ref):
        m._backbone = _RecordingBackbone(); m._backbone_device = "cpu"
    return new, ref


CASES = {
    "default": ({}, {}),
    "q0.9": ({"timesfm3": {"predict_quantile": 0.9}}, {}),
    "no_quantiles": ({"timesfm3": {"return_quantiles": False}}, {}),
    "nan_interp": ({}, {"nan": True}),
    "missing_pair": ({}, {"drop": ("general_temp_predict",)}),
    "past_only": ({"timesfm3": {"past_only_cols": ["general_temp"]}}, {}),
    "online_yaml": (yaml.safe_load((M5 / "config" / "configs.yaml").read_text(encoding="utf-8")), {}),
    "release_like": ({"run": {"model_name": "m5"}, "model": {}, "dataset": {"pv_col": "observe_power"}}, {}),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_legacy_forward_bit_identical(case):
    cfg, bkw = CASES[case]
    new, ref = _pair(cfg)
    assert new.mode == "legacy"
    a, b = new(_batch(**bkw), labels={"observe_power_future": torch.zeros(3, H)}), ref(_batch(**bkw))
    assert torch.equal(a.logits, b.logits) and torch.equal(a.predictions, b.predictions)
    assert a.loss is None and b.loss is None
    assert (a.aux is None) == (b.aux is None)
    if a.aux is not None:
        assert set(a.aux) == set(b.aux) == {"quantiles"} and torch.equal(a.aux["quantiles"], b.aux["quantiles"])
    (ia, ka), (ib, kb) = new._backbone.calls[0], ref._backbone.calls[0]
    assert len(ia) == len(ib) and all(np.array_equal(x, y) for x, y in zip(ia, ib))
    assert sorted(ka) == sorted(kb)
    for k in ka:
        va, vb = ka[k], kb[k]
        if isinstance(va, list):
            assert len(va) == len(vb) and all(np.array_equal(x, y) for x, y in zip(va, vb)), k
        else:
            assert va == vb, k


def test_legacy_attributes_used_by_callers():
    new, ref = _pair({})
    for k in ("target_col", "context_len", "pred_len", "checkpoint", "device_pref", "past_future_cols", "past_only_cols",
              "per_core_batch_size", "return_quantiles", "predict_quantile", "nan_policy", "make_positive", "use_znorm",
              "use_symmetric_averaging", "_warned_pairs"):
        assert getattr(new, k) == getattr(ref, k), k
    assert not hasattr(new, "config") and not hasattr(ref, "config")          # pipeline 按 hasattr 判断的契约不变
    new.past_future_cols = [("general_ghi", "general_ghi_predict")]            # experiments/nwp_covariate/backends.py 的用法
    new._warned_pairs = set()
    new(_batch())
    assert len(new._backbone.calls[-1][1]["past_future_covariates"][0]) == 1


def test_registry_and_alias():
    from pipelines.model_registry import import_model_class
    assert import_model_class("m5") is TransformersAdaptedM5Model
    assert issubclass(TransformersAdaptedTimesFM3Model, TransformersAdaptedM5Model)
    assert TransformersAdaptedTimesFM3Model.__name__ == "TransformersAdaptedTimesFM3Model"   # 线上按这个名字取预测列


def test_finetune_without_station_meta_raises():
    with pytest.raises(ValueError, match="station_meta"):
        TransformersAdaptedM5Model({"timesfm3": {"eager_init": False}, "finetune": {"enabled": True}})


def test_station_meta_selects_full_mode(tmp_path):
    m = TransformersAdaptedM5Model({"dataset": {"station_meta": str(write_station_meta(tmp_path))},
                                    "timesfm3": {"eager_init": False}})
    assert m.mode == "full" and m.backbone is None and m.cfg["finetune"]["enabled"] is True
    assert m.config is m.cfg and m.levels == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    assert m.set_known_stations([1002, 1001, 1001]) == [1001, 1002] and m.cfg["dataset"]["known_stations"] == [1001, 1002]


def test_full_mode_missing_station_meta_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="station_meta"):
        TransformersAdaptedM5Model({"dataset": {"station_meta": str(tmp_path / "nope.csv")}, "timesfm3": {"eager_init": False}})
