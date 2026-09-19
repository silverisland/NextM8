import importlib.util
import os

import numpy as np
import pytest
import torch

# 跳过闸必须在 import 时（collection 阶段）就生效，且不只看包装没装：本机常见场景是包装了
# 但没有权重（训练机路径 CKPT 在本机不存在）——两个条件任一不满足都跳过，理由里把两条都报出来，
# 方便一眼看出到底缺哪个。
CKPT = os.environ.get("TIMESFM3_CKPT", "/data/tangqingang/timesfm-3.0-pytorch")
_has_timesfm3 = importlib.util.find_spec("timesfm3") is not None
_has_ckpt = os.path.exists(CKPT)
pytestmark = pytest.mark.skipif(
    not (_has_timesfm3 and _has_ckpt),
    reason=f"需要 timesfm3 与本地权重（timesfm3 已安装={_has_timesfm3}, 权重路径存在={_has_ckpt}: {CKPT}）",
)

from .conftest import make_synthetic_frame, write_station_meta

# m5 与 m6 的 timesfm3.* 默认值里只有 per_core_batch_size 不同（m5=32, m6=64，其余
# checkpoint/context_len/pred_len/device/return_quantiles/nan_policy/make_positive/
# use_znorm/use_symmetric_averaging 两边默认值逐一核对一致）——两个测试配置都显式给同一个
# 值，保证喂给 predict_batch 的 timesfm3.* 设置真的相同，而不是"恰好都用各自默认值"。
SHARED_PER_CORE_BATCH_SIZE = 32


def _bd(df, rows):
    sub = df.iloc[rows]
    bd = {c: np.stack(sub[c].to_numpy()) for c in df.columns if c not in ("timestamp_win", "station_id", "cap_power_on", "observe_power_future")}
    bd["timestamp_win"] = (sub["timestamp_win"].astype("int64") // 10**9).to_numpy()
    bd["station_id"] = sub["station_id"].to_numpy(); bd["cap_power_on"] = sub["cap_power_on"].to_numpy()
    return bd


def test_m6_raw_median_matches_m5(tmp_path):
    """M6 关闭校准、关闭派生协变量 → 未校准的原始分位中位数（aux["quantiles"][...,4]，
    MODEL_LEVELS 里 0.5 那一档）与 m5（predict_quantile: 0.5）的 logits 用 atol=1e-5 近似相等。

    不是逐位比特相同：两边各自构造了一个独立的 TimesFM3Evaluator 实例，不保证浮点运算路径
    完全一致。也不经过 m6 的 physics_clip/select_point 决策层——夜间 p_clear 恒为 0，
    `clear_sky_cap_ratio` 再大也无法把 physics_clip 的夜间钳位关掉（upper = ratio * 0 = 0），
    所以比较必须发生在决策层之前，用 aux["quantiles"] 这个校准前的原始分位。"""
    from forecast_backbones.m5.transformer_models import TransformersAdaptedTimesFM3Model
    from forecast_backbones.m6.transformer_models import TransformersAdaptedM6Model
    from forecast_backbones.m6.decision import MODEL_LEVELS
    df = make_synthetic_frame(n_days=1, issue_every_min=720)
    bd = _bd(df, [0, 1])
    pf = [["general_ghi", "general_ghi_predict"], ["general_temp", "general_temp_predict"]]
    # m5 的 logits 默认取 forecast 点预报；这里显式取 q50，与 M6 的原始中位数口径可比
    m5 = TransformersAdaptedTimesFM3Model({"timesfm3": {"checkpoint": CKPT, "device": "cpu",
                                                        "per_core_batch_size": SHARED_PER_CORE_BATCH_SIZE,
                                                        "past_future_cols": pf, "predict_quantile": 0.5}})
    m6 = TransformersAdaptedM6Model({"timesfm3": {"checkpoint": CKPT, "device": "cpu",
                                                  "per_core_batch_size": SHARED_PER_CORE_BATCH_SIZE},
                                     "dataset": {"station_meta": str(write_station_meta(tmp_path))},
                                     "features": {"past_future": pf, "derived": {"solar_elevation": False, "clear_sky_power": False}},
                                     "calibration": {"enabled": False}})
    a = m5(bd).logits
    b = m6(bd).aux["quantiles"][:, :, MODEL_LEVELS.index(0.5)]
    assert torch.allclose(a, b, atol=1e-5), float((a - b).abs().max())


def test_m6_full_stack_runs_and_shrinks(tmp_path):
    from forecast_backbones.m6.transformer_models import TransformersAdaptedM6Model
    df = make_synthetic_frame(n_days=1, issue_every_min=720)
    m6 = TransformersAdaptedM6Model({"timesfm3": {"checkpoint": CKPT, "device": "cpu"},
                                     "dataset": {"station_meta": str(write_station_meta(tmp_path))},
                                     "calibration": {"enabled": False}})
    out = m6(_bd(df, [0]))
    day = ~out.aux["night"]
    assert (out.logits.numpy()[day] <= out.aux["median"].numpy()[day] + 1e-4).all()
    assert out.aux["quantiles"].shape == (1, 192, 9)


def test_torch_path_matches_evaluator_on_real_checkpoint(tmp_path):
    """真权重：inference_path=torch 的零样本输出与 evaluator 路径 allclose(1e-4)（spec §10 第 3 步）。"""
    from forecast_backbones.m6.transformer_models import TransformersAdaptedM6Model
    from .conftest import make_synthetic_frame, write_station_meta
    from .test_model_route_a import _batch
    df = make_synthetic_frame(n_days=1, issue_every_min=360)
    base = {"dataset": {"station_meta": str(write_station_meta(tmp_path))}, "calibration": {"enabled": False},
            "timesfm3": {"checkpoint": CKPT, "per_core_batch_size": SHARED_PER_CORE_BATCH_SIZE}}
    m_eval = TransformersAdaptedM6Model({**base, "finetune": {"enabled": False, "inference_path": "evaluator"}})
    m_torch = TransformersAdaptedM6Model({**base, "finetune": {"enabled": False, "inference_path": "torch"}})
    bd, _ = _batch(df, [0, 1, 2, 3])
    with torch.no_grad():
        a, b = m_eval(bd), m_torch(bd)
    assert np.allclose(a.aux["quantiles"].cpu().numpy(), b.aux["quantiles"].cpu().numpy(), atol=1e-4)
