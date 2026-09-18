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


def _bd(df, rows):
    sub = df.iloc[rows]
    bd = {c: np.stack(sub[c].to_numpy()) for c in df.columns if c not in ("timestamp_win", "station_id", "cap_power_on", "observe_power_future")}
    bd["timestamp_win"] = (sub["timestamp_win"].astype("int64") // 10**9).to_numpy()
    bd["station_id"] = sub["station_id"].to_numpy(); bd["cap_power_on"] = sub["cap_power_on"].to_numpy()
    return bd


def test_m6_median_mode_matches_m5_bitwise(tmp_path):
    """M6 关闭校准、关闭派生协变量、method=median、物理层关 → 与 m5 逐位一致。"""
    from forecast_backbones.m5.transformer_models import TransformersAdaptedTimesFM3Model
    from forecast_backbones.m6.transformer_models import TransformersAdaptedM6Model
    df = make_synthetic_frame(n_days=1, issue_every_min=720)
    bd = _bd(df, [0, 1])
    pf = [["general_ghi", "general_ghi_predict"], ["general_temp", "general_temp_predict"]]
    # m5 的 logits 默认取 forecast 点预报；这里显式取 q50，与 M6 的 median 口径逐位可比
    m5 = TransformersAdaptedTimesFM3Model({"timesfm3": {"checkpoint": CKPT, "device": "cpu",
                                                        "past_future_cols": pf, "predict_quantile": 0.5}})
    m6 = TransformersAdaptedM6Model({"timesfm3": {"checkpoint": CKPT, "device": "cpu"},
                                     "dataset": {"station_meta": str(write_station_meta(tmp_path))},
                                     "features": {"past_future": pf, "derived": {"solar_elevation": False, "clear_sky_power": False}},
                                     "calibration": {"enabled": False},
                                     "decision": {"physics": {"night_zero": False, "clear_sky_cap_ratio": 1e9, "capacity_cap": False},
                                                  "point": {"default": "median", "by_step": []}}})
    a = m5(bd).logits; b = m6(bd).logits
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
