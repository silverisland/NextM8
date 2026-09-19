import numpy as np
import pandas as pd
import pytest
import forecast_backbones.m6.features as features_mod
from forecast_backbones.m6.config import load_config
from forecast_backbones.m6.features import StationMeta, build_row_context

def _sm(tmp_path):
    p = tmp_path / "sm.csv"
    p.write_text("station_id,station,lat,lon,alt_m,cap_mw\n1001,A,30.0,101.0,4600,500\n1002,B,,,,300\n", encoding="utf-8")
    return StationMeta.from_csv(p)

def _batch(L=8, H=4):
    s = int(pd.Timestamp("2025-06-15 10:00:00").value // 10**9)
    return {
        "timestamp_win": np.array([s, s]),
        "station_id": np.array([1001, 1002]),
        "cap_power_on": np.array([500.0, 300.0]),
        "general_ghi_predict": np.full((2, H), 400.0),
    }

# 下面 6 个既有测试都用到 _sm()（站 1002 缺坐标）→ 会经过退化几何分支。build_row_context 现在对
# 退化站发一次性警告，且"是否真的发出"取决于本进程内 1002 是否已被别的测试标记过（模块级去重集合，
# 见下方专门测警告本身的用例）——用 recwarn 兜底吸收掉它，不让它以未捕获的形式混进 "-W default" 的
# pytest 警告汇总（其余断言与本次修复无关，保持原样）。

def test_row_context_shapes_and_degradation(tmp_path, recwarn):
    cfg = load_config({"timesfm3": {"context_len": 8, "pred_len": 4}})
    ctx = build_row_context(_batch(), cfg, _sm(tmp_path))
    assert ctx.elev_deg.shape == (2, 12) and ctx.p_clear.shape == (2, 12) and ctx.night_future.shape == (2, 4)
    assert ctx.degraded.tolist() == [False, True]
    assert ctx.cap.tolist() == [500.0, 300.0]
    assert ctx.elev_deg[0, -1] > 0 and not ctx.night_future[0].any()      # 10:15–11:00 白天
    assert ctx.p_clear[1, -1] == 300.0                                    # 退化站 p_clear = cap
    assert ctx.var_bin.shape == (2,) and set(ctx.var_bin.tolist()) <= {0, 1, 2}

def test_cap_column_vector_form(tmp_path, recwarn):
    cfg = load_config({"timesfm3": {"context_len": 8, "pred_len": 4}})
    b = _batch(); b["cap_power_on"] = np.tile(b["cap_power_on"][:, None], (1, 8))
    ctx = build_row_context(b, cfg, _sm(tmp_path))
    assert ctx.cap.tolist() == [500.0, 300.0]

def test_cap_falls_back_to_station_meta(tmp_path, recwarn):
    cfg = load_config({"timesfm3": {"context_len": 8, "pred_len": 4}})
    b = _batch(); del b["cap_power_on"]
    ctx = build_row_context(b, cfg, _sm(tmp_path))
    assert ctx.cap.tolist() == [500.0, 300.0]

def test_recent_kt_std_with_fixed_edges_bins(tmp_path, recwarn):
    cfg = load_config({"timesfm3": {"context_len": 8, "pred_len": 4},
                        "variability": {"signal": "recent_kt_std", "recent_kt_std_edges": [0.05, 0.15]}})
    b = _batch(); b["observe_power"] = np.array([[0., 50., 100., 150., 200., 250., 300., 350.],
                                                  [0., 0., 0., 0., 0., 0., 0., 0.]])
    ctx = build_row_context(b, cfg, _sm(tmp_path))
    assert ctx.var_bin.shape == (2,) and set(ctx.var_bin.tolist()) <= {0, 1, 2}

def test_variability_missing_edges_raises(tmp_path, recwarn):
    cfg = load_config({"timesfm3": {"context_len": 8, "pred_len": 4},
                        "variability": {"signal": "recent_kt_std"}})
    b = _batch(); b["observe_power"] = np.zeros((2, 8))
    with pytest.raises(ValueError, match="recent_kt_std_edges"):
        build_row_context(b, cfg, _sm(tmp_path))

def test_nwp_spread_unmet_prerequisite_raises(tmp_path, recwarn):
    cfg = load_config({"timesfm3": {"context_len": 8, "pred_len": 4},
                        "variability": {"signal": "nwp_spread", "nwp_spread_edges": [0.1, 0.2]}})
    with pytest.raises(ValueError, match="nwp_spread"):
        build_row_context(_batch(), cfg, _sm(tmp_path))


# ---------------------------------------------------------------- C1: 站元数据静默退化

def test_build_row_context_warns_once_for_degraded_station(tmp_path, monkeypatch):
    """station 缺坐标（不在 station_meta 里有效经纬度）→ 警告文本里列出该 station_id，且 ctx.degraded=True。
    显式清空模块级去重集合，保证本用例不受其它测试执行顺序影响，确定性地触发一次警告。"""
    monkeypatch.setattr(features_mod, "_DEGRADED_WARNED", set())
    cfg = load_config({"timesfm3": {"context_len": 8, "pred_len": 4}})
    with pytest.warns(UserWarning, match="1002"):
        ctx = build_row_context(_batch(), cfg, _sm(tmp_path))
    assert ctx.degraded.tolist() == [False, True]


# ---------------------------------------------------------------- I5.2 / I5.3: ghi_col 显式配置项

def test_ghi_col_missing_from_batch_raises(tmp_path):
    """variability.ghi_col（默认 general_ghi_predict）不在 batch_dict 里 → ValueError，不再是零值兜底。"""
    cfg = load_config({"timesfm3": {"context_len": 8, "pred_len": 4}})
    b = _batch(); del b["general_ghi_predict"]
    with pytest.raises(ValueError, match="ghi_col"):
        build_row_context(b, cfg, _sm(tmp_path))


def test_ghi_col_nan_is_interpolated_not_left_to_propagate(tmp_path):
    """ghi 预报列里单个 NaN 之前会直接传播成该行 var_signal=NaN，进而被 variability_bins 兜到中档 bin；
    现在先按 _col 同款逻辑插值，NaN 位置被邻居线性重建，signal 与无缺失时几乎完全一致，bin 也一致。"""
    cfg = load_config({"timesfm3": {"context_len": 8, "pred_len": 4}})
    sm = _sm(tmp_path)
    clean = np.array([[300.0, 320.0, 340.0, 360.0], [300.0, 320.0, 340.0, 360.0]])
    b_clean = _batch(); b_clean["general_ghi_predict"] = clean
    ctx_clean = build_row_context(b_clean, cfg, sm)

    with_nan = clean.copy(); with_nan[0, 1] = np.nan            # 邻居 300/340 线性插值应精确还原成 320
    b_nan = _batch(); b_nan["general_ghi_predict"] = with_nan
    ctx_nan = build_row_context(b_nan, cfg, sm)

    assert np.isfinite(ctx_nan.var_signal).all()
    assert np.isclose(ctx_nan.var_signal[0], ctx_clean.var_signal[0], atol=1e-6)
    assert ctx_nan.var_bin[0] == ctx_clean.var_bin[0]


def test_build_row_context_raises_for_nonfinite_cap(tmp_path):
    """station 在 station_meta 里完全不存在，且 batch_dict 没有 cap_power_on 兜底 → cap 非有限 → ValueError
    （而不是让 NaN 传播到 p_clear → physics_clip → NaN logits → NanwangMetric 静默丢站）。"""
    p = tmp_path / "sm_only_1001.csv"
    p.write_text("station_id,station,lat,lon,alt_m,cap_mw\n1001,A,30.0,101.0,4600,500\n", encoding="utf-8")
    sm = StationMeta.from_csv(p)
    cfg = load_config({"timesfm3": {"context_len": 8, "pred_len": 4}})
    b = _batch(); b["station_id"] = np.array([1001, 9999]); del b["cap_power_on"]
    with pytest.raises(ValueError, match="9999"):
        build_row_context(b, cfg, sm)
