import numpy as np
import pandas as pd
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

def test_row_context_shapes_and_degradation(tmp_path):
    cfg = load_config({"timesfm3": {"context_len": 8, "pred_len": 4}})
    ctx = build_row_context(_batch(), cfg, _sm(tmp_path))
    assert ctx.elev_deg.shape == (2, 12) and ctx.p_clear.shape == (2, 12) and ctx.night_future.shape == (2, 4)
    assert ctx.degraded.tolist() == [False, True]
    assert ctx.cap.tolist() == [500.0, 300.0]
    assert ctx.elev_deg[0, -1] > 0 and not ctx.night_future[0].any()      # 10:15–11:00 白天
    assert ctx.p_clear[1, -1] == 300.0                                    # 退化站 p_clear = cap
    assert ctx.var_bin.shape == (2,) and set(ctx.var_bin.tolist()) <= {0, 1, 2}

def test_cap_column_vector_form(tmp_path):
    cfg = load_config({"timesfm3": {"context_len": 8, "pred_len": 4}})
    b = _batch(); b["cap_power_on"] = np.tile(b["cap_power_on"][:, None], (1, 8))
    ctx = build_row_context(b, cfg, _sm(tmp_path))
    assert ctx.cap.tolist() == [500.0, 300.0]

def test_cap_falls_back_to_station_meta(tmp_path):
    cfg = load_config({"timesfm3": {"context_len": 8, "pred_len": 4}})
    b = _batch(); del b["cap_power_on"]
    ctx = build_row_context(b, cfg, _sm(tmp_path))
    assert ctx.cap.tolist() == [500.0, 300.0]
