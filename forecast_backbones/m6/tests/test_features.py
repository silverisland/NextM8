import numpy as np
import pandas as pd
import pytest
from forecast_backbones.m6.features import (window_timestamps, solar_geometry,
                                            clock_pseudo_geometry, StationMeta)
from forecast_backbones.m6.features import (k_nwp_signal, spread_signal, disagree_signal,
                                            recent_kt_std_signal, variability_bins)

EPOCH = int(pd.Timestamp("2025-06-15 10:00:00").value // 10**9)   # 北京墙钟 10:00

def test_window_timestamps_layout():
    ts = window_timestamps(np.array([EPOCH]), context_len=4, pred_len=3, step_min=15)
    assert ts.shape == (1, 7)
    assert ts[0, 3] == np.datetime64("2025-06-15T10:00:00")          # 历史最后一点 = S
    assert ts[0, 4] == np.datetime64("2025-06-15T10:15:00")          # 未来第 1 点 = S+15min
    assert ts[0, 0] == np.datetime64("2025-06-15T09:15:00")

def test_solar_geometry_beijing_noon_is_day_midnight_is_night():
    ts = np.array([np.datetime64("2025-06-15T12:00:00"), np.datetime64("2025-06-15T00:00:00")])
    elev, ghi = solar_geometry(ts, lat=30.0, lon=101.0, alt_m=4600, tz="Asia/Shanghai")
    assert elev[0] > 45 and ghi[0] > 500
    assert elev[1] < 0 and ghi[1] == 0.0

def test_solar_geometry_preserves_shape():
    ts = window_timestamps(np.array([EPOCH, EPOCH + 3600]), 8, 4)
    elev, ghi = solar_geometry(ts, 30.0, 101.0, 4600, "Asia/Shanghai")
    assert elev.shape == ts.shape == ghi.shape

def test_clock_pseudo_geometry():
    ts = np.array([np.datetime64("2025-06-15T12:00:00"), np.datetime64("2025-06-15T03:00:00")])
    elev, ghi = clock_pseudo_geometry(ts)
    assert elev[0] > 0 and ghi[0] > 0 and elev[1] < 0 and ghi[1] == 0

def test_station_meta(tmp_path):
    p = tmp_path / "sm.csv"
    p.write_text("station_id,station,lat,lon,alt_m,cap_mw\n1001,A,30.0,101.0,4600,500\n1002,B,,,,\n",
                 encoding="utf-8")
    sm = StationMeta.from_csv(p)
    assert sm.has_coords(1001) and not sm.has_coords(1002)
    assert sm.name(1001) == "A" and sm.name(9999) == "9999"
    assert sm.get(1001)["cap_mw"] == 500.0 and sm.get(7) is None

def test_k_nwp_signal_uses_day_steps_only():
    clear = np.array([[0.0, 100.0, 800.0, 0.0]])
    pred = np.array([[50.0, 50.0, 400.0, 9.0]])
    assert np.isclose(k_nwp_signal(pred, clear), (0.5 + 0.5) / 2)
    assert np.isnan(k_nwp_signal(pred, np.zeros_like(clear)))[0]

def test_spread_and_disagree():
    clear = np.array([[800.0, 800.0]])
    members = np.array([[[400.0, 400.0], [800.0, 800.0]]])         # [B=1, M=2, H=2]
    assert np.isclose(spread_signal(members, clear)[0], 200.0 / 800.0)
    assert np.isclose(disagree_signal(np.array([[400.0, 400.0]]), np.array([[800.0, 800.0]]), clear)[0], 0.5)

def test_recent_kt_std():
    p_clear = np.full((1, 4), 100.0)
    steady = np.full((1, 4), 50.0); jumpy = np.array([[10.0, 90.0, 10.0, 90.0]])
    assert recent_kt_std_signal(steady, p_clear, last_steps=4)[0] < recent_kt_std_signal(jumpy, p_clear, last_steps=4)[0]

def test_variability_bins_with_edges_and_nan():
    s = np.array([0.1, 0.5, 0.9, np.nan])
    assert variability_bins(s, edges=[0.35, 0.75]).tolist() == [0, 1, 2, 1]

def test_variability_bins_quantile_fallback():
    s = np.linspace(0, 1, 30)
    b = variability_bins(s, edges=None, n_bins=3)
    assert set(b.tolist()) == {0, 1, 2} and b[0] == 0 and b[-1] == 2
