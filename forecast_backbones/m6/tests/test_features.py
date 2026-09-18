import numpy as np
import pandas as pd
import pytest
from forecast_backbones.m6.features import (window_timestamps, solar_geometry,
                                            clock_pseudo_geometry, StationMeta)

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
