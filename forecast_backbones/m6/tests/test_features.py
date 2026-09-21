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

def test_variability_bins_all_nan_falls_back_to_middle_bin():
    s = np.array([np.nan, np.nan, np.nan])
    b = variability_bins(s, edges=None, n_bins=3)
    assert b.tolist() == [1, 1, 1]


# ---------------- 站名 → station_id 映射（框架数据只有 station 名字列时） ----------------
def test_station_meta_id_of_and_attach_station_id(tmp_path):
    from forecast_backbones.m6.features import StationMeta, attach_station_id
    from .conftest import write_station_meta, CAP
    meta = StationMeta.from_csv(write_station_meta(tmp_path))                     # 站名 S1001 / S1002
    assert meta.id_of("S1001") == 1001 and meta.id_of("nope") is None and meta.ids() == sorted(CAP)
    df = pd.DataFrame({"station": ["S1001", "S1002", "S1001"], "x": [1, 2, 3]})
    out = attach_station_id(df, meta)
    assert out["station_id"].tolist() == [1001, 1002, 1001] and "station_id" not in df.columns   # 不改原表
    assert attach_station_id(out, meta) is out                                       # 已有列原样返回
    with pytest.raises(ValueError, match="找不到"):
        attach_station_id(pd.DataFrame({"station": ["S1001", "S9"]}), meta)
    with pytest.raises(ValueError, match="既没有"):
        attach_station_id(pd.DataFrame({"x": [1]}), meta)



def test_station_meta_numeric_station_names_map(tmp_path):
    """站名全是数字时 read_csv 会读成整数；StationMeta 仍按文本站名映射（数据里整数站号 1001 同样能对上）。"""
    import pandas as pd
    from forecast_backbones.m6.features import StationMeta, attach_station_id
    p = tmp_path / "m.csv"
    p.write_text("station_id,station,lat,lon,alt_m,cap_mw\n1,1001,30,101,0,50\n2,1002,30,101,0,60\n", encoding="utf-8")
    meta = StationMeta.from_csv(p)
    assert meta.id_of("1001") == 1 and meta.name(2) == "1002"
    assert StationMeta(pd.read_csv(p)).id_of("1002") == 2                       # 不经 from_csv、列已是整数也一样
    out = attach_station_id(pd.DataFrame({"station": [1001, "1002"]}), meta)
    assert out["station_id"].tolist() == [1, 2]



def test_station_meta_rejects_text_or_duplicate_station_id(tmp_path):
    """station_id 是 M6 内部整数编号；雅砻江这类站名写在 station 列。写错位置时给出能照做的报错。"""
    import pandas as pd
    from forecast_backbones.m6.features import StationMeta
    p = tmp_path / "m.csv"
    p.write_text("station_id,station,lat,lon,alt_m,cap_mw\n雅砻江,雅砻江,30,101,0,50\n1002,1002,30,101,0,60\n", encoding="utf-8")
    with pytest.raises(ValueError, match="必须是整数编号.*雅砻江.*station 列"):
        StationMeta.from_csv(p)
    p.write_text("station_id,station,lat,lon,alt_m,cap_mw\n1,雅砻江,30,101,0,50\n1,1002,30,101,0,60\n", encoding="utf-8")
    with pytest.raises(ValueError, match="station_id 重复"):
        StationMeta.from_csv(p)
    p.write_text("station_id,station,lat,lon,alt_m,cap_mw\n1,雅砻江,30,101,0,50\n2,雅砻江,30,101,0,60\n", encoding="utf-8")
    with pytest.raises(ValueError, match="station 重复"):
        StationMeta.from_csv(p)
    p.write_text("station_id,station,lat,lon,alt_m,cap_mw\n1,雅砻江,30,101,0,50\n2.0,1002,30,101,0,60\n", encoding="utf-8")
    assert StationMeta.from_csv(p).id_of("雅砻江") == 1 and StationMeta.from_csv(p).id_of("1002") == 2



def test_station_lookup_by_name_then_by_numeric_code(tmp_path):
    """数据站列的值先按站表 station 列（站名）找；找不到且是纯数字时，按 station_id 列（编号）找。"""
    import pandas as pd
    from forecast_backbones.m6.features import StationMeta, attach_station_id
    p = tmp_path / "m.csv"
    p.write_text("station_id,station,lat,lon,cap_mw,alt_m\n1002,甲站,30,101,,0\n1003,乙站,30,101,,0\n7,雅砻江,30,101,,0\n",
                 encoding="utf-8")
    meta = StationMeta.from_csv(p)
    out = attach_station_id(pd.DataFrame({"station": ["1002", 1003, "雅砻江", " 1002 ", "1003.0"]}), meta)
    assert out["station_id"].tolist() == [1002, 1003, 7, 1002, 1003]
    assert attach_station_id(pd.DataFrame({"station": ["甲站"]}), meta)["station_id"].tolist() == [1002]
    assert meta.id_of("1003") == 1003 and meta.id_of("雅砻江") == 7 and meta.id_of("没这个站") is None
    with pytest.raises(ValueError, match="找不到.*'1004'"):
        attach_station_id(pd.DataFrame({"station": ["1004"]}), meta)
    p.write_text("station_id,station,lat,lon,cap_mw,alt_m\n1002,甲站,30,101,,0\n1003,雅砻江,30,101,,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'1003'.*雅砻江.*换一个"):                # 雅砻江的自编号撞上了数据里的站 1003
        attach_station_id(pd.DataFrame({"station": ["1003", "雅砻江"]}), StationMeta.from_csv(p))


def test_station_meta_reads_excel_encodings_and_messy_names(tmp_path):
    from forecast_backbones.m6.features import StationMeta
    text = "station_id,station,lat,lon,alt_m,cap_mw\n1,雅砻江 ,30,101,0,50\n2,1002.0,30,101,0,60\n"
    for enc in ("utf-8-sig", "gbk"):
        p = tmp_path / f"m_{enc}.csv"; p.write_bytes(text.encode(enc))
        m = StationMeta.from_csv(p)
        assert m.id_of("雅砻江") == 1 and m.id_of("1002") == 2 and m.name(1) == "雅砻江", enc
