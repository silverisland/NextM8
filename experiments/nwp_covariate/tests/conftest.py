# -*- coding: utf-8 -*-
"""合成 DS 轨迹表：列名按 config.columns 的前缀 + 原始字段名，形状与线上 DS 表一致
（每行一个站一个起报窗，历史 672 点 list、未来 P 点 list）。数值有物理结构：晴空 × 云况，功率由 GHI 换算。"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))

from ops import clear_sky_ghi, window_timestamps  # noqa: E402

STATIONS = [("plant_guangfu1025", 120.0, 22.82, 108.32), ("plant_guangfu1358", 80.0, 24.33, 109.42),
            ("plant_guangfu1780", 200.0, 25.28, 110.29)]
C, P = 672, 192


def _ar1(rng, n, scale, corr=0.9):
    e = rng.normal(0, scale, n); out = np.empty(n); out[0] = e[0]
    for i in range(1, n):
        out[i] = corr * out[i - 1] + np.sqrt(1 - corr**2) * e[i]
    return out


def make_ds(report_times, stations=STATIONS, seed=7, missing=None) -> pd.DataFrame:
    """missing: dict{col: 行索引列表} 把这些单元格置 None，用来测补全。"""
    rng = np.random.default_rng(seed)
    rows = []
    for st, cap, lat, lon in stations:
        for t in report_times:
            ts = window_timestamps(np.array([np.datetime64(pd.Timestamp(t))]), C, P)[0]
            cs = clear_sky_ghi(ts[None, :], np.array([lat]), np.array([lon]))[0]
            kt = np.clip(0.7 + _ar1(rng, C + P, 0.2), 0.05, 1.05)
            ghi = cs * kt
            t2m = 26 + 6 * np.sin((pd.DatetimeIndex(ts).hour.to_numpy() - 8) / 24 * 2 * np.pi) + _ar1(rng, C + P, 0.5)
            power = np.clip(cap * ghi / 1000 * (1 - 0.004 * (t2m + 0.03 * ghi - 25)), 0, cap)
            # EC 累积 J/m²：15min 步长上累积（模拟 DS 层已内插到 15min 的累积量）
            ssrd_acc = np.cumsum(ghi * 1.02 * 900.0)
            fdir_acc = np.cumsum(ghi * 0.7 * 900.0)
            cdir_acc = np.cumsum(cs * 0.8 * 900.0)
            tcc = np.clip(1 - kt + rng.normal(0, 0.05, C + P), 0, 1)
            row = {"timestamp_win": pd.Timestamp(t), "station": st, "cap_power_on": [cap] * 4,
                   "observe_power": power[:C], "observe_power_future": power[C:]}
            def put(name, arr, hist_only=False):
                row[name] = arr[:C]
                if not hist_only:
                    row[name + "_predict"] = arr[C:]
            put("EC_JTKRD7RO_ssrd", ssrd_acc); put("EC_JTKRD7RO_fdir", fdir_acc); put("EC_JTKRD7RO_cdir", cdir_acc)
            put("EC_JTKRD7RO_tcc", tcc); put("EC_JTKRD7RO_lcc", tcc * 0.8)
            put("EC_JTKRD7RO_t2m", t2m + 273.15); put("EC_JTKRD7RO_skt", t2m + 8 + 273.15)
            put("EC_JTKRD7RO_u10", np.full(C + P, 1.5)); put("EC_JTKRD7RO_v10", np.full(C + P, 2.0))
            put("EC_JTKRD7RO_tp", np.cumsum(np.where(kt < 0.3, 0.2, 0.0)))
            put("GDFS_JTHFGNED_MID_JTNWQISX_sr", ghi * 0.97); put("GDFS_JTHFGNED_MID_JTNWQISX_tcc", tcc * 100)
            put("GDFS_JTHFGNED_MID_JTNWQISX_tem_hl_10", t2m + 0.5)
            put("general_GHI", ghi * 1.0); put("general_TEMP", t2m)
            put("solar_southern_GHI", ghi * 0.95); put("solar_southern_TEMP", t2m - 0.3)
            put("observe_weather_GHI", ghi * 0.99, hist_only=True); put("observe_weather_TEMP", t2m, hist_only=True)
            rows.append({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in row.items()})
    df = pd.DataFrame(rows)
    for col, idx in (missing or {}).items():
        for i in idx:
            df.at[i, col] = None
    return df


@pytest.fixture
def base_cfg():
    cfg = yaml.safe_load(open(PKG / "config.yaml", encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    cfg["data"]["predict_type"] = "ultra_short"
    cfg["data"]["time_range"] = None
    cfg["report_times"]["ultra_short"] = {"start": "2026-08-26 10:00:00", "end": "2026-08-28 10:00:00", "step": "1D"}
    cfg["columns"]["sources"]["ec"]["fields"] = {k: v for k, v in cfg["columns"]["sources"]["ec"]["fields"].items()
                                                 if k in ("ghi_ec", "fdir_ec", "cdir_ec", "tcc_ec", "lcc_ec", "t2m_ec", "skt_ec", "u10_ec", "v10_ec", "tp_ec", "ghi_xr")}
    cfg["columns"]["sources"]["gdfs"]["fields"] = {k: v for k, v in cfg["columns"]["sources"]["gdfs"]["fields"].items()
                                                   if k in ("ghi_gdfs", "tcc_gdfs", "t2m_gdfs")}
    return cfg


@pytest.fixture
def ds_dir(tmp_path, base_cfg):
    """写 3 个起报时刻的 DS parquet + station_meta.csv，把 cfg 指过去。"""
    rt = base_cfg["report_times"]["ultra_short"]
    times = pd.date_range(rt["start"], rt["end"], freq=rt["step"])
    df = make_ds(times)
    root = tmp_path / "ds"
    for t, g in df.groupby("timestamp_win"):
        d = root / f"date={t:%Y-%m-%d}" / f"time={t:%H:%M}"
        d.mkdir(parents=True)
        g.to_parquet(d / f"{t:%Y%m%d%H%M}.parquet", index=False)
    meta = pd.DataFrame([{"station": s, "lat": la, "lon": lo, "alt_m": 100.0} for s, _, la, lo in STATIONS])
    meta.to_csv(tmp_path / "station_meta.csv", index=False)
    base_cfg["data"]["input_mode"] = "ds_parquet"
    base_cfg["data"]["ds_trajectory_glob"] = str(root / "date=*" / "time=*" / "*.parquet")
    base_cfg["data"]["station_meta"] = str(tmp_path / "station_meta.csv")
    base_cfg["data"]["feature_store"] = str(tmp_path / "fs" / "{predict_type}")
    return base_cfg, df
