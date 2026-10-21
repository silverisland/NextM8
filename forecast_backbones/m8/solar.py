# forecast_backbones/m8/solar.py（2026-10-13 起自带一份，不再引用 m6）
"""站表（station_meta.csv）与太阳几何：时间戳网格、pvlib 晴空 GHI / 太阳高度角、缺坐标站的钟表几何、站名→站号映射、
每站固定容量。纯函数，不依赖 torch。"""
from __future__ import annotations


import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd


from pipelines.forecasting.stations import window_timestamps as window_timestamps
from pipelines.forecasting.stations import norm_station_name as norm_station_name
from pipelines.forecasting.stations import StationMeta as StationMeta
from pipelines.forecasting.stations import attach_station_id as attach_station_id
from pipelines.forecasting.stations import station_cap_mw as station_cap_mw


def solar_geometry(ts: np.ndarray, lat: float, lon: float, alt_m: float, tz: str):
    """ts: datetime64 数组（任意形状，naive 本地墙钟）→ (elev_deg, ghi_clear W/m²)，同形状。
    夜间（elev<=0）ghi 置 0。Ineichen 需 Linke 浊度表（pvlib 自带），失败退到 Haurwitz。"""
    import pvlib  # Station metadata / capacity helpers do not need solar geometry.

    shape = ts.shape
    idx = pd.DatetimeIndex(pd.to_datetime(ts.reshape(-1))).tz_localize(tz)
    loc = pvlib.location.Location(lat, lon, tz=tz, altitude=float(alt_m))
    sp = loc.get_solarposition(idx)
    elev = sp["apparent_elevation"].to_numpy(dtype=float)
    try:
        ghi = loc.get_clearsky(idx, model="ineichen")["ghi"].to_numpy(dtype=float)
    except Exception:                                   # noqa: BLE001
        ghi = pvlib.clearsky.haurwitz(sp["apparent_zenith"])["ghi"].to_numpy(dtype=float)
    ghi = np.where(elev > 0, np.maximum(np.nan_to_num(ghi), 0.0), 0.0)
    return elev.reshape(shape), ghi.reshape(shape)


def clock_pseudo_geometry(ts: np.ndarray, day_start: str = "06:15", day_end: str = "20:00"):
    """缺坐标站的退化几何：时段窗内 elev=+10/ghi=800，窗外 elev=-10/ghi=0。"""
    shape = ts.shape
    t = pd.DatetimeIndex(pd.to_datetime(ts.reshape(-1)))
    tod = t.hour * 60 + t.minute
    lo = dt.time.fromisoformat(day_start); hi = dt.time.fromisoformat(day_end)
    inside = (tod >= lo.hour * 60 + lo.minute) & (tod <= hi.hour * 60 + hi.minute)
    elev = np.where(inside, 10.0, -10.0)
    ghi = np.where(inside, 800.0, 0.0)
    return elev.reshape(shape), ghi.reshape(shape)
