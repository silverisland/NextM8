# forecast_backbones/m6/features.py
"""M6 特征层：时间戳、太阳几何（pvlib）、站元数据、多变度信号、RowContext。纯函数，不依赖 torch。"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib


def window_timestamps(ts_win_epoch_s, context_len: int, pred_len: int, step_min: int = 15) -> np.ndarray:
    """[B] epoch 秒 → [B, context_len+pred_len] datetime64[s]（naive 本地墙钟）。
    历史段最后一点 = 起报时刻 S；未来第 i 点 = S + (i+1)·step（与 nanwang_metric 对齐）。"""
    s = np.asarray(ts_win_epoch_s, dtype=np.int64).reshape(-1, 1)
    offs = np.concatenate([np.arange(-(context_len - 1), 1), np.arange(1, pred_len + 1)]) * (step_min * 60)
    return (s + offs[None, :]).astype("datetime64[s]")


def solar_geometry(ts: np.ndarray, lat: float, lon: float, alt_m: float, tz: str):
    """ts: datetime64 数组（任意形状，naive 本地墙钟）→ (elev_deg, ghi_clear W/m²)，同形状。
    夜间（elev<=0）ghi 置 0。Ineichen 需 Linke 浊度表（pvlib 自带），失败退到 Haurwitz。"""
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


class StationMeta:
    """station_meta.csv：station_id, station, lat, lon, alt_m, cap_mw。缺值为 NaN。"""

    COLS = ("station_id", "station", "lat", "lon", "alt_m", "cap_mw")

    def __init__(self, df: pd.DataFrame):
        missing = [c for c in self.COLS if c not in df.columns]
        if missing:
            raise KeyError(f"station_meta 缺列: {missing}")
        d = df.copy()
        d["station_id"] = d["station_id"].astype(int)
        for c in ("lat", "lon", "alt_m", "cap_mw"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d["alt_m"] = d["alt_m"].fillna(0.0)
        self._rows = {int(r.station_id): r._asdict() for r in d.itertuples(index=False)}

    @classmethod
    def from_csv(cls, path: str | Path) -> "StationMeta":
        return cls(pd.read_csv(path))

    def get(self, station_id) -> dict | None:
        return self._rows.get(int(station_id))

    def has_coords(self, station_id) -> bool:
        r = self.get(station_id)
        return r is not None and np.isfinite(r["lat"]) and np.isfinite(r["lon"])

    def name(self, station_id) -> str:
        r = self.get(station_id)
        return str(r["station"]) if r is not None and isinstance(r["station"], str) else str(int(station_id))
