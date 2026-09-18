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


def _day_mean(values: np.ndarray, ghi_clear: np.ndarray, min_clear: float) -> np.ndarray:
    """按行对白天步取均值；无白天步 → NaN。"""
    day = ghi_clear > min_clear
    num = np.where(day, values, 0.0).sum(axis=1)
    cnt = day.sum(axis=1)
    return np.where(cnt > 0, num / np.maximum(cnt, 1), np.nan)


def k_nwp_signal(ghi_pred, ghi_clear_future, min_clear: float = 50.0) -> np.ndarray:
    ghi_pred = np.asarray(ghi_pred, float); gc = np.asarray(ghi_clear_future, float)
    k = np.where(gc > min_clear, ghi_pred / np.maximum(gc, 1e-6), 0.0)
    return _day_mean(k, gc, min_clear)


def spread_signal(members_pred, ghi_clear_future, min_clear: float = 50.0) -> np.ndarray:
    m = np.asarray(members_pred, float); gc = np.asarray(ghi_clear_future, float)
    sd = m.std(axis=1)                                          # [B,H]
    return _day_mean(sd / np.maximum(gc, 1e-6), gc, min_clear)


def disagree_signal(ghi_a, ghi_b, ghi_clear_future, min_clear: float = 50.0) -> np.ndarray:
    a = np.asarray(ghi_a, float); b = np.asarray(ghi_b, float); gc = np.asarray(ghi_clear_future, float)
    return _day_mean(np.abs(a - b) / np.maximum(gc, 1e-6), gc, min_clear)


def recent_kt_std_signal(power_hist, p_clear_hist, last_steps: int = 192) -> np.ndarray:
    p = np.asarray(power_hist, float)[:, -last_steps:]; pc = np.asarray(p_clear_hist, float)[:, -last_steps:]
    day = pc > 0
    kt = np.where(day, p / np.maximum(pc, 1e-6), np.nan)
    with np.errstate(all="ignore"):
        return np.nanstd(kt, axis=1)


def variability_bins(signal, edges=None, n_bins: int = 3) -> np.ndarray:
    s = np.asarray(signal, float)
    if edges is None:
        qs = np.nanquantile(s, np.linspace(0, 1, n_bins + 1)[1:-1]) if np.isfinite(s).any() else []
        edges = list(np.atleast_1d(qs))
    b = np.digitize(s, np.asarray(edges, float))          # NaN 落到最后一档，下一行改写成中档
    mid = n_bins // 2
    return np.where(np.isfinite(s), b, mid).astype(int)


try:
    from .masks import night_from_elevation
except ImportError:
    from masks import night_from_elevation


def _to_np(v) -> np.ndarray:
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    return np.asarray(v)


def _scalar_per_row(v) -> np.ndarray:
    a = _to_np(v).astype(float)
    return a[:, 0] if a.ndim == 2 else a.reshape(-1)


@dataclass
class RowContext:
    station_id: np.ndarray
    cap: np.ndarray
    ts: np.ndarray
    elev_deg: np.ndarray
    ghi_clear: np.ndarray
    p_clear: np.ndarray
    night_future: np.ndarray
    k_steps: np.ndarray
    var_signal: np.ndarray
    var_bin: np.ndarray
    degraded: np.ndarray


def build_row_context(batch_dict: dict, cfg: dict, station_meta: StationMeta) -> RowContext:
    ds = cfg["dataset"]; t3 = cfg["timesfm3"]; L = int(t3["context_len"]); H = int(t3["pred_len"])
    ts_epoch = _to_np(batch_dict[ds["ts_col"]]).astype(np.int64).reshape(-1)
    B = ts_epoch.shape[0]
    sid = _to_np(batch_dict[ds["station_id_col"]]).astype(int).reshape(-1) if ds["station_id_col"] in batch_dict \
        else np.zeros(B, int)
    if ds["cap_col"] in batch_dict:
        cap = _scalar_per_row(batch_dict[ds["cap_col"]])
    else:
        cap = np.array([(station_meta.get(s) or {}).get("cap_mw", np.nan) for s in sid], float)
    ts = window_timestamps(ts_epoch, L, H, int(ds["step_min"]))
    elev = np.empty((B, L + H)); ghi = np.empty((B, L + H)); degraded = np.zeros(B, bool)
    for s in np.unique(sid):
        rows = np.flatnonzero(sid == s)
        if station_meta.has_coords(s):
            m = station_meta.get(s)
            e, g = solar_geometry(ts[rows], m["lat"], m["lon"], m["alt_m"], ds["tz"])
        else:
            e, g = clock_pseudo_geometry(ts[rows]); degraded[rows] = True
        elev[rows] = e; ghi[rows] = g
    p_clear = np.where(degraded[:, None], np.where(ghi > 0, cap[:, None], 0.0), ghi / 1000.0 * cap[:, None])
    night_future = night_from_elevation(elev[:, L:])
    # 多变度信号
    vcfg = cfg["variability"]; fcfg = cfg["features"]
    ghi_pred_col = fcfg["past_future"][0][1]
    ghi_pred = _to_np(batch_dict[ghi_pred_col]).astype(float)[:, :H] if ghi_pred_col in batch_dict else np.zeros((B, H))
    gc_fut = ghi[:, L:]
    k_steps = np.where(gc_fut > 50.0, ghi_pred / np.maximum(gc_fut, 1e-6), 0.0)
    kind = vcfg["signal"]
    if kind == "nwp_spread" and fcfg["nwp_members"]["enabled"]:
        suf = fcfg["nwp_members"]["future_suffix"]
        members = np.stack([_to_np(batch_dict[c + suf]).astype(float)[:, :H] for c in fcfg["nwp_members"]["cols"]], axis=1)
        sig = spread_signal(members, gc_fut)
    elif kind == "src_disagree" and len(fcfg["past_future"]) >= 3 and fcfg["past_future"][2][1] in batch_dict:
        sig = disagree_signal(ghi_pred, _to_np(batch_dict[fcfg["past_future"][2][1]]).astype(float)[:, :H], gc_fut)
    elif kind == "recent_kt_std":
        hist = _to_np(batch_dict[ds["pv_col"]]).astype(float)[:, -L:]
        sig = recent_kt_std_signal(hist, p_clear[:, :L])
    else:
        sig = k_nwp_signal(ghi_pred, gc_fut)
    edges = vcfg.get("k_nwp_edges") if kind == "k_nwp" else None
    vbin = variability_bins(sig, edges=edges, n_bins=int(vcfg["bins"]))
    return RowContext(station_id=sid, cap=cap, ts=ts, elev_deg=elev, ghi_clear=ghi, p_clear=p_clear,
                      night_future=night_future, k_steps=k_steps, var_signal=sig, var_bin=vbin, degraded=degraded)
