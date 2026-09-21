# forecast_backbones/m6/features.py
"""M6 特征层：时间戳、太阳几何（pvlib）、站元数据、多变度信号、RowContext。纯函数，不依赖 torch。"""
from __future__ import annotations

import datetime as dt
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

# 已经就哪些站发过“缺坐标退化几何”警告——跨调用去重，同一进程内每个站只提示一次。
_DEGRADED_WARNED: set[int] = set()


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


def norm_station_name(x):
    """站名归一成文本：去首尾空格；1002、1002.0、"1002.0" 都成 "1002"；缺值返回 None。"""
    if x is None or (not isinstance(x, str) and pd.isna(x)):
        return None
    t = str(x).strip()
    if t == "":
        return None
    try:
        f = float(t)
        if np.isfinite(f) and f == int(f):
            return str(int(f))
    except ValueError:
        pass
    return t


class StationMeta:
    """station_meta.csv：station_id, station, lat, lon, alt_m, cap_mw（列序随意）。缺值为 NaN。
    station_id 是 M6 内部整数编号；station 是数据站列里的值。按站查编号见 resolve：先按 station 列的站名找，
    找不到且是纯数字时按 station_id 找。"""

    COLS = ("station_id", "station", "lat", "lon", "alt_m", "cap_mw")

    def __init__(self, df: pd.DataFrame):
        missing = [c for c in self.COLS if c not in df.columns]
        if missing:
            raise KeyError(f"station_meta 缺列: {missing}")
        d = df.copy()
        ids = pd.to_numeric(d["station_id"], errors="coerce")
        bad = d.loc[ids.isna() | (ids != ids.round()), "station_id"].astype(str).tolist()
        if bad:
            raise ValueError(f"station_meta 的 station_id 必须是整数编号，这些不是：{bad[:10]}。"
                             f"站名（数据站列里的值，如 雅砻江、1001）写在 station 列；station_id 自己编一个不重复的整数即可")
        d["station_id"] = ids.astype(int)
        # 站名一律按文本：去空格；全是数字的站名（如 1001）被读成整数/浮点或写成 1001.0 时归一成 '1001'
        d["station"] = d["station"].map(norm_station_name)
        for c in ("lat", "lon", "alt_m", "cap_mw"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d["alt_m"] = d["alt_m"].fillna(0.0)
        dup = sorted(d.loc[d["station_id"].duplicated(), "station_id"].astype(int).unique().tolist())
        if dup:
            raise ValueError(f"station_meta 里 station_id 重复：{dup}")
        names = d["station"].dropna()
        dup = sorted(names[names.duplicated()].astype(str).unique().tolist())
        if dup:
            raise ValueError(f"station_meta 里 station 重复：{dup}（一个站名只能对应一个 station_id）")
        self._rows = {int(r.station_id): r._asdict() for r in d.itertuples(index=False)}
        self._by_name = {str(r["station"]): sid for sid, r in self._rows.items() if isinstance(r["station"], str)}

    @classmethod
    def from_csv(cls, path: str | Path) -> "StationMeta":
        """UTF-8（含 Excel 的带 BOM 版）或 GBK（中文 Windows 下 Excel 另存的 CSV）都能读。"""
        try:
            df = pd.read_csv(path, dtype={"station": str}, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path, dtype={"station": str}, encoding="gbk")
        df.columns = [str(c).strip() for c in df.columns]
        return cls(df)

    def get(self, station_id) -> dict | None:
        return self._rows.get(int(station_id))

    def has_coords(self, station_id) -> bool:
        r = self.get(station_id)
        return r is not None and np.isfinite(r["lat"]) and np.isfinite(r["lon"])

    def name(self, station_id) -> str:
        r = self.get(station_id)
        return str(r["station"]) if r is not None and isinstance(r["station"], str) else str(int(station_id))

    def resolve(self, values) -> tuple[dict, list, list]:
        """数据站列的取值 → station_id。返回 (映射 {归一后的值: id}, 找不到的值, 冲突说明)。
        先按 station 列的站名找；找不到且是纯数字时按 station_id 找。若那个编号所在行的站名也出现在数据里
        （即它已被另一个站按名认领），记为冲突，不静默合并两个站。"""
        vals = sorted({v for v in (norm_station_name(x) for x in values) if v is not None})
        claimed = {self._by_name[v]: v for v in vals if v in self._by_name}
        mapping, unknown, conflicts = {}, [], []
        for v in vals:
            if v in self._by_name:
                mapping[v] = self._by_name[v]
            elif v.lstrip("-").isdigit() and int(v) in self._rows:
                sid = int(v)
                if sid in claimed:
                    conflicts.append(f"数据里的站 {v!r} 按编号对上 station_id={sid}，但这个编号已属于站名 {claimed[sid]!r}"
                                     f"：请给 {claimed[sid]!r} 换一个不重复的 station_id")
                else:
                    mapping[v] = sid
            else:
                unknown.append(v)
        return mapping, unknown, conflicts

    def id_of(self, station_name) -> int | None:
        """站名（或纯数字编号）→ station_id；查不到或有冲突返回 None。"""
        mapping, _, _ = self.resolve([station_name])
        return mapping.get(norm_station_name(station_name))

    def ids(self) -> list[int]:
        return sorted(self._rows)


def attach_station_id(df: pd.DataFrame, meta: StationMeta, id_col: str = "station_id", name_col: str = "station") -> pd.DataFrame:
    """给数据表补 station_id：已有 id_col 原样返回；只有站名列 name_col 时按 station_meta 映射；
    名字在表里查不到、或两列都没有，直接报错（M6 的校准分层与路由都按 station_id 走，不能猜）。"""
    if id_col in df.columns:
        return df
    if name_col not in df.columns:
        raise ValueError(f"数据里既没有 {id_col!r} 也没有 {name_col!r}，无法确定站点（列：{list(df.columns)[:12]}…）")
    names = df[name_col].map(norm_station_name)
    mapping, unknown, conflicts = meta.resolve(names.dropna().unique())
    if conflicts:
        raise ValueError("站点编号冲突：" + "；".join(conflicts))
    if names.isna().any():
        unknown = unknown + ["<空站名>"]
    if unknown:
        raise ValueError(f"这些站在 station_meta 里找不到：{unknown}。查找规则：先按 station 列的站名找，"
                         f"找不到且是纯数字时按 station_id 列找；请在 station_meta.csv 补上这些站")
    return df.assign(**{id_col: names.map(mapping).astype(np.int64)})


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
    from ._arrays import interp_nan
except ImportError:
    from masks import night_from_elevation
    from _arrays import interp_nan


def _to_np(v) -> np.ndarray:
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    return np.asarray(v)


def _scalar_per_row(v) -> np.ndarray:
    """多列（逐时刻）形式先按 _col 同款 NaN 插值再取第 0 列；本就是逐行标量的形式不插值
    （单点没有邻居可插，NaN 会被上游的 cap 非有限性检查捕获）。"""
    a = _to_np(v).astype(float)
    if a.ndim == 2:
        return interp_nan(a)[:, 0]
    return a.reshape(-1)


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
    bad_cap = ~np.isfinite(cap)
    if bad_cap.any():
        bad_ids = sorted(int(s) for s in np.unique(sid[bad_cap]))
        raise ValueError(f"cap 非有限（station_meta 与 {ds['cap_col']!r} 均无有效值）：station_id={bad_ids}")
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
    if degraded.any():
        degraded_ids = sorted(int(s) for s in np.unique(sid[degraded]))
        new_ids = [s for s in degraded_ids if s not in _DEGRADED_WARNED]
        if new_ids:
            warnings.warn(
                f"station(s) without coordinates in station_meta fell back to clock_pseudo_geometry "
                f"(degraded geometry) for station_id={new_ids}: p_clear = nameplate capacity, "
                f"elev = ±10, and calibration elevation strata collapse for these stations"
            )
            _DEGRADED_WARNED.update(new_ids)
    p_clear = np.where(degraded[:, None], np.where(ghi > 0, cap[:, None], 0.0), ghi / 1000.0 * cap[:, None])
    night_future = night_from_elevation(elev[:, L:])
    # 多变度信号
    vcfg = cfg["variability"]; fcfg = cfg["features"]
    ghi_pred_col = vcfg["ghi_col"]
    if ghi_pred_col not in batch_dict:
        raise ValueError(f"variability.ghi_col={ghi_pred_col!r} 不在 batch_dict 里（无零值兜底）")
    ghi_pred = interp_nan(_to_np(batch_dict[ghi_pred_col]).astype(float))[:, :H]
    gc_fut = ghi[:, L:]
    k_steps = np.where(gc_fut > 50.0, ghi_pred / np.maximum(gc_fut, 1e-6), 0.0)
    kind = vcfg["signal"]
    if kind == "nwp_spread":
        if not fcfg["nwp_members"]["enabled"]:
            raise ValueError("variability.signal=nwp_spread requires features.nwp_members.enabled=true")
        suf = fcfg["nwp_members"]["future_suffix"]
        members = np.stack([_to_np(batch_dict[c + suf]).astype(float)[:, :H] for c in fcfg["nwp_members"]["cols"]], axis=1)
        sig = spread_signal(members, gc_fut)
    elif kind == "src_disagree":
        if not (len(fcfg["past_future"]) >= 3 and fcfg["past_future"][2][1] in batch_dict):
            raise ValueError("variability.signal=src_disagree requires a third features.past_future column present in batch_dict")
        sig = disagree_signal(ghi_pred, _to_np(batch_dict[fcfg["past_future"][2][1]]).astype(float)[:, :H], gc_fut)
    elif kind == "recent_kt_std":
        hist = interp_nan(_to_np(batch_dict[ds["pv_col"]]).astype(float))[:, -L:]
        sig = recent_kt_std_signal(hist, p_clear[:, :L])
    elif kind == "k_nwp":
        sig = k_nwp_signal(ghi_pred, gc_fut)
    else:
        raise ValueError(f"variability.signal 未知取值: {kind!r}")
    edges = vcfg.get(f"{kind}_edges")
    if edges is None:
        raise ValueError(f"variability.{kind}_edges must be set (derive once on the calibration set); "
                          f"data-derived edges are not allowed at inference")
    vbin = variability_bins(sig, edges=edges, n_bins=int(vcfg["bins"]))
    return RowContext(station_id=sid, cap=cap, ts=ts, elev_deg=elev, ghi_clear=ghi, p_clear=p_clear,
                      night_future=night_future, k_steps=k_steps, var_signal=sig, var_bin=vbin, degraded=degraded)
