# forecast_backbones/m5/solar.py（2026-10-13 起自带一份，不再引用 m6）
"""站表（station_meta.csv）与太阳几何：时间戳网格、pvlib 晴空 GHI / 太阳高度角、缺坐标站的钟表几何、站名→站号映射、
每站固定容量。纯函数，不依赖 torch。"""
from __future__ import annotations

import datetime as dt
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
    OPTIONAL = ("lat", "lon", "alt_m")          # 映射里写 null = 表里没有这一列（当全缺：缺坐标站退化、海拔 0）

    @classmethod
    def apply_cols(cls, df: pd.DataFrame, cols: dict | None) -> pd.DataFrame:
        """dataset.station_meta_cols = {内部名: 你表里的列名}（2026-10-14）：表头随意，没写的内部名按原名找；
        值为 null 的列视作没有，只允许 lat / lon / alt_m。缺的映射列报错时内部名和表里列名一起给。"""
        if not cols:
            return df
        bad = [k for k in cols if k not in cls.COLS]
        if bad:
            raise ValueError(f"dataset.station_meta_cols 里有不认识的内部名 {bad}；只能是 {list(cls.COLS)}")
        d = df.copy()
        d.columns = [str(c).strip() for c in d.columns]
        absent = [k for k, v in cols.items() if v is None]
        not_opt = [k for k in absent if k not in cls.OPTIONAL]
        if not_opt:
            raise ValueError(f"dataset.station_meta_cols 里 {not_opt} 不能为 null：只有 {list(cls.OPTIONAL)} 可以没有")
        rename = {str(v).strip(): k for k, v in cols.items() if v is not None}
        missing = [f"{k}（映射到 {v}）" for v, k in rename.items() if v not in d.columns]
        if missing:
            raise KeyError(f"station_meta 缺列: {missing}")
        clash = [k for src, k in rename.items() if k in d.columns and src != k] + [k for k in absent if k in d.columns]
        d = d.drop(columns=clash).rename(columns=rename)
        for k in absent:
            d[k] = np.nan
        return d

    def __init__(self, df: pd.DataFrame, cols: dict | None = None):
        df = self.apply_cols(df, cols)
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
    def from_csv(cls, path: str | Path, cols: dict | None = None) -> "StationMeta":
        """UTF-8（含 Excel 的带 BOM 版）或 GBK（中文 Windows 下 Excel 另存的 CSV）都能读。cols 见 apply_cols。"""
        name_col = str((cols or {}).get("station") or "station").strip()
        try:
            df = pd.read_csv(path, dtype={name_col: str}, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path, dtype={name_col: str}, encoding="gbk")
        df.columns = [str(c).strip() for c in df.columns]
        return cls(df, cols)

    def get(self, station_id) -> dict | None:
        return self._rows.get(int(station_id))

    def ids(self) -> list[int]:
        return sorted(self._rows)

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
    名字在表里查不到、或两列都没有，直接报错（M6 的校准分层按 station_id 走，不能猜）。"""
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




def station_cap_mw(meta: StationMeta, station_id) -> float:
    """每站一个固定容量：只取 station_meta 的 cap_mw。数据里的 cap_power_on 是随时间变的开机容量
    （白马湖并网首月 8.88 → 68），训练 / 验证 / 测试一律不用它，模型归一化与指标分母才是同一个数。"""
    row = meta.get(int(station_id)) or {}
    cap = float(row.get("cap_mw", np.nan))
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError(f"station_meta 里没有 station_id={int(station_id)} 的 cap_mw（或非正）；容量只来自站表，请补 station_meta.csv")
    return cap


