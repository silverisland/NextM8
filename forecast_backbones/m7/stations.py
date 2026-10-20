# -*- coding: utf-8 -*-
# forecast_backbones/m7/stations.py（2026-10-20，自 m5/solar.py 复制站表部分；M7 不依赖其它模型包，不需要 pvlib）
"""站表（station_meta.csv / 南网 info.csv）：站名 → 站号映射、每站固定容量 cap_mw。纯 pandas / numpy。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


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
    station_id 是内部整数编号；station 是数据站列里的值。按站查编号见 resolve：先按 station 列的站名找，
    找不到且是纯数字时按 station_id 找。"""

    COLS = ("station_id", "station", "lat", "lon", "alt_m", "cap_mw")
    OPTIONAL = ("lat", "lon", "alt_m")          # 映射里写 null = 表里没有这一列（当全缺：缺坐标站退化、海拔 0）
    # 线上站点信息表 info.csv 的表头（2026-10-20，自 M8）：没给 station_meta_cols、表里又没有标准列时按它自动映射。
    # station = plant_pointname（数据 station 列里的 plant_guangfu1358 这种），容量 GCCAPACITY（MW），没有海拔列
    INFO_CSV_COLS = {"station_id": "plantid", "station": "plant_pointname", "lat": "LATITUDE", "lon": "LONGITUDE",
                     "alt_m": None, "cap_mw": "GCCAPACITY"}

    @classmethod
    def detect_cols(cls, columns, cols: dict | None) -> dict | None:
        """显式映射优先；否则标准列不全而 info.csv 表头齐全时返回 INFO_CSV_COLS。"""
        if cols:
            return cols
        names = {str(c).strip() for c in columns}
        if not set(cls.COLS) <= names and {v for v in cls.INFO_CSV_COLS.values() if v} <= names:
            return dict(cls.INFO_CSV_COLS)
        return cols

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
        df = self.apply_cols(df, self.detect_cols(df.columns, cols))
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
        try:
            head = pd.read_csv(path, nrows=0, encoding="utf-8-sig"); enc = "utf-8-sig"
        except UnicodeDecodeError:
            head = pd.read_csv(path, nrows=0, encoding="gbk"); enc = "gbk"
        name_col = str((cls.detect_cols(head.columns, cols) or {}).get("station") or "station").strip()
        df = pd.read_csv(path, dtype={name_col: str}, encoding=enc)
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


def station_cap_mw(meta: StationMeta, station_id) -> float:
    """每站一个固定容量：只取 station_meta 的 cap_mw。数据里的 cap_power_on 是随时间变的开机容量
    （白马湖并网首月 8.88 → 68），训练 / 验证 / 测试一律不用它，模型归一化与指标分母才是同一个数。"""
    row = meta.get(int(station_id)) or {}
    cap = float(row.get("cap_mw", np.nan))
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError(f"station_meta 里没有 station_id={int(station_id)} 的 cap_mw（或非正）；容量只来自站表，请补 station_meta.csv")
    return cap
