# forecast_backbones/m8/_parquet.py
"""M8 自带的拼表与站号归一（2026-10-14 从 pipelines.time_splits 搬来，行为相同；m8 训练不再依赖切分工具）：
  concat_parquets：文件夹里一站一个的 parquet 拼成一张表，各文件列集合必须一致，站标识列类型不一致时统一成文本站号；
  station_keys：站标识归一成文本（1001 / 1001.0 / '1001.0' 都成 '1001'）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

STATION_COLS = ("station_id", "station")


def _arrow_kind(t) -> str:
    """parquet 列类型归大类：同一大类里的差别（int32/float64、list<float>/list<double>）pandas 拼接能兼容。"""
    import pyarrow as pa
    if pa.types.is_null(t):
        return "null"
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return "文本"
    if pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_boolean(t) or pa.types.is_decimal(t):
        return "数值"
    if pa.types.is_timestamp(t) or pa.types.is_date(t):
        return "时间"
    if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
        return "数组"
    return str(t)


def station_keys(values) -> np.ndarray:
    """站标识归一成字符串：数值型且全是整数（含 1001.0 这种被物化成浮点的 id）→ '1001'；其他 → str。"""
    arr = np.asarray(values)
    if arr.dtype.kind in "iu":
        return arr.astype(np.int64).astype(str)
    if arr.dtype.kind == "f":
        if np.isnan(arr).any():
            raise ValueError("站标识列含 NaN")
        if np.all(arr == np.round(arr)):
            return arr.astype(np.int64).astype(str)
        return arr.astype(str)
    out = arr.astype(str)
    # 字符串形态的 '1001.0' 同样归一，与数值列保持一致
    fixed = []
    for x in out.tolist():
        try:
            f = float(x)
            fixed.append(str(int(f)) if f == int(f) and x.strip() != "" else x)
        except ValueError:
            fixed.append(x)
    return np.asarray(fixed, dtype=str)


def concat_parquets(paths, station_col=None) -> pd.DataFrame:
    """读若干 parquet 拼成一张表（一站一个文件时用）。各文件列集合必须一致，否则报错（不静默补 NaN）。
    同一列在各文件里类型大类不一致时：站标识列（station_col，缺省 station_id / station）统一成文本站号
    （station_keys：1001 与 1001.0 都成 '1001'）再拼；其他列直接报错并指明列与文件，
    否则 pandas 拼出混类型的 object 列，落盘时 pyarrow 才报 ArrowTypeError。"""
    paths = [str(x) for x in paths]
    schemas = [pq.read_schema(p) for p in paths]
    cols0 = set(schemas[0].names)
    for pth, sc in zip(paths[1:], schemas[1:]):
        if set(sc.names) != cols0:
            raise ValueError(f"{pth} 的列与 {paths[0]} 不一致：多出 {sorted(set(sc.names) - cols0)}，"
                             f"缺少 {sorted(cols0 - set(sc.names))}")
    frames = [pd.read_parquet(p) for p in paths]
    if len(frames) == 1:
        return frames[0]
    station_like = {station_col} if station_col is not None else set(STATION_COLS)
    for c in schemas[0].names:
        kinds: dict = {}
        for pth, sc in zip(paths, schemas):
            k = _arrow_kind(sc.field(c).type)
            if k != "null":
                kinds.setdefault(k, []).append(pth)
        if len(kinds) <= 1:
            continue
        detail = "；".join(f"{k} {len(v)} 个文件，如 {Path(v[0]).name}" for k, v in kinds.items())
        if c in station_like and set(kinds) <= {"文本", "数值"}:
            print(f"[m8] 站标识列 {c!r} 在各文件里类型不一致（{detail}）：统一转成文本站号后再拼接")
            frames = [f.assign(**{c: station_keys(f[c].to_numpy())}) for f in frames]
            continue
        raise ValueError(f"列 {c!r} 在各文件里类型不一致（{detail}）：请先把这一列在所有文件里统一成同一种类型")
    return pd.concat(frames, ignore_index=True)
