# forecast_backbones/m5/data.py（2026-10-13 起自带一份，不再引用 m6）
"""训练 parquet（文件夹里的若干文件，或切分目录里的 train/valid.parquet）→ 按 M5 口径建 UnifiedDataset（arrow 后端）。
拼表与站号归一自带（_parquet.py），不依赖 pipelines.time_splits。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .solar import attach_station_id
    from ._parquet import station_keys, concat_parquets, _arrow_kind
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from forecast_backbones.m5.solar import attach_station_id
    from forecast_backbones.m5._parquet import station_keys, concat_parquets, _arrow_kind
try:
    from ...pipelines.transformer_dataset import UnifiedDataset
except ImportError:
    from pipelines.transformer_dataset import UnifiedDataset


def _dataset_with_station_id(paths, cfg: dict, station_meta, scratch_dir, reuse: bool = False) -> UnifiedDataset:
    """建 UnifiedDataset（arrow 后端）。数据没有 station_id 列而只有站名列时，按 station_meta 映射补上并落到 scratch_dir 下的
    临时 parquet 再装载（arrow 后端不能就地加列）。reuse=True：临时 parquet 已存在就直接读（多卡时主进程写好，其余进程不再转换）。"""
    ds_cfg = cfg["dataset"]; id_col = str(ds_cfg["station_id_col"])
    import pyarrow.parquet as pq
    schemas = [pq.read_schema(p) for p in paths]
    cols = schemas[0].names
    # 数据自带的 station_id 列是文本（站名，或文本存的数字）：collate 会丢掉文本列，所有行会静默落成站 0，所以先转成整数编号
    text_id = id_col in cols and any(_arrow_kind(sc.field(id_col).type) == "文本" for sc in schemas if id_col in sc.names)
    tmp = Path(scratch_dir) / f"{Path(paths[0]).stem}_with_{id_col}.parquet"
    if (id_col not in cols or text_id) and reuse and tmp.exists():
        paths = [str(tmp)]
    elif id_col not in cols or text_id:
        name_col = id_col if text_id else "station"
        df = concat_parquets(paths, station_col=name_col)
        if text_id:
            vals = pd.Series(station_keys(df[id_col].to_numpy()), index=df.index)
            as_int = pd.to_numeric(vals, errors="coerce")
            if as_int.notna().all():
                df[id_col] = as_int.astype(np.int64)
                how = f"{id_col!r} 列是文本存的整数编号：直接转成整数"
            else:
                df = attach_station_id(df.drop(columns=[id_col]).assign(_station_name=vals), station_meta,
                                       id_col=id_col, name_col="_station_name").drop(columns=["_station_name"])
                how = f"{id_col!r} 列是站名文本：按 station_meta 的 station 列映射成整数编号"
        else:
            df = attach_station_id(df, station_meta, id_col=id_col, name_col="station")
            how = f"数据无 {id_col!r} 列：按 station_meta 从站名映射补上"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(tmp, index=False)
        print(f"[m5] {how}，写到 {tmp}")
        paths = [str(tmp)]
    return UnifiedDataset(paths if len(paths) > 1 else paths[0], timestamp_cols=[ds_cfg["ts_col"]], entity_cols=[],
                          label_cols=[ds_cfg["label_col"]], timestamp_output="epoch_s", epoch_unit="s",
                          for_hf_trainer=True)


def _check_presplit_dir(splits_dir, need_cal: bool = False) -> Path:
    d = Path(splits_dir)
    need = ["train.parquet", "valid.parquet", "split_manifest.json"] + (["cal.parquet"] if need_cal else [])
    missing = [f for f in need if not (d / f).exists()]
    if missing:
        raise FileNotFoundError(f"{d} 缺 {missing}")
    return d
