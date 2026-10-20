# forecast_backbones/m5/data.py（2026-10-13 起自带一份，不再引用 m6）
"""训练 parquet（文件夹里的若干文件，或切分目录里的 train/valid.parquet）→ 按 M5 口径建 UnifiedDataset（arrow 后端）。
拼表与站号归一自带（_parquet.py），不依赖 pipelines.time_splits。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .solar import attach_station_id, norm_station_name
    from ._parquet import station_keys, concat_parquets, _arrow_kind
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from forecast_backbones.m5.solar import attach_station_id, norm_station_name
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


def unified_feature_cols(model, columns) -> list[str]:
    """南网训练入口（train_shell/m5_train_cli.py）建 UnifiedDataset 用的 feature_cols：配置要的列（历史功率、past_future 对、
    past_only、包络开时的 GHI 列对）按 model.resolve_col（精确 → col_aliases → 大小写不敏感）对到数据里的列名，数据里没有的跳过
    （NWP 对缺列 = 不喂这一对）。历史功率列缺了直接报错。"""
    columns = [str(c) for c in columns]
    if model.resolve_col(columns, model.target_col) is None:
        raise KeyError(f"数据里没有历史功率列 {model.target_col!r}（dataset.pv_col）；数据列 {sorted(columns)}")
    fe = model.cfg["features"]
    want = [model.target_col] + [c for pair in model.past_future_cols for c in pair] + list(model.past_only_cols)
    if fe["derived"].get("history_envelope") or fe["derived"].get("envelope_nwp"):
        want += list(fe["envelope"]["ghi"])
    out: list[str] = []
    for c in want:
        k = model.resolve_col(columns, c)
        if k is not None and k not in out:
            out.append(k)
    return out


def station_ids_from_names(station_meta, names) -> list[int]:
    """数据 station 列（站名）→ 训练集里出现过的整数站号（升序）；站表里找不到或有冲突直接报错（与 rowctx 同一映射）。"""
    names = [n for n in dict.fromkeys(names) if norm_station_name(n) is not None]
    mapping, unknown, conflicts = station_meta.resolve(names)
    if unknown or conflicts:
        raise ValueError(f"站名映射失败：station_meta 里找不到 {unknown[:10]} {'；'.join(conflicts)}"
                         f"（station_meta 的 station 列要写数据里的站名，或换站表）")
    return sorted({int(mapping[norm_station_name(n)]) for n in names})


def _check_presplit_dir(splits_dir, need_cal: bool = False) -> Path:
    d = Path(splits_dir)
    need = ["train.parquet", "valid.parquet", "split_manifest.json"] + (["cal.parquet"] if need_cal else [])
    missing = [f for f in need if not (d / f).exists()]
    if missing:
        raise FileNotFoundError(f"{d} 缺 {missing}")
    return d
