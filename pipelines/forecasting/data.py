"""Public dataset loaders for presplit, refit and partitioned parquet inputs."""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.transformer_dataset import UnifiedDataset

from .parquet import arrow_kind, concat_parquets, station_keys
from .stations import attach_station_id


def dataset_with_station_id(
    paths, cfg: dict, station_meta, scratch_dir, reuse: bool = False
) -> UnifiedDataset:
    """建 UnifiedDataset（arrow 后端）。数据没有 station_id 列而只有站名列时，按 station_meta 映射补上并落到 scratch_dir 下的
    临时 parquet 再装载（arrow 后端不能就地加列）。reuse=True：临时 parquet 已存在就直接读（多卡时主进程写好，其余进程不再转换）。"""
    ds_cfg = cfg["dataset"]
    id_col = str(ds_cfg["station_id_col"])
    import pyarrow.parquet as pq

    schemas = [pq.read_schema(p) for p in paths]
    cols = schemas[0].names
    # 数据自带的 station_id 列是文本（站名，或文本存的数字）：collate 会丢掉文本列，所有行会静默落成站 0，所以先转成整数编号
    text_id = id_col in cols and any(
        arrow_kind(sc.field(id_col).type) == "文本"
        for sc in schemas
        if id_col in sc.names
    )
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
                df = attach_station_id(
                    df.drop(columns=[id_col]).assign(_station_name=vals),
                    station_meta,
                    id_col=id_col,
                    name_col="_station_name",
                ).drop(columns=["_station_name"])
                how = f"{id_col!r} 列是站名文本：按 station_meta 的 station 列映射成整数编号"
        else:
            df = attach_station_id(df, station_meta, id_col=id_col, name_col="station")
            how = f"数据无 {id_col!r} 列：按 station_meta 从站名映射补上"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(tmp, index=False)
        print(f"[forecast] {how}，写到 {tmp}")
        paths = [str(tmp)]
    return UnifiedDataset(
        paths if len(paths) > 1 else paths[0],
        timestamp_cols=[ds_cfg["ts_col"]],
        entity_cols=[],
        label_cols=[ds_cfg["label_col"]],
        timestamp_output="epoch_s",
        epoch_unit="s",
        for_hf_trainer=True,
    )


def check_presplit_dir(splits_dir, need_cal: bool = False) -> Path:
    d = Path(splits_dir)
    need = ["train.parquet", "valid.parquet", "split_manifest.json"] + (
        ["cal.parquet"] if need_cal else []
    )
    missing = [f for f in need if not (d / f).exists()]
    if missing:
        raise FileNotFoundError(f"{d} 缺 {missing}")
    return d


def load_presplit(
    splits_dir,
    cfg: dict,
    station_meta,
    scratch_dir,
    reuse: bool = False,
    refit: bool = False,
):
    """refit=True：全量重训，train.parquet ∪ valid.parquet 全部当训练行（valid 可以是 0 行），不建验证集（返回 ds_val=None）。"""
    import pyarrow.parquet as pq

    d = check_presplit_dir(splits_dir, need_cal=False)
    man = json.loads((d / "split_manifest.json").read_text(encoding="utf-8"))
    load = functools.partial(
        dataset_with_station_id,
        cfg=cfg,
        station_meta=station_meta,
        scratch_dir=scratch_dir,
        reuse=reuse,
    )
    if refit:
        ds_train = load([str(_refit_rows(d, scratch_dir, reuse))])
        if len(ds_train) == 0:
            raise ValueError(f"{d} 的 train ∪ valid 为空")
        return ds_train, None, man
    if pq.read_metadata(d / "valid.parquet").num_rows == 0:
        raise ValueError(
            f"{d} 的 valid.parquet 为 0 行：这是全量重训切分，训练请加 --refit（不验证、不早停、固定步数）"
        )
    ds_train, ds_val = (
        load([str(d / "train.parquet")]),
        load([str(d / "valid.parquet")]),
    )
    if len(ds_train) == 0 or len(ds_val) == 0:
        raise ValueError(f"{d} 的 train/valid 为空")
    return ds_train, ds_val, man


def expand_parquets(items) -> list[str]:
    """目录 → 其下全部 *.parquet（递归，train/date=2026-06-01/data_0.parquet 这种分区目录也行）；通配符 → 展开；文件 → 原样。
    按路径排序，去重。"""
    import glob

    out: list[str] = []
    for it in items or ():
        p = Path(str(it))
        if p.is_dir():
            out += sorted(
                str(f)
                for f in p.rglob("*.parquet")
                if not any(part.startswith(".") for part in f.relative_to(p).parts)
            )
        else:
            hits = sorted(glob.glob(str(it)))
            if not hits:
                raise FileNotFoundError(f"--train-parquet {it!r} 没有匹配到文件")
            out += hits
    return sorted(dict.fromkeys(out))


def load_full(train_parquet, cfg: dict, station_meta, scratch_dir, reuse: bool = False):
    """全量训练：现成的训练 parquet（文件夹 / 通配符 / 文件列表，一站一个文件也行）全部当训练行，一行不删、没有验证集。
    站列是站名文本时按 station_meta 映射成整数站号（dataset_with_station_id）。返回 (ds_train, None, manifest)。"""
    files = expand_parquets(train_parquet)
    if not files:
        raise FileNotFoundError(
            f"--train-parquet {train_parquet!r} 下没有 parquet 文件"
        )
    ds = dataset_with_station_id(
        files, cfg, station_meta, scratch_dir=scratch_dir, reuse=reuse
    )
    if len(ds) == 0:
        raise ValueError(f"--train-parquet {files} 里没有行")
    man = {
        "generated_by": "pipelines.forecasting.data --train-parquet",
        "plan": None,
        "train_parquet": files,
    }
    return ds, None, man


def _refit_rows(d: Path, scratch_dir, reuse: bool) -> Path:
    """train.parquet ∪ valid.parquet（去掉 split_block 列）→ scratch_dir/refit_rows.parquet；valid 为 0 行时直接用 train.parquet。
    reuse=True（多卡的非主进程）：主进程已写好就直接用。"""
    import pyarrow.parquet as pq

    if pq.read_metadata(d / "valid.parquet").num_rows == 0:
        return d / "train.parquet"
    out = Path(scratch_dir) / "refit_rows.parquet"
    if reuse and out.exists():
        return out
    tr = pd.read_parquet(d / "train.parquet")
    va = pd.read_parquet(d / "valid.parquet").drop(
        columns=["split_block"], errors="ignore"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([tr, va], ignore_index=True).to_parquet(out, index=False)
    return out
