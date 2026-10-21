"""Forecast issue selection and station-day sampling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

SHORT_ISSUE_TIME = "10:00"


def row_times(ds, ts_col: str) -> pd.Series:
    """UnifiedDataset 每行的起报时刻，取到秒（与 evaluate_model 从 epoch 秒还原出的时刻一致）。"""
    ts = (
        np.asarray(ds.column(ts_col))
        if ds.backend == "numpy"
        else ds._ts_as_datetime64(ts_col)
    )
    return pd.Series(np.asarray(ts).astype("datetime64[ns]")).dt.floor("s")


def row_stations(ds, station_col: str) -> np.ndarray:
    """UnifiedDataset 每行的整数站号；数据没有站号列时全部记为站 0。"""
    known = (
        set(ds.timestamp_cols)
        | set(ds.entity_cols)
        | set(ds.feature_cols)
        | set(ds.label_cols)
    )
    if station_col not in known:
        return np.zeros(len(ds), np.int64)
    return np.asarray(ds.column(station_col)).astype(np.int64)


def scored_row_index(
    ds, station_col: str, ts_col: str, issue_time: str = SHORT_ISSUE_TIME
) -> np.ndarray:
    """short_d1_daily 每站每个起报日只用一行：当日 timestamp_win <= issue_time 的最晚一行。返回这些行在 ds 里的
    下标（升序）。其余行不影响分数，不必推理。"""
    ts, st = row_times(ds, ts_col), row_stations(ds, station_col)
    day = ts.dt.normalize()
    ok = ts.notna() & ((ts - day) <= pd.Timedelta(f"{issue_time}:00"))
    f = pd.DataFrame({"i": np.arange(len(ts)), "st": st, "day": day, "ts": ts})[
        ok.to_numpy()
    ]
    f = f.sort_values(["ts", "i"], kind="stable").drop_duplicates(
        ["st", "ts"], keep="first"
    )
    return np.sort(f.groupby(["st", "day"], sort=False)["i"].last().to_numpy())


def issue_grid_row_index(ds, ts_col: str, every_min: int) -> np.ndarray:
    """起报时刻落在当日 every_min 分钟网格上的行（every_min=60：每小时整点一行；15：全部行）。返回下标（升序）。"""
    ts = row_times(ds, ts_col)
    minute = (ts - ts.dt.normalize()) // pd.Timedelta(minutes=1)
    return np.flatnonzero((ts.notna() & (minute % int(every_min) == 0)).to_numpy())


class StationDaySampler(torch.utils.data.Sampler):
    """训练行采样：每个（站, 起报日）被抽中的概率相同，抽中后在该站日的行里均匀取一行（有放回），每轮抽 len(ds) 次。
    序列只由 seed 与轮次决定：多卡时每个进程生成同一序列，再由 accelerate 按批分给各卡。"""

    def __init__(self, ds, station_col: str, ts_col: str, seed: int = 0):
        day = row_times(ds, ts_col).dt.normalize()
        grp = (
            pd.DataFrame({"st": row_stations(ds, station_col), "day": day})
            .groupby(["st", "day"], sort=False)
            .ngroup()
        )
        grp = grp.to_numpy()
        self.weights = torch.as_tensor(1.0 / np.bincount(grp)[grp], dtype=torch.double)
        self.n_groups = int(grp.max()) + 1 if len(grp) else 0
        self.seed, self.epoch = int(seed), 0

    def __len__(self) -> int:
        return len(self.weights)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        yield from torch.multinomial(
            self.weights, len(self.weights), replacement=True, generator=g
        ).tolist()
