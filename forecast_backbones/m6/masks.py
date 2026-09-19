"""训练损失掩码与提交侧夜间归零共用的规则。1=计入损失。"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd


def night_from_elevation(elev_deg) -> np.ndarray:
    return np.asarray(elev_deg, float) <= 0.0


def night_from_clock(ts, day_start: str = "06:15", day_end: str = "20:00") -> np.ndarray:
    t = pd.DatetimeIndex(pd.to_datetime(np.asarray(ts).reshape(-1)))
    tod = t.hour * 60 + t.minute
    lo = dt.time.fromisoformat(day_start); hi = dt.time.fromisoformat(day_end)
    inside = (tod >= lo.hour * 60 + lo.minute) & (tod <= hi.hour * 60 + hi.minute)
    return (~inside).reshape(np.asarray(ts).shape)


def _min_run_filter(cond: np.ndarray, min_run: int) -> np.ndarray:
    """逐行只保留长度 >= min_run 的连续 True 段。"""
    out = np.zeros_like(cond, dtype=bool)
    for i in range(cond.shape[0]):
        row = cond[i]; j = 0; n = row.shape[0]
        while j < n:
            if row[j]:
                k = j
                while k < n and row[k]:
                    k += 1
                if k - j >= min_run:
                    out[i, j:k] = True
                j = k
            else:
                j += 1
    return out


def curtail_from_rule(y, p_clear, k_steps, plateau_ratio: float = 0.05, k_min: float = 0.7,
                      min_run: int = 4) -> np.ndarray:
    """疑似弃光：出力 < plateau_ratio·晴空 且 NWP 晴空指数 > k_min 且白天，连续 >= min_run 步。"""
    y = np.asarray(y, float); pc = np.asarray(p_clear, float); k = np.asarray(k_steps, float)
    cond = (pc > 0) & (y < plateau_ratio * pc) & (k > k_min)
    return _min_run_filter(cond, min_run)


def missing_mask(y) -> np.ndarray:
    return ~np.isfinite(np.asarray(y, float))


def loss_mask(night, curtail=None, missing=None) -> np.ndarray:
    drop = np.asarray(night, bool).copy()
    if curtail is not None:
        drop |= np.asarray(curtail, bool)
    if missing is not None:
        drop |= np.asarray(missing, bool)
    return (~drop).astype(np.float32)
