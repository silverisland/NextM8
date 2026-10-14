# forecast_backbones/m5/_arrays.py
"""M5 内部的小型数组工具（自带一份，不再引用 m6）。"""
from __future__ import annotations

import numpy as np


def interp_nan(arr: np.ndarray) -> np.ndarray:
    """[B,T] 逐行沿时间轴（最后一维）线性插值填补非有限值；一整行都非有限时该行填 0。"""
    out = np.array(arr, dtype=float, copy=True)
    idx = np.arange(out.shape[1])
    for i in range(out.shape[0]):
        row = out[i]; m = np.isfinite(row)
        if m.all():
            continue
        if not m.any():
            out[i] = 0.0; continue
        row[~m] = np.interp(idx[~m], idx[m], row[m])
    return out
