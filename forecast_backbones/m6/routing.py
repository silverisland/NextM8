# -*- coding: utf-8 -*-
# forecast_backbones/m6/routing.py
"""站级回退路由：训后逐站比较 LoRA 与零样本（各自校准后的南网短期准确率），输的站回退零样本。
表：{station_id: "lora" | "zeroshot"}，JSON 落盘（键为字符串），推理时按行选路。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROUTES = ("lora", "zeroshot")


def decide_routing(table: pd.DataFrame, min_gain_pct: float = 0.0) -> dict[int, str]:
    out: dict[int, str] = {}
    for r in table.itertuples(index=False):
        zs, lo = float(r.acc_zeroshot), float(r.acc_lora)
        ok = np.isfinite(zs) and np.isfinite(lo) and (lo - zs) > float(min_gain_pct)
        out[int(r.station_id)] = "lora" if ok else "zeroshot"
    return out


def save_routing(path, routing: dict[int, str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({str(int(k)): v for k, v in sorted(routing.items())}, indent=2, ensure_ascii=False), encoding="utf-8")


def load_routing(path) -> dict[int, str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for k, v in raw.items():
        if v not in ROUTES:
            raise ValueError(f"station_routing[{k}]={v!r} 不在 {ROUTES}")
        out[int(k)] = v
    return out
