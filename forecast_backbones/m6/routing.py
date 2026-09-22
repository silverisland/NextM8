# -*- coding: utf-8 -*-
# forecast_backbones/m6/routing.py
"""站级回退路由：训后逐站比较 LoRA 与零样本（各自校准后的南网短期准确率），输的站回退零样本。
表：{station_id: "lora" | "zeroshot"}，JSON 落盘（键为字符串），推理时按行选路。
route_by_days（训练入口用）按起报日配对后做 bootstrap，LoRA 须稳定胜出才选；decide_routing 只比逐站均值。"""
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


ROUTE_DETAIL_COLS = ["station_id", "n_days_route", "gain_mean", "gain_lower", "route"]


def route_by_days(daily_zs: pd.DataFrame, daily_lo: pd.DataFrame, stations, min_gain_pct: float = 0.0,
                  n_boot: int = 2000, alpha: float = 0.1, min_days: int = 5, seed: int = 0) -> tuple[dict[int, str], pd.DataFrame]:
    """逐站按起报日配对 LoRA 与零样本的日准确率（同站同日两边都有限才配对），差 d = LoRA − 零样本。
    n_boot > 0：对配对天有放回重抽 n_boot 次，取重抽均值的 alpha 分位作下界；n_boot = 0：下界就是 d 的均值。
    配对天数 ≥ min_days 且下界 > min_gain_pct 才走 lora。stations：要出路由的全部站，没有配对天的站走 zeroshot。
    返回 (路由表, 逐站明细[station_id, n_days_route, gain_mean, gain_lower, route])。"""
    key = ["station_id", "date"]
    pair = daily_zs[key + ["accuracy_pct"]].merge(daily_lo[key + ["accuracy_pct"]], on=key, suffixes=("_zs", "_lo"))
    zs, lo = pair["accuracy_pct_zs"].to_numpy(float), pair["accuracy_pct_lo"].to_numpy(float)
    ok = np.isfinite(zs) & np.isfinite(lo)
    diffs = {int(s): g.to_numpy(float) for s, g in pd.Series(lo[ok] - zs[ok]).groupby(pair["station_id"].to_numpy()[ok])}
    rng = np.random.default_rng(int(seed))
    rows, out = [], {}
    for s in sorted({int(x) for x in stations}):
        d = diffs.get(s, np.empty(0))
        n = d.size
        mean = float(d.mean()) if n else float("nan")
        if n and int(n_boot) > 0:
            lower = float(np.quantile(d[rng.integers(0, n, size=(int(n_boot), n))].mean(axis=1), float(alpha)))
        else:
            lower = mean
        out[s] = "lora" if n >= int(min_days) and np.isfinite(lower) and lower > float(min_gain_pct) else "zeroshot"
        rows.append({"station_id": s, "n_days_route": int(n), "gain_mean": mean, "gain_lower": lower, "route": out[s]})
    return out, pd.DataFrame(rows, columns=ROUTE_DETAIL_COLS)


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
