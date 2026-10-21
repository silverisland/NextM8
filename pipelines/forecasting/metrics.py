"""Station-day D+1 and capacity-normalized point metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipelines.nanwang_metric import NanwangMetric

from .stations import station_cap_mw

TABLE_COLS = ["station_id", "station", "accuracy_pct", "n_days"]


def nrmse_pct(res: pd.DataFrame) -> float:
    if res.empty:
        return float("nan")
    y = np.stack(res["observe_power_future"].to_numpy()).astype(float)
    p = np.stack(res["observe_power_predict"].to_numpy()).astype(float)
    r = (y - p) / res["cap"].to_numpy(float)[:, None]
    ok = np.isfinite(r)
    return float(100.0 * np.sqrt(np.mean(r[ok] ** 2))) if ok.any() else float("nan")


def seen_unseen_scores(table: pd.DataFrame) -> tuple[float, float, int]:
    """逐站表 → (已见站站日均值, 未见站站日均值, 未见站数)；按 n_days 加权等于站日均值。"""
    if table.empty or "seen" not in table:
        return float("nan"), float("nan"), 0

    def wmean(t):
        return (
            float((t["accuracy_pct"] * t["n_days"]).sum() / t["n_days"].sum())
            if len(t) and t["n_days"].sum()
            else float("nan")
        )

    seen, unseen = table[table["seen"]], table[~table["seen"]]
    return wmean(seen), wmean(unseen), int(len(unseen))


def score_nanwang_result(model, res: pd.DataFrame, cfg):
    """Score prepared point forecasts using the shared Nanwang metric."""
    if res.empty:
        return float("nan"), pd.DataFrame(columns=TABLE_COLS), res
    gccap = {
        model.station_meta.name(int(s)): station_cap_mw(model.station_meta, int(s))
        for s in res["station_id"].unique()
    }  # 每站固定容量
    sd = NanwangMetric().short_d1_daily(res, gccap)
    name_to_id = {
        model.station_meta.name(int(s)): int(s) for s in res["station_id"].unique()
    }
    if len(sd) == 0:
        return float("nan"), pd.DataFrame(columns=TABLE_COLS), res
    g = sd.groupby("station")["accuracy_pct"].agg(["mean", "count"]).reset_index()
    table = pd.DataFrame(
        {
            "station_id": g["station"].map(name_to_id).astype(int),
            "station": g["station"],
            "accuracy_pct": g["mean"].astype(float),
            "n_days": g["count"].astype(int),
        }
    )
    known = cfg["dataset"].get("known_stations")
    table["seen"] = (
        True if not known else table["station_id"].isin([int(s) for s in known])
    )
    # 诊断列：这站的打分行有多少走了退化几何（缺坐标）；这站打分行上的 nrmse
    by_sid = {int(s): g for s, g in res.groupby("station_id")}
    table["degraded_frac"] = (
        table["station_id"]
        .map(
            lambda s: float(by_sid[s]["degraded"].mean()) if "degraded" in res else 0.0
        )
        .round(3)
    )
    table["nrmse_pct"] = (
        table["station_id"].map(lambda s: nrmse_pct(by_sid[s])).round(2)
    )
    return float(sd["accuracy_pct"].mean()), table, res
