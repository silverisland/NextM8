"""Common station workbooks and JSON summaries for point forecasts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.metric import MIN_DAYS, SolarMetricCalculator
from pipelines.nanwang_metric import NanwangMetric

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_COLS = [
    "timestamp_win",
    "station",
    "observe_power_predict",
    "observe_power_future",
]
ODS_SHEETS = ("ods_ultra_short", "ods_short")
ODS_METRICS = ("rmse_acc", "mae_acc", "ACC_acc")
DS_SHEETS = ("ultra_short", "short", "48hours")
DS_METRICS = ("RMSE_SCORE", "RMSE", "MAE")
NANWANG_ROWS = ("nanwang_ultra_short", "nanwang_short_D+1")


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, pd.Timestamp):
            return str(o)
        return super().default(o)


def summary_helpers():
    """-> (extract_summary, JSON encoder, 来源)。eval_station_script 在仓库根（服务器）时用它，键与 M1–M7 逐个一致。"""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from eval_station_script import NumpyJSONEncoder as ExternalEncoder
        from eval_station_script import extract_summary

        return extract_summary, ExternalEncoder, "eval_station_script"
    except ImportError:
        return fallback_extract_summary, NumpyJSONEncoder, "fallback"


def _month_mean(table: pd.DataFrame, rows, day_row: str = "day_count") -> dict:
    """ODS/DS 月表（.T 后，列 = 月份位置编号）→ {行名: 月均值}。有 mean 列（generate_report 给 ODS 表加的，已剔小月）直接用；
    否则剔除样本天数 < MIN_DAYS 的月份后各月等权。"""
    out = {}
    if "mean" in table.columns:
        for r in rows:
            v = table.at[r, "mean"] if r in table.index else np.nan
            out[r] = None if pd.isna(v) else float(v)
        return out
    months = [c for c in table.columns if c != "mean"]
    if day_row in table.index:
        days = pd.to_numeric(table.loc[day_row, months], errors="coerce")
        months = [c for c in months if pd.notna(days[c]) and float(days[c]) >= MIN_DAYS]
    for r in rows:
        if r not in table.index or not months:
            out[r] = None
            continue
        v = pd.to_numeric(table.loc[r, months], errors="coerce").mean()
        out[r] = None if pd.isna(v) else float(v)
    return out


def main_metrics(report: dict, nanwang_table: pd.DataFrame) -> dict:
    """station_metrics.csv 的指标列：直接从表里取，不依赖 extract_summary 的键名。"""
    out = {}
    for k in ODS_SHEETS:
        for m, v in _month_mean(report[k], ODS_METRICS).items():
            out[f"{k}_{m}"] = v
    for r in NANWANG_ROWS:
        v = (
            nanwang_table.at[r, "mean"]
            if r in nanwang_table.index and "mean" in nanwang_table.columns
            else np.nan
        )
        out[r] = None if pd.isna(v) else float(v)
    return out


def fallback_extract_summary(report: dict, nanwang_tables: dict) -> dict:
    """<sheet>_<指标> 的扁平字典：ODS 两张表 + DS 三张表的月均值，再加南网两项（该站月度表的 mean 列）。"""
    s = {}
    for k in ODS_SHEETS:
        for m, v in _month_mean(report[k], ODS_METRICS).items():
            s[f"{k}_{m}"] = v
    for k in DS_SHEETS:
        for m, v in _month_mean(report[k], DS_METRICS).items():
            s[f"{k}_{m}"] = v
    tab = next(iter(nanwang_tables.values())) if nanwang_tables else None
    for r in NANWANG_ROWS:
        v = (
            tab.at[r, "mean"]
            if tab is not None and r in tab.index and "mean" in tab.columns
            else np.nan
        )
        s[r] = None if pd.isna(v) else float(v)
    return s


def write_station_report(
    res: pd.DataFrame, st: str, cap: float, out_dir, extract_summary, encoder
) -> tuple[dict, dict]:
    """与 eval_shell/m6_station_script.run_station 同一套：五张 solar 表 + nanwang 表合写 {站}-metrics.xlsx，{站}-summary.json。
    -> (summary.json 的字典, station_metrics.csv 的一行)。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_df = res[RESULT_COLS].copy().assign(station=st)
    result_df["timestamp_win"] = pd.to_datetime(result_df["timestamp_win"])
    nanwang_tables = NanwangMetric().monthly_table(result_df, gccap=cap)
    xlsx = out_dir / f"{st}-metrics.xlsx"
    with pd.ExcelWriter(xlsx, engine="xlsxwriter") as writer:
        ev = SolarMetricCalculator(
            model_version=st,
            cap_power=cap,
            save_dir=str(out_dir),
            time_col="timestamp_win",
            groundtruth_col="observe_power_future",
            predict_col="observe_power_predict",
            station_col="station",
        )
        report = ev.generate_report(result_df, writer=writer)
        nanwang_tables[st].to_excel(writer, sheet_name="nanwang")
    summary = extract_summary(report, nanwang_tables)
    summary["station"], summary["cap_power"], summary["rows"] = (
        st,
        cap,
        int(len(result_df)),
    )
    (out_dir / f"{st}-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, cls=encoder), encoding="utf-8"
    )
    return summary, {
        "station": st,
        "cap": cap,
        "rows": int(len(result_df)),
        **main_metrics(report, nanwang_tables[st]),
    }
