"""Nanwang reports when a station's first-value capacity varies across issues."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.nanwang_metric import NanwangMetric


def write_variable_capacity_report(group, station, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = NanwangMetric(capacity_col="cap").monthly_table(group, gccap=None)[station]
    with pd.ExcelWriter(
        out_dir / f"{station}-metrics.xlsx", engine="xlsxwriter"
    ) as writer:
        table.to_excel(writer, sheet_name="nanwang")
    result = {
        "station": station,
        "cap": None,
        "cap_min": float(group.cap.min()),
        "cap_max": float(group.cap.max()),
        "rows": len(group),
        "capacity_policy": "row_first; D+1 issue row; ultra-short truth-source row",
        "ods_unavailable_reason": "Legacy ODS reports require a fixed station capacity",
        "ods_short_rmse_acc": None,
        "ods_ultra_short_rmse_acc": None,
    }
    for key in ("nanwang_short_D+1", "nanwang_ultra_short"):
        value = table.at[key, "mean"]
        result[key] = float(value) if np.isfinite(value) else None
    (out_dir / f"{station}-summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    return result
