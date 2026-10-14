# -*- coding: utf-8 -*-
# forecast_backbones/m8/collect_runs.py
"""把若干 M8 run 目录（各含 reports/summary.csv 与 train_log.json）汇总成一张表，按南网 D+1 降序。
用法：python -m forecast_backbones.m8.collect_runs runs/m8/variants_xxx [更多目录…] --out summary.csv"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def collect(roots) -> pd.DataFrame:
    rows = []
    for root in roots:
        for s in sorted(Path(root).glob("**/reports/summary.csv")):
            run = s.parent.parent
            r = pd.read_csv(s).iloc[0].to_dict()
            r["run_dir"] = str(run)
            log = run / "train_log.json"
            if log.exists():
                h = json.loads(log.read_text(encoding="utf-8")).get("history", [])
                r["n_evals"] = len(h)
                r["curve"] = " ".join(f"{e['step']}:{e.get('nanwang_after_pstar', e.get('nrmse', float('nan'))):.2f}" for e in h)
            rows.append(r)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values("nanwang_short_D+1", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = collect(a.roots)
    if df.empty:
        print("没有找到任何 reports/summary.csv")
    else:
        cols = [c for c in ["tag", "nanwang_short_D+1", "nanwang_seen", "nanwang_unseen", "nrmse_pct", "nanwang_ultra_proxy", "best_step", "params", "blocks",
                            "patch_len", "d_model", "target_scale", "cpm", "cov_dropout", "future_only", "residual", "train_minutes", "run_dir"] if c in df.columns]
        print(df[cols].to_string(index=False))
        if a.out:
            df.to_csv(a.out, index=False); print(f"→ {a.out}")
