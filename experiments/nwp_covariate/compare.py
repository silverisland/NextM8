# -*- coding: utf-8 -*-
"""compare.py —— 把各实验与 baseline（或各自 parent）做逐站逐日配对检验，出 report.md + comparison.csv。

判定规则来自 config.evaluation：
    paired_test.min_mean_gain       主指标配对差均值下限（分）
    paired_test.min_station_win_frac 月度 mean 不下降的站占比下限
    paired_test.bootstrap / ci      站-日重抽样置信区间；下界 > 0 才算显著
    veto.max_drop_secondary         次指标下降超过此值则否决
    ablation_alert_drop             消融实验相对 parent 掉超过此值告警

用法（cwd=仓库根）：
    python experiments/nwp_covariate/compare.py [--ids C1,C2a,...] [--baseline C1] [--predict_type ultra_short]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_feature_store import load_config, _abs  # noqa: E402
from run_experiments import METRICS  # noqa: E402

log = logging.getLogger("nwp_cov.compare")


def out_root(cfg: dict) -> Path:
    return _abs(cfg["run"]["out_dir"].replace("${run.name}", cfg["run"]["name"]))


def load_exp(root: Path, exp_id: str) -> Optional[dict]:
    p = root / exp_id / "summary.json"
    if not p.exists():
        return None
    s = json.load(open(p, encoding="utf-8"))
    s["daily"] = {}
    for m in METRICS:
        f = root / exp_id / f"daily_{m}.csv"
        s["daily"][m] = pd.read_csv(f, parse_dates=["date"]) if f.exists() else pd.DataFrame(columns=["station", "date", "accuracy_pct"])
    return s


def paired(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """b − a 的逐站逐日配对差。"""
    m = a[["station", "date", "accuracy_pct"]].merge(b[["station", "date", "accuracy_pct"]], on=["station", "date"], suffixes=("_a", "_b"))
    m["diff"] = m["accuracy_pct_b"] - m["accuracy_pct_a"]
    return m


def bootstrap_ci(x: np.ndarray, n: int, ci: float, seed: int = 0) -> tuple:
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n)])
    lo, hi = np.percentile(means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)


def compare_one(base: dict, exp: dict, primary: str, secondary: str, ev: dict) -> dict:
    pt = ev["paired_test"]
    row = {"id": exp["id"], "vs": base["id"], "past_future": "; ".join(f"{h}" for h, _ in exp["past_future_cols"]),
           "past_only": "; ".join(exp["past_only_cols"]), "ablation": json.dumps(exp["ablation"], ensure_ascii=False) if exp["ablation"] else ""}
    d = paired(base["daily"][primary], exp["daily"][primary])
    row["n_station_days"] = int(len(d))
    row["primary_mean_gain"] = float(d["diff"].mean()) if len(d) else np.nan
    lo, hi = bootstrap_ci(d["diff"].to_numpy(float), int(pt.get("bootstrap", 2000)), float(pt.get("ci", 0.95)))
    row["primary_ci_lo"], row["primary_ci_hi"] = lo, hi
    st_gain = d.groupby("station")["diff"].mean()
    row["station_win_frac"] = float((st_gain >= 0).mean()) if len(st_gain) else np.nan
    row["n_stations"] = int(len(st_gain))
    fb, fe = base["fleet_mean"].get(primary), exp["fleet_mean"].get(primary)
    row["primary_fleet_base"], row["primary_fleet_exp"] = fb, fe
    d2 = paired(base["daily"][secondary], exp["daily"][secondary])
    row["secondary_mean_gain"] = float(d2["diff"].mean()) if len(d2) else np.nan
    keep = (np.isfinite(row["primary_mean_gain"]) and row["primary_mean_gain"] >= float(pt["min_mean_gain"])
            and np.isfinite(lo) and lo > 0 and row["station_win_frac"] >= float(pt["min_station_win_frac"]))
    veto = np.isfinite(row["secondary_mean_gain"]) and row["secondary_mean_gain"] < -float(ev["veto"]["max_drop_secondary"])
    row["keep"] = bool(keep and not veto)
    row["veto_secondary"] = bool(veto)
    row["ablation_alert"] = bool(exp["ablation"] and np.isfinite(row["primary_mean_gain"])
                                 and row["primary_mean_gain"] < -float(ev.get("ablation_alert_drop", 3.0)))
    return row


def run(cfg: dict, predict_type: str, ids: Optional[list] = None, baseline: Optional[str] = None) -> pd.DataFrame:
    ev = cfg["evaluation"]
    primary = ev["primary_by_type"][predict_type]
    secondary = [m for m in METRICS if m != primary][0]
    root = out_root(cfg)
    baseline = baseline or cfg["experiments"]["baseline"]
    all_ids = ids or [e["id"] for e in cfg["experiments"]["list"]]
    exps = {i: load_exp(root, i) for i in all_ids + [baseline]}
    exps = {k: v for k, v in exps.items() if v is not None}
    if baseline not in exps:
        raise FileNotFoundError(f"baseline {baseline} 没有 summary.json（{root}）")
    rows = []
    for i in all_ids:
        if i == baseline or i not in exps:
            continue
        e = exps[i]
        ref_id = e.get("parent") if e.get("ablation") and e.get("parent") in exps else baseline
        rows.append(compare_one(exps[ref_id], e, primary, secondary, ev))
    tab = pd.DataFrame(rows).sort_values("primary_mean_gain", ascending=False) if rows else pd.DataFrame()
    tab.to_csv(root / "comparison.csv", index=False)
    write_report(root / "report.md", cfg, predict_type, primary, secondary, exps[baseline], tab)
    return tab


def write_report(path: Path, cfg: dict, predict_type: str, primary: str, secondary: str, base: dict, tab: pd.DataFrame):
    ev = cfg["evaluation"]; pt = ev["paired_test"]
    lines = [f"# NWP 协变量实验报告（{cfg['run']['name']}，{predict_type}）", "",
             f"主指标 **{primary}**，次指标 {secondary}。baseline = **{base['id']}**（fleet mean {base['fleet_mean'].get(primary)}）。",
             f"保留条件：配对差均值 ≥ {pt['min_mean_gain']} 分，bootstrap {int(pt.get('ci', 0.95) * 100)}% 下界 > 0，"
             f"≥ {pt['min_station_win_frac'] * 100:.0f}% 站不退化；次指标下降 > {ev['veto']['max_drop_secondary']} 分否决；"
             f"消融相对 parent 掉 > {ev.get('ablation_alert_drop', 3.0)} 分告警。", ""]
    if tab.empty:
        lines.append("没有可比较的实验。")
    else:
        cols = ["id", "vs", "past_future", "primary_mean_gain", "primary_ci_lo", "primary_ci_hi", "station_win_frac",
                "secondary_mean_gain", "keep", "veto_secondary", "ablation_alert", "n_station_days"]
        t = tab[cols].copy()
        for c in ("primary_mean_gain", "primary_ci_lo", "primary_ci_hi", "secondary_mean_gain"):
            t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{v:+.2f}")
        t["station_win_frac"] = t["station_win_frac"].map(lambda v: "" if pd.isna(v) else f"{v:.0%}")
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for _, r in t.iterrows():
            lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        lines += ["", "## 结论", ""]
        kept = tab[tab["keep"]]
        lines.append(f"- 满足保留条件的实验：{', '.join(kept['id']) if len(kept) else '无'}。")
        vet = tab[tab["veto_secondary"]]
        if len(vet):
            lines.append(f"- 被次指标否决：{', '.join(vet['id'])}。")
        al = tab[tab["ablation_alert"]]
        if len(al):
            lines.append(f"- 消融告警（补全链需要调整）：{', '.join(al['id'])}。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("报告 -> %s", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--ids", default=None)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--predict_type", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config)
    pt = a.predict_type or cfg["data"]["predict_type"]
    ids = [s.strip() for s in a.ids.split(",")] if a.ids else None
    tab = run(cfg, pt, ids, a.baseline)
    pd.set_option("display.width", 250)
    print(tab.to_string(index=False) if len(tab) else "无结果")


if __name__ == "__main__":
    main()
