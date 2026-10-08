# -*- coding: utf-8 -*-
"""run_experiments.py —— 按 config.experiments 逐个实验：解析协变量组合 → （需要时）构建消融特征库 →
生成该实验的 M5 configs.yaml → 逐站推理 → 南网逐日指标 → 落盘。

每个实验产出 ${run.out_dir}/<exp_id>/：
    config/configs.yaml                 M5 配置（只改 timesfm3.past_future_cols / past_only_cols）
    result/station=<st>.parquet         timestamp_win, station, observe_power_predict, observe_power_future
    daily_nanwang_short_D+1.csv         [station, date, accuracy_pct, n_points]
    daily_nanwang_ultra_short.csv       [station, date, accuracy_pct, n_targets]
    summary.json                        每站每指标的月度 mean + fleet 均值 + 协变量清单 + 消融

推理后端：
    m5    训练机：pipelines.model_registry.import_model_class("m5") + pipelines.pipeline_wrapper_inference.unified_inference
    mock  本机自测：预测 = p_phys_predict（或 ghi_cons_predict 缩放）+ 噪声，只为跑通流程与指标

用法（cwd=仓库根）：
    python experiments/nwp_covariate/run_experiments.py --ids C0,C1,C3a [--backend m5] [--predict_type ultra_short] [--force]
    python experiments/nwp_covariate/run_experiments.py --all
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (str(HERE), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from build_feature_store import build, feature_store_dir, load_config, write_feature_store, _abs  # noqa: E402

log = logging.getLogger("nwp_cov.run")
METRICS = ("nanwang_short_D+1", "nanwang_ultra_short")


# ----------------------------------------------------------------------------- 实验解析
def _pair(item, suffix: str) -> List[str]:
    if isinstance(item, str):
        return [item, item + suffix]
    if isinstance(item, (list, tuple)) and len(item) == 2:
        return [str(item[0]), str(item[1])]
    raise ValueError(f"past_future 元素格式错误: {item!r}")


def resolve_experiment(cfg: dict, exp_id: str, _seen=None) -> dict:
    """处理 extends / add_past_future / ablation 继承 -> {id, past_future: [[h,f],...], past_only: [...], ablation, requires}。"""
    suffix = cfg["columns"].get("future_suffix", "_predict")
    table = {e["id"]: e for e in cfg["experiments"]["list"]}
    if exp_id not in table:
        raise KeyError(f"实验 {exp_id!r} 不在 config.experiments.list 里")
    _seen = _seen or set()
    if exp_id in _seen:
        raise ValueError(f"extends 循环: {exp_id}")
    e = table[exp_id]
    base = {"past_future": [], "past_only": [], "ablation": {}, "requires": []}
    if e.get("extends"):
        base = resolve_experiment(cfg, e["extends"], _seen | {exp_id})
    out = {"id": exp_id,
           "past_future": [list(x) for x in base["past_future"]],
           "past_only": list(base["past_only"]),
           "ablation": dict(base["ablation"]),
           "requires": list(base["requires"])}
    if "past_future" in e:
        out["past_future"] = [_pair(x, suffix) for x in (e["past_future"] or [])]
    for x in e.get("add_past_future", []) or []:
        p = _pair(x, suffix)
        if p not in out["past_future"]:
            out["past_future"].append(p)
    if "past_only" in e:
        out["past_only"] = list(e["past_only"] or [])
    for x in e.get("add_past_only", []) or []:
        if x not in out["past_only"]:
            out["past_only"].append(x)
    if e.get("ablation"):
        out["ablation"] = dict(e["ablation"])
    out["requires"] = sorted(set(out["requires"]) | set(e.get("requires", []) or []))
    out["parent"] = e.get("extends")
    return out


def m5_config(cfg: dict, exp: dict) -> dict:
    d = cfg["data"]
    t = copy.deepcopy(cfg.get("m5_base", {}) or {})
    t.update({"context_len": int(d["context_len"]), "pred_len": int(d["pred_len"]),
              "past_future_cols": [list(p) for p in exp["past_future"]],
              "past_only_cols": list(exp["past_only"])})
    return {"run": {"model_name": "m5", "experiment": exp["id"]},
            "model": {}, "dataset": {"pv_col": d["target"]}, "timesfm3": t}


# ----------------------------------------------------------------------------- 特征库
def ensure_feature_store(cfg: dict, predict_type: str, exp: dict, rebuild: bool = False) -> Path:
    abl = exp.get("ablation") or {}
    fs = feature_store_dir(cfg, predict_type, exp["id"] if abl else None)
    if rebuild or not glob.glob(str(fs / "station=*.parquet")):
        log.info("构建特征库 %s (ablation=%s)", fs, abl or "无")
        frames, rep = build(cfg, predict_type, abl or None)
        write_feature_store(frames, fs, rep)
    return fs


def check_requires(fs: Path, required: List[str], pairs: List[List[str]], past_only: List[str]) -> List[str]:
    f = sorted(glob.glob(str(fs / "station=*.parquet")))[0]
    cols = set(pd.read_parquet(f).columns)
    need = set(required) | {c for p in pairs for c in p} | set(past_only)
    return sorted(need - cols)


# ----------------------------------------------------------------------------- 推理后端
class MockBackend:
    name = "mock"

    def __init__(self, cfg: dict, exp: dict, seed: int = 0):
        self.suffix = cfg["columns"].get("future_suffix", "_predict")
        self.pred_len = int(cfg["data"]["pred_len"])
        self.rng = np.random.default_rng(seed)
        self.pairs = exp["past_future"]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        B = len(df)
        cap = df["cap_power_on"].to_numpy(float)[:, None]
        futs = [f for _, f in self.pairs if f in df.columns]
        if "p_phys" + self.suffix in futs:
            base = np.stack(df["p_phys" + self.suffix].to_numpy())
        elif any(f.startswith("ghi") for f in futs):
            col = next(f for f in futs if f.startswith("ghi"))
            base = np.stack(df[col].to_numpy()) / 1000.0 * cap
        else:                                                # 无协变量：历史尾部持平
            base = np.repeat(np.stack(df["observe_power"].to_numpy())[:, -1:], self.pred_len, axis=1)
        noise = self.rng.normal(0, 0.03, base.shape) * cap
        return np.clip(np.nan_to_num(base) + noise, 0, None).astype(np.float32)


class M5Backend:
    name = "m5"

    def __init__(self, cfg: dict, exp: dict, model_config: dict, device: str = "auto", batch_size: int = 512):
        import torch
        from pipelines.model_registry import import_model_class
        self.batch_size = batch_size
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.model = import_model_class("m5")(model_config).to(self.device).eval()
        self.pred_col = type(self.model).__name__
        self.label = cfg["data"]["label"]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        from pipelines.pipeline_wrapper_inference import unified_inference
        infer_df = df.drop(columns=[c for c in (self.label, "observe_power_true") if c in df.columns])
        out = unified_inference(pipeline_backbone=self.model, ds_input=infer_df,
                                timestamp_cols=["timestamp_win"], entity_cols=["timestamp_win", "station"],
                                feature_cols=None, batch_size=self.batch_size, prediction_to_numpy=True)
        out["timestamp_win"] = pd.to_datetime(out["timestamp_win"])
        m = df[["timestamp_win", "station"]].merge(out[["timestamp_win", "station", self.pred_col]],
                                                    on=["timestamp_win", "station"], how="left")
        return np.stack([np.asarray(v, dtype=np.float32) for v in m[self.pred_col].to_numpy()])


# ----------------------------------------------------------------------------- 指标
def daily_metrics(result: pd.DataFrame, gccap: Dict[str, float]) -> Dict[str, pd.DataFrame]:
    from pipelines.nanwang_metric import NanwangMetric
    nm = NanwangMetric()
    short = nm.short_d1_daily(result, gccap)
    ultra = nm.ultra_short_daily(result, gccap)
    return {"nanwang_short_D+1": short, "nanwang_ultra_short": ultra}


def monthly_means(result: pd.DataFrame, gccap: Dict[str, float]) -> Dict[str, Dict[str, Optional[float]]]:
    """{metric: {station: 月度 mean（剔小月后）}}，与 eval_pretrain 的 nanwang sheet 同口径。"""
    from pipelines.nanwang_metric import NanwangMetric
    tables = NanwangMetric().monthly_table(result, gccap=gccap)
    out = {m: {} for m in METRICS}
    for st, tab in tables.items():
        for m in METRICS:
            v = tab.at[m, "mean"] if m in tab.index else np.nan
            out[m][st] = None if pd.isna(v) else float(v)
    return out


# ----------------------------------------------------------------------------- 单个实验
def run_experiment(cfg: dict, exp_id: str, predict_type: str, backend: str = "m5", device: str = "auto",
                   force: bool = False, rebuild_fs: bool = False, out_root: Optional[Path] = None) -> dict:
    exp = resolve_experiment(cfg, exp_id)
    out_root = out_root or _abs(cfg["run"]["out_dir"].replace("${run.name}", cfg["run"]["name"]))
    out = out_root / exp_id
    summary_path = out / "summary.json"
    if summary_path.exists() and not force:
        log.info("SKIP %s（summary 已存在）", exp_id)
        return json.load(open(summary_path, encoding="utf-8"))
    t0 = time.time()
    fs = ensure_feature_store(cfg, predict_type, exp, rebuild_fs)
    missing = check_requires(fs, exp["requires"], exp["past_future"], exp["past_only"])
    if missing:
        log.warning("SKIP %s：特征库缺列 %s", exp_id, missing)
        return {"id": exp_id, "skipped": True, "missing": missing}

    mcfg = m5_config(cfg, exp)
    (out / "config").mkdir(parents=True, exist_ok=True)
    with open(out / "config" / "configs.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(mcfg, f, allow_unicode=True, sort_keys=False)

    be = MockBackend(cfg, exp) if backend == "mock" else M5Backend(cfg, exp, mcfg, device, int(cfg["run"].get("batch_size", 512)))
    label = cfg["data"]["label"]
    results, gccap = [], {}
    (out / "result").mkdir(exist_ok=True)
    for f in sorted(glob.glob(str(fs / "station=*.parquet"))):
        df = pd.read_parquet(f)
        st = str(df["station"].iloc[0])
        pred = be.predict(df)
        r = df[["timestamp_win", "station"]].copy()
        r["observe_power_predict"] = list(pred)
        r[label] = df[label]
        r.to_parquet(out / "result" / f"station={st}.parquet", index=False)
        results.append(r)
        gccap[st] = float(np.nanmax(df["cap_power_on"].to_numpy(float)))
        log.info("[%s] %s 行=%d cap=%.1f", exp_id, st, len(df), gccap[st])
    result = pd.concat(results, ignore_index=True)

    daily = daily_metrics(result, gccap)
    for m, tab in daily.items():
        tab.to_csv(out / f"daily_{m}.csv", index=False)
    mm = monthly_means(result, gccap)
    summary = {"id": exp_id, "predict_type": predict_type, "backend": be.name, "parent": exp.get("parent"),
               "past_future_cols": exp["past_future"], "past_only_cols": exp["past_only"], "ablation": exp["ablation"],
               "feature_store": str(fs), "stations": sorted(gccap), "rows": int(len(result)),
               "monthly_mean": mm,
               "fleet_mean": {m: (float(np.nanmean([v for v in mm[m].values() if v is not None]))
                                  if any(v is not None for v in mm[m].values()) else None) for m in METRICS},
               "daily_days": {m: int(daily[m]["date"].nunique()) if len(daily[m]) else 0 for m in METRICS},
               "elapsed_s": round(time.time() - t0, 1)}
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info("DONE %s fleet_mean=%s (%.1fs)", exp_id, summary["fleet_mean"], summary["elapsed_s"])
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--ids", default=None, help="逗号分隔的实验 id")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--predict_type", default=None)
    ap.add_argument("--backend", choices=["m5", "mock"], default="m5")
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rebuild_fs", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config)
    pt = a.predict_type or cfg["data"]["predict_type"]
    ids = [e["id"] for e in cfg["experiments"]["list"]] if a.all else [s.strip() for s in (a.ids or "").split(",") if s.strip()]
    if not ids:
        raise SystemExit("给 --ids 或 --all")
    for i in ids:
        run_experiment(cfg, i, pt, a.backend, a.device or cfg["run"].get("device", "auto"), a.force, a.rebuild_fs)


if __name__ == "__main__":
    main()
