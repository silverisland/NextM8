# -*- coding: utf-8 -*-
# forecast_backbones/m5/evaluate.py
"""M5 测试集评估：训练产物（一个或多个 run 目录）→ 测试 parquet 推理 → 南网短期 D+1 / nrmse → result_test.parquet + reports。

  python -m forecast_backbones.m5.evaluate --runs runs/m5/final/e1_s0 --test "datasets/test/*.parquet"
  python -m forecast_backbones.m5.evaluate --runs runs/m5/final/e1_s0 runs/m5/final/e1_s1 runs/m5/final/e1_s2 \\
         --test datasets/test --out runs/m5/final/ens_eval --decision-yaml forecast_backbones/m5/config/variants/d2_pstar_point.yaml

多个 run 目录 = 集成：各成员的原始分位（真实单位）逐分位平均后再过一次决策层；reports/members.csv 另报每个成员单独的分数。
决策规则可在评估时覆盖（--point / --decision-yaml），不用重训：决策层没有可训练参数。
seen / unseen 按第一个成员 model_config.yaml 里的 known_stations 判定。"""
from __future__ import annotations

import argparse
import copy
import functools
import glob
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

try:
    from .config import deep_merge, raw_override
    from .transformer_models import TransformersAdaptedM5Model
    from .train import score_nanwang, seen_unseen_scores, nrmse_pct, ultra_proxy_pct, WEIGHTS_FILE, CONFIG_FILE
    from .data import _dataset_with_station_id
    from .rows import scored_row_index, issue_grid_row_index
    from .decision import quantile_cols
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from forecast_backbones.m5.config import deep_merge, raw_override
    from forecast_backbones.m5.transformer_models import TransformersAdaptedM5Model
    from forecast_backbones.m5.train import score_nanwang, seen_unseen_scores, nrmse_pct, ultra_proxy_pct, WEIGHTS_FILE, CONFIG_FILE
    from forecast_backbones.m5.data import _dataset_with_station_id
    from forecast_backbones.m5.rows import scored_row_index, issue_grid_row_index
    from forecast_backbones.m5.decision import quantile_cols
try:
    from ...pipelines.transformer_dataset import UnifiedDataset
except ImportError:
    from pipelines.transformer_dataset import UnifiedDataset


def _np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def expand_paths(items) -> list[str]:
    """目录 → 其下全部 *.parquet；通配符 → 展开；文件 → 原样。"""
    out: list[str] = []
    for it in items:
        p = Path(it)
        if p.is_dir():
            out += sorted(str(f) for f in p.glob("*.parquet"))
        else:
            hits = sorted(glob.glob(str(it)))
            out += hits if hits else [str(it)]
    missing = [x for x in out if not Path(x).exists()]
    if missing or not out:
        raise FileNotFoundError(f"测试 parquet 不存在：{missing or items}")
    return out


def load_member(run_dir, device) -> TransformersAdaptedM5Model:
    """train_m5 的产物目录 → 推理态模型。LoRA run 必须有 model.safetensors；零样本 run（finetune.enabled=false）没有权重文件。"""
    run = Path(run_dir)
    cfg_path, w = run / CONFIG_FILE, run / WEIGHTS_FILE
    if not cfg_path.exists():
        raise FileNotFoundError(f"{run} 缺 {CONFIG_FILE}（要 train_m5 的产物目录）")
    raw = raw_override(str(cfg_path))
    finetune = bool((raw.get("finetune") or {}).get("enabled", True))
    if finetune and not w.exists():
        raise FileNotFoundError(f"{run} 缺 {WEIGHTS_FILE}：finetune.enabled=true 的 run 必须有 LoRA 权重")
    cfg = deep_merge(raw, {"train": {"weights_path": str(w)}, "timesfm3": {"device": str(device)}})
    m = TransformersAdaptedM5Model(cfg)
    if finetune and not m.weights_loaded:
        raise RuntimeError(f"{w} 没有加载上")
    if not m.cfg["dataset"].get("known_stations"):
        warnings.warn(f"{cfg_path} 没有 known_stations：所有站都按已见站处理")
    m.to(device); m.eval()
    return m


def decision_override(base: dict, point: str | None = None, decision_yaml: str | None = None) -> dict:
    """评估时覆盖决策规则：--decision-yaml 合并该 yaml 的 decision 段；--point 设 point.default 并清空 by_step。"""
    dec = copy.deepcopy(base)
    if decision_yaml:
        y = yaml.safe_load(Path(decision_yaml).read_text(encoding="utf-8")) or {}
        dec = deep_merge(dec, y.get("decision") or {})
    if point:
        dec["point"]["default"] = str(point); dec["point"]["by_step"] = []
    return dec


def describe_point(dec: dict) -> str:
    pt = dec["point"]
    steps = ";".join(f"{b['steps'][0]}-{b['steps'][1]}:{b['method']}" for b in (pt.get("by_step") or ()))
    return pt["default"] + (f"[{steps}]" if steps else "")


def run_inference(members, ds, idx, collate, decision: dict, batch_size: int, show: bool = True) -> pd.DataFrame:
    """members 各自算原始分位 → 逐分位平均 → 一次决策层。返回与 evaluate_model 同 schema 的结果表（多成员时另带 predict_m<k>）。"""
    m0 = members[0]; cfg = m0.cfg
    ts_col, label_col = cfg["dataset"]["ts_col"], cfg["dataset"]["label_col"]
    L, H = m0.context_len, m0.pred_len
    qcols = quantile_cols(m0.levels)
    loader = DataLoader(Subset(ds, [int(i) for i in idx]), batch_size=batch_size, shuffle=False, collate_fn=collate)
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"[m5] 测试推理 ×{len(members)}", unit="batch", disable=not show, mininterval=2.0, dynamic_ncols=True):
            bd, labels = (batch["batch_dict"], batch.get("labels")) if isinstance(batch, dict) and "batch_dict" in batch else (batch, None)
            bd = {k: v for k, v in bd.items() if k != label_col}                  # 绝不喂未来真值
            truth = _np(labels[label_col]) if labels is not None and label_col in labels else None
            qs, ctx = [], None
            for m in members:
                q, c = m.predict_raw(bd); qs.append(q)
                ctx = ctx if ctx is not None else c
            q_mean = np.mean(qs, axis=0)
            night_fut, p_clear_fut = ctx.night[:, L:], ctx.p_clear[:, L:]
            point, q_cal, median, pstar, night_physics = m0.decide(q_mean, night_fut, p_clear_fut, ctx.cap, decision)
            member_points = [m0.decide(q, night_fut, p_clear_fut, ctx.cap, decision)[0] for q in qs] if len(members) > 1 else []
            ts = _np(bd[ts_col]).astype(np.int64).reshape(-1)
            for i in range(point.shape[0]):
                sid = int(ctx.station_id[i])
                rec = {
                    "timestamp_win": pd.Timestamp(int(ts[i]), unit="s"),
                    "station": m0.station_meta.name(sid), "station_id": sid, "cap": float(ctx.cap[i]),
                    "observe_power_predict": point[i].astype(np.float32),
                    "observe_power_future": (truth[i][:H].astype(np.float32) if truth is not None else np.full(H, np.nan, np.float32)),
                    "p_star": pstar[i].astype(np.float32), "median": median[i].astype(np.float32),
                    "night": night_physics[i].astype(bool), "elev_deg": ctx.elev_deg[i, L:].astype(np.float32),
                    "p_clear": p_clear_fut[i].astype(np.float32),
                    "var_bin": 0, "var_signal": float(ctx.var_signal[i]), "degraded": bool(ctx.degraded[i]),
                    "cs_ratio": float(ctx.cs_ratio[i]) if ctx.cs_ratio is not None else 1.0,
                }
                for j, c in enumerate(qcols):
                    rec[c] = q_mean[i, :, j].astype(np.float32)
                for k, pm in enumerate(member_points):
                    rec[f"predict_m{k}"] = pm[i].astype(np.float32)
                rows.append(rec)
    return pd.DataFrame(rows)


def evaluate(runs, test_paths, out_dir=None, point: str | None = None, decision_yaml: str | None = None,
             issue_every_min: int = 60, batch_size: int | None = None, device: str | None = None,
             tag: str | None = None, progress: bool = True) -> dict:
    t0 = time.time()
    runs = [str(r) for r in runs]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    members = [load_member(r, device) for r in runs]
    m0 = members[0]; cfg = m0.cfg
    out = Path(out_dir) if out_dir else (Path(runs[0]) / "eval_test" if len(runs) == 1 else Path(runs[0]).parent / "ensemble_eval")
    out.mkdir(parents=True, exist_ok=True); rep = out / "reports"; rep.mkdir(exist_ok=True)
    paths = expand_paths(test_paths)
    ds = _dataset_with_station_id(paths, cfg, m0.station_meta, scratch_dir=out / ".data", reuse=False)
    if len(ds) == 0:
        raise ValueError("测试集为空")
    st_col, ts_col = str(cfg["dataset"]["station_id_col"]), str(cfg["dataset"]["ts_col"])
    idx = sorted(set(int(i) for i in scored_row_index(ds, st_col, ts_col)) | set(int(i) for i in issue_grid_row_index(ds, ts_col, issue_every_min)))
    if not idx:
        raise ValueError("测试集里没有可打分的起报行（10:00 行 / 整点行都没有）")
    decision = decision_override(cfg["decision"], point, decision_yaml)
    bs = int(batch_size or cfg["eval"].get("batch_size", 256))
    print(f"[m5-eval] 成员 {len(members)} 个，测试文件 {len(paths)} 个，行 {len(ds)}，打分行 {len(idx)}，决策 {describe_point(decision)}，device={device}", flush=True)
    res = run_inference(members, ds, idx, functools.partial(UnifiedDataset.collate_fn, for_hf_trainer=True), decision, bs, progress)
    res.to_parquet(out / "result_test.parquet", index=False)

    hourly = res[(res["timestamp_win"].dt.minute + 60 * res["timestamp_win"].dt.hour) % issue_every_min == 0]
    nw, table, _ = score_nanwang(m0, None, None, cfg, res=res)
    seen, unseen, n_unseen = seen_unseen_scores(table)
    tag = tag or ("+".join(Path(r).name for r in runs) if len(runs) <= 3 else f"ens{len(runs)}_{Path(runs[0]).name}")
    summary = {"tag": tag, "nanwang_short_D+1": nw, "nanwang_seen": seen, "nanwang_unseen": unseen, "n_unseen_stations": n_unseen,
               "nrmse_pct": nrmse_pct(hourly), "nanwang_ultra_proxy": ultra_proxy_pct(hourly),
               "n_members": len(members), "point_rule": describe_point(decision),
               "n_stations": int(table["station_id"].nunique()) if len(table) else 0, "n_station_days": int(table["n_days"].sum()) if len(table) else 0,
               "n_scored_rows": len(res), "eval_minutes": (time.time() - t0) / 60.0, "members": "|".join(runs), "test": "|".join(paths)}
    members_tab = pd.DataFrame()
    if len(members) > 1:
        recs = []
        for k, r in enumerate(runs):
            rk = res.assign(observe_power_predict=res[f"predict_m{k}"])
            nw_k, tab_k, _ = score_nanwang(m0, None, None, cfg, res=rk)
            s_k, u_k, _ = seen_unseen_scores(tab_k)
            recs.append({"member": Path(r).name, "nanwang_short_D+1": nw_k, "nanwang_seen": s_k, "nanwang_unseen": u_k,
                         "nrmse_pct": nrmse_pct(rk[rk.index.isin(hourly.index)]), "run_dir": r})
        recs.append({"member": "ensemble", "nanwang_short_D+1": nw, "nanwang_seen": seen, "nanwang_unseen": unseen,
                     "nrmse_pct": summary["nrmse_pct"], "run_dir": ""})
        members_tab = pd.DataFrame(recs); members_tab.to_csv(rep / "members.csv", index=False)
    pd.DataFrame([summary]).to_csv(rep / "summary.csv", index=False)
    table.to_csv(rep / "station_table.csv", index=False)
    (rep / "decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(pd.DataFrame([{k: v for k, v in summary.items() if k not in ("members", "test")}]).to_string(index=False))
        print(table.to_string(index=False))
        if len(members_tab):
            print(members_tab.drop(columns=["run_dir"]).to_string(index=False))
    print(f"[m5-eval] 产物：{out / 'result_test.parquet'} / {rep}")
    return {"summary": summary, "station_table": table, "members": members_tab, "result": res}


def main(argv=None):
    ap = argparse.ArgumentParser(description="M5 测试集评估（支持多 run 集成与决策规则覆盖）")
    ap.add_argument("--runs", nargs="+", required=True, help="train_m5 的产物目录，多个 = 集成（分位平均）")
    ap.add_argument("--test", nargs="+", required=True, help="测试 parquet：目录 / 文件 / 通配符")
    ap.add_argument("--out", default=None, help="输出目录（默认 单成员 <run>/eval_test，多成员 <runs 的父目录>/ensemble_eval）")
    ap.add_argument("--point", default=None, help="覆盖 decision.point.default（median / p_star / p_star_l1），并清空 by_step")
    ap.add_argument("--decision-yaml", default=None, help="合并该 yaml 的 decision 段（如 config/variants/d2_pstar_point.yaml）")
    ap.add_argument("--issue-every-min", type=int, default=60, help="nrmse 用的起报间隔（分钟）")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-progress", action="store_true")
    a = ap.parse_args(argv)
    evaluate(a.runs, a.test, a.out, a.point, a.decision_yaml, a.issue_every_min, a.batch_size, a.device, a.tag, not a.no_progress)


if __name__ == "__main__":
    main()
