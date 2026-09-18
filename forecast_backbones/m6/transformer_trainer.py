# -*- coding: utf-8 -*-
# forecast_backbones/m6/transformer_trainer.py
"""M6 推理/评估入口。evaluate_model 与 m1–m4 同签名：
    evaluate_model(model, test_dataset, HF_collate_fn, config) -> pd.DataFrame
result_df 与 pipelines/nanwang_metric.py 的 schema 兼容（timestamp_win, station, observe_power_predict,
observe_power_future），并追加原始分位 q_0.1..q_0.9、p_star、median、night、elev_deg、p_clear、var_bin 等列。"""
from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    from .calibration import quantile_cols
    from .decision import MODEL_LEVELS
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from calibration import quantile_cols
    from decision import MODEL_LEVELS

try:
    from ...pipelines.nanwang_metric import NanwangMetric
    from ...pipelines.transformer_dataset import UnifiedDataset
except ImportError:
    from pipelines.nanwang_metric import NanwangMetric
    from pipelines.transformer_dataset import UnifiedDataset
try:
    from .calibration import Calibrator, CalibrationConfig, result_to_long
    from .decision import (enforce_monotone, physics_clip, select_point, level_value, global_shrink_factor)
    from .config import load_config
except ImportError:
    from calibration import Calibrator, CalibrationConfig, result_to_long
    from decision import enforce_monotone, physics_clip, select_point, level_value, global_shrink_factor
    from config import load_config


def _np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def evaluate_model(model, test_dataset, HF_collate_fn, config) -> pd.DataFrame:
    cfg = getattr(model, "cfg", None) or config
    bs = int(cfg["eval"].get("batch_size", 64))
    ts_col = cfg["dataset"]["ts_col"]; label_col = cfg["dataset"]["label_col"]; H = int(cfg["timesfm3"]["pred_len"])
    qcols = quantile_cols(MODEL_LEVELS)
    loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, collate_fn=HF_collate_fn)
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict) and "batch_dict" in batch:
                bd, labels = batch["batch_dict"], batch.get("labels")
            else:
                bd, labels = batch, None
            bd = {k: v for k, v in bd.items() if k != label_col}          # 绝不喂未来真值
            truth = None
            if labels is not None and label_col in labels:
                truth = _np(labels[label_col])
            out = model(bd)
            q = _np(out.aux["quantiles"]); logits = _np(out.logits)
            pstar = _np(out.aux["p_star"]); med = _np(out.aux["median"])
            ts = _np(bd[ts_col]).astype(np.int64).reshape(-1)
            B = logits.shape[0]
            for i in range(B):
                sid = int(out.aux["station_id"][i])
                rec = {
                    "timestamp_win": pd.Timestamp(int(ts[i]), unit="s"),
                    "station": model.station_meta.name(sid), "station_id": sid,
                    "cap": float(out.aux["cap"][i]),
                    "observe_power_predict": logits[i].astype(np.float32),
                    "observe_power_future": (truth[i][:H].astype(np.float32) if truth is not None
                                             else np.full(H, np.nan, np.float32)),
                    "p_star": pstar[i].astype(np.float32), "median": med[i].astype(np.float32),
                    "night": out.aux["night"][i].astype(bool), "elev_deg": out.aux["elev_deg"][i].astype(np.float32),
                    "p_clear": out.aux["p_clear"][i].astype(np.float32),
                    "var_bin": int(out.aux["var_bin"][i]), "var_signal": float(out.aux["var_signal"][i]),
                    "degraded": bool(out.aux["degraded"][i]),
                }
                for j, c in enumerate(qcols):
                    rec[c] = q[i, :, j].astype(np.float32)
                rows.append(rec)
    return pd.DataFrame(rows)


def fit_calibrator_from_result(result_df: pd.DataFrame, cfg: dict) -> Calibrator:
    long = result_to_long(result_df, cfg["calibration"]["fit_steps"], MODEL_LEVELS)
    return Calibrator(CalibrationConfig.from_cfg(cfg)).fit(long)


def fit_shrink_from_result(result_df: pd.DataFrame, cfg: dict) -> float:
    lo, hi = cfg["data"]["scored_steps"]["short"]
    ys, ps, caps = [], [], []
    for r in result_df.itertuples(index=False):
        y = np.asarray(r.observe_power_future, float)[lo:hi]; p = np.asarray(r.median, float)[lo:hi]
        night = np.asarray(r.night, bool)[lo:hi]
        ok = np.isfinite(y) & ~night
        ys.append(y[ok]); ps.append(p[ok]); caps.append(np.full(ok.sum(), float(r.cap)))
    y = np.concatenate(ys); p = np.concatenate(ps); cap = np.concatenate(caps)
    d = np.maximum(y, float(cfg["decision"]["point"]["floor_ratio"]) * cap)
    den = np.sum(p ** 2 / d ** 2)
    return float(np.sum(y * p / d ** 2) / den) if den > 0 else 1.0


def _stack_q(result_df: pd.DataFrame) -> np.ndarray:
    return np.stack([np.stack([np.asarray(row[c], float) for c in quantile_cols(MODEL_LEVELS)], axis=-1)
                     for _, row in result_df.iterrows()])                       # [N,192,9]


def recompute_points(result_df: pd.DataFrame, cfg: dict, calibrator, variant: str, shrink: float = 1.0) -> pd.DataFrame:
    q = _stack_q(result_df); N, H, _ = q.shape
    night = np.stack(result_df["night"].to_numpy()).astype(bool)
    p_clear = np.stack(result_df["p_clear"].to_numpy()).astype(float)
    cap = result_df["cap"].to_numpy(float)
    pt = cfg["decision"]["point"]; ph = cfg["decision"]["physics"]
    if variant == "raw_median":
        point = level_value(q, MODEL_LEVELS, 0.5)
    elif variant == "global_shrink":
        point = shrink * level_value(q, MODEL_LEVELS, 0.5)
    elif variant == "physics_median":
        qc = physics_clip(enforce_monotone(q), night, p_clear, cap[:, None], float(ph["clear_sky_cap_ratio"]), bool(ph["capacity_cap"]))
        point = level_value(qc, MODEL_LEVELS, 0.5)
    elif variant == "calibrated_pstar":
        if calibrator is not None:
            lead = np.tile(np.arange(H), N)
            elev = np.stack(result_df["elev_deg"].to_numpy()).astype(float)
            qc = calibrator.apply(q.reshape(N * H, -1), np.repeat(result_df["station_id"].to_numpy(int), H),
                                  elev.reshape(-1), np.repeat(result_df["var_bin"].to_numpy(int), H), lead, p_clear.reshape(-1))
            qc = qc.reshape(N, H, -1); levels = tuple(calibrator.cfg.levels)
        else:
            qc, levels = q, MODEL_LEVELS
        qc = physics_clip(enforce_monotone(qc), night, p_clear, cap[:, None], float(ph["clear_sky_cap_ratio"]), bool(ph["capacity_cap"]))
        point = select_point(qc, levels, cap, pt["default"], pt.get("by_step") or (), shrink,
                             float(pt["floor_ratio"]), float(pt["tau_grid_step"]))
    else:
        raise ValueError(variant)
    out = result_df.copy()
    out["observe_power_predict"] = [p.astype(np.float32) for p in point]
    return out


def _nrmse_short_pct(df: pd.DataFrame, cfg: dict) -> float:
    lo, hi = cfg["data"]["scored_steps"]["short"]
    se, n = 0.0, 0
    for _, r in df.iterrows():
        y = np.asarray(r["observe_power_future"], float)[lo:hi]; p = np.asarray(r["observe_power_predict"], float)[lo:hi]
        ok = np.isfinite(y) & np.isfinite(p)
        se += float(np.sum(((y[ok] - p[ok]) / float(r["cap"])) ** 2)); n += int(ok.sum())
    return 100.0 * float(np.sqrt(se / n)) if n else float("nan")


LADDER = ("raw_median", "global_shrink", "physics_median", "calibrated_pstar")


def ladder_report(result_df: pd.DataFrame, gccap: dict, cfg: dict, calibrator, shrink: float) -> pd.DataFrame:
    nm = NanwangMetric()
    rows = []
    for v in LADDER:
        df = recompute_points(result_df, cfg, calibrator, v, shrink)
        sd = nm.short_d1_daily(df, gccap); ud = nm.ultra_short_daily(df, gccap)
        rows.append({"variant": v,
                     "nanwang_short_D+1": float(sd["accuracy_pct"].mean()) if len(sd) else float("nan"),
                     "nanwang_ultra_short": float(ud["accuracy_pct"].mean()) if len(ud) else float("nan"),
                     "nrmse_short_pct": _nrmse_short_pct(df, cfg),
                     "n_days": int(sd["date"].nunique()) if len(sd) else 0})
    return pd.DataFrame(rows)


def backfill_and_report(config_path, parquet_path, cal_start, cal_end, val_start, val_end, out_dir=None) -> pd.DataFrame:
    """零样本回填 cal+val → 在 cal 上拟合校准与 f* → 在 val 上出阶梯报表。
    STATE 产物（calibration.parquet + sidecar / decision_state.json / result_cal.parquet /
    result_val.parquet）落 run.out_dir（或 out_dir 覆盖）；REPORT 产物（ladder.csv）落
    cfg['eval']['report_dir']（已插值为 <out_dir>/reports）。calibration.parquet 的存盘路径
    取 cfg['calibration']['table_path']，与模型 calibration.enabled=True 时的加载路径一致。"""
    cfg = load_config(config_path)
    out = Path(out_dir or cfg["run"]["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    report_dir = Path(cfg["eval"]["report_dir"]); report_dir.mkdir(parents=True, exist_ok=True)
    try:
        from .transformer_models import TransformersAdaptedM6Model
    except ImportError:
        from transformer_models import TransformersAdaptedM6Model
    model = TransformersAdaptedM6Model({**cfg, "calibration": {**cfg["calibration"], "enabled": False}})
    ds_all = UnifiedDataset(str(parquet_path), timestamp_cols=[cfg["dataset"]["ts_col"]], entity_cols=[],
                            label_cols=[cfg["dataset"]["label_col"]], timestamp_output="epoch_s", epoch_unit="s",
                            for_hf_trainer=True)
    collate = functools.partial(UnifiedDataset.collate_fn, for_hf_trainer=True)
    res_cal = evaluate_model(model, ds_all.filter_by_time(cal_start, cal_end), collate, cfg)
    res_val = evaluate_model(model, ds_all.filter_by_time(val_start, val_end), collate, cfg)
    calib = fit_calibrator_from_result(res_cal, cfg); calib.save(cfg["calibration"]["table_path"])
    f = fit_shrink_from_result(res_cal, cfg)
    gccap = {model.station_meta.name(s): float(c) for s, c in
             res_val.drop_duplicates("station_id")[["station_id", "cap"]].itertuples(index=False)}
    rep = ladder_report(res_val, gccap, cfg, calib, f)
    rep.to_csv(report_dir / "ladder.csv", index=False)
    res_cal.to_parquet(out / "result_cal.parquet", index=False); res_val.to_parquet(out / "result_val.parquet", index=False)
    (out / "decision_state.json").write_text(json.dumps({"shrink_factor": f}, indent=2), encoding="utf-8")
    print(rep.to_string(index=False)); print(f"\n拟合 f* = {f:.4f}（写入 decision.point.shrink_factor 可启用 global_shrink）")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="M6 路线 A：回填 + 校准 + 阶梯报表")
    ap.add_argument("--config", default="forecast_backbones/m6/model_config.yaml")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--cal-start", required=True); ap.add_argument("--cal-end", required=True)
    ap.add_argument("--val-start", required=True); ap.add_argument("--val-end", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    backfill_and_report(a.config, a.parquet, a.cal_start, a.cal_end, a.val_start, a.val_end, a.out)
