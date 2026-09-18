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
import yaml
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
    from .config import load_config, deep_merge
except ImportError:
    from calibration import Calibrator, CalibrationConfig, result_to_long
    from decision import enforce_monotone, physics_clip, select_point, level_value, global_shrink_factor
    from config import load_config, deep_merge


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
    """短期打分窗上的 f*：先按 night 掩夜间（与旧实现一致），再交给 decision.global_shrink_factor
    统一做 isfinite(y)&isfinite(p)&isfinite(cap) 掩码——不能只掩 y：median 里单点 NaN（如某步分位
    退化）若不经 isfinite(p) 过滤会直接把 f* 整体 NaN 掉。result_df 为空、或全部点被过滤剩 0 个时
    （无收缩信息），返回 1.0（不收缩），不抛错。"""
    lo, hi = cfg["data"]["scored_steps"]["short"]
    ys, ps, caps = [], [], []
    for r in result_df.itertuples(index=False):
        y = np.asarray(r.observe_power_future, float)[lo:hi]; p = np.asarray(r.median, float)[lo:hi]
        night = np.asarray(r.night, bool)[lo:hi]
        ok = np.isfinite(y) & ~night
        ys.append(y[ok]); ps.append(p[ok]); caps.append(np.full(ok.sum(), float(r.cap)))
    if not ys or sum(a.size for a in ys) == 0:
        return 1.0
    y = np.concatenate(ys); p = np.concatenate(ps); cap = np.concatenate(caps)
    return global_shrink_factor(y, p, cap, float(cfg["decision"]["point"]["floor_ratio"]))


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


def _clip_calibration_window(res_cal: pd.DataFrame, cal_end, window_days: int) -> pd.DataFrame:
    """按 calibration.window_days 把 res_cal 收窄到滚动窗口 [cal_end - window_days, cal_end)（上界已由
    filter_by_time(cal_start, cal_end) 保证，这里只再加下界）。窗口内一行都没有时抛 ValueError 报出区间，
    而不是静默在过长的 cal 段全量上拟合。"""
    lo = pd.Timestamp(cal_end) - pd.Timedelta(days=int(window_days))
    hi = pd.Timestamp(cal_end)
    out = res_cal[res_cal["timestamp_win"] >= lo]
    if out.empty:
        raise ValueError(f"calibration.window_days={window_days} 对应的窗口 [{lo}, {hi}) 内没有样本"
                         f"（res_cal 覆盖的 timestamp_win 范围: "
                         f"{res_cal['timestamp_win'].min() if len(res_cal) else None} .. "
                         f"{res_cal['timestamp_win'].max() if len(res_cal) else None}）")
    return out


def _raw_override(config_path) -> dict:
    """把 config_path（yaml 路径 / dict / None）解析成尚未与 DEFAULT_CFG 合并、未插值的 override dict。
    只用于 out_dir 覆盖时重建 cfg：不能直接改已插值 cfg 的 run.out_dir 后再调一次 interpolate ——
    ${run.out_dir} 占位符字面量在第一次 interpolate 时已被替换掉的字符串里不会再出现，二次插值是空操作，
    calibration.table_path / eval.report_dir 仍会停在旧 out_dir 下。必须回到合并前（占位符还在）的
    dict，把新 out_dir 合进去，再整体走一遍 load_config 的 deep_merge+interpolate。"""
    if config_path is None:
        return {}
    if isinstance(config_path, dict):
        return config_path
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}


def backfill_and_report(config_path, parquet_path, cal_start, cal_end, val_start, val_end, out_dir=None) -> pd.DataFrame:
    """零样本回填 cal+val → 在 cal 上按 calibration.window_days 收窄的滚动窗口拟合校准与 f* →
    在 val 上出阶梯报表。cal_end 不得晚于 val_start（否则拒绝：校准/验证段重叠即时间泄漏）。
    STATE 产物（calibration.parquet + sidecar / decision_state.json / result_cal.parquet /
    result_val.parquet）与 REPORT 产物（ladder.csv，落 cfg['eval']['report_dir']）都从同一份
    cfg 派生路径；给了 out_dir 时会先把它合进 run.out_dir 再重新整体插值，五个产物一起跟着挪，
    不会出现 STATE 跟着 out_dir 走、REPORT/calibration.table_path 仍停在旧 out_dir 的分裂。"""
    if pd.Timestamp(cal_end) > pd.Timestamp(val_start):
        raise ValueError(f"cal_end={cal_end!r} 晚于 val_start={val_start!r}：校准段与验证段重叠会造成时间泄漏")
    if out_dir is not None:
        config_path = deep_merge(_raw_override(config_path), {"run": {"out_dir": str(out_dir)}})
    cfg = load_config(config_path)
    out = Path(cfg["run"]["out_dir"]); out.mkdir(parents=True, exist_ok=True)
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
    res_cal = _clip_calibration_window(res_cal, cal_end, cfg["calibration"]["window_days"])
    calib = fit_calibrator_from_result(res_cal, cfg); calib.save(cfg["calibration"]["table_path"])
    f = fit_shrink_from_result(res_cal, cfg)
    gccap = {model.station_meta.name(s): float(c) for s, c in
             res_val.drop_duplicates("station_id")[["station_id", "cap"]].itertuples(index=False)}
    rep = ladder_report(res_val, gccap, cfg, calib, f)
    rep.to_csv(report_dir / "ladder.csv", index=False)
    res_cal.to_parquet(out / "result_cal.parquet", index=False); res_val.to_parquet(out / "result_val.parquet", index=False)
    (out / "decision_state.json").write_text(json.dumps({"shrink_factor": f}, indent=2), encoding="utf-8")
    print(rep.to_string(index=False))
    print(f"\n拟合 f* = {f:.4f}（写入 decision.point.shrink_factor 后，还须把 decision.point.default"
          f" 或某条 by_step 规则的 method 设为 'global_shrink' 才会生效——p_star/p_star_l1 不读它）")
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
