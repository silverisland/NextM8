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
    """[N,192,9]，向量化装配（不用 iterrows()）：每个分位列先各自 stack 成 [N,192]，再沿新尾轴叠成 [N,192,9]。"""
    cols = quantile_cols(MODEL_LEVELS)
    per_col = [np.stack(result_df[c].to_numpy()) for c in cols]                 # 每列 [N,192]
    return np.stack(per_col, axis=-1)                                          # [N,192,9]


def _recompute_points_chunk(result_df: pd.DataFrame, cfg: dict, calibrator, variant: str,
                            shrink: float, q: np.ndarray | None) -> pd.DataFrame:
    if q is None:
        q = _stack_q(result_df)
    N, H, _ = q.shape
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


def recompute_points(result_df: pd.DataFrame, cfg: dict, calibrator, variant: str, shrink: float = 1.0,
                     chunk_rows: int = 256, q: np.ndarray | None = None) -> pd.DataFrame:
    """按 chunk_rows 行分块处理 result_df 再 pd.concat（限制单次峰值内存），输出与不分块时逐位相同。
    q：可选的预先 _stack_q 好的 [N,192,9] 数组（ladder_report 用它把 _stack_q 提到打分轮次循环外，
    避免同一 result_df 在每个 rung 都被重新堆叠一次）；按同样的行区间切片喂给每个分块。"""
    tail = cfg["decision"]["point"].get("tail", "flat")
    if tail != "flat":
        raise ValueError(f"decision.point.tail: only 'flat' is implemented (got {tail!r})")
    N = len(result_df)
    if N <= chunk_rows:
        return _recompute_points_chunk(result_df, cfg, calibrator, variant, shrink, q)
    parts = []
    track = variant == "calibrated_pstar" and calibrator is not None
    n_seen, uncal_counts = 0, None                      # 跨分块累加 apply() 的未校准格计数（不拼接 last_levels）
    for start in range(0, N, chunk_rows):
        end = min(start + chunk_rows, N)
        chunk = result_df.iloc[start:end]
        q_chunk = q[start:end] if q is not None else None
        parts.append(_recompute_points_chunk(chunk, cfg, calibrator, variant, shrink, q_chunk))
        if track:
            uncal = calibrator.last_levels == -1                 # [n_chunk*H, L]
            uncal_counts = uncal.sum(axis=0) if uncal_counts is None else uncal_counts + uncal.sum(axis=0)
            n_seen += int(uncal.shape[0])
    if track:
        calibrator.last_levels = None                    # 只保留全集聚合统计；逐格层号不跨分块保存（内存）
        calibrator.last_stats = _aggregate_calibration_stats(calibrator, n_seen, uncal_counts)
    return pd.concat(parts)


def _aggregate_calibration_stats(calibrator, n_seen: int, uncal_counts: np.ndarray | None) -> dict:
    """把分块 apply() 的未校准格计数汇成与 Calibrator.apply 单次调用同构的 last_stats（全集口径）。"""
    taus = [float(t) for t in calibrator.cfg.levels]
    L = len(taus)
    counts = np.zeros(L, float) if uncal_counts is None else np.asarray(uncal_counts, float)
    return {
        "n": int(n_seen),
        "uncalibrated_cells": int(counts.sum()),
        "uncalibrated_frac": float(counts.sum() / (n_seen * L)) if n_seen * L > 0 else 0.0,
        "per_tau_uncalibrated_frac": {t: (float(counts[j] / n_seen) if n_seen > 0 else 0.0) for j, t in enumerate(taus)},
    }


def _nrmse_short_pct(df: pd.DataFrame, cfg: dict) -> float:
    lo, hi = cfg["data"]["scored_steps"]["short"]
    se, n = 0.0, 0
    for _, r in df.iterrows():
        y = np.asarray(r["observe_power_future"], float)[lo:hi]; p = np.asarray(r["observe_power_predict"], float)[lo:hi]
        ok = np.isfinite(y) & np.isfinite(p)
        se += float(np.sum(((y[ok] - p[ok]) / float(r["cap"])) ** 2)); n += int(ok.sum())
    return 100.0 * float(np.sqrt(se / n)) if n else float("nan")


LADDER = ("raw_median", "global_shrink", "physics_median", "calibrated_pstar")


def ladder_report(result_df: pd.DataFrame, gccap: dict, cfg: dict, calibrator, shrink: float,
                  chunk_rows: int = 256) -> pd.DataFrame:
    nm = NanwangMetric()
    q = _stack_q(result_df)                    # 提到打分轮次循环外：4 个 rung 共用同一份堆叠结果
    rows = []
    for v in LADDER:
        df = recompute_points(result_df, cfg, calibrator, v, shrink, chunk_rows=chunk_rows, q=q)
        sd = nm.short_d1_daily(df, gccap); ud = nm.ultra_short_daily(df, gccap)
        rows.append({"variant": v,
                     "nanwang_short_D+1": float(sd["accuracy_pct"].mean()) if len(sd) else float("nan"),
                     "nanwang_ultra_short": float(ud["accuracy_pct"].mean()) if len(ud) else float("nan"),
                     "nrmse_short_pct": _nrmse_short_pct(df, cfg),
                     "n_days": int(sd["date"].nunique()) if len(sd) else 0})
    return pd.DataFrame(rows)


def _diagnostics_block(res_cal: pd.DataFrame, res_val: pd.DataFrame, calibrator, gccap: dict,
                       shrink: float, cfg: dict) -> dict:
    """I4：ladder.csv 旁 diagnostics.json 的内容。calibrator_last_stats 取 calibrator.last_stats——
    recompute_points 的 calibrated_pstar 变体分块调用 apply() 后会把各块的未校准格计数汇成全集口径写回
    calibrator.last_stats，所以 ladder_report 跑完后直接读即可，不再对整个 val 集单独重跑一次不分块的
    apply()（那会重新引入 C2 修掉的整集内存峰值）。若调用时 last_stats 尚未填充（没先跑 ladder_report），
    则按分块方式跑一遍 calibrated_pstar 来填充。"""
    var_bin_hist = {int(b): 0 for b in range(int(cfg["variability"]["bins"]))}
    if calibrator is not None and len(res_val) > 0 and calibrator.last_stats is None:
        recompute_points(res_val, cfg, calibrator, "calibrated_pstar", shrink)
    for b, n in res_val["var_bin"].value_counts().items():
        var_bin_hist[int(b)] = var_bin_hist.get(int(b), 0) + int(n)
    return {
        "calibrator_last_stats": calibrator.last_stats if calibrator is not None else None,
        "degraded_frac": float(res_val["degraded"].mean()) if len(res_val) else 0.0,
        "n_rows_val": int(len(res_val)),
        "n_rows_cal": int(len(res_cal)),
        "var_bin_hist": var_bin_hist,
        "stations_scored": sorted(gccap.keys()),
        "shrink_factor": float(shrink),
    }


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
    diagnostics = _diagnostics_block(res_cal, res_val, calib, gccap, f, cfg)
    (report_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(f"\ndiagnostics: {json.dumps(diagnostics, indent=2)}")
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
