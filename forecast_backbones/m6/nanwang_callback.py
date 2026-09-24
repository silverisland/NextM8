# -*- coding: utf-8 -*-
# forecast_backbones/m6/nanwang_callback.py
"""训练期打分 + 早停（spec §5.2 第 3 步）。早停指标 finetune.early_stop.metric：
nanwang_after_pstar（默认，越高越好）= 南网短期 D+1；nrmse（越低越好）= 提交曲线全部 192 步的容量归一 RMSE（%），
只推理起报时刻落在 nrmse_issue_every_min 分钟网格上的行。以下按南网口径描述，nrmse 的流程相同，只是打分换掉。不走 HF compute_metrics（labels 是 dict，
Trainer 的 eval loop 拼不起来），而是独立回调：每 eval_every_steps 个优化器步（未设则每个 epoch 末）cal 段推理拟合 δ →
各验证块推理 → 校准 → p* → 夜间归零 → NanwangMetric.short_d1_daily → 块均值即 nanwang_after_pstar。
训练开始先记零样本基线并存为初始最优，所以保存的权重在验证集上永远不输零样本（全体口径）。
验证只推理南网短期指标用到的行（每站每个起报日一行），分数与推理全部行相同。
多卡时只在主进程打分、存盘、写日志，是否早停广播给所有进程。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import TrainerCallback

try:
    from .transformer_trainer import evaluate_model, fit_calibrator_from_result, recompute_points
    from .lora import lora_state_dict, save_lora_weights
    from .rows import SHORT_ISSUE_TIME, issue_grid_row_index, scored_row_index
    from . import _dist
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from transformer_trainer import evaluate_model, fit_calibrator_from_result, recompute_points
    from lora import lora_state_dict, save_lora_weights
    from rows import SHORT_ISSUE_TIME, issue_grid_row_index, scored_row_index
    import _dist
try:
    from ...pipelines.nanwang_metric import NanwangMetric
except ImportError:
    from pipelines.nanwang_metric import NanwangMetric

TABLE_COLS = ["station_id", "station", "accuracy_pct", "n_days"]
EARLY_STOP_METRICS = {"nanwang_after_pstar": True, "nrmse": False}      # 指标 → 是否越高越好


def fit_split_calibrator(model, ds_cal, collate, cfg):
    """ds_cal 上推理并拟合校准表；ds_cal=None（不做 conformal）返回 None。"""
    if ds_cal is None:
        return None
    was_training = bool(model.training)
    try:
        return fit_calibrator_from_result(evaluate_model(model, ds_cal, collate, cfg), cfg)
    finally:
        model.train(was_training)


def score_split(model, ds_cal, ds_val, collate, cfg, *, calib=None,
                scored_rows_only: bool = True) -> tuple[float, pd.DataFrame, object]:
    """ds_cal 上拟合校准表 → ds_val 上按 decision.point 取点 → 南网短期日准确率。ds_cal=None 时不校准，
    点预测在未校准的原生分位上取，返回的校准表为 None。calib：同一模型状态下已拟合好的校准表，给了就不再推理 ds_cal。
    scored_rows_only：只推理指标用到的行（每站每个起报日一行），分数与推理全部行相同，推理量约为 1/96。
    返回 (日准确率均值, 逐站表, 校准表)。"""
    if getattr(model, "calibrator", None) is not None:
        raise ValueError("score_split 需要未挂 calibrator 的模型（训练模型应输出原始分位，校准在这里拟合）")
    if calib is None:
        calib = fit_split_calibrator(model, ds_cal, collate, cfg)
    was_training = bool(model.training)
    try:
        if scored_rows_only:
            idx = scored_row_index(ds_val, str(cfg["dataset"]["station_id_col"]), str(cfg["dataset"]["ts_col"]))
            if len(idx) == 0:
                return float("nan"), pd.DataFrame(columns=TABLE_COLS), calib
            ds_val = torch.utils.data.Subset(ds_val, idx.tolist())
        res_val = evaluate_model(model, ds_val, collate, cfg)
        gccap = {model.station_meta.name(int(s)): float(c) for s, c in
                 res_val.drop_duplicates("station_id")[["station_id", "cap"]].itertuples(index=False)}
        df = recompute_points(res_val, cfg, calib, "calibrated_pstar", shrink=1.0)
        sd = NanwangMetric().short_d1_daily(df, gccap)
        name_to_id = {model.station_meta.name(int(s)): int(s) for s in res_val["station_id"].unique()}
        if len(sd):
            g = sd.groupby("station")["accuracy_pct"].agg(["mean", "count"]).reset_index()
            table = pd.DataFrame({"station_id": g["station"].map(name_to_id).astype(int), "station": g["station"],
                                  "accuracy_pct": g["mean"].astype(float), "n_days": g["count"].astype(int)})
            score = float(sd["accuracy_pct"].mean())
        else:
            table, score = pd.DataFrame(columns=TABLE_COLS), float("nan")
        return score, table, calib
    finally:
        model.train(was_training)


def nrmse_split(model, ds_val, collate, cfg, *, calib=None, issue_every_min: int = 60) -> float:
    """ds_val 中起报时刻落在 issue_every_min 网格上的行 → 推理 → 按 calib 校准、decision.point 取提交曲线 →
    全部 192 步（含夜间，与 ODS 口径一致）的容量归一 RMSE，单位 %。没有可用的点返回 NaN。"""
    idx = issue_grid_row_index(ds_val, str(cfg["dataset"]["ts_col"]), issue_every_min)
    if len(idx) == 0:
        return float("nan")
    was_training = bool(model.training)
    try:
        res = evaluate_model(model, torch.utils.data.Subset(ds_val, idx.tolist()), collate, cfg)
        df = recompute_points(res, cfg, calib, "calibrated_pstar", shrink=1.0)
    finally:
        model.train(was_training)
    y = np.stack(df["observe_power_future"].to_numpy()).astype(float)
    p = np.stack(df["observe_power_predict"].to_numpy()).astype(float)
    r = (y - p) / df["cap"].to_numpy(float)[:, None]
    ok = np.isfinite(r)
    return float(100.0 * np.sqrt(np.mean(r[ok] ** 2))) if ok.any() else float("nan")


class NanwangEvalCallback(TrainerCallback):
    """ds_val_blocks：验证块数据集列表；每块各自打分，块均值（等权、跳过非有限）作早停指标，逐块分数记入 history。
    eval_every_steps：每多少个优化器步打一次分，训练的最后一步必打；None = 每个 epoch 末打。patience 按打分次数计。"""

    def __init__(self, model, ds_cal, ds_val_blocks, collate, cfg, best_path, patience: int, log_path,
                 eval_every_steps: int | None = None):
        if not isinstance(ds_val_blocks, (list, tuple)) or len(ds_val_blocks) == 0:
            raise ValueError("ds_val_blocks 须是非空列表（每个元素一个验证块数据集）")
        self.model, self.ds_cal, self.ds_val_blocks, self.collate, self.cfg = model, ds_cal, list(ds_val_blocks), collate, cfg
        self.best_path, self.log_path, self.patience = Path(best_path), Path(log_path), int(patience)
        self.eval_every_steps = int(eval_every_steps) if eval_every_steps else None
        es = (cfg.get("finetune") or {}).get("early_stop") or {}
        self.metric = str(es.get("metric", "nanwang_after_pstar"))
        if self.metric not in EARLY_STOP_METRICS:
            raise ValueError(f"finetune.early_stop.metric={self.metric!r} 未知，可选 {sorted(EARLY_STOP_METRICS)}")
        self.higher_better = EARLY_STOP_METRICS[self.metric]
        self.issue_every_min = int(es.get("nrmse_issue_every_min", 60))
        self.best = -np.inf if self.higher_better else np.inf
        self.best_epoch, self.best_step, self.bad_evals, self.history = None, None, 0, []

    def _score(self) -> tuple[float, list[float]]:
        per, n = [], len(self.ds_val_blocks)
        self.model._progress_desc = "[m6-lora] 校准段"
        try:
            calib = fit_split_calibrator(self.model, self.ds_cal, self.collate, self.cfg)     # 各块共用同一张校准表
        finally:
            self.model._progress_desc = None
        for i, ds in enumerate(self.ds_val_blocks):
            self.model._progress_desc = f"[m6-lora] 验证块 {i + 1}/{n}"          # evaluate_model 的进度条标题
            try:
                if self.metric == "nrmse":
                    per.append(nrmse_split(self.model, ds, self.collate, self.cfg, calib=calib,
                                           issue_every_min=self.issue_every_min))
                else:
                    per.append(float(score_split(self.model, None, ds, self.collate, self.cfg, calib=calib)[0]))
            finally:
                self.model._progress_desc = None
        finite = [s for s in per if np.isfinite(s)]
        return (float(np.mean(finite)) if finite else float("nan")), per

    def _save_best(self):
        save_lora_weights(self.best_path, lora_state_dict(self.model.backbone.model))

    def _record(self, step: int, epoch: float, scored: tuple[float, list[float]]) -> bool:
        """记一次打分；返回是否已连续 patience 次没有超过最优。"""
        score, per = scored
        better = score > self.best if self.higher_better else score < self.best
        if np.isfinite(score) and better:
            self.best, self.best_epoch, self.best_step, self.bad_evals = score, float(epoch), int(step), 0
            self._save_best()
        else:
            self.bad_evals += 1
        self.history.append({"step": int(step), "epoch": float(epoch), self.metric: score, "blocks": per,
                             "best": self.best, "bad_evals": self.bad_evals})
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(json.dumps({"history": self.history, "best": self.best, "best_epoch": self.best_epoch,
                                             "best_step": self.best_step, "patience": self.patience,
                                             "metric": self.metric, "higher_better": self.higher_better,
                                             "eval_every_steps": self.eval_every_steps},
                                            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[m6-lora] step {step} (epoch {epoch:.3g}): {self.metric}={score:.4f} "
              f"blocks={[round(s, 4) for s in per]} best={self.best:.4f}@step {self.best_step} "
              f"bad={self.bad_evals}/{self.patience}", flush=True)
        return self.bad_evals >= self.patience

    def _evaluate(self, state, control):
        stop = self._record(int(state.global_step or 0), float(state.epoch or 0.0), self._score()) if _dist.is_main() else False
        if _dist.broadcast_from_main(stop) and control is not None:
            control.should_training_stop = True

    def on_train_begin(self, args, state, control, **kwargs):
        if _dist.is_main():
            rows = (f"每站每天推理起报时刻在 {self.issue_every_min} 分钟网格上的行" if self.metric == "nrmse"
                    else f"每站每天只推理 {SHORT_ISSUE_TIME} 前最晚的一行")
            print(f"[m6-lora] 开训前先在 {len(self.ds_val_blocks)} 个验证块上打零样本基线分，早停指标 {self.metric}"
                  f"（{rows}）…", flush=True)
            self._record(0, 0.0, self._score())         # 零样本基线：LoRA B=0，等价零样本；永远是初始最优，不触发早停
        _dist.barrier()                                 # 其余进程等主进程打完基线
        return control

    def on_step_end(self, args, state, control, **kwargs):
        every = self.eval_every_steps
        if every and (state.global_step % every == 0 or state.global_step >= state.max_steps):
            self._evaluate(state, control)
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if not self.eval_every_steps:
            self._evaluate(state, control)
        return control
