# -*- coding: utf-8 -*-
# forecast_backbones/m6/nanwang_callback.py
"""训练期南网口径打分 + 早停（spec §5.2 第 3 步）。不走 HF compute_metrics（labels 是 dict，
Trainer 的 eval loop 拼不起来），而是独立回调：每个 epoch 末 cal 段推理拟合 δ → val 段推理 → 校准 → p* →
夜间归零 → NanwangMetric.short_d1_daily → 均值即 nanwang_after_pstar。epoch 0 先记零样本基线并存为初始最优，
所以保存的权重在 val 上永远不输零样本（全体口径）；逐站回退在 train.py 训后做。"""
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
    from .lora import lora_state_dict
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from transformer_trainer import evaluate_model, fit_calibrator_from_result, recompute_points
    from lora import lora_state_dict
try:
    from ...pipelines.nanwang_metric import NanwangMetric
except ImportError:
    from pipelines.nanwang_metric import NanwangMetric


def score_split(model, ds_cal, ds_val, collate, cfg) -> tuple[float, pd.DataFrame, object]:
    if getattr(model, "calibrator", None) is not None:
        raise ValueError("score_split 需要未挂 calibrator 的模型（训练模型应输出原始分位，校准在这里拟合）")
    was_training = bool(model.training)
    try:
        res_cal = evaluate_model(model, ds_cal, collate, cfg)
        calib = fit_calibrator_from_result(res_cal, cfg)
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
            table = pd.DataFrame(columns=["station_id", "station", "accuracy_pct", "n_days"]); score = float("nan")
        return score, table, calib
    finally:
        model.train(was_training)


class NanwangEvalCallback(TrainerCallback):
    def __init__(self, model, ds_cal, ds_val, collate, cfg, best_path, patience: int, log_path):
        self.model, self.ds_cal, self.ds_val, self.collate, self.cfg = model, ds_cal, ds_val, collate, cfg
        self.best_path, self.log_path, self.patience = Path(best_path), Path(log_path), int(patience)
        self.best, self.best_epoch, self.bad_epochs, self.history = -np.inf, None, 0, []

    def _score(self) -> float:
        score, _, _ = score_split(self.model, self.ds_cal, self.ds_val, self.collate, self.cfg)
        return float(score)

    def _save_best(self):
        self.best_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(lora_state_dict(self.model.backbone.model), self.best_path)

    def _record(self, epoch: float, score: float, control):
        if np.isfinite(score) and score > self.best:
            self.best, self.best_epoch, self.bad_epochs = score, float(epoch), 0
            self._save_best()
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience and control is not None:
                control.should_training_stop = True
        self.history.append({"epoch": float(epoch), "nanwang_after_pstar": score, "best": self.best, "bad_epochs": self.bad_epochs})
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(json.dumps({"history": self.history, "best": self.best, "best_epoch": self.best_epoch,
                                             "patience": self.patience}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[m6-lora] epoch {epoch:g}: nanwang_after_pstar={score:.4f} best={self.best:.4f}@{self.best_epoch} bad={self.bad_epochs}")

    def on_train_begin(self, args, state, control, **kwargs):
        self._record(0.0, self._score(), None)          # 零样本基线：LoRA B=0，等价零样本；永远是初始最优
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        self._record(float(state.epoch or 0.0), self._score(), control)
        return control
