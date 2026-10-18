# -*- coding: utf-8 -*-
# forecast_backbones/m5/transformer_trainer.py
"""M5 的框架契约入口：evaluate_model(model, test_dataset, HF_collate_fn, config) -> DataFrame（registry.build_evaluator("m5")）。
  legacy 模式（线上零样本配置）：结果表与重构前相同（timestamp_win / station_id / observe_power_predict / observe_power_future / q_*）。
  full 模式（给了站表）：结果表与 M8 同 schema（另带 station / cap / p_star / median / night / elev_deg / p_clear / var_bin /
  var_signal / degraded），train.py 的打分（score_nanwang / nrmse）直接吃。
build_training_args：train.py 用的 HF TrainingArguments（与 M8 相同）。"""
from __future__ import annotations

import inspect
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

MODEL_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
QUANTILE_COLS = [f"q_{float(l):g}" for l in MODEL_LEVELS]


def _np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def evaluate_model(model, test_dataset, HF_collate_fn, config) -> pd.DataFrame:
    if getattr(model, "mode", "legacy") == "full":
        return _evaluate_full(model, test_dataset, HF_collate_fn)
    return _evaluate_legacy(model, test_dataset, HF_collate_fn, config)


def _evaluate_legacy(model, test_dataset, HF_collate_fn, config) -> pd.DataFrame:
    """逐 batch 零样本推理。列名从 config.dataset 取（ts_col / label_col / station_id_col），缺省与 M6 一致；
    绝不把 label_col 喂给模型。站标识列若被 collate 丢掉（字符串站名）则结果表里没有 station_id 列。"""
    cfg = config or {}
    ds_cfg = cfg.get("dataset") or {}
    ts_col = ds_cfg.get("ts_col", "timestamp_win")
    label_col = ds_cfg.get("label_col", "observe_power_future")
    sid_col = ds_cfg.get("station_id_col", "station_id")
    bs = int((cfg.get("eval") or {}).get("batch_size", 64))
    H = int(model.pred_len)
    loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, collate_fn=HF_collate_fn)
    rows = []
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, dict) and "batch_dict" in batch:
                    bd, labels = batch["batch_dict"], batch.get("labels")
                else:
                    bd, labels = batch, None
                bd = {k: v for k, v in bd.items() if k != label_col}
                truth = _np(labels[label_col]) if labels is not None and label_col in labels else None
                out = model(bd)
                logits = _np(out.logits)
                q = _np(out.aux["quantiles"]) if out.aux and out.aux.get("quantiles") is not None else None
                ts = _np(bd[ts_col]).astype(np.int64).reshape(-1)
                sid = _np(bd[sid_col]).reshape(-1) if sid_col in bd else None
                for i in range(logits.shape[0]):
                    rec = {"timestamp_win": pd.Timestamp(int(ts[i]), unit="s")}
                    if sid is not None:
                        rec["station_id"] = int(sid[i])
                    rec["observe_power_predict"] = logits[i].astype(np.float32)
                    rec["observe_power_future"] = (truth[i][:H].astype(np.float32) if truth is not None
                                                  else np.full(H, np.nan, np.float32))
                    if q is not None:
                        for j, c in enumerate(QUANTILE_COLS):
                            rec[c] = q[i, :, j].astype(np.float32)
                    rows.append(rec)
    finally:
        model.train(was_training)
    return pd.DataFrame(rows)


def _evaluate_full(model, test_dataset, HF_collate_fn) -> pd.DataFrame:
    cfg = model.cfg
    bs = int(cfg["eval"].get("batch_size", 64))
    ts_col, label_col, H = cfg["dataset"]["ts_col"], cfg["dataset"]["label_col"], int(cfg["dataset"]["pred_len"])
    loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, collate_fn=HF_collate_fn)
    show = bool(cfg["train"]["schedule"].get("progress_bar", True))
    desc = str(getattr(model, "_progress_desc", None) or "[m5] 推理")
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, unit="batch", disable=not show, mininterval=2.0, dynamic_ncols=True):
            if isinstance(batch, dict) and "batch_dict" in batch:
                bd, labels = batch["batch_dict"], batch.get("labels")
            else:
                bd, labels = batch, None
            bd = {k: v for k, v in bd.items() if k != label_col}          # 绝不喂未来真值
            truth = _np(labels[label_col]) if labels is not None and label_col in labels else None
            out = model(bd)
            q = _np(out.aux["quantiles"]); logits = _np(out.logits)
            pstar = _np(out.aux["p_star"]); med = _np(out.aux["median"])
            ts = _np(bd[ts_col]).astype(np.int64).reshape(-1)
            for i in range(logits.shape[0]):
                sid = int(out.aux["station_id"][i])
                rec = {
                    "timestamp_win": pd.Timestamp(int(ts[i]), unit="s"),
                    "station": model.station_meta.name(sid), "station_id": sid, "cap": float(out.aux["cap"][i]),
                    "observe_power_predict": logits[i].astype(np.float32),
                    "observe_power_future": (truth[i][:H].astype(np.float32) if truth is not None else np.full(H, np.nan, np.float32)),
                    "p_star": pstar[i].astype(np.float32), "median": med[i].astype(np.float32),
                    "night": out.aux["night"][i].astype(bool), "elev_deg": out.aux["elev_deg"][i].astype(np.float32),
                    "p_clear": out.aux["p_clear"][i].astype(np.float32),
                    "var_bin": int(out.aux["var_bin"][i]), "var_signal": float(out.aux["var_signal"][i]),
                    "degraded": bool(out.aux["degraded"][i]),
                }
                for j, c in enumerate(QUANTILE_COLS):
                    rec[c] = q[i, :, j].astype(np.float32)
                rows.append(rec)
    return pd.DataFrame(rows)


def per_device_batch(schedule: dict) -> int:
    """train.schedule.batch_size 是全局 batch（所有卡合计）：每卡 = batch_size / WORLD_SIZE，卡数变了配方不变。"""
    gb, world = int(schedule["batch_size"]), int(os.environ.get("WORLD_SIZE") or 1)   # accelerate launch / torchrun 建进程组前就设好
    if gb % world:
        raise ValueError(f"train.schedule.batch_size={gb} 是全局 batch，必须能被卡数 {world} 整除（换 --gpus 的卡数，或换 --batch_size）")
    return gb // world


def build_training_args(cfg: dict, out_dir, max_steps=None):
    """max_steps：None = 用 train.schedule.max_steps；0（或配置里 null）= 不限步数，按 epochs_max 训。"""
    from transformers import TrainingArguments
    s = cfg["train"]["schedule"]
    steps = s.get("max_steps") if max_steps is None else max_steps
    cuda = torch.cuda.is_available()
    bf16 = str(s.get("precision", "fp32")) == "bf16" and cuda and torch.cuda.is_bf16_supported()
    fp16 = str(s.get("precision", "fp32")) == "fp16" and cuda
    eval_kw = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters else "evaluation_strategy"
    return TrainingArguments(
        output_dir=str(Path(out_dir) / "hf"), per_device_train_batch_size=per_device_batch(s),
        num_train_epochs=float(s["epochs_max"]), learning_rate=float(s["lr"]), weight_decay=float(s["weight_decay"]),
        max_grad_norm=float(s["grad_clip"]), bf16=bf16, fp16=fp16, use_cpu=not cuda,
        lr_scheduler_type=str(s.get("lr_scheduler") or "cosine"), warmup_steps=int(s.get("warmup_steps", 0)),
        save_strategy="no", logging_steps=int(s.get("logging_steps", 50)), report_to=[],
        remove_unused_columns=False, disable_tqdm=not bool(s.get("progress_bar", True)), dataloader_pin_memory=cuda,
        dataloader_num_workers=int(s.get("num_workers", 0)),
        max_steps=(int(steps) if steps else -1), seed=int((cfg["run"].get("seeds") or [0])[0]),
        ddp_timeout=7200, **{eval_kw: "no"})
