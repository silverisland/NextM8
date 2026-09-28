# -*- coding: utf-8 -*-
# forecast_backbones/m7/transformer_trainer.py
"""M7（PatchTST-FM-r2 零样本）的框架契约入口：evaluate_model(model, test_dataset, HF_collate_fn, config) -> DataFrame。
M7 与 M5 同为零样本基础模型、输出契约相同（logits [B,H] + aux["quantiles"] [B,H,9]），
所以直接复用 M5 的 evaluate_model，结果表列完全一致（timestamp_win / station_id /
observe_power_predict / observe_power_future / q_0.1..q_0.9）。
registry.build_evaluator("m7") 只要求本文件与 evaluate_model 存在。"""
from __future__ import annotations

from forecast_backbones.m5.transformer_trainer import MODEL_LEVELS, QUANTILE_COLS, evaluate_model

__all__ = ["MODEL_LEVELS", "QUANTILE_COLS", "evaluate_model"]
