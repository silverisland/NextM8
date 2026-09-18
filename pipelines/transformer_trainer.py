"""
framework.py —— 框架工程师维护，在这里优化训练流程和训练框架。
模型设计 员工不碰这个文件。

提供：
  FrameworkTrainer(Trainer)  — 委托 compute_loss 给模型
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
from pathlib import Path
from typing import Dict, List, Callable

from transformers import Trainer, TrainingArguments

try:
    from .transformer_utils import assert_cuda_available, print_run_fingerprint
    from .transformer_dataset import UnifiedDataset
except ImportError:
    from transformer_utils import assert_cuda_available, print_run_fingerprint
    from transformer_dataset import UnifiedDataset

def train_model(
    training_config:       str | Path | Dict | TrainingArguments,       # output_dir = output_dir / （函数内部model_name）/ run_id ("") / ...
    model: nn.Module,
    # model_config:          str | Path | Dict, # | ***Config,       # model_name: str | None, 
    train_dataset:         str | Path | pd.DataFrame | UnifiedDataset | None, 
    eval_dataset:          str | Path | pd.DataFrame | UnifiedDataset | None, 
    collate_fn: Callable | None, 
    dataset_config: str | Path | Dict | None = None , # | ***Config,       # 使用现有的dataset config 即可 
    **kwargs
) -> Trainer:

    """
    训练函数。所有差异由  training_args + model 提供。

    Args:
        training_args:  HF的 TrainArgument 类, 需要和配置文件进行绑定。
        model:          nn.Module，需实现 forward + compute_loss + compute_metrics_numpy + get_training_config（后两者可缺省）
        train_dataset:  HuggingFace Dataset
        eval_dataset:   HuggingFace Dataset
        output_dir:     输出目录
        num_epochs:     覆盖模型配置中的 epoch 数
        batch_size:     覆盖模型配置中的 batch_size

    Returns:
        trainer: 完成训练的 FrameworkTrainer 实例
    """

    # ── 判断训练设备是否和设置参数相符 ──
    if not training_config.use_cpu:
        assert_cuda_available()

    # ── 重新设置保存路径 ──
    
    # ── 组装 Trainer ──
    trainer = Trainer(
        model=model,
        args=training_config,
        data_collator = collate_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=model.compute_metrics
    )

    # ── 保留训练指纹 ──
    print_run_fingerprint(trainer)

    # ── 开始训练 ──
    trainer.train()

    # ── 输出完整训练日志 ──
    # print("\n" + "=" * 60)
    # print("训练完成 — 日志历史")
    # print("=" * 60)
    # for entry in trainer.state.log_history:
    #     if any(
    #         k in str(entry)
    #         for k in ("eval_loss", "eval_mse", "eval_accuracy", "eval_mae", "loss")
    #     ):
    #         print(entry)

    # ── 输出最佳指标 ──
    # best_metric = trainer.state.best_metric
    # if best_metric is not None:
    #     print(f"\n最佳 {config['metric_for_best_model']}: {best_metric:.6f}")

    # ── 退出训练，清除accelerate 子进程 ──
    # 优雅退出，减少内存消耗 
    # 优先走 Trainer 内部的 Accelerator
    if trainer is not None and hasattr(trainer, "accelerator"):
        trainer.accelerator.end_training()
    else:
        # 兜底：纯 PyTorch 分布式清理
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    return trainer


# 
class FrameworkTrainer(Trainer):
    """
    不写任何 loss / metric 逻辑。只做委托 + backward + optimizer.step。
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1. 提取 labels（不 pop，保留给 prediction_step 用）
        labels = inputs.get("labels")

        # 2. forward（过滤掉 labels，不让它作为模型参数进入 forward）
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)

        # 3. 委托给模型开发者 —— 多态入口
        if hasattr(model, "compute_loss") and labels is not None:
            loss = model.compute_loss(outputs, labels, num_items_in_batch)
        elif labels is not None:
            # fallback: 模型 forward 已在 outputs 中返回 loss
            loss = outputs.get("loss", None)
            if loss is None:
                raise ValueError(
                    "模型的 forward 没有返回 'loss'，且没有实现 compute_loss 方法。"
                    "请在模型中实现 compute_loss(self, outputs, labels, num_items_in_batch=None)。"
                )
        else:
            # 纯推理，没有 labels
            loss = outputs.get("loss", torch.tensor(0.0, requires_grad=True))

        return (loss, outputs) if return_outputs else loss