"""
framework.py —— 框架工程师维护，只写一次。
模型设计 员工不碰这个文件。

提供：
  FrameworkTrainer(Trainer)  — 委托 compute_loss 给模型
  build_and_train()          — 一锤子训练入口
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

from transformers import TrainingArguments


BASE_PATH = str(Path(__file__).resolve().parent)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from .transformer_models import TransformersAdaptedM4Model
except ImportError:
    from transformer_models import TransformersAdaptedM4Model

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def Q90(pred, true, q, idlist):
    q = q[:true.shape[0]]
    for id in set(idlist):
        index = np.where(idlist == id)
        id_true = true[index]
        id_pred = pred[index]
        id_q = q[index]
        id_q90 = 1 - np.sqrt(mean_squared_error(id_true.reshape(-1, 24), id_pred.reshape(-1, 24))) /id_q[0]
        print(f'id:{id}    q90:{id_q90}')
    return 

def C(pred, true, c, idlist):

    for id in set(idlist):
        index = np.where(idlist == id)
        id_true = true[index]
        id_pred = pred[index]
        id_c = c[index]
        id_cvalue = 1 - np.sqrt(mean_squared_error(id_true.reshape(-1, 24), id_pred.reshape(-1, 24))) / id_c[0]
        print(f'id:{id}    q90:{id_cvalue}')
    return 

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


# 这里需要每个模型自己写，因此需要出一个简单的说明文档辅助对齐 
def M4_build_training_args(config_dict: dict, result_dir):

    # ── 这里写一些config 内部没有的配置，后续建议合并 ──
    config = {
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "early_stopping_patience": 10,
        "learning_rate": 1e-3,
        "num_train_epochs": 100,
        "per_device_train_batch_size": 256,
        "per_device_eval_batch_size": 256,
        "weight_decay": 1e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.01,
        "report_to": "none",
        "logging_dir": None,
        "run_name": None,
        "output_dir":  Path(result_dir) / "hf_trainer_output"
    }

    # 允许调用方覆盖
    config["num_train_epochs"] = config_dict['optimization']['train_epochs']
    config["learning_rate"] = config_dict["optimization"]["learning_rate"]
    config["per_device_train_batch_size"] = config_dict["optimization"]["batch_size"] // (torch.cuda.device_count() if torch.cuda.is_available() else 1 )
    config["fp16"] = config_dict["optimization"]["fp16"]

    # ── 构建 TrainingArguments ──
    training_args = TrainingArguments(
        output_dir = config['output_dir'],
        num_train_epochs= config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        # find_unused_parameters=False,

        # 梯度相关操作 
        # gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        # max_grad_norm=10.0,

        # 训练设备以及训练策略
        use_cpu = False,
        fp16=config["fp16"],

        # 优化器
        optim = "adamw_torch",
        learning_rate = config["learning_rate"],
        weight_decay=config["weight_decay"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],

        # eval / save 策略
        # eval_strategy="steps",
        save_strategy="epoch",
        metric_for_best_model=config["metric_for_best_model"],
        greater_is_better=config["greater_is_better"],
        # load_best_model_at_end=True,

        # logging 策略
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,


        # 其他工程相关
        remove_unused_columns=False,
        dataloader_drop_last=False,
          
        report_to=config["report_to"],
        logging_dir=config["logging_dir"],
        run_name=config["run_name"],
        save_total_limit=3,
    )

    return training_args


@torch.no_grad()
def evaluate_model(
    model: TransformersAdaptedM4Model,
    test_dataset,
    HF_collate_fn,
    config: dict,
) -> pd.DataFrame:
    """
    使用训练好的 HF Trainer 模型在测试集上做推理，输出 result_df。

    输出 result_df 格式（与 Exp_Main.inference 保持一致）:
        timestamp_win:         (N,)      时间戳
        observe_power_predict: (N,)      每行是 (pred_len,) array — 反归一化预测值
        observe_power_future:  (N,)      每行是 (pred_len,) array — 真实值

    Args:
        model:          已训练的 TransformersAdaptedM4Model
        test_dataset:   HFPVDataset(test) 实例
        config:         完整配置字典
        batch_size:     推理 batch size
        device:         计算设备

    Returns:
        result_df:      与 Exp_Main.inference 一致格式的 DataFrame
    """
    model.eval()
    device = config["optimization"]["device"]
    model.to(device)
    batch_size = config["optimization"]["batch_size"]

    # ── 用 DataLoader 包装 HFPVDataset，使用 HF_collate_fn ──
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=HF_collate_fn,
    )

    preds = []
    trues = []
    with torch.no_grad():
        # with torch.cuda.amp.autocast():        # ← 加这一行，和训练 fp16 对齐
            for batch_dict in dataloader:

                # 模型前向推理（test 模式下，batch_y 为 None → flag='test'）
                outputs = model(batch_dict["batch_dict"])
                # outputs.prediction: (B, S') — 已经是反归一化后的原始功率值

                pred = outputs.predictions.detach().cpu().numpy()
                # m4 模型返回的 predictions 形状是 (B, 1, S')，而 labels 是 (B, S')；
                # 若不去掉中间的 1 维，metric(preds, trues) 会把两者广播成 (N, N, S')≈940GB，导致 OOM Killed
                pred = np.squeeze(pred, axis=1) if pred.ndim == 3 else pred   # (B, 1, S') → (B, S')
                true = batch_dict["labels"]["observe_power_future"].detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

    # ── 拼接所有 batch 的预测结果 ──
    trues = np.concatenate(trues)
    preds = np.concatenate(preds)  # (N, pred_len)

    # ── 提取时间戳和真值 ──
    timestamps = test_dataset.timestamp             # (N,)

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print(f'mae:{mae}   mse:{mse},  rmse:{rmse}')

    # ── 组装 result_df ──
    result_df = pd.DataFrame({
        "timestamp_win": timestamps,
        "observe_power_predict": list(preds),
        "observe_power_future": list(trues),
    })

    return result_df

