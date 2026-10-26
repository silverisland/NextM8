import warnings

import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from functools import partial

import numpy as np

from pipelines.transformer_dataset import UnifiedDataset


def _run_model_logits(model, bd):
    """跑一次 forward，返回 logits；logits 为 None 视作失败。"""
    logits = model(bd).logits
    if logits is None:
        raise ValueError("ModelOutput.logits 为 None")
    return logits


def _sub_batch(bd, idx):
    """从 batch_dict 中取第 idx 个样本，构造 batch=1 的 dict。

    只对 tensor / list / dict 按 batch 维切片；无法切分的标量型字段原样保留。
    """
    out = {}
    for k, v in bd.items():
        if isinstance(v, torch.Tensor):
            out[k] = v[idx: idx + 1]          # 保持 batch 维
        elif isinstance(v, list):
            out[k] = [v[idx]]
        elif isinstance(v, dict):
            out[k] = {kk: (vv[idx: idx + 1] if isinstance(vv, torch.Tensor) else vv)
                      for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def _fallback_per_sample(model, bd, bs):
    """batch 失败时逐样本兜底。返回长度 == bs 的 list，元素为 [w] 张量或 None。"""
    out = []
    for i in range(bs):
        try:
            sbd = _sub_batch(bd, i)
            logits = _run_model_logits(model, sbd)      # [1, w]
            out.append(logits.detach().cpu()[0])        # [w]
        except Exception:
            out.append(None)                            # 单样本也失败 -> None
    return out


def unified_inference(
    pipeline_backbone,
    ds_input,
    timestamp_cols=('timestamp_win',),
    entity_cols=('timestamp_win', 'station'),
    feature_cols=None,
    batch_size: int = 512,
    num_workers: int = 0,
    timestamp_output: str = 'epoch_s',
    prediction_to_numpy: bool = True,
    *,
    error_policy: str = 'fallback',
    progress: bool = True,
    collate_fn=None,
    exclude_input_cols=(),
    on_batch_output=None,
) -> pd.DataFrame:
    """基于pipeline的核心模块的通用预测函数

    数据为DS(Dataset)层的表的格式，模型为符合pipeline标准的 nn.Module。
    Dataset类使用 UnifiedDataset 类。

    Args:
        pipeline_backbone: 具备 pipeline 规范的nn.Module对象，
            1) 统一 forward() 签名(``forward(batch_dict, labels=None)``)；
            2) 已经load模型参数的模型；3) 统一的ModelOutput输出格式
        ds_input: Path | str | pd.DataFrame，满足UnifiedDataset的要求
        timestamp_cols: list of str, 时间戳列，UnifiedDataset的入参之一
        entity_cols: list of str, 实体列，UnifiedDataset的入参之一
        feature_cols: list of str | None, 显式指定特征列；None 表示
            "除 timestamp/entity/label 以外的所有列"
        batch_size: DataLoader 的 batch 大小
        num_workers: DataLoader 的 worker 数
        timestamp_output: 传给 UnifiedDataset/collate_fn 的时间戳输出单位，
            仅支持 'epoch_s' / 'epoch_ns'
        prediction_to_numpy: True 时预测列单元格存 np.ndarray，
            False 时存 torch.Tensor
        error_policy: 默认 fallback 保持逐样本兜底；raise 遇异常、无效形状或非有限预测立即报错。
        progress: 是否显示推理进度条。
        collate_fn: 已构造的 torch Dataset / Subset 可复用原 collate；不会重新物化整个数据集。
        exclude_input_cols: 从送入模型的 batch_dict 中移除这些列（例如真实标签）。
        on_batch_output: 严格模式下调用 callback(original_batch, model_output)，可收集
            分位数、诊断及原始标签；不重复 forward，回调异常直接上抛。

    Return:
        ds_output: pd.DataFrame，行序与 ds_input 的读入顺序一致（shuffle=False），
            包含 entity_cols 列和 predictions 列。predictions 列名为预测模块类名，
            每个单元格是形状为 (prediction_length,) 的序列（即一条完整的预测曲线）。
    """
    if not isinstance(pipeline_backbone, torch.nn.Module):
        raise TypeError(
            f"pipeline_backbone 必须是 nn.Module，收到 {type(pipeline_backbone)}"
        )

    timestamp_cols = list(timestamp_cols)
    entity_cols = list(entity_cols)
    if not entity_cols:
        raise ValueError("entity_cols must contain at least one column")
    if error_policy not in ('fallback', 'raise'):
        raise ValueError("error_policy must be fallback or raise")
    if on_batch_output is not None and error_policy != 'raise':
        raise ValueError("on_batch_output requires error_policy='raise'")
    if timestamp_output not in ('epoch_s', 'epoch_ns'):
        raise ValueError(
            f"timestamp_output 仅支持 'epoch_s'/'epoch_ns'，收到: {timestamp_output!r}"
        )

    # ---- 1. 构建 UnifiedDataset 与 DataLoader -----------------------------
    # for_hf_trainer=True 是必要的: 模型与训练侧共用 "batch_dict"/"labels" 约定。
    # 推理不需要 label，故不指定 label_cols；batch_dict 里只会包含所需的输入列。
    input_dataset = ds_input if isinstance(ds_input, Dataset) else UnifiedDataset(
        ds_input,
        timestamp_cols=timestamp_cols,
        entity_cols=entity_cols,
        feature_cols=feature_cols,
        for_hf_trainer=True,
    )

    custom_collate_fn = collate_fn or partial(
        UnifiedDataset.collate_fn,
        timestamp_output=timestamp_output,
        drop_str_type=False,   # 保留实体列(如 station), 便于回填
        for_hf_trainer=True,
    )

    # 推理必须保持行序稳定: shuffle=False + 按 batch 顺序逐块回填,
    # 才能保证 "哪一行预测对应哪一行实体" 的对齐。
    dataloader = DataLoader(
        input_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        drop_last=False,       # 最后一小批也要推理
    )

    prediction_col = type(pipeline_backbone).__name__

    # ---- 2. 推理循环 ------------------------------------------------------
    # 采用"先收集、再一次性拼接"的方式: 避免逐行写 DataFrame(慢) 且天然按序对齐。
    pipeline_backbone.eval()
    id_cols_records: dict[str, list] = {c: [] for c in entity_cols}
    pred_records: list = []

    with torch.no_grad():
        for batch_dict in tqdm(dataloader, desc="Predict", disable=not progress):
            original_bd = batch_dict.get("batch_dict", batch_dict)
            bd = {k: v for k, v in original_bd.items() if k not in exclude_input_cols}
            bs = len(bd[entity_cols[0]])

            # ---- 批量优先 ----
            try:
                output = pipeline_backbone(bd)
                logits = output.logits
                if logits is None:
                    raise ValueError("ModelOutput.logits 为 None")
                if error_policy == 'raise':
                    if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[0] != bs:
                        raise ValueError("Expected logits shaped [batch, horizon]")
                    horizon = getattr(pipeline_backbone, 'prediction_length', None)
                    if horizon is not None and logits.shape[1] != horizon:
                        raise ValueError("Prediction horizon does not match model.prediction_length")
                    if not torch.isfinite(logits).all():
                        raise ValueError("Model returned non-finite predictions")
                fallback = None
            except Exception as e:
                if error_policy == 'raise':
                    raise
                # ---- 失败逐条兜底 ----（把原因打出来：否则特征/配置错误会被静默吞掉、全部行变 None）
                warnings.warn(f"批量推理失败，逐样本兜底：{type(e).__name__}: {e}")
                logits = None
                fallback = _fallback_per_sample(pipeline_backbone, bd, bs)

            if on_batch_output is not None:
                on_batch_output(batch_dict, output)

            # bd 中实体列是 int64 tensor(station) / epoch(int64)(timestamp_win)
            # 实体列始终从整批 extend，保证行对齐 (bs 行)
            for c in entity_cols:
                v = bd[c]
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().numpy()
                id_cols_records[c].extend(np.asarray(v).tolist())

            # 预测列: 批量成功 -> 每行一条序列; 兜底 -> 每条或 None
            if fallback is None:
                pred = logits.detach().cpu()
                if prediction_to_numpy:
                    pred_records.extend([arr for arr in pred.numpy()])
                else:
                    pred_records.extend(pred)
            else:
                for p in fallback:
                    if p is None:
                        pred_records.append(None)
                    elif prediction_to_numpy:
                        pred_records.append(p.detach().cpu().numpy())
                    else:
                        pred_records.append(p)

    # ---- 3. 组装结果表 ----------------------------------------------------
    n = len(pred_records)
    if n == 0:
        raise ValueError("推理结果为空，请检查 ds_input 是否为空")

    # 校验实体列长度与预测数一致
    for c in entity_cols:
        if len(id_cols_records[c]) != n:
            raise RuntimeError(
                f"实体列 '{c}' 行数({len(id_cols_records[c])})与预测行数({n})不一致，"
                f"对齐失败"
            )

    ds_output = pd.DataFrame({c: id_cols_records[c] for c in entity_cols})

    # timestamp 列: epoch(int64) -> datetime64, 便于读取
    for c in timestamp_cols:
        if c in ds_output.columns and prediction_to_numpy:
            unit = "s" if timestamp_output == "epoch_s" else "ns"
            ds_output[c] = pd.to_datetime(ds_output[c], unit=unit)

    pred_series = pd.Series(pred_records, name=prediction_col)
    ds_output = pd.concat([ds_output, pred_series.to_frame()], axis=1)

    return ds_output
