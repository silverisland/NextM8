# fourier_moba_config.py
# -*- coding: utf-8 -*-

"""
Model 相关的共有的独立nn.module文件。

适配：
    transformers==4.40.1

设计要求和使用要求：
    1、所有模型必须要继承 nn.Module。
    2、模型 forward 方法 有且只能有batch_dict一个方法。
    3、模型 forward 方法 最开始部分必须是 feature_collator，通过该类组织当前模型需要消费的数据。
    4、模型 forward 方法 内部必须有损失计算函数。
    5、模型 forward 方法 输出只能是来源于 transformer_utils.py 的 TransformersAdaptedModelOutput
        输出部分必须包括损失。
"""

import os, sys
from pathlib import Path

import torch
import numpy as np
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from transformers.trainer_utils import EvalPrediction
from typing import List, Optional, Dict, Optional, Any, Tuple
BASE_PATH = str(Path(__file__).resolve().parent) 

try:
    from .layers.RevIN import RevIN
    from .layers.PatchTST_backbone import PatchTST_backbone
    from .layers.feature_collator import HFFeatureCollator
except ImportError:
    from layers.RevIN import RevIN
    from layers.PatchTST_backbone import PatchTST_backbone
    from layers.feature_collator import HFFeatureCollator


try:
    from ...interfaces.modeloutput import TransformersAdaptedModelOutput
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from interfaces.modeloutput import TransformersAdaptedModelOutput


# ===== 加入南网损失之后的结果：复用 m1 的南网损失组件 PVNormalizedLoss =====
try:
    from ...m1.fourier_moba_transformer.nw_pv_loss import PVNormalizedLoss
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from m1.fourier_moba_transformer.nw_pv_loss import PVNormalizedLoss
# ===== 加入南网损失之后的结果：复用 m1 的南网损失组件 PVNormalizedLoss END =====


## 模型侧配置
class TransformersAdaptedM4Model(nn.Module):
    """Expose the core NTP model through a Hugging Face style forward surface."""
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:str='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True,
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()

        self.config = configs

        # load parameters
        # self.capacity = configs["dataset"]["capacity"]
        # ===== 加入南网损失之后的结果：新增 m1 风格 主损失(criterion) + 南网损失(nw_loss) =====
        m_cfg = configs["model"]
        self.loss_type    = m_cfg.get("loss_type", 'mae')
        _main_loss_map    = {'mae': nn.L1Loss, 'mse': nn.MSELoss, 'huber': nn.HuberLoss}
        if self.loss_type not in _main_loss_map:
            raise ValueError(f"loss_type='{self.loss_type}' 不支持，可选: {list(_main_loss_map)}")
        self.criterion    = _main_loss_map[self.loss_type]()
        self.nw_loss      = PVNormalizedLoss(loss_form=self.loss_type)
        self.lambda_main  = m_cfg.get("lambda_main", 100)
        self.lambda_nw    = m_cfg.get("lambda_nw", 10)
        # ===== 加入南网损失之后的结果：新增 m1 风格 主损失(criterion) + 南网损失(nw_loss) END =====
        self.channels = configs["model"]["enc_in"]
        c_in = configs["model"]["enc_in"]
        context_window = configs["inference"]["seq_len"]
        self.prediction_length = configs["inference"]["pred_len"]
        self.context_length = context_window - self.prediction_length
        self.labels_name = [self.config["dataset"]["target_col"]]
        
        n_layers = configs["model"]["e_layers"]
        n_heads = configs["model"]["n_heads"]
        d_model = configs["model"]["d_model"]
        d_ff = configs["model"]["d_ff"]
        dropout = configs["model"]["dropout"]
        fc_dropout = configs["model"]["fc_dropout"]
        head_dropout = configs["model"]["head_dropout"]
        
        embedding_dim = configs["model"].get("embedding_dim", 256)
        

        individual = configs["model"]["individual"]
    
        patch_len = configs["model"]["patch_len"]
        stride = configs["model"]["stride"]
        padding_patch = configs["model"]["padding_patch"]
        
        revin = configs["model"]["revin"]
        affine = configs["model"]["affine"]
        subtract_last = configs["model"]["subtract_last"]
        
        act = configs["model"]["activation"]
        self.revin = revin
        if self.revin: 
            self.weather_revin_layer = RevIN(c_in - 1, affine=affine, subtract_last=subtract_last)
            self.pv_revin_layer = RevIN(1, affine=affine, subtract_last=subtract_last)

        # model
        self.featurecollator = HFFeatureCollator(configs) 
        self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=self.prediction_length, patch_len=patch_len, stride=stride,
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, embedding_dim=embedding_dim, **kwargs)


    def forward(
        self,
        batch_dict: Dict[str, Any],
        labels: Dict[str, Any] | None = None,
    ) -> TransformersAdaptedModelOutput:
        """
        HuggingFace Trainer 兼容的前向传播。

        从 batch_dict 解包 collator 输出的各分量，维度已对齐 Model.forward 输入。
        内部流程与 model/patchtst.py 的 Model.forward 完全一致。

        Collator 输出 → Forward 输入映射:
            batch_x1: (B, 4, S)      -- 历史天气特征
                dim=1: [general_ghi(0), GHI_SOLARGIS(1), general_temp(2), TEMP_SOLARGIS(3)]
            batch_x2: (B, 4, S')     -- 未来天气特征
                dim=1: [general_ghi_predict(0), GHI_SOLARGIS_predict(1), general_temp_predict(2), TEMP_SOLARGIS_predict(3)]
            batch_x3: (B, S)         -- 历史 PV 功率（已 squeeze）
            batch_x4: (B, S)         -- tsfm 预测 PV（已 squeeze）
            batch_y:  (B, S')        -- 未来 PV 真值/标签（已 squeeze，推理时为 None）

        Shape 参数:
            B = batch_size
            S = seq_len = 864   输入序列长度
            S' = pred_len = 192 预测序列长度
        """
        # ── 解包 batch_dict ──
        x1, x2, x3, x4 = self.featurecollator(batch_dict)

        # 读取输入的装机容量作为归一化系数，如果没有则为None
        # cap_power = batch_dict.get("cap_power_on", 1)

        # ── 模式判断 ──
        # flag = 'test' if y is None else 'train'

        # ── 维度扩展与归一化 ──
        # x3 = x3.unsqueeze(1) / cap_power if cap_power else x3.unsqueeze(1) # (B, S) → (B, 1, S)
        # x4 = x4.unsqueeze(1) / cap_power if cap_power else x4.unsqueeze(1) # (B, S) → (B, 1, S)
        x3 = x3.unsqueeze(1)
        x4 = x4.unsqueeze(1)

        # ── 拼接天气与 PV 特征 ──
        weather = torch.concat([x1, x2], dim=2)   # (B, 4, S) + (B, 4, S') → (B, 4, S+S')
        pv = torch.concat([x3, x4], dim=2)         # (B, 1, S) + (B, 1, S) → (B, 1, S+S')

        # ── RevIN 归一化 ──
        if self.revin:
            weather = weather.permute(0, 2, 1)         # (B, 4, S+S') → (B, S+S', 4)
            weather = self.weather_revin_layer(weather, 'norm')  # (B, S+S', 4)
            weather = weather.permute(0, 2, 1)         # (B, S+S', 4) → (B, 4, S+S')

            pv = pv.permute(0, 2, 1)                   # (B, 1, S+S') → (B, S+S', 1)
            pv = self.pv_revin_layer(pv, 'norm')       # (B, S+S', 1)
            pv = pv.permute(0, 2, 1)                   # (B, S+S', 1) → (B, 1, S+S')

        # ── PatchTST 主干 ──
        output, embedding = self.model(weather, pv)    # output: (B, 1, S'), embedding: (B, embedding_dim)

        # ── RevIN 逆归一化 ──
        if self.revin:
            output = output.permute(0, 2, 1)           # (B, 1, S') → (B, S', 1)
            output = self.pv_revin_layer(output, 'denorm')  # (B, S', 1)
            output = output.permute(0, 2, 1)           # (B, S', 1) → (B, 1, S')

        # ── 反归一化 ──
        result_ = output.squeeze(1)
        # result_ = result_ * cap_power if cap_power else result_  # (B, 1, S') → (B, S')  反归一化预测值

       
        # ── 计算 loss（训练/验证模式） ──
        loss = None
        if labels is not None:
            label_key = self.config["dataset"]["target_col"]
            labels_tensor = labels[label_key]

            # ===== 加入南网损失之后的结果：原 9 分位数损失保留（注释标明旧逻辑）=====
            # loss = self.compute_loss(output, labels_tensor.unsqueeze(1) / self.capacity)   # 标量 9 分位数损失
            # ===== 加入南网损失之后的结果：原 9 分位数损失保留（注释标明旧逻辑）END =====

            # ===== 加入南网损失之后的结果：新增 m1 风格 主损失(main) + 南网损失(nw) 加权 =====
            # 三部分/两部分损失逻辑已整体移入 compute_loss 内部（含 m1 式动态容量兜底），
            # forward 仅把真实装机容量列传入；列名由 config[dataset].label_norm 配置（m1 同款语义），
            # compute_loss 内优先用它，拿不到时按 m1 从 label 动态估计（见 compute_loss，原 9 分位数代码保留）。
            # 注：m4 的 labels_tensor 为原始功率 (B,S')，故兜底估计出的容量接近真实装机容量量级。
            cap_col = (self.config["dataset"].get("label_norm") or [None])[0]
            capacity = batch_dict.get(cap_col) if cap_col else None
            loss = self.compute_loss(result_,
                                     labels_tensor,
                                     capacity=capacity)
            # ===== 加入南网损失之后的结果：新增 m1 风格 主损失(main) + 南网损失(nw) 加权 END =====
        
       
        # ── 统一返回 HF 风格 ModelOutput ──
        return TransformersAdaptedModelOutput(
            loss = loss,                      # train/valid 时有值，test 时为 None
            logits = result_.squeeze(1),                  # (B, S')  与 prediction 一致
            predictions = result_.squeeze(1),             # (B, S')  反归一化预测值
            embeddings = embedding
        )
    
    def compute_loss(self, y_pred, y_true, capacity=None):
        """
        计算损失，返回标量。

        新逻辑（m1 风格 主损失 + 南网损失，代码块前置 return，原有 9 分位数代码保留在下方）:
            y_pred : [B, 1, S']  模型输出的归一化 PV 预测值（0~1 量纲）
            y_true : [B, S']     未来 PV 真值（原始功率，非归一化）
            capacity : 可选，真实装机容量 batch["cap_power_on"]（(B,)/标量/tensor）；
                       None 时按 m1 做法用 torch.max(y_true, ones) 从 label 动态估计容量。

            计算:
                _ynorm = y_true.unsqueeze(1) / _cap_v          # 按容量归一化真值 (B,1,S')
                _main  = criterion(y_pred, _ynorm)             # 归一化主损失
                _raw_pred = y_pred * _cap_v                    # 反归一化原始功率预测
                _nw    = nw_loss(_raw_pred.squeeze(1), y_true, _cap)  # 南网损失(原始功率+容量)
                loss   = lambda_main*_main + lambda_nw*_nw

        Returns:
            loss: 标量。
        """
        # ===== 加入南网损失之后的结果：新增 m1 风格 主损失(main) + 南网损失(nw) 加权 =====
        # 容量来源：优先真实 cap_power_on；拿不到时按 m1 的做法从 label 动态估计
        # （m1: s = torch.max(label, torch.ones_like(label))，不用 config 静态容量兜底）。
        # 注：y_true 为原始功率 (B,S')，故兜底估计出的容量接近真实装机容量量级。

        _cap = capacity
        if _cap is None:
            _cap   = torch.max(y_true, torch.ones_like(y_true))   # m1 式从 label 动态估计容量 (B,S')
            _cap_v = _cap.unsqueeze(1)                            # (B,S') -> (B,1,S') 广播
        elif isinstance(_cap, torch.Tensor) and _cap.dim() == 1:
            _cap_v = _cap.view(-1, 1, 1)                          # (B,) -> (B,1,1) 广播
        else:
            _cap_v = _cap
        
        
        _ynorm    = y_true.unsqueeze(1) / _cap_v                  # (B,1,S') 按容量归一化真值
        # y_pred 为 (B,S'), 归一化需用 (B,1) 形状容量, 避免 _cap_v=(B,1,1) 把 y_pred 广播成 (B,B,S')。
        _cap_row  = _cap.view(-1, 1) if (isinstance(_cap, torch.Tensor) and _cap.dim() == 1) else _cap
        y_pred_norm = (y_pred / _cap_row).unsqueeze(1)            # (B,1,S') 与 _ynorm 同形, 无广播告警
        # _main     = self.criterion(y_pred, _ynorm)                # 归一化主损失 (B,1,S') -> 标量
        _main     = self.criterion(y_pred_norm, _ynorm)           # (B,1,S') vs (B,1,S')
        # _raw_pred = y_pred * _cap_v                               # (B,1,S') 反归一化原始功率预测
        # _nw       = self.nw_loss(_raw_pred.squeeze(1), y_true, _cap)  # 南网损失(原始功率 + 容量)
        _nw       = self.nw_loss(y_pred, y_true, _cap)  # 南网损失(原始功率 + 容量)
        loss      = self.lambda_main * _main + self.lambda_nw * _nw
        print(
            f"[M4N-LOSS] main={_main.item():.6f} (λ={self.lambda_main:.3f}) | "
            f"nw={_nw.item():.6f} (λ={self.lambda_nw:.3f}) | total={loss.item():.6f}",
            flush=True,
        )
        return loss
        # ===== 加入南网损失之后的结果：新增 m1 风格 主损失(main) + 南网损失(nw) 加权 END =====

        # 下方为 m4 原有 9 分位数损失代码，逐字未改；仅因上方新增 return 不再执行。
        # error = y_true.unsqueeze(1) / self.capacity - y_pred
        # loss = 0
        # for quantile in np.arange(0.1, 1.0, 0.1):
        #     loss += torch.mean(torch.max(quantile * error, (quantile - 1) * error))
        # # 打印格式参考 m1 的 [M1-LOSS]（9 分位数损失只有一项，total 即 loss）
        # print(f"[M4-LOSS] quantile9={loss.item():.6f} | total={loss.item():.6f}", flush=True)
        # return loss


    # =========================================================================
    # Metrics
    # =========================================================================

    def compute_metrics(
        self,
        eval_pred: EvalPrediction,
    ) -> Dict[str, float]:
        """验证集和测试集共用的聚合指标。"""

        # NOTE: transformers 会把所有非loss列收集起来放在这里
        predictions = eval_pred.predictions[0]
        labels = eval_pred.label_ids

        if isinstance(labels, dict):
            target_col_name = self.config["dataset"]["target_col"]
            labels = labels[target_col_name]


        if predictions is None or labels is None:
            raise ValueError("compute_metrics 要求 predictions 和 labels 均不为空。")

        predictions = np.asarray(predictions)
        labels      = np.asarray(labels)

        if predictions.shape != labels.shape:
            raise ValueError(f"M4 指标计算时预测与标签形状不一致：predictions={predictions.shape}, labels={labels.shape}。")

        error = predictions - labels
        mse  = float(np.mean(error ** 2))
        mae  = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(mse))

        return {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
        }
    