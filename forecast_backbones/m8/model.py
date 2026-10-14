# -*- coding: utf-8 -*-
# forecast_backbones/m8/model.py
"""PVRexCore：逐步通道 x[B,T,C] + 静态向量 + 站槽 → 每步 K 档分位（归一化单位）。

token：一个 patch 的 P 步 × 全部 C 通道展平成一个 token（TiRex / Reverso 式单流），加静态投影、站点嵌入、位置嵌入。
输出头（step_cov）：每步拼 [token 隐状态, patch 内位置嵌入, 该步原始通道向量] → ResidualMLP → K
（TiDE temporal decoder / ChronosX OIB：未来 NWP 在输出侧二次注入）。
2026-10-14 瘦身：split 双流布局、token 头、双向混合器、sLSTM、mean_head / recent_skip / ultra_head 超短期模块都删了
（三轮消融里全在种子噪声内或从未跑过；要找回去看 commit 203c735 之前的历史）。"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .layers import Block, RMSNorm, ResidualMLP
except ImportError:
    from layers import Block, RMSNorm, ResidualMLP


class PVRexCore(nn.Module):
    def __init__(self, n_channels: int, n_static: int, n_slots: int, context_len: int, pred_len: int, model_cfg: dict):
        super().__init__()
        m = model_cfg
        self.P = int(m["patch_len"]); self.C = int(n_channels)
        self.L = int(context_len)
        self.T = int(context_len) + int(pred_len)
        self.n_tok = self.T // self.P
        d = int(m["d_model"]); self.d = d
        self.K = len(m["quantiles"])
        dropout = float(m.get("dropout", 0.0))
        self.embed = ResidualMLP(self.P * self.C, int(m["d_ff"]), d, dropout)
        self.static_proj = nn.Linear(int(n_static), d)
        self.station_emb = nn.Embedding(int(n_slots) + 1, d)
        nn.init.normal_(self.station_emb.weight, std=0.02)
        self.pos = nn.Parameter(torch.randn(self.n_tok, d) * 0.02)
        self.blocks = nn.ModuleList([
            Block(kind, d, int(m["n_heads"]), int(m["d_ff"]), self.n_tok, int(m.get("short_conv_kernel", 4)),
                  bool(m.get("deltanet_gated", True)), dropout)
            for kind in m["blocks"]])
        self.norm = RMSNorm(d)
        dp = int(m.get("pos_in_patch_dim", 32))
        self.pos_in_patch = nn.Parameter(torch.randn(self.P, dp) * 0.02)
        self.head = ResidualMLP(d + dp + self.C, int(m.get("head_hidden", 256)), self.K, dropout)

    def encode(self, x: torch.Tensor, static: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        """→ token 隐状态 [B, n_tok, d]。"""
        B, T, C = x.shape
        assert T == self.T and C == self.C, f"期望 x[B,{self.T},{self.C}]，得到 {tuple(x.shape)}"
        cond = self.static_proj(static)[:, None, :] + self.station_emb(slot)[:, None, :] + self.pos[None]
        tok = self.embed(x.reshape(B, self.n_tok, self.P * C)) + cond
        for blk in self.blocks:
            tok = blk(tok)
        return self.norm(tok)

    def forward(self, x: torch.Tensor, static: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        """→ q [B, T, K] 分位（归一化单位）。"""
        B, T, C = x.shape
        h = self.encode(x, static, slot)
        h_step = h[:, :, None, :].expand(B, self.n_tok, self.P, self.d).reshape(B, T, self.d)
        pip = self.pos_in_patch[None, None].expand(B, self.n_tok, self.P, self.pos_in_patch.shape[-1]).reshape(B, T, -1)
        return self.head(torch.cat([h_step, pip, x], dim=-1))
