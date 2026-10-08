# -*- coding: utf-8 -*-
# forecast_backbones/m8/model.py
"""PVRexCore：逐步通道 x[B,T,C] + 静态向量 + 站槽 → 每步 K 档分位（归一化单位）。

token 布局（model.layout）：
  fused —— 一个 patch 的 P 步 × 全部 C 通道展平成一个 token（TiRex/Reverso 式单流）。
  split —— 目标流（y, y_mask, nwp_resid…）与协变量流（NWP、太阳几何、时间编码…）各自嵌入成两条 token 序列，
           时间混合器共享权重逐流处理，每个 Block 后做一次同 patch 位置的跨流混合（Chronos-2 group attention /
           TimesFM-3 variate attention 的两流简化版）。头用目标流的 token。
输出头（model.head_mode）：
  step_cov —— 每步拼 [token 隐状态, patch 内位置嵌入, 该步原始通道向量] → ResidualMLP → K（TiDE temporal decoder /
              ChronosX OIB：未来 NWP 在输出侧二次注入）。
  token    —— TiRex 式：token 隐状态 → ResidualMLP → P×K，不再看逐步协变量（用于消融输出侧注入的价值）。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .layers import Block, RMSNorm, ResidualMLP
except ImportError:
    from layers import Block, RMSNorm, ResidualMLP


class VariateMix(nn.Module):
    """split 布局：同一 patch 位置的两条流拼接 → MLP → 残差回各流（pre-norm）。"""

    def __init__(self, d: int, dropout: float = 0.0):
        super().__init__()
        self.norm = RMSNorm(d)
        self.mlp = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, 2 * d))

    def forward(self, tok: torch.Tensor) -> torch.Tensor:                        # [B, n_tok, 2, d]
        B, n, s, d = tok.shape
        z = self.norm(tok).reshape(B, n, s * d)
        return tok + self.mlp(z).view(B, n, s, d)


class PVRexCore(nn.Module):
    def __init__(self, n_channels: int, n_static: int, n_slots: int, context_len: int, pred_len: int, model_cfg: dict,
                 target_channels: list[int] | None = None, cov_channels: list[int] | None = None):
        super().__init__()
        m = model_cfg
        self.P = int(m["patch_len"]); self.C = int(n_channels)
        self.T = int(context_len) + int(pred_len)
        self.n_tok = self.T // self.P
        d = int(m["d_model"]); self.d = d
        K = len(m["quantiles"]); self.K = K
        dropout = float(m.get("dropout", 0.0))
        self.layout = str(m.get("layout", "fused"))
        self.head_mode = str(m.get("head_mode", "step_cov"))
        if self.layout not in ("fused", "split"):
            raise ValueError(f"model.layout 须是 fused 或 split，得到 {self.layout!r}")
        if self.head_mode not in ("step_cov", "token"):
            raise ValueError(f"model.head_mode 须是 step_cov 或 token，得到 {self.head_mode!r}")
        if self.layout == "fused":
            self.embed = ResidualMLP(self.P * self.C, int(m["d_ff"]), d, dropout)
        else:
            if not target_channels or not cov_channels:
                raise ValueError("layout=split 需要 target_channels / cov_channels")
            self.register_buffer("tgt_idx", torch.as_tensor(target_channels, dtype=torch.long), persistent=False)
            self.register_buffer("cov_idx", torch.as_tensor(cov_channels, dtype=torch.long), persistent=False)
            self.embed_t = ResidualMLP(self.P * len(target_channels), int(m["d_ff"]), d, dropout)
            self.embed_c = ResidualMLP(self.P * len(cov_channels), int(m["d_ff"]), d, dropout)
            self.stream_emb = nn.Parameter(torch.randn(2, d) * 0.02)
            self.vmix = nn.ModuleList([VariateMix(d, dropout) for _ in m["blocks"]])
        self.static_proj = nn.Linear(int(n_static), d)
        self.station_emb = nn.Embedding(int(n_slots) + 1, d)
        nn.init.normal_(self.station_emb.weight, std=0.02)
        self.pos = nn.Parameter(torch.randn(self.n_tok, d) * 0.02)
        self.blocks = nn.ModuleList([
            Block(kind, d, int(m["n_heads"]), int(m["d_ff"]), self.n_tok, int(m.get("short_conv_kernel", 4)),
                  bool(m.get("deltanet_gated", True)), dropout, bool(m.get("bidirectional", False)))
            for kind in m["blocks"]])
        self.norm = RMSNorm(d)
        if self.head_mode == "step_cov":
            dp = int(m.get("pos_in_patch_dim", 32))
            self.pos_in_patch = nn.Parameter(torch.randn(self.P, dp) * 0.02)
            self.head = ResidualMLP(d + dp + self.C, int(m.get("head_hidden", 256)), K, dropout)
        else:
            self.head = ResidualMLP(d, int(m.get("head_hidden", 256)), self.P * K, dropout)

    def encode(self, x: torch.Tensor, static: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        """→ 目标流 token 隐状态 [B, n_tok, d]。"""
        B, T, C = x.shape
        assert T == self.T and C == self.C, f"期望 x[B,{self.T},{self.C}]，得到 {tuple(x.shape)}"
        cond = self.static_proj(static)[:, None, :] + self.station_emb(slot)[:, None, :] + self.pos[None]
        if self.layout == "fused":
            tok = self.embed(x.reshape(B, self.n_tok, self.P * C)) + cond
            for blk in self.blocks:
                tok = blk(tok)
            return self.norm(tok)
        xt = x.index_select(-1, self.tgt_idx).reshape(B, self.n_tok, -1)
        xc = x.index_select(-1, self.cov_idx).reshape(B, self.n_tok, -1)
        tok = torch.stack([self.embed_t(xt), self.embed_c(xc)], dim=2) + cond[:, :, None, :] + self.stream_emb[None, None]
        for blk, vm in zip(self.blocks, self.vmix):
            tok = blk(tok.permute(0, 2, 1, 3).reshape(B * 2, self.n_tok, self.d)).view(B, 2, self.n_tok, self.d).permute(0, 2, 1, 3)
            tok = vm(tok)
        return self.norm(tok[:, :, 0])

    def forward(self, x: torch.Tensor, static: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        h = self.encode(x, static, slot)
        if self.head_mode == "token":
            return self.head(h).view(B, self.n_tok, self.P, self.K).reshape(B, T, self.K)
        h_step = h[:, :, None, :].expand(B, self.n_tok, self.P, self.d).reshape(B, T, self.d)
        pip = self.pos_in_patch[None, None].expand(B, self.n_tok, self.P, self.pos_in_patch.shape[-1]).reshape(B, T, -1)
        return self.head(torch.cat([h_step, pip, x], dim=-1))                    # [B, T, K]
