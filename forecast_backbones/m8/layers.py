# -*- coding: utf-8 -*-
# forecast_backbones/m8/layers.py
"""M8 的序列混合器与基础层，全部纯 PyTorch（无 Triton / CUDA 扩展），fp32 可跑，全部因果。
  GatedLongConv  —— Reverso 的门控长卷积：SiLU(ShortConv(W_a x)) ⊙ CausalDepthwiseConv_{kernel=T}(W_b x) → W_o。
                    与 Reverso 的区别：线性（零填充）卷积而不是 FFT 循环卷积，T 只有几十个 token，直接 conv1d 即可。
  GatedDeltaNet  —— Gated DeltaNet：S_t = α_t S_{t-1} + β_t (v_t − α_t S_{t-1} k_t) k_tᵀ，o_t = S_t q_t；
                    q/k/v 经因果短卷积 + SiLU，q、k 逐头 L2 归一，逐头 RMSNorm 后乘 SiLU 门。token 数少，逐步循环。
  CausalAttention—— 因果多头注意力，作对照。
  Block          —— pre-RMSNorm：x += Mixer(norm(x)); x += FFN(norm(x))。
（sLSTM 混合器与双向混合器已于 2026-10-14 瘦身删除。）"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class ResidualMLP(nn.Module):
    """TiRex 的 ResidualBlock：Linear → GELU → Linear，加一条线性跳接。"""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden); self.fc2 = nn.Linear(d_hidden, d_out)
        self.skip = nn.Linear(d_in, d_out); self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(F.gelu(self.fc1(x)))) + self.skip(x)


class CausalShortConv(nn.Module):
    """逐通道因果卷积（kernel k，左填充 k-1）。输入 [B, T, d]。"""

    def __init__(self, d: int, k: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(d, d, k, groups=d, padding=k - 1); self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        return self.conv(x.transpose(1, 2))[:, :, :T].transpose(1, 2)


class GatedLongConv(nn.Module):
    def __init__(self, d: int, seq_len: int, short_kernel: int = 4, dropout: float = 0.0):
        super().__init__()
        self.seq_len = seq_len
        self.w_a = nn.Linear(d, d); self.w_b = nn.Linear(d, d); self.w_o = nn.Linear(d, d)
        self.short = CausalShortConv(d, short_kernel)
        self.kernel = nn.Parameter(torch.randn(d, seq_len) * (0.5 / math.sqrt(seq_len)))
        self.drop = nn.Dropout(dropout)

    def long_conv(self, u: torch.Tensor) -> torch.Tensor:
        """u [B, T, d] → 因果逐通道卷积，kernel 长 seq_len（T ≤ seq_len）。"""
        T = u.shape[1]
        k = self.kernel[:, None, :T] if T < self.seq_len else self.kernel[:, None, :]
        out = F.conv1d(u.transpose(1, 2), k, padding=T - 1, groups=u.shape[-1])[:, :, :T]
        return out.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = F.silu(self.short(self.w_a(x)))
        b = self.long_conv(self.w_b(x))
        return self.w_o(self.drop(a * b))


class GatedDeltaNet(nn.Module):
    def __init__(self, d: int, n_heads: int, short_kernel: int = 4, gated: bool = True, dropout: float = 0.0):
        super().__init__()
        assert d % n_heads == 0
        self.h, self.dh, self.gated = n_heads, d // n_heads, gated
        self.q_proj = nn.Linear(d, d); self.k_proj = nn.Linear(d, d); self.v_proj = nn.Linear(d, d)
        self.conv_q = CausalShortConv(d, short_kernel); self.conv_k = CausalShortConv(d, short_kernel); self.conv_v = CausalShortConv(d, short_kernel)
        self.beta_proj = nn.Linear(d, n_heads)
        self.alpha_proj = nn.Linear(d, n_heads)
        nn.init.zeros_(self.alpha_proj.weight); nn.init.constant_(self.alpha_proj.bias, 4.0)   # α≈0.98 起步：几乎不遗忘
        self.g_proj = nn.Linear(d, d); self.o_norm = RMSNorm(self.dh); self.o_proj = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape; H, dh = self.h, self.dh
        q = F.silu(self.conv_q(self.q_proj(x))).view(B, T, H, dh)
        k = F.silu(self.conv_k(self.k_proj(x))).view(B, T, H, dh)
        v = F.silu(self.conv_v(self.v_proj(x))).view(B, T, H, dh)
        q = F.normalize(q, dim=-1); k = F.normalize(k, dim=-1)
        beta = torch.sigmoid(self.beta_proj(x))                                  # [B, T, H]
        alpha = torch.sigmoid(self.alpha_proj(x)) if self.gated else torch.ones_like(beta)
        S = x.new_zeros(B, H, dh, dh)                                             # S[b,h] : k → v（[dv, dk]）
        outs = []
        for t in range(T):
            kt, vt, qt = k[:, t], v[:, t], q[:, t]                                # [B, H, dh]
            at = alpha[:, t][:, :, None, None]; bt = beta[:, t][:, :, None]
            S = at * S
            Sk = torch.einsum("bhvk,bhk->bhv", S, kt)
            S = S + ((bt * (vt - Sk))[..., :, None] * kt[..., None, :])
            outs.append(torch.einsum("bhvk,bhk->bhv", S, qt))
        o = torch.stack(outs, 1)                                                  # [B, T, H, dh]
        o = self.o_norm(o) * F.silu(self.g_proj(x)).view(B, T, H, dh)
        return self.o_proj(self.drop(o.reshape(B, T, d)))


class CausalAttention(nn.Module):
    """因果多头注意力（对照用）。"""

    def __init__(self, d: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        return self.attn(x, x, x, attn_mask=mask, need_weights=False)[0]


def make_mixer(kind: str, d: int, n_heads: int, seq_len: int, short_kernel: int, gated: bool, dropout: float) -> nn.Module:
    if kind == "conv":
        return GatedLongConv(d, seq_len, short_kernel, dropout)
    if kind == "deltanet":
        return GatedDeltaNet(d, n_heads, short_kernel, gated, dropout)
    if kind == "attention":
        return CausalAttention(d, n_heads, dropout)
    raise ValueError(f"未知混合器 {kind!r}（可选 conv / deltanet / attention）")


class Block(nn.Module):
    def __init__(self, kind: str, d: int, n_heads: int, d_ff: int, seq_len: int, short_kernel: int = 4,
                 gated: bool = True, dropout: float = 0.0):
        super().__init__()
        self.kind = kind
        self.norm1 = RMSNorm(d); self.norm2 = RMSNorm(d)
        self.mixer = make_mixer(kind, d, n_heads, seq_len, short_kernel, gated, dropout)
        self.ffn = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.mixer(self.norm1(x)))
        return x + self.drop(self.ffn(self.norm2(x)))
