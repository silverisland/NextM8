# forecast_backbones/m5/losses.py（自带一份，不再引用 m6）
"""训练损失：pinball（9 档分位，proper scoring rule）。"""
from __future__ import annotations

from typing import Sequence

import torch


def pinball_loss(q: torch.Tensor, y: torch.Tensor, levels: Sequence[float], mask: torch.Tensor) -> torch.Tensor:
    """q [B,T,K]、y [B,T]、mask [B,T]（1=计入）→ 标量，按有效 (t,τ) 对数归一。"""
    lv = torch.as_tensor(list(levels), dtype=q.dtype, device=q.device)          # [K]
    e = y.unsqueeze(-1) - q                                                      # [B,T,K]
    loss = torch.maximum(lv * e, (lv - 1.0) * e)
    m = mask.to(dtype=q.dtype, device=q.device).unsqueeze(-1)
    return (loss * m).sum() / (m.sum() * lv.numel()).clamp_min(1.0)
