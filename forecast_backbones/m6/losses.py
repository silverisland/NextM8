"""训练损失：pinball（主头，proper）与南网 r²（辅助头/堆叠器）。"""
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


def nanwang_r2_loss(p: torch.Tensor, y: torch.Tensor, cap, mask: torch.Tensor, floor_ratio: float = 0.2) -> torch.Tensor:
    """mean over mask of ((y-p)/max(y, floor_ratio·cap))²。cap 标量或 [B,1]。"""
    cap_t = torch.as_tensor(cap, dtype=p.dtype, device=p.device)
    den = torch.maximum(y, floor_ratio * cap_t)
    r = (y - p) / den
    m = mask.to(dtype=p.dtype, device=p.device)
    return ((r ** 2) * m).sum() / m.sum().clamp_min(1.0)
