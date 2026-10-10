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
  token    —— TiRex 式：token 隐状态 → ResidualMLP → P×K，不再看逐步协变量（用于消融输出侧注入的价值）。
超短期 / 指标对齐模块（默认关）：
  mean_head   —— 头多出一路点预测（第 K+1 个输出），由 transformer_models 按南网相对 L2 训练，决策层可在 D+1 段直接用它。
  recent_skip —— step_cov 头的每步输入再拼 [最近 k 步可见观测, 可见掩码, 末次晴空指数 k_last, k_last·p_clear_t]：
                 最近观测与"晴空指数持久化"直连输出头（NLinear / scaled persistence 的做法），不再只靠 token 隐状态传递。
  ultra_head  —— 前 n 步的直出 MLP 头（TabM / TiDE 式：最近窗口 + 这 n 步的原始协变量 + 第一个未来 token）→ n×K，
                 与主头按步 sigmoid 门控融合（门初值 0.5，可学）。"""
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
                 target_channels: list[int] | None = None, cov_channels: list[int] | None = None,
                 chan_idx: dict | None = None):
        super().__init__()
        m = model_cfg
        self.P = int(m["patch_len"]); self.C = int(n_channels)
        self.L = int(context_len)
        self.T = int(context_len) + int(pred_len)
        self.n_tok = self.T // self.P
        d = int(m["d_model"]); self.d = d
        K = len(m["quantiles"]); self.K = K
        dropout = float(m.get("dropout", 0.0))
        # ---- 超短期 / 指标对齐模块 ----
        self.mean_head = bool(m.get("mean_head", False))
        self.n_out = K + (1 if self.mean_head else 0)
        rs = m.get("recent_skip") or {}
        uh = m.get("ultra_head") or {}
        self.rs_k = int(rs.get("k", 16)) if bool(rs.get("enabled", False)) else 0
        self.uh_n = int(uh.get("n_steps", 16)) if bool(uh.get("enabled", False)) else 0
        self.rec_k = self.rs_k or (int(rs.get("k", 16)) if self.uh_n else 0)        # ultra_head 也要最近窗口
        if self.rec_k:
            if str(m.get("head_mode", "step_cov")) != "step_cov":
                raise ValueError("model.recent_skip / ultra_head 只支持 head_mode=step_cov")
            need = ("y", "y_mask", "p_clear")
            if not chan_idx or any(k not in chan_idx for k in need):
                raise ValueError(f"model.recent_skip / ultra_head 需要 chan_idx 给出 {need} 的通道下标")
            if not (0 < self.rec_k <= self.L) or not (0 <= self.uh_n <= int(pred_len)):
                raise ValueError(f"recent_skip.k={self.rec_k} 须在 (0, context_len]，ultra_head.n_steps={self.uh_n} 须在 [0, pred_len]")
            self.ci_y, self.ci_ym, self.ci_pc = int(chan_idx["y"]), int(chan_idx["y_mask"]), int(chan_idx["p_clear"])
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
            rec_dim = (2 * self.rec_k + 2) if self.rs_k else 0
            self.head = ResidualMLP(d + dp + self.C + rec_dim, int(m.get("head_hidden", 256)), self.n_out, dropout)
        else:
            self.head = ResidualMLP(d, int(m.get("head_hidden", 256)), self.P * self.n_out, dropout)
        if self.uh_n:
            n = self.uh_n
            self.ultra = ResidualMLP(2 * self.rec_k + 1 + n + n * self.C + d, int(uh.get("hidden", 256)), n * self.n_out, dropout)
            self.uh_gate = nn.Parameter(torch.zeros(n))                           # sigmoid(0) = 0.5：一开始主头与直出头各半

    def _recent(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """最近窗口：→ (rec [B, 2k+1] = [最近 k 步可见观测, 可见掩码, 末次晴空指数 k_last], persist [B, T] = k_last·p_clear_t)。
        k_last 取最近 k 步里最后一个可见且 p_clear>0.05 的 y/p_clear（夹到 [0, 1.5]），没有则 0。"""
        L, k = self.L, self.rec_k
        y = x[:, :L, self.ci_y]
        vis = 1.0 - x[:, :L, self.ci_ym]
        pc = x[:, :, self.ci_pc]
        yk, vk, pck = y[:, L - k:], vis[:, L - k:], pc[:, L - k:L]
        valid = (vk > 0.5) & (pck > 0.05)
        ratio = torch.where(valid, yk / pck.clamp_min(0.05), torch.zeros_like(yk))
        pos = torch.arange(k, device=x.device)[None].expand(valid.shape)
        last = torch.where(valid, pos, torch.full_like(pos, -1)).max(dim=1).values           # [B]，-1 = 没有可见点
        k_last = ratio.gather(1, last.clamp_min(0)[:, None])[:, 0]
        k_last = torch.where(last >= 0, k_last, torch.zeros_like(k_last)).clamp(0.0, 1.5)
        return torch.cat([yk, vk, k_last[:, None]], dim=1), k_last[:, None] * pc

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

    def forward(self, x: torch.Tensor, static: torch.Tensor, slot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """→ (q [B, T, K] 分位, mean [B, T] 或 None（model.mean_head 关时）)，都是归一化单位。"""
        B, T, C = x.shape
        h = self.encode(x, static, slot)
        if self.head_mode == "token":
            out = self.head(h).view(B, self.n_tok, self.P, self.n_out).reshape(B, T, self.n_out)
        else:
            h_step = h[:, :, None, :].expand(B, self.n_tok, self.P, self.d).reshape(B, T, self.d)
            pip = self.pos_in_patch[None, None].expand(B, self.n_tok, self.P, self.pos_in_patch.shape[-1]).reshape(B, T, -1)
            parts = [h_step, pip, x]
            rec = persist = None
            if self.rec_k:
                rec, persist = self._recent(x)
            if self.rs_k:
                parts.append(torch.cat([rec[:, None, :].expand(B, T, rec.shape[1]), persist[:, :, None]], dim=-1))
            out = self.head(torch.cat(parts, dim=-1))                              # [B, T, n_out]
            if self.uh_n:
                L, n = self.L, self.uh_n
                z = torch.cat([rec, persist[:, L:L + n], x[:, L:L + n].reshape(B, n * C), h[:, L // self.P]], dim=-1)
                u = self.ultra(z).view(B, n, self.n_out)
                g = torch.sigmoid(self.uh_gate)[None, :, None]
                out = torch.cat([out[:, :L], (1.0 - g) * out[:, L:L + n] + g * u, out[:, L + n:]], dim=1)
        q = out[..., :self.K]
        return q, (out[..., self.K] if self.mean_head else None)
