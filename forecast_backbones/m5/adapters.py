# -*- coding: utf-8 -*-
# forecast_backbones/m5/adapters.py
"""M5 的 LoRA 之外的适配器（2026-10-18，全部默认关；关时不挂任何模块、不注册 hook，前向与原模型逐位相同）。
全部挂在底座根模块的 ModuleDict `m5_adapters` 下（lora.ADAPTER_ROOT），所以构造即可训、随 LoRA 一起存档 / 加载 / EMA / WiSE；
插入点都在 TimesFM-3 自己的 forward 里用 hook 完成，不改 timesfm3 源码：

  pre_transformer_resblock 的输出（RevIN 之后、进 transformer 之前的 token [b, v, n, d]）
      static_embed      零初始化加性嵌入：W·s 加到目标变量的全部 token（s = 经纬度 / 海拔 / log 容量 / has_coords，
                        训练时整行按 dropout 置零，让"没有元数据"也是见过的输入）。W=0 → 初始与原模型逐位相同。
                        对称平均下 −x 分支加 −W·s（取奇）：同号注入在 (f(x) − f(−x))/2 里大部分相减抵消（小模型实测效应
                        34 → 取奇 42，不开对称平均 52）。
      resblock_adapter  残差块非线性适配器：x + up(relu(down(x)))，up 零初始化 → 初始逐位相同。
      prompt            虚拟变量 soft prompt：在变量轴末尾追加 n_tokens 个可学的"变量"，每个变量在所有 patch 上是同一个
                        可学向量（时间位置靠 seq RoPE），经 var_attn 被功率与 NWP 看到；per_station 时 = 全局 + 逐站增量
                        （增量零初始化；训练集外的站 / station_dropout 命中的行只用全局）。追加的变量不进 head：
                        transformer_stack 前把 patch mask 补 False，出来后切掉。注意：多出的 token 会参与 softmax，
                        所以 prompt 开时第 0 步不再与零样本逐位相同（WiSE α=0 也不是零样本）。
  LoRALinear 的条件权重（lora.LoRALinear.cond）
      lora_cond         形势特征 z（envelope.regime_features）→ 每个 LoRA 模块的逐样本权重 [B, M, K]：
                        gate（K=1）：g = 2·sigmoid(Wz + b)，W、b 零初始化 → g ≡ 1，初始与普通 LoRA 相同（Gated-LoRA）；
                        router（K=experts>1）：softmax(Wz + b)，W、b 小随机初始化打破对称（HydraLoRA 式共享 A + K 个 B）。
逐样本量统一存在 CondState（每次前向前由模型写入，长度 = 原始 batch B），hook / LoRA 里按 lora.per_sample 广播：
TimesFM-3 里所有张量的第 0 维都是"样本为主序"（对称平均把 B 交错成 2B，注意力再展平成 b·v / b·n）。"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

try:
    from .lora import ADAPTER_ROOT, LoRALinear
except ImportError:
    from forecast_backbones.m5.lora import ADAPTER_ROOT, LoRALinear

N_STATIC = 5                                      # lat / lon / alt / log 容量 / has_coords


def static_features(meta, cap_of, sid: np.ndarray) -> np.ndarray:
    """[B] 站号 → s [B, N_STATIC] float32，量纲大致归一到 O(1)。缺坐标站的 lat/lon/alt 记 0、has_coords = 0。"""
    out = np.zeros((len(sid), N_STATIC), np.float32)
    for i, s in enumerate(np.asarray(sid, int)):
        r = meta.get(int(s))
        if r is not None and meta.has_coords(int(s)):
            out[i, 0] = (float(r["lat"]) - 30.0) / 10.0
            out[i, 1] = (float(r["lon"]) - 105.0) / 15.0
            out[i, 2] = float(r.get("alt_m") or 0.0) / 1000.0
            out[i, 4] = 1.0
        out[i, 3] = math.log(max(float(cap_of(int(s))), 1e-3)) - 4.0
    return out


class CondState:
    """一次前向的逐样本条件量（长度都是原始 batch B）：模型在调底座前 set()，hook 与 LoRALinear 读。"""

    def __init__(self):
        self.B = None
        self.static = None          # torch [B, N_STATIC]
        self.slot = None            # torch long [B]：prompt 逐站表的行号（0 = 未知站 / 只用全局）
        self.z = None               # torch [B, Z]：形势特征
        self._lora_w = None         # torch [B, M, K]
        self.v_orig = None          # transformer_stack 前的变量数（prompt 切回用）

    def set(self, B: int, static=None, slot=None, z=None, lora_w=None) -> None:
        self.B, self.static, self.slot, self.z, self._lora_w = int(B), static, slot, z, lora_w

    def lora_weights(self, idx):
        return None if self._lora_w is None else self._lora_w[:, idx]


class StaticEmbed(nn.Module):
    def __init__(self, d: int, n_in: int = N_STATIC, dropout: float = 0.25):
        super().__init__()
        self.proj = nn.Linear(n_in, d, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.dropout = float(dropout)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        if self.training and self.dropout > 0:
            keep = (torch.rand(s.shape[0], 1, device=s.device) >= self.dropout).to(s.dtype)
            s = s * keep
        return self.proj(s)                                     # [B, d]


class ResblockAdapter(nn.Module):
    def __init__(self, d: int, rank: int):
        super().__init__()
        self.down = nn.Linear(d, rank, bias=True)
        self.up = nn.Linear(rank, d, bias=True)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(torch.relu(self.down(x)))


class PromptVariates(nn.Module):
    def __init__(self, d: int, n_tokens: int, n_slots: int = 0, init_std: float = 0.02, station_dropout: float = 0.25):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(n_tokens, d) * float(init_std))
        self.station = nn.Parameter(torch.zeros(n_slots, n_tokens, d)) if n_slots > 0 else None
        self.station_dropout = float(station_dropout)

    def forward(self, B: int, slot: torch.Tensor | None) -> torch.Tensor:
        """→ [B, k, d]。slot 0 = 只用全局；训练时按 station_dropout 把行随机落到 slot 0。"""
        t = self.tokens[None].expand(B, -1, -1)
        if self.station is None or slot is None:
            return t
        if self.training and self.station_dropout > 0:
            slot = torch.where(torch.rand(B, device=slot.device) < self.station_dropout, torch.zeros_like(slot), slot)
        delta = self.station[slot.clamp_min(0)] * (slot > 0).to(t.dtype)[:, None, None]
        return t + delta


class LoRACond(nn.Module):
    def __init__(self, n_z: int, n_modules: int, experts: int, mode: str):
        super().__init__()
        if mode not in ("gate", "router"):
            raise ValueError(f"LoRACond mode 须是 gate / router，得到 {mode!r}")
        self.mode, self.M, self.K = mode, int(n_modules), int(experts)
        self.net = nn.Linear(n_z, self.M * self.K)
        if mode == "gate":
            nn.init.zeros_(self.net.weight); nn.init.zeros_(self.net.bias)
        else:
            nn.init.normal_(self.net.weight, std=1.0); nn.init.normal_(self.net.bias, std=0.5)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        a = self.net(z).view(z.shape[0], self.M, self.K)
        return 2.0 * torch.sigmoid(a) if self.mode == "gate" else torch.softmax(a, dim=-1)


ODD = {"on": True}                               # 对称平均下常数注入取奇（见 _repeat）；测试里可关掉做对照


def _repeat(t: torch.Tensor, n: int, odd: bool = False) -> torch.Tensor:
    """[B, ...] → [n, ...]（n = B 的整数倍，样本为主序）。
    odd=True 且 n = 2B（对称平均把 [x, −x] 交错成 2B）：−x 分支取 −t。底座输出 = (f(x) − f(−x) 翻转)/2，
    同号的常数注入在两支里的效果大部分相减抵消；取奇后两支同向，注入的效果才完整留在输出里。"""
    if n % t.shape[0]:
        raise ValueError(f"逐样本量 B={t.shape[0]} 与张量首维 {n} 不整除")
    k = n // t.shape[0]
    out = t.repeat_interleave(k, dim=0)
    if odd and k == 2 and ODD["on"]:
        sign = torch.tensor([1.0, -1.0], device=t.device, dtype=t.dtype).repeat(t.shape[0])
        out = out * sign.reshape((-1,) + (1,) * (out.dim() - 1))
    return out


def attach(root: nn.Module, ft: dict, n_slots: int, n_z: int) -> CondState | None:
    """按 finetune 段挂适配器、绑 LoRA 条件权重、注册 hook。什么都没开 → 不动模型，返回 None。
    必须在 inject_lora 之后、freeze_all_but_lora 之前调用（适配器参数靠名字前缀判可训）。"""
    lo, pr, se, ra = ft["lora"], ft.get("prompt") or {}, ft.get("static_embed") or {}, ft.get("resblock_adapter") or {}
    experts, gate = int(lo.get("experts", 1) or 1), bool(lo.get("gate", False))
    n_tok, rank_ra = int(pr.get("n_tokens", 0) or 0), int(ra.get("rank", 0) or 0)
    use_static = bool(se.get("enabled", False))
    if not (experts > 1 or gate or n_tok > 0 or rank_ra > 0 or use_static):
        return None
    if experts > 1 and gate:
        raise ValueError("finetune.lora.gate 与 experts>1 二选一：experts>1 时路由权重本身就是逐样本门")
    d = int(root.output_head.in_features)
    mods = nn.ModuleDict()
    state = CondState()
    loras = [m for _, m in root.named_modules() if isinstance(m, LoRALinear)]
    if experts > 1 or gate:
        if not loras:
            raise ValueError("finetune.lora.gate / experts>1 需要 LoRA 靶点（lora.target_modules 非空）")
        mods["lora_cond"] = LoRACond(n_z, len(loras), experts, "router" if experts > 1 else "gate")
        for i, m in enumerate(loras):
            m.cond, m.cond_idx = state, i
    if use_static:
        mods["static_embed"] = StaticEmbed(d, N_STATIC, float(se.get("dropout", 0.25)))
    if rank_ra > 0:
        mods["resblock_adapter"] = ResblockAdapter(d, rank_ra)
    if n_tok > 0:
        mods["prompt"] = PromptVariates(d, n_tok, n_slots if bool(pr.get("per_station", False)) else 0,
                                        float(pr.get("init_std", 0.02)), float(pr.get("station_dropout", 0.25)))
    dev = root.output_head.weight.device
    setattr(root, ADAPTER_ROOT, mods.to(dev))
    if use_static or rank_ra > 0 or n_tok > 0:
        _install_hooks(root, mods, state)
    return state


def _install_hooks(root: nn.Module, mods: nn.ModuleDict, state: CondState) -> None:
    def after_resblock(_m, _inp, out):                         # out [Bq, v, n, d]
        if "static_embed" in mods:
            e = _repeat(mods["static_embed"](state.static), out.shape[0], odd=True)   # [Bq, d]
            out = torch.cat([out[:, :1] + e[:, None, None, :], out[:, 1:]], dim=1)
        if "resblock_adapter" in mods:
            out = mods["resblock_adapter"](out)
        if "prompt" in mods:
            p = _repeat(mods["prompt"](state.B, state.slot), out.shape[0])           # [Bq, k, d]（独立变量，取奇与否小模型上无差别）
            out = torch.cat([out, p[:, :, None, :].expand(-1, -1, out.shape[2], -1)], dim=1)
        return out

    root.pre_transformer_resblock.register_forward_hook(after_resblock)
    if "prompt" not in mods:
        return

    def before_stack(_m, args):
        x, mask, *rest = args                                   # x [Bq, v+k, n, d]；mask [Bq, v, n]
        k = x.shape[1] - mask.shape[1]
        state.v_orig = mask.shape[1]
        if k > 0:
            mask = torch.cat([mask, torch.zeros(mask.shape[0], k, mask.shape[2], dtype=mask.dtype, device=mask.device)], dim=1)
        return (x, mask, *rest)

    def after_stack(_m, _args, out):
        y, *rest = out
        return (y[:, : state.v_orig], *rest)

    root.transformer_stack.register_forward_pre_hook(before_stack)
    root.transformer_stack.register_forward_hook(after_stack)
