# -*- coding: utf-8 -*-
# forecast_backbones/m5/lora.py
"""LoRA 最小实现（不依赖 peft）。LoRALinear 包装 nn.Linear：base 冻结，A[r,in] kaiming、B[out,r] 零初始化，
所以挂载瞬间输出与原模型逐位相同；scaling = alpha / rank。inject_lora 按模块名尾段匹配替换（'re:' 开头的靶点按完整路径
fullmatch，如 're:.*\\.var_attn\\..*_proj'），非空靶点零匹配即 raise，空靶点不挂（只训 head / 解冻块）。
存取只走 lora_state_dict / load_lora_state_dict（LoRA 参数 + 全部可训的整块参数：output_head、unfreeze_modules），不保存底座；
落盘走 save_lora_weights / load_lora_weights：.safetensors 后缀存成 safetensors，其他后缀用 torch.save。
2026-10-18 扩展（都默认关，关时与原 LoRA 逐位相同）：
  experts>1  HydraLoRA 式：共享 A、K 个 B，逐样本按路由权重混合；
  条件权重    LoRALinear 绑定一个 CondState（adapters.py），每次前向读逐样本权重 [B, M, K]（Gated-LoRA 的门 / HydraLoRA 的路由）；
  IA3Linear  (IA)³：输出逐通道乘可训缩放向量（初值 1 = 原模型）。
"适配器参数" = LoRA 因子 + (IA)³ 缩放 + m5_adapters 子模块（prompt / 静态嵌入 / 残差块适配器 / 门控网络）：都可训、都存档、加载时都必须齐。"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn


ADAPTER_ROOT = "m5_adapters"                       # adapters.attach 挂在底座根模块下的 ModuleDict 名


def _is_lora_name(name: str) -> bool:
    """Check if a parameter name corresponds to a LoRA factor (lora_A or lora_B)."""
    return name.endswith(".lora_A") or name.endswith(".lora_B")


def _is_adapter_name(name: str) -> bool:
    """LoRA 因子、(IA)³ 缩放、m5_adapters 下的参数：构造即可训，存档必带。"""
    return _is_lora_name(name) or name.endswith(".ia3_scale") or name == ADAPTER_ROOT or name.startswith(ADAPTER_ROOT + ".")


def per_sample(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """逐样本量 w [B, ...] 广播到 x 的行：TimesFM-3 里所有线性层输入的第 0 维都是"样本为主序"的展平
    （b·v、b·n、b，对称平均再 ×2 交错），所以 repeat_interleave(x.shape[0] // B) 后补够维数即可。"""
    B = w.shape[0]
    if x.shape[0] % B:
        raise ValueError(f"逐样本权重 B={B} 与输入首维 {x.shape[0]} 不整除：CondState 没在本次前向前更新？")
    w = w.repeat_interleave(x.shape[0] // B, dim=0)
    return w.reshape(w.shape[:1] + (1,) * (x.dim() - 2) + w.shape[1:])


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float, experts: int = 1):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank 必须 > 0（得到 {rank}）")
        if experts < 1:
            raise ValueError(f"experts 必须 ≥ 1（得到 {experts}）")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        kw = dict(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, **kw))
        shape_b = (base.out_features, rank) if experts == 1 else (experts, base.out_features, rank)
        self.lora_B = nn.Parameter(torch.zeros(*shape_b, **kw))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = float(alpha) / float(rank)
        self.experts = int(experts)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.cond, self.cond_idx = None, None          # adapters.CondState 与本模块在 [B, M, K] 里的下标 M

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(x) @ self.lora_A.t()                                   # [..., r]
        w = None if self.cond is None else self.cond.lora_weights(self.cond_idx)   # [B, K] 或 None
        if self.experts == 1:
            d = h @ self.lora_B.t()
            if w is not None:
                d = d * per_sample(w[:, :1], x)
        else:
            if w is None:
                raise RuntimeError("experts>1 的 LoRA 需要路由权重：模型没有绑定 CondState（adapters.attach）")
            # Σ_k w_k·(h B_kᵀ) = [w_1 h, …, w_K h] @ [B_1ᵀ; …; B_Kᵀ]：只多出 [..., K·r] 的中间量，不物化 [..., K, out]
            hw = (h[..., None, :] * per_sample(w, x)[..., None]).flatten(-2)          # [..., K·r]
            d = hw @ self.lora_B.permute(0, 2, 1).reshape(-1, self.lora_B.shape[1])
        return self.base(x) + d * self.scaling

    def extra_repr(self) -> str:
        return f"rank={self.lora_A.shape[0]}, scaling={self.scaling}" + (f", experts={self.experts}" if self.experts > 1 else "")


class IA3Linear(nn.Module):
    """(IA)³：base 冻结，输出逐通道 × ia3_scale（初值全 1，挂上瞬间与原模型逐位相同）。"""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.ia3_scale = nn.Parameter(torch.ones(base.out_features, device=base.weight.device, dtype=base.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) * self.ia3_scale


def _matches(name: str, targets: Sequence[str]) -> bool:
    """普通靶点 = 模块名尾段相等；'re:<正则>' = 完整模块路径 fullmatch。"""
    tail = name.split(".")[-1]
    return any(re.fullmatch(t[3:], name) if t.startswith("re:") else tail == t for t in targets)


def _under(name: str, prefixes: Sequence[str]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def inject_lora(root: nn.Module, target_modules: Sequence[str], rank: int, alpha: float, dropout: float,
                experts: int = 1) -> list[str]:
    targets = list(target_modules)
    if not targets:
        return []
    hits: list[tuple[str, nn.Module, str]] = []
    for name, mod in root.named_modules():
        if name == "" or not isinstance(mod, nn.Linear) or not _matches(name, targets):
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = root.get_submodule(parent_name) if parent_name else root
        if isinstance(parent, (LoRALinear, IA3Linear)):   # 已挂：mod 是包装层的 base，跳过
            continue
        hits.append((name, parent, attr))
    # Check if any LoRALinear modules already exist whose tail segment is in current targets
    already_wrapped = any(_matches(name, targets) for name, m in root.named_modules() if isinstance(m, LoRALinear))
    if not hits and not already_wrapped:
        raise ValueError(f"inject_lora: 没有任何 nn.Linear 的名字（尾段 / 're:' 完整路径）命中 {sorted(targets)}；"
                         f"可用尾段示例：{sorted({n.split('.')[-1] for n, m in root.named_modules() if isinstance(m, nn.Linear)})[:12]}")
    for name, parent, attr in hits:
        setattr(parent, attr, LoRALinear(getattr(parent, attr), rank, alpha, dropout, experts))
    return [h[0] for h in hits]


def inject_ia3(root: nn.Module, target_modules: Sequence[str]) -> list[str]:
    """(IA)³ 挂载：靶点规则同 inject_lora（尾段 / 're:' 完整路径）；已被 LoRA 包装的层不再匹配（其 nn.Linear 名尾段是 base），
    所以与 LoRA 靶点重叠时零命中、直接报错。空靶点不挂。"""
    targets = list(target_modules)
    if not targets:
        return []
    hits = []
    for name, mod in root.named_modules():
        if name == "" or not isinstance(mod, nn.Linear) or not _matches(name, targets):
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = root.get_submodule(parent_name) if parent_name else root
        if isinstance(parent, (LoRALinear, IA3Linear)):
            continue
        hits.append((name, parent, attr))
    if not hits:
        raise ValueError(f"inject_ia3: 没有任何（未被 LoRA 包装的）nn.Linear 命中 {sorted(targets)}；(IA)³ 与 LoRA 靶点不能重叠")
    for name, parent, attr in hits:
        setattr(parent, attr, IA3Linear(getattr(parent, attr)))
    return [h[0] for h in hits]


def freeze_all_but_lora(root: nn.Module, unfreeze_output_head: bool, output_head_name: str = "output_head",
                        unfreeze_modules: Sequence[str] = ()) -> None:
    """只留适配器参数（LoRA / (IA)³ / m5_adapters）、（可选）output_head 与 unfreeze_modules 里的整块参数可训；其余冻结。"""
    dense = ([output_head_name] if unfreeze_output_head else []) + list(unfreeze_modules)
    for name, p in root.named_parameters():
        p.requires_grad_(_is_adapter_name(name) or _under(name, dense))


def lora_state_dict(root: nn.Module) -> dict[str, torch.Tensor]:
    """适配器参数 + 全部可训的整块参数（output_head、unfreeze_modules）。"""
    out = {}
    for name, p in root.named_parameters():
        if _is_adapter_name(name) or p.requires_grad:
            out[name] = p.detach().cpu().clone()
    return out


def save_lora_weights(path, sd: dict[str, torch.Tensor]) -> None:
    """lora_state_dict 的结果落盘。path 以 .safetensors 结尾存 safetensors（键名相对 backbone.model），否则 torch.save。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".safetensors":
        from safetensors.torch import save_file
        save_file({k: v.detach().cpu().contiguous() for k, v in sd.items()}, str(path),
                  metadata={"format": "pt", "content": "m5 lora + output_head", "root": "backbone.model"})
    else:
        torch.save(sd, path)


def load_lora_weights(path) -> dict[str, torch.Tensor]:
    """读 save_lora_weights 存的文件：.safetensors 后缀按 safetensors 读，其他后缀 torch.load。"""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu")


def load_lora_state_dict(root: nn.Module, sd: dict, allowed_prefixes: Sequence[str] = ("output_head",)) -> None:
    """键只能是适配器参数或 allowed_prefixes 下的整块参数；适配器参数必须齐，allowed_prefixes 里除 output_head 外的块也必须齐
    （output_head 可缺：冻结 head 的 run 不存它）。"""
    names = {n for n, _ in root.named_parameters()}
    invalid_keys = [k for k in sd.keys() if not (_is_adapter_name(k) or _under(k, allowed_prefixes)) or k not in names]
    if invalid_keys:
        raise ValueError(f"LoRA 权重含 unexpected 键 {invalid_keys[:5]}：不是本模型的 LoRA 存档（允许前缀 {list(allowed_prefixes)}）")
    blocks = [p for p in allowed_prefixes if p != "output_head"]
    expected = {n for n, _ in root.named_parameters() if _is_adapter_name(n) or _under(n, blocks)}
    missing = sorted(expected - set(sd))
    if missing:
        raise ValueError(f"LoRA 权重缺键 {missing[:5]}{'…' if len(missing) > 5 else ''}")
    root.load_state_dict(sd, strict=False)


def trainable_summary(root: nn.Module) -> dict:
    n_tr = sum(p.numel() for p in root.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in root.parameters())
    return {"n_trainable": int(n_tr), "n_total": int(n_all), "frac": (n_tr / n_all) if n_all else 0.0,
            "n_lora_modules": sum(isinstance(m, LoRALinear) for m in root.modules()),
            "n_ia3_modules": sum(isinstance(m, IA3Linear) for m in root.modules())}
