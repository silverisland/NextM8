# -*- coding: utf-8 -*-
# forecast_backbones/m6/lora.py
"""LoRA 最小实现（不依赖 peft）。LoRALinear 包装 nn.Linear：base 冻结，A[r,in] kaiming、B[out,r] 零初始化，
所以挂载瞬间输出与原模型逐位相同；scaling = alpha / rank。inject_lora 按模块名尾段匹配替换，零匹配即 raise。
存取只走 lora_state_dict / load_lora_state_dict（LoRA 参数 + 可训的 output_head），不保存底座；
落盘走 save_lora_weights / load_lora_weights：.safetensors 后缀存成 safetensors，其他后缀用 torch.save。"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn


def _is_lora_name(name: str) -> bool:
    """Check if a parameter name corresponds to a LoRA factor (lora_A or lora_B)."""
    return name.endswith(".lora_A") or name.endswith(".lora_B")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank 必须 > 0（得到 {rank}）")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling

    def extra_repr(self) -> str:
        return f"rank={self.lora_A.shape[0]}, scaling={self.scaling}"


def inject_lora(root: nn.Module, target_modules: Sequence[str], rank: int, alpha: float, dropout: float) -> list[str]:
    targets = set(target_modules)
    hits: list[tuple[str, nn.Module, str]] = []
    for name, mod in root.named_modules():
        if name == "" or not isinstance(mod, nn.Linear) or name.split(".")[-1] not in targets:
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = root.get_submodule(parent_name) if parent_name else root
        if isinstance(parent, LoRALinear):          # 已挂：mod 是 LoRALinear.base，跳过
            continue
        hits.append((name, parent, attr))
    # Check if any LoRALinear modules already exist whose tail segment is in current targets
    already_wrapped = any(name.split(".")[-1] in targets for name, m in root.named_modules() if isinstance(m, LoRALinear))
    if not hits and not already_wrapped:
        raise ValueError(f"inject_lora: 没有任何 nn.Linear 的名字尾段命中 {sorted(targets)}；"
                         f"可用尾段示例：{sorted({n.split('.')[-1] for n, m in root.named_modules() if isinstance(m, nn.Linear)})[:12]}")
    for name, parent, attr in hits:
        setattr(parent, attr, LoRALinear(getattr(parent, attr), rank, alpha, dropout))
    return [h[0] for h in hits]


def freeze_all_but_lora(root: nn.Module, unfreeze_output_head: bool, output_head_name: str = "output_head") -> None:
    for name, p in root.named_parameters():
        is_lora = _is_lora_name(name)
        is_head = unfreeze_output_head and (name == output_head_name or name.startswith(output_head_name + "."))
        p.requires_grad_(is_lora or is_head)


def lora_state_dict(root: nn.Module) -> dict[str, torch.Tensor]:
    out = {}
    for name, p in root.named_parameters():
        is_lora = _is_lora_name(name)
        is_head = p.requires_grad and (name == "output_head" or name.startswith("output_head."))
        if is_lora or is_head:
            out[name] = p.detach().cpu().clone()
    return out


def save_lora_weights(path, sd: dict[str, torch.Tensor]) -> None:
    """lora_state_dict 的结果落盘。path 以 .safetensors 结尾存 safetensors（键名相对 backbone.model），否则 torch.save。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".safetensors":
        from safetensors.torch import save_file
        save_file({k: v.detach().cpu().contiguous() for k, v in sd.items()}, str(path),
                  metadata={"format": "pt", "content": "m6 lora + output_head", "root": "backbone.model"})
    else:
        torch.save(sd, path)


def load_lora_weights(path) -> dict[str, torch.Tensor]:
    """读 save_lora_weights 存的文件：.safetensors 后缀按 safetensors 读，其他后缀 torch.load。"""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu")


def load_lora_state_dict(root: nn.Module, sd: dict) -> None:
    # Check that all keys in sd are valid (lora_* or output_head.*)
    invalid_keys = [k for k in sd.keys() if not (_is_lora_name(k) or k == "output_head" or k.startswith("output_head."))]
    if invalid_keys:
        raise ValueError(f"LoRA 权重含 unexpected 键 {invalid_keys[:5]}：不是本模型的 LoRA 存档")

    expected = {n for n, _ in root.named_parameters() if _is_lora_name(n)}
    missing_lora = sorted(expected - set(sd))
    if missing_lora:
        raise ValueError(f"LoRA 权重缺键 {missing_lora[:5]}{'…' if len(missing_lora) > 5 else ''}")
    root.load_state_dict(sd, strict=False)


def trainable_summary(root: nn.Module) -> dict:
    n_tr = sum(p.numel() for p in root.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in root.parameters())
    return {"n_trainable": int(n_tr), "n_total": int(n_all), "frac": (n_tr / n_all) if n_all else 0.0,
            "n_lora_modules": sum(isinstance(m, LoRALinear) for m in root.modules())}
