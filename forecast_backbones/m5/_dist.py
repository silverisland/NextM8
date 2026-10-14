# -*- coding: utf-8 -*-
# forecast_backbones/m5/_dist.py（自带一份，不再引用 m6）
"""多进程训练（accelerate launch / torchrun 起的 DDP）下的进程身份与同步；单进程时全部退化成无操作。"""
from __future__ import annotations

import torch.distributed as dist


def _on() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if _on() else 0


def world_size() -> int:
    return dist.get_world_size() if _on() else 1


def is_main() -> bool:
    return rank() == 0


def barrier() -> None:
    if _on():
        dist.barrier()


def broadcast_from_main(obj):
    """把主进程的 obj 发给所有进程，其余进程传入的值被覆盖。"""
    if not _on():
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=0)
    return box[0]
