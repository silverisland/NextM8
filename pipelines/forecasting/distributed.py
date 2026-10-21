"""Distributed rank and synchronization helpers; no-op in one process."""

from __future__ import annotations

import torch
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
    if not _on():
        return
    if dist.get_backend() == "nccl":
        # NCCL may not know the bound device before its first collective.
        # current_device is a logical index after CUDA_VISIBLE_DEVICES remapping.
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def broadcast_from_main(obj):
    """把主进程的 obj 发给所有进程，其余进程传入的值被覆盖。"""
    if not _on():
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=0)
    return box[0]
