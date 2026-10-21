# -*- coding: utf-8 -*-
# forecast_backbones/m8/_dist.py（自带一份，不再引用 m6）
"""多进程训练（accelerate launch / torchrun 起的 DDP）下的进程身份与同步；单进程时全部退化成无操作。"""
from __future__ import annotations


import torch.distributed as dist


from pipelines.forecasting.distributed import _on as _on
from pipelines.forecasting.distributed import rank as rank
from pipelines.forecasting.distributed import world_size as world_size
from pipelines.forecasting.distributed import is_main as is_main
from pipelines.forecasting.distributed import barrier as barrier
from pipelines.forecasting.distributed import broadcast_from_main as broadcast_from_main
