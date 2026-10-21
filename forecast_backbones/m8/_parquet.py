# forecast_backbones/m8/_parquet.py
"""M8 自带的拼表与站号归一（2026-10-14 从 pipelines.time_splits 搬来，行为相同；m8 训练不再依赖切分工具）：
  concat_parquets：文件夹里一站一个的 parquet 拼成一张表，各文件列集合必须一致，站标识列类型不一致时统一成文本站号；
  station_keys：站标识归一成文本（1001 / 1001.0 / '1001.0' 都成 '1001'）。"""
from __future__ import annotations


from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

STATION_COLS = ("station_id", "station")


from pipelines.forecasting.parquet import arrow_kind as _arrow_kind
from pipelines.forecasting.parquet import station_keys as station_keys
from pipelines.forecasting.parquet import concat_parquets as concat_parquets
