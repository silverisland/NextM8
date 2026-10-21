# -*- coding: utf-8 -*-
# forecast_backbones/m8/rows.py（2026-10-13 起自带一份，不再引用 m6）
"""按行取子集：南网短期打分实际用到的行、按（站, 起报日）等概率抽训练行。"""
from __future__ import annotations


import numpy as np
import pandas as pd
import torch

SHORT_ISSUE_TIME = "10:00"          # NanwangMetric.short_d1_daily 的默认起报时刻


from pipelines.forecasting.rows import row_times as row_times
from pipelines.forecasting.rows import row_stations as row_stations
from pipelines.forecasting.rows import scored_row_index as scored_row_index
from pipelines.forecasting.rows import issue_grid_row_index as issue_grid_row_index
from pipelines.forecasting.rows import StationDaySampler as StationDaySampler
