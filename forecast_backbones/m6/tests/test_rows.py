import numpy as np
import pandas as pd

from pipelines.transformer_dataset import UnifiedDataset
from forecast_backbones.m6.rows import StationDaySampler, row_stations
from .conftest import make_synthetic_frame


def _ds(df):
    return UnifiedDataset(df.reset_index(drop=True), timestamp_cols=["timestamp_win"], entity_cols=[],
                          label_cols=["observe_power_future"], for_hf_trainer=True)


def test_station_day_sampler_gives_every_station_day_equal_mass():
    """站 1001 每小时一个起报、站 1002 每 6 小时一个：按行均匀抽时 1001 占 80%，按站日等概率时各占一半。"""
    dense = make_synthetic_frame(n_days=4, stations=(1001,), issue_every_min=60, seed=0)
    sparse = make_synthetic_frame(n_days=4, stations=(1002,), issue_every_min=360, seed=1)
    ds = _ds(pd.concat([dense, sparse]))
    s = StationDaySampler(ds, "station_id", "timestamp_win", seed=0)
    assert len(s) == len(ds) == 4 * 24 + 4 * 4 and s.n_groups == 8
    st = row_stations(ds, "station_id")
    draws = np.concatenate([list(iter(s)) for _ in range(20)])
    assert abs((st[draws] == 1001).mean() - 0.5) < 0.02
    assert np.isclose(s.weights.sum().item(), 8.0)                       # 每个站日总质量 1


def test_station_day_sampler_is_reproducible_per_epoch():
    ds = _ds(make_synthetic_frame(n_days=2, issue_every_min=60, seed=0))
    a, b = StationDaySampler(ds, "station_id", "timestamp_win", seed=3), StationDaySampler(ds, "station_id", "timestamp_win", seed=3)
    first_a, first_b = list(iter(a)), list(iter(b))
    assert first_a == first_b                                            # 各进程同一序列
    assert list(iter(a)) != first_a                                      # 下一轮换一批
    a.set_epoch(0)
    assert list(iter(a)) == first_a
