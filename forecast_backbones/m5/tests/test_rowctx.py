import numpy as np
import pandas as pd
import pytest

from forecast_backbones.m5.config import load_config
from forecast_backbones.m5 import rowctx
from forecast_backbones.m5.rowctx import RowContextBuilder
from forecast_backbones.m5.solar import StationMeta
from .conftest import write_station_meta, L, H


def _bd(sids, ts):
    return {"station_id": np.asarray(sids), "timestamp_win": np.asarray([pd.Timestamp(t).value // 10**9 for t in ts])}


def _builder(tmp_path, drop=()):
    meta = StationMeta.from_csv(write_station_meta(tmp_path, drop_coords_for=drop))
    return RowContextBuilder(load_config({"dataset": {"station_meta": "unused.csv"}}), meta)


def test_caps_shapes_and_night(tmp_path):
    ctx = _builder(tmp_path).build(_bd([1001, 1002], ["2025-06-20 10:00", "2025-06-20 10:00"]))
    assert ctx.cap.tolist() == [500.0, 300.0] and ctx.station_id.tolist() == [1001, 1002]
    assert ctx.elev_deg.shape == (2, L + H) and ctx.p_clear.shape == (2, L + H) and ctx.night.shape == (2, L + H)
    assert not ctx.night[0, L - 1]                       # 起报时刻 10:00 是白天
    assert ctx.night[0, L + 63]                          # 未来第 64 步 = 次日 02:00 是夜间
    # 高海拔站晴空 GHI 可超 1000 W/m²，p_clear 与 M8 一致不截断：= ghi_clear/1000·cap，夜间为 0
    assert np.allclose(ctx.p_clear, ctx.ghi_clear / 1000.0 * ctx.cap[:, None]) and (ctx.p_clear[ctx.night] == 0).all()
    assert (ctx.p_clear >= 0).all() and not ctx.degraded.any()
    assert ctx.k_steps.shape == (2, H) and (ctx.k_steps == 0).all()
    assert (ctx.var_signal == 0).all() and (ctx.cs_ratio == 1).all() and not ctx.envelope_used.any()


def test_missing_coords_degrade_with_warning(tmp_path):
    rowctx._DEGRADED_WARNED.clear()                      # 模块级去重集合：避免被同进程其他测试的状态吞掉警告
    with pytest.warns(UserWarning, match="1002"):
        ctx = _builder(tmp_path, drop=(1002,)).build(_bd([1001, 1002], ["2025-06-20 12:00", "2025-06-20 12:00"]))
    assert ctx.degraded.tolist() == [False, True]
    assert ctx.p_clear[1, L - 1] == pytest.approx(300.0)  # 钟表几何：白天 p_clear = 容量


def test_unknown_station_raises(tmp_path):
    with pytest.raises(ValueError, match="station_meta"):
        _builder(tmp_path).build(_bd([9999], ["2025-06-20 10:00"]))


def test_missing_station_id_column_raises(tmp_path):
    with pytest.raises(ValueError, match="station_id"):
        _builder(tmp_path).build({"timestamp_win": np.asarray([0])})
