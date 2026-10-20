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


def test_geometry_off_skips_pvlib_and_warning(tmp_path, recwarn):
    """geometry=False（M5 默认）：不算太阳几何、不出缺坐标 warning；容量 / 站号 / 时刻槽照旧，degraded 仍标出缺坐标的站。"""
    rowctx._DEGRADED_WARNED.clear()
    b = _builder(tmp_path, drop=(1002,))
    bd = _bd([1001, 1002], ["2025-06-20 12:00", "2025-06-20 12:00"])
    off = b.build(bd, geometry=False)
    assert not [w for w in recwarn if "缺坐标" in str(w.message)]
    assert off.elev_deg is None and off.ghi_clear is None and off.p_clear is None and off.night is None
    assert off.degraded.tolist() == [False, True]
    with pytest.warns(UserWarning, match="1002"):
        on = b.build(bd)
    assert off.cap.tolist() == on.cap.tolist() and np.array_equal(off.slot, on.slot) and np.array_equal(off.station_id, on.station_id)


def _stub(**over):
    """geometry_needed / decide 只读 cfg 与 levels：用桩对象调，不建 TimesFM。"""
    from types import SimpleNamespace
    from forecast_backbones.m5.config import deep_merge
    from forecast_backbones.m5.decision import MODEL_LEVELS
    cfg = load_config(deep_merge({"dataset": {"station_meta": "unused.csv"}}, over))
    return SimpleNamespace(cfg=cfg, levels=tuple(float(q) for q in MODEL_LEVELS))


@pytest.mark.parametrize("over,need", [
    ({}, False),
    ({"decision": {"point": {"default": "p_star"}}}, False),
    ({"decision": {"physics": {"night_zero": True}}}, True),
    ({"decision": {"physics": {"clear_sky_cap_ratio": 1.05}}}, True),
    ({"features": {"derived": {"solar_elevation": True}}}, True),
    ({"features": {"derived": {"clear_sky_power": True}}}, True),
    ({"features": {"derived": {"history_envelope": True}}}, False),      # 包络只用时刻槽，不用太阳几何
    ({"train": {"loss": {"night_weight": 0.5}}}, True),
])
def test_geometry_needed(over, need):
    from forecast_backbones.m5.transformer_models import TransformersAdaptedM5Model as M
    st = _stub(**over)
    assert M.geometry_needed(st) is need
    assert M.geometry_needed(st, {"physics": {"night_zero": True}}) is True            # 评估时换的规则也算


@pytest.mark.parametrize("point", ["median", "p_star", "p_star_l1"])
def test_decide_without_geometry_is_bit_identical(tmp_path, point):
    """几何没用上时，不算几何（None）与算了几何的决策结果逐位相同。"""
    from forecast_backbones.m5.transformer_models import TransformersAdaptedM5Model as M
    st = _stub(decision={"point": {"default": point}})
    ctx = _builder(tmp_path).build(_bd([1001, 1002], ["2025-06-20 10:00", "2025-06-20 22:00"]))
    rng = np.random.default_rng(0)
    q = np.sort(rng.uniform(-20, 600, (2, H, 9)), -1)
    a = M.decide(st, q, ctx.night[:, L:], ctx.p_clear[:, L:], ctx.cap)
    b = M.decide(st, q, None, None, ctx.cap)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
    with pytest.raises(ValueError, match="太阳几何"):
        M.decide(st, q, None, None, ctx.cap, {**st.cfg["decision"], "physics": {**st.cfg["decision"]["physics"], "night_zero": True}})
