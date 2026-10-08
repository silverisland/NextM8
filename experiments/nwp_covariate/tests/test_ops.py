import numpy as np
import pandas as pd

from ops import (BuildContext, Pair, clear_sky_ghi, convert_kind, op_consensus, op_kt, op_p_phys,
                 window_timestamps, run_op)


def _ctx(B=2, C=8, P=4):
    ts = np.array([np.datetime64("2026-08-26T10:00:00")] * B)
    return BuildContext(ts_win=ts, context_len=C, pred_len=P, lat=np.full(B, 23.0), lon=np.full(B, 108.0),
                        alt_m=np.zeros(B), cap=np.array([100.0, 50.0]))


def test_window_timestamps_last_hist_is_issue_time():
    ts = window_timestamps(np.array([np.datetime64("2026-08-26T10:00:00")]), 672, 192)
    assert ts.shape == (1, 864)
    assert ts[0, 671] == np.datetime64("2026-08-26T10:00:00")
    assert ts[0, 672] == np.datetime64("2026-08-26T10:15:00")
    assert ts[0, 0] == np.datetime64("2026-08-19T10:15:00")


def test_clear_sky_zero_at_night_and_peaks_near_noon():
    ts = window_timestamps(np.array([np.datetime64("2026-06-21T00:00:00")]), 1, 96)
    cs = clear_sky_ghi(ts, np.array([23.0]), np.array([108.0]))[0, 1:]
    hours = pd.DatetimeIndex(ts[0, 1:]).hour
    assert cs[hours == 2].max() == 0
    assert 700 < cs.max() < 1100
    assert 11 <= hours[cs.argmax()] <= 14


def test_convert_kind_accum_and_units():
    full = np.arange(12, dtype=float)[None, :] * 900.0  # 每 15min 累积 900 J/m² -> 1 W/m²
    p = convert_kind(Pair(full[:, :8], full[:, 8:]), "accum_jm2")
    assert np.allclose(p.full()[0, 1:], 1.0)
    k = convert_kind(Pair(np.full((1, 2), 300.15), np.full((1, 1), 273.15)), "kelvin")
    assert np.allclose(k.hist, 27.0) and np.allclose(k.fut, 0.0)
    pc = convert_kind(Pair(np.full((1, 2), 50.0), np.full((1, 1), 100.0)), "pct")
    assert np.allclose(pc.full(), [[0.5, 0.5, 1.0]])


def test_kt_and_p_phys():
    ctx = _ctx()
    ctx.cols["ghi"] = Pair(np.full((2, 8), 500.0), np.full((2, 4), 250.0))
    ctx.cols["clear"] = Pair(np.full((2, 8), 1000.0), np.full((2, 4), 10.0))
    kt = op_kt(ctx, "ghi", "clear", min_clear=50)
    assert np.allclose(kt.hist, 0.5) and np.allclose(kt.fut, 0.0)  # 未来晴空<50 -> 夜间填 0
    ctx.cols["tc"] = Pair(np.full((2, 8), 45.0), np.full((2, 4), 25.0))
    p = op_p_phys(ctx, "ghi", "tc", gamma=-0.004)
    assert np.allclose(p.hist[0], 100 * 0.5 * (1 - 0.004 * 20))
    assert np.allclose(p.fut[1], 50 * 0.25)


def test_consensus_modes_and_missing_member():
    ctx = _ctx()
    ctx.cols["a"] = Pair(np.full((2, 8), 1.0), np.full((2, 4), 1.0))
    ctx.cols["b"] = Pair(np.full((2, 8), 3.0), np.full((2, 4), np.nan))
    eq = op_consensus(ctx, ["a", "b", "zzz"], mode="equal")
    assert np.allclose(eq.hist, 2.0) and np.allclose(eq.fut, 1.0)      # 未来 b 全 NaN -> 只剩 a
    med = op_consensus(ctx, ["a", "b"], mode="median")
    assert np.allclose(med.hist, 2.0)
    ctx.cols["ref"] = Pair(np.full((2, 8), 1.0), np.full((2, 4), np.nan))
    sk = op_consensus(ctx, ["a", "b"], mode="skill", reference="ref")
    assert np.all(sk.hist < 2.0)                                        # a 更贴近 ref，权重更大
    assert any("成员缺失" in w for w in ctx.warnings)


def test_run_op_unknown_raises():
    ctx = _ctx()
    try:
        run_op(ctx, "nope", {})
        assert False
    except KeyError:
        pass
