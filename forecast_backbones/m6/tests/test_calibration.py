import numpy as np
import pandas as pd
import pytest
from forecast_backbones.m6.calibration import (CalibrationConfig, Calibrator, quantile_cols,
                                               nearest_model_level, n_min, bin_elev, bin_lead)
from forecast_backbones.m6.decision import MODEL_LEVELS, CAL_LEVELS

def _cfg(**kw):
    base = dict(levels=CAL_LEVELS, model_levels=MODEL_LEVELS, normalization="clear_sky",
                elev_edges=(20.0, 45.0), lead_edges=(4, 8, 16, 55, 103), use_variability=True, n_min_factor=10.0)
    base.update(kw)
    return CalibrationConfig(**base)

def _table(n=6000, seed=0, bias=0.0, scale=1.0):
    """真值 y = q50·(1+ε)，ε~N(bias, 0.3·scale)；模型分位 = q50·[0.6..1.4]（偏窄/偏宽由 scale 控制）。"""
    rng = np.random.default_rng(seed)
    q50 = rng.uniform(30, 90, n); p_clear = np.full(n, 100.0)
    y = np.clip(q50 * (1 + rng.normal(bias, 0.3 * scale, n)), 0, None)
    q = q50[:, None] * np.linspace(0.6, 1.4, 9)[None, :]
    df = pd.DataFrame({"station_id": 1, "lead": rng.integers(0, 16, n), "elev_deg": 60.0,
                       "var_bin": 1, "y": y, "p_clear": p_clear})
    for c, j in zip(quantile_cols(MODEL_LEVELS), range(9)):
        df[c] = q[:, j]
    return df

def test_helpers():
    assert quantile_cols([0.1, 0.5]) == ["q_0.1", "q_0.5"]
    assert nearest_model_level(0.05, MODEL_LEVELS) == 0.1 and nearest_model_level(0.95, MODEL_LEVELS) == 0.9
    assert n_min(0.05, 10) == 200 and n_min(0.5, 10) == 20
    assert bin_elev(np.array([5.0, 30.0, 60.0]), (20, 45)).tolist() == [0, 1, 2]
    assert bin_lead(np.array([0, 4, 60, 191]), (4, 8, 16, 55, 103)).tolist() == [0, 1, 4, 5]

def test_fit_apply_calibrates_coverage():
    cal = Calibrator(_cfg()).fit(_table(seed=0, scale=1.5))          # 模型分位偏窄
    test = _table(seed=1, scale=1.5)
    qc = cal.apply(test[quantile_cols(MODEL_LEVELS)].to_numpy(), test.station_id.to_numpy(),
                   test.elev_deg.to_numpy(), test.var_bin.to_numpy(), test.lead.to_numpy(), test.p_clear.to_numpy())
    assert qc.shape == (len(test), len(CAL_LEVELS))
    assert (np.diff(qc, axis=1) >= 0).all()                             # 单调
    for j, tau in enumerate(CAL_LEVELS):
        cov = (test.y.to_numpy() <= qc[:, j]).mean()
        assert abs(cov - tau) < 0.03, f"tau={tau} cov={cov}"

def test_uncalibrated_narrow_quantiles_fail_coverage_so_calibration_matters():
    test = _table(seed=1, scale=1.5)
    raw_cov10 = (test.y <= test["q_0.1"]).mean()
    assert raw_cov10 > 0.13                                             # 偏窄：q10 覆盖 > 10%

def test_hierarchical_fallback_used_for_sparse_bins():
    df = _table(n=300)                                                   # 单层 n 小，尾档必回退
    cal = Calibrator(_cfg()).fit(df)
    t = cal.table
    assert set(t["level"].unique()) >= {0, 3, 4}
    qc = cal.apply(df[quantile_cols(MODEL_LEVELS)].to_numpy(), df.station_id.to_numpy(),
                   df.elev_deg.to_numpy(), df.var_bin.to_numpy(), df.lead.to_numpy(), df.p_clear.to_numpy())
    assert np.isfinite(qc).all()

def test_save_load_roundtrip(tmp_path):
    cal = Calibrator(_cfg()).fit(_table())
    p = tmp_path / "cal.parquet"; cal.save(p)
    cal2 = Calibrator.load(p, _cfg())
    assert cal2.is_fitted and len(cal2.table) == len(cal.table)
    t = _table(seed=2)
    a = cal.apply(t[quantile_cols(MODEL_LEVELS)].to_numpy(), t.station_id.to_numpy(), t.elev_deg.to_numpy(),
                  t.var_bin.to_numpy(), t.lead.to_numpy(), t.p_clear.to_numpy())
    b = cal2.apply(t[quantile_cols(MODEL_LEVELS)].to_numpy(), t.station_id.to_numpy(), t.elev_deg.to_numpy(),
                   t.var_bin.to_numpy(), t.lead.to_numpy(), t.p_clear.to_numpy())
    assert np.allclose(a, b)

def test_unknown_station_falls_back_to_global():
    cal = Calibrator(_cfg()).fit(_table())
    t = _table(seed=3); t["station_id"] = 999
    qc = cal.apply(t[quantile_cols(MODEL_LEVELS)].to_numpy(), t.station_id.to_numpy(), t.elev_deg.to_numpy(),
                   t.var_bin.to_numpy(), t.lead.to_numpy(), t.p_clear.to_numpy())
    assert np.isfinite(qc).all()
