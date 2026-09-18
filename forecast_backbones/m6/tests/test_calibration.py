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
    qc = cal.apply(df[quantile_cols(MODEL_LEVELS)].to_numpy(), df.station_id.to_numpy(),
                   df.elev_deg.to_numpy(), df.var_bin.to_numpy(), df.lead.to_numpy(), df.p_clear.to_numpy())
    assert np.isfinite(qc).all()
    taus = list(CAL_LEVELS)
    levels = cal.last_levels
    # 独立算层0（=层1，var 已恒定）按时效档分箱后的计数，不依赖 Calibrator 内部状态
    lead_bin_counts = pd.Series(bin_lead(df.lead.to_numpy(), cal.cfg.lead_edges)).value_counts()
    nmin_50 = n_min(0.5, cal.cfg.n_min_factor)
    nmin_05 = n_min(0.05, cal.cfg.n_min_factor)
    assert (lead_bin_counts >= nmin_50).all()               # 层0对 τ=0.5 处处够数 → 应选层0
    assert (levels[:, taus.index(0.5)] == 0).all()
    assert (lead_bin_counts < nmin_05).all()                # 层0/1 对 τ=0.05 处处不够数 → 必回退
    assert len(df) >= nmin_05                               # 层2（站×高度角，忽略时效档）总量够数
    assert (levels[:, taus.index(0.05)] == 2).all()

def test_apply_tracks_calibration_level_and_stats():
    df = _table(n=300)
    cal = Calibrator(_cfg()).fit(df)
    qc = cal.apply(df[quantile_cols(MODEL_LEVELS)].to_numpy(), df.station_id.to_numpy(),
                   df.elev_deg.to_numpy(), df.var_bin.to_numpy(), df.lead.to_numpy(), df.p_clear.to_numpy())
    taus = list(CAL_LEVELS)
    levels = cal.last_levels
    assert levels.shape == (len(df), len(CAL_LEVELS))
    for tau in taus:                                         # 尾档 n_min 超过样本总量：任何层都不够 → -1
        if n_min(tau, cal.cfg.n_min_factor) > len(df):
            assert (levels[:, taus.index(tau)] == -1).all(), f"tau={tau}"
    assert (levels[:, taus.index(0.5)] >= 0).all()           # τ=0.5 全体层足够 → 必被校准
    stats = cal.last_stats
    assert stats["n"] == len(df)
    assert stats["uncalibrated_cells"] == int((levels == -1).sum())
    assert abs(stats["uncalibrated_frac"] - stats["uncalibrated_cells"] / (len(df) * len(taus))) < 1e-9
    assert set(stats["per_tau_uncalibrated_frac"].keys()) == set(taus)
    for tau in taus:
        j = taus.index(tau)
        expect = float((levels[:, j] == -1).mean())
        assert abs(stats["per_tau_uncalibrated_frac"][tau] - expect) < 1e-9

def test_delta_pins_ceil_n1_tau_order_statistic():
    n = 9
    df = pd.DataFrame({"station_id": 1, "lead": 0, "elev_deg": 60.0, "var_bin": 1,
                       "y": np.arange(1, n + 1, dtype=float), "p_clear": 1.0})
    for c in quantile_cols(MODEL_LEVELS):
        df[c] = 0.0                                          # q_τ0≡0，norm≡1 → 分数 s = y = 1..9
    cal = Calibrator(_cfg()).fit(df)
    g = cal.table[cal.table.level == 0].set_index("tau")["delta"]
    assert g.loc[0.3] == 3.0                                  # k=ceil(10·0.3)=3 → 第3小
    assert g.loc[0.5] == 5.0                                  # k=5
    assert g.loc[0.9] == 9.0                                  # k=9（=n，最大）
    assert g.loc[0.99] == 9.0                                 # k=ceil(9.9)=10，clip 到 n=9

def test_save_load_roundtrip(tmp_path):
    cal = Calibrator(_cfg()).fit(_table())
    p = tmp_path / "cal.parquet"; cal.save(p)
    assert (tmp_path / "cal.parquet.json").exists()
    cal2 = Calibrator.load(p, _cfg())
    assert cal2.is_fitted and len(cal2.table) == len(cal.table)
    t = _table(seed=2)
    a = cal.apply(t[quantile_cols(MODEL_LEVELS)].to_numpy(), t.station_id.to_numpy(), t.elev_deg.to_numpy(),
                  t.var_bin.to_numpy(), t.lead.to_numpy(), t.p_clear.to_numpy())
    b = cal2.apply(t[quantile_cols(MODEL_LEVELS)].to_numpy(), t.station_id.to_numpy(), t.elev_deg.to_numpy(),
                   t.var_bin.to_numpy(), t.lead.to_numpy(), t.p_clear.to_numpy())
    assert np.allclose(a, b)

def test_save_load_metadata_mismatch_raises(tmp_path):
    cal = Calibrator(_cfg()).fit(_table())
    p = tmp_path / "cal.parquet"; cal.save(p)
    bad_levels = tuple(0.55 if l == 0.5 else l for l in CAL_LEVELS)
    with pytest.raises(ValueError, match="levels"):
        Calibrator.load(p, _cfg(levels=bad_levels))

def test_save_load_missing_sidecar_raises(tmp_path):
    cal = Calibrator(_cfg()).fit(_table())
    p = tmp_path / "cal.parquet"; cal.save(p)
    (tmp_path / "cal.parquet.json").unlink()
    with pytest.raises(ValueError):
        Calibrator.load(p, _cfg())

def test_unknown_station_falls_back_to_global():
    cal = Calibrator(_cfg()).fit(_table())
    t = _table(seed=3); t["station_id"] = 999
    q_raw = t[quantile_cols(MODEL_LEVELS)].to_numpy()
    qc = cal.apply(q_raw, t.station_id.to_numpy(), t.elev_deg.to_numpy(),
                   t.var_bin.to_numpy(), t.lead.to_numpy(), t.p_clear.to_numpy())
    assert np.isfinite(qc).all()
    assert (cal.last_levels == 4).all()                       # 未知站：所有 τ 都回退到全体层
    delta_g = cal.table.loc[(cal.table.level == 4) & (cal.table.tau == 0.5), "delta"].iloc[0]
    assert abs(delta_g) > 1e-9                                 # 全体层校准非零（非静默直通）
    expected50 = q_raw[:, list(MODEL_LEVELS).index(0.5)] + delta_g * np.maximum(t.p_clear.to_numpy(), 1e-6)
    assert np.any(np.isclose(qc, expected50[:, None]), axis=1).all()   # τ=0.5 的校准值确实出现在输出行中


# ---------------------------------------------------------------- Minor: levels/model_levels 必须升序

def test_calibration_config_rejects_unsorted_levels():
    with pytest.raises(ValueError, match="levels"):
        _cfg(levels=(0.5, 0.1, 0.9))


def test_calibration_config_rejects_unsorted_model_levels():
    with pytest.raises(ValueError, match="model_levels"):
        _cfg(model_levels=(0.5, 0.1, 0.9))
