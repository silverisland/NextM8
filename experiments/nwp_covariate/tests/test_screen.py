import numpy as np
import pandas as pd

from build_feature_store import build, feature_store_dir, write_feature_store
from screen import ridge_fit_predict, run, window_scores


def test_ridge_recovers_linear_relation():
    rng = np.random.default_rng(0)
    Xh = rng.normal(size=(200, 2)); yh = 3 * Xh[:, 0] - 2 * Xh[:, 1] + 5 + rng.normal(0, 0.01, 200)
    Xf = rng.normal(size=(50, 2)); yf = 3 * Xf[:, 0] - 2 * Xf[:, 1] + 5
    yhat = ridge_fit_predict(Xh, yh, Xf, alpha=1e-3)
    assert np.allclose(yhat, yf, atol=0.1)
    r2, nrmse = window_scores(Xh, yh, Xf, yf, cap=10.0, alpha=1e-3)
    assert r2 > 0.99 and nrmse < 1.0


def test_ridge_handles_nan_and_too_few_rows():
    Xh = np.full((5, 1), np.nan); yh = np.ones(5)
    assert np.isnan(ridge_fit_predict(Xh, yh, np.ones((3, 1)), 1.0)).all()


def test_screen_ranks_irradiance_above_cloud_and_writes(ds_dir, tmp_path):
    cfg, raw = ds_dir
    frames, rep = build(cfg, "ultra_short")
    out = feature_store_dir(cfg, "ultra_short")
    write_feature_store(frames, out, rep)
    cfg["run"]["out_dir"] = str(tmp_path / "runs")
    cfg["screening"]["out"] = str(tmp_path / "runs" / "screening.xlsx")
    cfg["screening"]["ridge"]["candidates"] = ["ghi_cons", "p_phys", "ghi_ec", "tcc_ec", "t2m_ec", "spread"]
    cfg["screening"]["ridge"]["base"] = ["ghi_cons"]
    summary = run(cfg, "ultra_short", out)
    s = summary.set_index("candidate")
    assert s.loc["p_phys", "r2_single"] > 0.9
    assert s.loc["ghi_ec", "r2_single"] > s.loc["t2m_ec", "r2_single"]
    assert s.loc["ghi_ec", "r2_single"] > s.loc["tcc_ec", "r2_single"]
    # base 自身的 ΔR² 为 NaN；其余候选有数
    assert np.isnan(s.loc["ghi_cons", "delta_r2_over_base"])
    assert np.isfinite(s.loc["t2m_ec", "delta_r2_over_base"])
    assert (tmp_path / "runs" / "screening.csv").exists()
    assert (tmp_path / "runs" / "screening_summary.csv").exists()
    by_station = pd.read_csv(tmp_path / "runs" / "screening.csv")
    assert set(by_station["station"]) == set(frames)
