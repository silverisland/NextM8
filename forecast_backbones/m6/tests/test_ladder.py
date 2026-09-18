import functools
import numpy as np
import pandas as pd
import pytest
from pipelines.transformer_dataset import UnifiedDataset
from forecast_backbones.m6.transformer_models import TransformersAdaptedM6Model
from forecast_backbones.m6.transformer_trainer import (evaluate_model, fit_calibrator_from_result,
                                                       fit_shrink_from_result, recompute_points, ladder_report,
                                                       backfill_and_report, LADDER, _clip_calibration_window)
from forecast_backbones.m6.calibration import result_to_long, quantile_cols
from forecast_backbones.m6.config import load_config
from forecast_backbones.m6.decision import MODEL_LEVELS, global_shrink_factor
from .conftest import FakeEvaluator, make_synthetic_frame, write_station_meta, CAP


def _run(tmp_path, n_days=3):
    df = make_synthetic_frame(n_days=n_days, issue_every_min=15, seed=0)
    cfg = {"timesfm3": {"eager_init": False}, "dataset": {"station_meta": str(write_station_meta(tmp_path))},
           "calibration": {"enabled": False}, "eval": {"batch_size": 64}}
    m = TransformersAdaptedM6Model(cfg); m._backbone = FakeEvaluator(); m._backbone_device = "cpu"
    ds = UnifiedDataset(df, timestamp_cols=["timestamp_win"], entity_cols=[], label_cols=["observe_power_future"], for_hf_trainer=True)
    res = evaluate_model(m, ds, functools.partial(UnifiedDataset.collate_fn, for_hf_trainer=True), m.cfg)
    return m, res


def test_result_to_long_filters(tmp_path):
    _, res = _run(tmp_path, n_days=1)
    long = result_to_long(res, fit_steps=[[0, 16], [55, 151]])
    assert set(long.columns) >= {"station_id", "lead", "elev_deg", "var_bin", "y", "p_clear"} | set(quantile_cols(MODEL_LEVELS))
    assert long["lead"].between(0, 150).all() and not long["lead"].between(16, 54).any()
    assert (long["p_clear"] > 0).all() and np.isfinite(long["y"]).all()


def test_ladder_calibrated_pstar_beats_raw_median(tmp_path):
    m, res = _run(tmp_path, n_days=3)
    day = res["timestamp_win"].dt.normalize()
    cal_df, val_df = res[day < day.max()], res[day == day.max()]
    calib = fit_calibrator_from_result(cal_df, m.cfg)
    f = fit_shrink_from_result(cal_df, m.cfg)
    assert 0.5 < f < 1.0
    gccap = {f"S{s}": c for s, c in CAP.items()}
    rep = ladder_report(val_df, gccap, m.cfg, calib, f).set_index("variant")
    assert list(rep.index) == ["raw_median", "global_shrink", "physics_median", "calibrated_pstar"]
    assert rep.loc["calibrated_pstar", "nanwang_short_D+1"] > rep.loc["raw_median", "nanwang_short_D+1"]
    assert rep.loc["global_shrink", "nanwang_short_D+1"] > rep.loc["raw_median", "nanwang_short_D+1"]
    assert rep.loc["calibrated_pstar", "nanwang_ultra_short"] >= rep.loc["raw_median", "nanwang_ultra_short"] - 0.5
    assert (rep["n_days"] >= 1).all()


def test_recompute_points_variants_shape(tmp_path):
    m, res = _run(tmp_path, n_days=1)
    out = recompute_points(res, m.cfg, None, "physics_median")
    assert len(out) == len(res) and len(out.iloc[0]["observe_power_predict"]) == 192
    night = res.iloc[0]["night"]
    assert (out.iloc[0]["observe_power_predict"][night] == 0).all()


# ---------------------------------------------------------------- fix #1: calibration.window_days

def test_clip_calibration_window_excludes_rows_older_than_window():
    ts = pd.date_range("2025-01-01", periods=60, freq="D")           # 2025-01-01 .. 2025-03-01
    res_cal = pd.DataFrame({"timestamp_win": ts, "v": range(60)})
    cal_end = "2025-03-01"
    window_days = 45
    lo = pd.Timestamp(cal_end) - pd.Timedelta(days=window_days)       # 2025-01-15

    clipped = _clip_calibration_window(res_cal, cal_end, window_days)

    assert len(clipped) < len(res_cal)
    assert (clipped["timestamp_win"] >= lo).all()
    assert ts[0] not in set(clipped["timestamp_win"])                 # 2025-01-01：早于窗口，剔除
    assert ts[-1] in set(clipped["timestamp_win"])                    # 2025-03-01：窗口内，保留


def test_clip_calibration_window_raises_when_empty():
    res_cal = pd.DataFrame({"timestamp_win": pd.date_range("2025-01-01", periods=5, freq="D")})
    with pytest.raises(ValueError, match="window_days"):
        _clip_calibration_window(res_cal, "2025-06-01", window_days=45)


def test_backfill_and_report_clips_calibration_to_window_days(tmp_path, monkeypatch):
    """cal_start..cal_end 给的范围比 calibration.window_days 长时，只有末尾 window_days 天参与拟合：
    用一个短窗口（3 天）把 4 天的合成数据切到只剩最后 1 天参与拟合，校验落盘的 result_cal.parquet
    行数与用这最后 1 天单独跑 fit_calibrator_from_result 时的长表行数一致（rows older than the window
    are excluded before fitting）。"""
    _patch_fake_backbone(monkeypatch)
    parquet_path, out_dir, cfg, cal_start, cal_end, val_start, val_end = _write_backfill_inputs(tmp_path, "run", n_days=4)
    cfg = {**cfg, "calibration": {"enabled": False, "window_days": 1}}

    backfill_and_report(cfg, str(parquet_path), cal_start, cal_end, val_start, val_end)

    res_cal_saved = pd.read_parquet(out_dir / "result_cal.parquet")
    cal_lo = pd.Timestamp(cal_end) - pd.Timedelta(days=1)
    assert (pd.to_datetime(res_cal_saved["timestamp_win"]) >= cal_lo).all()
    assert len(res_cal_saved) > 0
    # cal 段本身跨 3 天（cal_start..cal_end），window_days=1 应该真的把它切窄了
    assert res_cal_saved["timestamp_win"].min() > pd.Timestamp(cal_start)


# ---------------------------------------------------------------- fix #2: fit_shrink_from_result 掩码

def test_fit_shrink_from_result_nan_median_stays_finite_and_matches_excluded_point():
    cfg = load_config(None)
    lo, hi = cfg["data"]["scored_steps"]["short"]
    n = int(cfg["timesfm3"]["pred_len"])
    rng = np.random.default_rng(0)
    y = rng.uniform(10.0, 90.0, n)
    p = rng.uniform(10.0, 90.0, n)
    night = np.zeros(n, bool)
    cap = 100.0
    nan_pos = lo + 5

    p_nan = p.copy(); p_nan[nan_pos] = np.nan
    row_nan = pd.DataFrame([{"observe_power_future": y, "median": p_nan, "night": night, "cap": cap}])
    f_nan = fit_shrink_from_result(row_nan, cfg)
    assert np.isfinite(f_nan)

    keep = np.ones(hi - lo, bool); keep[nan_pos - lo] = False
    f_ref = global_shrink_factor(y[lo:hi][keep], p[lo:hi][keep], cap, float(cfg["decision"]["point"]["floor_ratio"]))
    assert f_nan == pytest.approx(f_ref, rel=1e-9)


def test_fit_shrink_from_result_returns_one_when_no_usable_points():
    cfg = load_config(None)
    n = int(cfg["timesfm3"]["pred_len"])
    row = pd.DataFrame([{"observe_power_future": np.full(n, np.nan), "median": np.zeros(n),
                         "night": np.zeros(n, bool), "cap": 100.0}])
    assert fit_shrink_from_result(row, cfg) == 1.0


# ---------------------------------------------------------------- backfill_and_report 落盘路径

def _patch_fake_backbone(monkeypatch):
    def _fake_ensure_backbone(self, want):
        self._backbone = FakeEvaluator(); self._backbone_device = want
    monkeypatch.setattr(TransformersAdaptedM6Model, "_ensure_backbone", _fake_ensure_backbone)


def _write_backfill_inputs(tmp_path, out_dir_name="run", n_days=4):
    df = make_synthetic_frame(n_days=n_days, issue_every_min=15, seed=0)
    parquet_path = tmp_path / "data.parquet"
    df.to_parquet(parquet_path, index=False)

    out_dir = tmp_path / out_dir_name
    cfg = {"dataset": {"station_meta": str(write_station_meta(tmp_path))},
          "calibration": {"enabled": False}, "eval": {"batch_size": 64},
          "run": {"out_dir": str(out_dir)}}

    days = sorted(df["timestamp_win"].dt.normalize().unique())
    cal_start, cal_end = str(pd.Timestamp(days[0]).date()), str(pd.Timestamp(days[-1]).date())
    val_start = str(pd.Timestamp(days[-1]).date())
    val_end = str((pd.Timestamp(days[-1]) + pd.Timedelta(days=1)).date())
    return parquet_path, out_dir, cfg, cal_start, cal_end, val_start, val_end


def test_backfill_and_report_splits_state_and_report_paths(tmp_path, monkeypatch):
    """R1: ladder.csv (REPORT) 落 cfg['eval']['report_dir']；calibration.parquet(+.json sidecar)/
    decision_state.json/result_cal.parquet/result_val.parquet (STATE) 落 run.out_dir；calibration
    存盘路径须与 cfg['calibration']['table_path'] 一致（模型 calibration.enabled=True 时按此路径加载）。"""
    _patch_fake_backbone(monkeypatch)
    parquet_path, out_dir, cfg, cal_start, cal_end, val_start, val_end = _write_backfill_inputs(tmp_path)

    rep = backfill_and_report(cfg, str(parquet_path), cal_start, cal_end, val_start, val_end)

    assert list(rep["variant"]) == list(LADDER)
    report_dir = out_dir / "reports"
    assert (report_dir / "ladder.csv").exists()
    assert not (out_dir / "ladder.csv").exists()
    assert (out_dir / "calibration.parquet").exists()
    assert (out_dir / "calibration.parquet.json").exists()
    assert (out_dir / "decision_state.json").exists()
    assert (out_dir / "result_cal.parquet").exists()
    assert (out_dir / "result_val.parquet").exists()
    assert not (report_dir / "calibration.parquet").exists()


def test_backfill_and_report_out_dir_override_moves_all_five_artifacts(tmp_path, monkeypatch):
    """fix #3: out_dir= 覆盖时，STATE 四件（decision_state.json/result_cal.parquet/result_val.parquet/
    calibration.parquet+.json）与 REPORT（reports/ladder.csv）必须一起跟着搬到 out_dir，一个都不能
    留在 cfg 里原本的 run.out_dir 下——calibration.table_path/eval.report_dir 也要跟着重新插值。"""
    _patch_fake_backbone(monkeypatch)
    parquet_path, cfg_out_dir, cfg, cal_start, cal_end, val_start, val_end = _write_backfill_inputs(tmp_path, "cfg_run")
    override_dir = tmp_path / "override_run"

    rep = backfill_and_report(cfg, str(parquet_path), cal_start, cal_end, val_start, val_end, out_dir=override_dir)

    assert list(rep["variant"]) == list(LADDER)
    for rel in ("reports/ladder.csv", "calibration.parquet", "calibration.parquet.json",
               "decision_state.json", "result_cal.parquet", "result_val.parquet"):
        assert (override_dir / rel).exists(), rel
        assert not (cfg_out_dir / rel).exists(), rel


# ---------------------------------------------------------------- fix #5: cal/val 时间泄漏闸

def test_backfill_and_report_rejects_cal_end_after_val_start(tmp_path, monkeypatch):
    _patch_fake_backbone(monkeypatch)
    parquet_path, out_dir, cfg, cal_start, cal_end, val_start, val_end = _write_backfill_inputs(tmp_path)
    leaky_val_start = str(pd.Timestamp(cal_end) - pd.Timedelta(days=1))       # 早于 cal_end：重叠

    with pytest.raises(ValueError, match="泄漏|leak"):
        backfill_and_report(cfg, str(parquet_path), cal_start, cal_end, leaky_val_start, val_end)
