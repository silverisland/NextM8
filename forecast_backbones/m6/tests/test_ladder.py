import functools
import numpy as np
import pandas as pd
from pipelines.transformer_dataset import UnifiedDataset
from forecast_backbones.m6.transformer_models import TransformersAdaptedM6Model
from forecast_backbones.m6.transformer_trainer import (evaluate_model, fit_calibrator_from_result,
                                                       fit_shrink_from_result, recompute_points, ladder_report,
                                                       backfill_and_report, LADDER)
from forecast_backbones.m6.calibration import result_to_long, quantile_cols
from forecast_backbones.m6.decision import MODEL_LEVELS
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


def test_backfill_and_report_splits_state_and_report_paths(tmp_path, monkeypatch):
    """R1: ladder.csv (REPORT) 落 cfg['eval']['report_dir']；calibration.parquet(+.json sidecar)/
    decision_state.json/result_cal.parquet/result_val.parquet (STATE) 落 run.out_dir；calibration
    存盘路径须与 cfg['calibration']['table_path'] 一致（模型 calibration.enabled=True 时按此路径加载）。"""
    def _fake_ensure_backbone(self, want):
        self._backbone = FakeEvaluator(); self._backbone_device = want
    monkeypatch.setattr(TransformersAdaptedM6Model, "_ensure_backbone", _fake_ensure_backbone)

    df = make_synthetic_frame(n_days=4, issue_every_min=15, seed=0)
    parquet_path = tmp_path / "data.parquet"
    df.to_parquet(parquet_path, index=False)

    out_dir = tmp_path / "run"
    cfg = {"dataset": {"station_meta": str(write_station_meta(tmp_path))},
          "calibration": {"enabled": False}, "eval": {"batch_size": 64},
          "run": {"out_dir": str(out_dir)}}

    days = sorted(df["timestamp_win"].dt.normalize().unique())
    cal_start, cal_end = str(pd.Timestamp(days[0]).date()), str(pd.Timestamp(days[-1]).date())
    val_start = str(pd.Timestamp(days[-1]).date())
    val_end = str((pd.Timestamp(days[-1]) + pd.Timedelta(days=1)).date())

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
