import functools
import json
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


# ---------------------------------------------------------------- C2: 分块不改变数值

def test_recompute_points_chunking_matches_unchunked(tmp_path):
    """chunk_rows=7（真的会分好几块）与 chunk_rows=10**9（等价于不分块）在同一份数据上必须逐位相同。"""
    m, res = _run(tmp_path, n_days=3)
    assert len(res) > 7                      # 确保 chunk_rows=7 真的触发多块，不是退化成单块
    out_small = recompute_points(res, m.cfg, None, "physics_median", chunk_rows=7)
    out_big = recompute_points(res, m.cfg, None, "physics_median", chunk_rows=10**9)
    pts_small = np.stack(out_small["observe_power_predict"].to_numpy())
    pts_big = np.stack(out_big["observe_power_predict"].to_numpy())
    assert np.array_equal(pts_small, pts_big)


def test_recompute_points_chunked_calibration_stats_match_unchunked(tmp_path):
    """calibrated_pstar 分块跑完后 calibrator.last_stats 必须是全集口径，且与不分块单次 apply() 的统计一致。"""
    m, res = _run(tmp_path, n_days=3)
    assert len(res) > 5
    calib = fit_calibrator_from_result(res, m.cfg)
    recompute_points(res, m.cfg, calib, "calibrated_pstar", chunk_rows=10**9)
    whole = calib.last_stats
    recompute_points(res, m.cfg, calib, "calibrated_pstar", chunk_rows=5)
    chunked = calib.last_stats
    assert chunked["n"] == whole["n"] == len(res) * 192
    assert chunked["uncalibrated_cells"] == whole["uncalibrated_cells"]
    assert chunked["uncalibrated_frac"] == pytest.approx(whole["uncalibrated_frac"])
    assert set(chunked["per_tau_uncalibrated_frac"]) == set(whole["per_tau_uncalibrated_frac"])
    for k, v in whole["per_tau_uncalibrated_frac"].items():
        assert chunked["per_tau_uncalibrated_frac"][k] == pytest.approx(v)
    assert calib.last_levels is None            # 分块路径不跨块拼接逐格层号


# ---------------------------------------------------------------- I2: decision.point.tail 只认 flat

def test_recompute_points_raises_when_tail_not_flat(tmp_path):
    m, res = _run(tmp_path, n_days=1)
    cfg = {**m.cfg, "decision": {**m.cfg["decision"],
                                 "point": {**m.cfg["decision"]["point"], "tail": "extrapolate"}}}
    with pytest.raises(ValueError, match="tail"):
        recompute_points(res, cfg, None, "physics_median")


# ---------------------------------------------------------------- I1: 阶梯 ↔ 部署模型的口径不变量

def test_ladder_calibrated_pstar_matches_deployed_model_with_calibration(tmp_path):
    """不变量：阶梯的 calibrated_pstar 档与打开校准的部署模型 forward 必须算出同一个点——
    calibrate → enforce_monotone → physics_clip → select_point，同一套 levels/cap 口径。
    用小 chunk_rows（5）跑阶梯这一侧，同时覆盖 C2 分块路径不改变数值这条。"""
    from .test_model_route_a import _batch
    df = make_synthetic_frame(n_days=3, issue_every_min=15, seed=0)
    sm_path = str(write_station_meta(tmp_path))
    m_off = TransformersAdaptedM6Model({"timesfm3": {"eager_init": False}, "dataset": {"station_meta": sm_path},
                                        "calibration": {"enabled": False}, "eval": {"batch_size": 64}})
    m_off._backbone = FakeEvaluator(); m_off._backbone_device = "cpu"
    ds = UnifiedDataset(df, timestamp_cols=["timestamp_win"], entity_cols=[], label_cols=["observe_power_future"], for_hf_trainer=True)
    res = evaluate_model(m_off, ds, functools.partial(UnifiedDataset.collate_fn, for_hf_trainer=True), m_off.cfg)

    day = res["timestamp_win"].dt.normalize()
    cal_res, val_res = res[day < day.max()], res[day == day.max()]
    calib = fit_calibrator_from_result(cal_res, m_off.cfg)
    f = fit_shrink_from_result(cal_res, m_off.cfg)
    cal_path = tmp_path / "calibration.parquet"
    calib.save(cal_path)

    ladder_out = recompute_points(val_res, m_off.cfg, calib, "calibrated_pstar", f, chunk_rows=5)
    ladder_points = np.stack(ladder_out["observe_power_predict"].to_numpy())

    m_on = TransformersAdaptedM6Model({"timesfm3": {"eager_init": False}, "dataset": {"station_meta": sm_path},
                                       "calibration": {"enabled": True, "table_path": str(cal_path)},
                                       "decision": {"point": {"shrink_factor": f}}})
    m_on._backbone = FakeEvaluator(); m_on._backbone_device = "cpu"

    val_idx = np.flatnonzero((df["timestamp_win"].dt.normalize() == day.max()).to_numpy())
    bd, _ = _batch(df, val_idx)
    deployed_points = m_on(bd).logits.numpy()

    assert np.allclose(ladder_points, deployed_points, rtol=1e-5, atol=1e-4)


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


# ---------------------------------------------------------------- I4: diagnostics.json

def test_backfill_and_report_writes_diagnostics_json(tmp_path, monkeypatch):
    _patch_fake_backbone(monkeypatch)
    parquet_path, out_dir, cfg, cal_start, cal_end, val_start, val_end = _write_backfill_inputs(tmp_path)

    backfill_and_report(cfg, str(parquet_path), cal_start, cal_end, val_start, val_end)

    diag_path = out_dir / "reports" / "diagnostics.json"
    assert diag_path.exists()
    diag = json.loads(diag_path.read_text(encoding="utf-8"))
    assert set(diag) >= {"calibrator_last_stats", "degraded_frac", "n_rows_val", "n_rows_cal",
                         "var_bin_hist", "stations_scored", "shrink_factor"}
    assert sum(diag["var_bin_hist"].values()) == diag["n_rows_val"]
    assert diag["calibrator_last_stats"]["n"] == diag["n_rows_val"] * 192
    assert "per_tau_uncalibrated_frac" in diag["calibrator_last_stats"]


# ---------------------------------------------------------------- 路线 B 兼容：回填走 LoRA


def test_backfill_uses_lora_weights_when_finetune_enabled(tmp_path, monkeypatch):
    """回填 CLI 与路线 B 产物兼容：finetune.enabled=True + lora_path 存在 → 模型经 torch 路径加载 LoRA 后回填，
    且回填结果真的随 LoRA 权重变化（不是零样本回填碰巧也能算出同样形状的阶梯表）。

    证明手段三件套：
    (a) 训练后手动把 model.safetensors 里所有 lora_B 改成非零常数——不依赖训练本身是否真的带来提升，
        权重必然与零样本不同；
    (b) 监视 backfill_and_report 内部构造的模型（手法与 test_train.py::test_second_run_in_same_out_dir_starts_fresh
        一致），断言它确实 lora_loaded=True 且 torch_path=True；
    (c) 零样本基线回填（finetune.enabled=False）用独立 out_dir、且必须在训练/改权重之前用同一个
        evaluator 跑——LoRA 注入是对 evaluator.model 的原地 monkey-patch，之后再用同一个 evaluator
        构造"enabled=False"的模型不会真正回到零样本（已注入的 LoRALinear 层默认仍是 enabled=True），
        必须在任何注入发生前先取这份基线，才能保证它是干净的零样本对照，再与 LoRA 回填的阶梯表比较
        至少一个数值列不同。"""
    timesfm3 = pytest.importorskip("timesfm3")
    import torch
    import forecast_backbones.m6.transformer_models as tm_mod
    from .test_train import _patch_tiny_backbone, _inputs
    from forecast_backbones.m6.train import train_m6
    from forecast_backbones.m6.backbone import TorchBackbone
    d = tmp_path / "tiny"; TorchBackbone.tiny_for_tests().model.save_pretrained(str(d))
    from timesfm3 import ModelConfig, TimesFM3Evaluator
    ev = TimesFM3Evaluator(ModelConfig(checkpoint_path=str(d), per_core_batch_size=4, device="cpu"))
    _patch_tiny_backbone(monkeypatch, ev)
    df, parquet, out, cfg = _inputs(tmp_path)

    # (c) 零样本基线：finetune.enabled=False，独立 out_dir，必须在下面 train_m6 第一次给 ev.model 挂
    # LoRA 之前跑，否则这个 evaluator 就不再是干净的零样本底座了。
    cfg_off = {**cfg, "finetune": {**cfg["finetune"], "enabled": False}}
    out_zeroshot = tmp_path / "run_zeroshot"
    rep_off = backfill_and_report(cfg_off, str(parquet), "2025-06-11", "2025-06-12", "2025-06-12", "2025-06-14",
                                  out_dir=out_zeroshot)

    with pytest.warns(UserWarning, match="LoRA 权重|站级路由"):
        train_m6(cfg, str(parquet), max_steps=1)

    # (a) 训后把 lora_B 全部改成非零常数，保证 LoRA 输出与零样本必然不同
    from forecast_backbones.m6.lora import load_lora_weights, save_lora_weights
    lora_path = out / "model.safetensors"
    sd = load_lora_weights(lora_path)
    for k, v in sd.items():
        if k.endswith(".lora_B"):
            sd[k] = torch.full_like(v, 0.05)
    save_lora_weights(lora_path, sd)

    # (b) 监视 backfill_and_report 内部构造的模型
    RealModel = tm_mod.TransformersAdaptedM6Model
    recorded = []

    class _SpyModel(RealModel):
        def __init__(self, model_config=None, **kw):
            super().__init__(model_config, **kw)
            recorded.append({"lora_loaded": self.lora_loaded, "torch_path": self.torch_path})

    monkeypatch.setattr(tm_mod, "TransformersAdaptedM6Model", _SpyModel)

    rep_on = backfill_and_report(cfg, str(parquet), "2025-06-11", "2025-06-12", "2025-06-12", "2025-06-14")

    assert len(recorded) == 1, "backfill_and_report 应只构造一次模型"
    assert recorded[0]["lora_loaded"] is True and recorded[0]["torch_path"] is True
    assert list(rep_on["variant"]) == ["raw_median", "global_shrink", "physics_median", "calibrated_pstar"]
    assert (out / "reports" / "ladder.csv").exists()

    on_row = rep_on.set_index("variant").loc["raw_median"]
    off_row = rep_off.set_index("variant").loc["raw_median"]
    numeric_cols = ["nanwang_short_D+1", "nanwang_ultra_short", "nrmse_short_pct"]
    assert any(on_row[c] != pytest.approx(off_row[c]) for c in numeric_cols), (on_row, off_row)
