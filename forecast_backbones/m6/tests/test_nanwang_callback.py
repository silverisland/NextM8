import functools
import json
import numpy as np
import pandas as pd
import pytest
import torch
from transformers import TrainerControl, TrainerState, TrainingArguments

pytest.importorskip("timesfm3")
from pipelines.transformer_dataset import UnifiedDataset
from forecast_backbones.m6 import nanwang_callback as nc
from forecast_backbones.m6.nanwang_callback import score_split, NanwangEvalCallback
from forecast_backbones.m6.rows import scored_row_index
from forecast_backbones.m6.lora import LoRALinear
from .conftest import make_synthetic_frame, CAP
from .test_model_route_b import _model_b


def _datasets():
    df = make_synthetic_frame(n_days=3, issue_every_min=60, seed=0)
    day = df["timestamp_win"].dt.normalize()
    mk = lambda d: UnifiedDataset(d.reset_index(drop=True), timestamp_cols=["timestamp_win"], entity_cols=[],
                                  label_cols=["observe_power_future"], for_hf_trainer=True)
    return mk(df[day < day.max()]), mk(df[day == day.max()])


def test_score_split_returns_station_table_and_restores_train_mode(tmp_path, tiny_evaluator):
    ds_cal, ds_val = _datasets()
    with pytest.warns(UserWarning):
        m = _model_b(tmp_path, tiny_evaluator)
    collate = functools.partial(UnifiedDataset.collate_fn, for_hf_trainer=True)
    m.train()
    score, table, calib = score_split(m, ds_cal, ds_val, collate, m.cfg)
    assert m.training is True                                           # 恢复
    assert np.isfinite(score) and 0 <= score <= 100
    assert set(table.columns) >= {"station_id", "station", "accuracy_pct", "n_days"} and set(table["station_id"]) == set(CAP)
    assert calib is not None and calib.last_stats is not None
    assert np.isclose((table["accuracy_pct"] * table["n_days"]).sum() / table["n_days"].sum(), score)


def test_score_split_rejects_model_with_calibrator(tmp_path, tiny_evaluator):
    ds_cal, ds_val = _datasets()
    with pytest.warns(UserWarning):
        m = _model_b(tmp_path, tiny_evaluator)
    from forecast_backbones.m6.transformer_trainer import fit_calibrator_from_result, evaluate_model
    collate = functools.partial(UnifiedDataset.collate_fn, for_hf_trainer=True)
    m.set_calibrator(fit_calibrator_from_result(evaluate_model(m, ds_cal, collate, m.cfg), m.cfg))
    with pytest.raises(ValueError, match="calibrator"):
        score_split(m, ds_cal, ds_val, collate, m.cfg)


def _cb(tmp_path, m, patience, scores, monkeypatch, n_blocks=1, **kw):
    """scores：每次 _score 调用依次消费 n_blocks 个分数。"""
    it = iter(scores)
    monkeypatch.setattr(nc, "score_split", lambda *a, **k: (next(it), None, None))
    return NanwangEvalCallback(m, None, [None] * n_blocks, None, m.cfg, best_path=tmp_path / "best.pt", patience=patience,
                               log_path=tmp_path / "log.json", **kw)


def test_callback_keeps_best_and_stops_on_patience(tmp_path, tiny_evaluator, monkeypatch):
    with pytest.warns(UserWarning):
        m = _model_b(tmp_path, tiny_evaluator)
    cb = _cb(tmp_path, m, patience=2, scores=[50.0, 52.0, 51.0, 51.5], monkeypatch=monkeypatch)
    args = TrainingArguments(output_dir=str(tmp_path / "hf"), report_to=[]); state = TrainerState(); control = TrainerControl()
    cb.on_train_begin(args, state, control)                              # epoch 0 = 零样本基线 50.0
    assert cb.best == 50.0 and (tmp_path / "best.pt").exists()
    with torch.no_grad():
        for x in m.backbone.model.modules():
            if isinstance(x, LoRALinear):
                x.lora_B.fill_(0.3)
    state.epoch = 1.0; cb.on_epoch_end(args, state, control)             # 52 → 新最优，存盘
    saved = torch.load(tmp_path / "best.pt")
    assert cb.best == 52.0 and cb.best_epoch == 1.0 and all(float(v.abs().sum()) > 0 for k, v in saved.items() if k.endswith("lora_B"))
    with torch.no_grad():
        for x in m.backbone.model.modules():
            if isinstance(x, LoRALinear):
                x.lora_B.fill_(0.9)
    state.epoch = 2.0; cb.on_epoch_end(args, state, control)             # 51 → bad=1
    assert control.should_training_stop is False
    state.epoch = 3.0; cb.on_epoch_end(args, state, control)             # 51.5 → bad=2 → 停
    assert control.should_training_stop is True
    assert torch.allclose(torch.load(tmp_path / "best.pt")[next(k for k in saved if k.endswith("lora_B"))], torch.full_like(saved[next(k for k in saved if k.endswith("lora_B"))], 0.3))
    log = json.loads((tmp_path / "log.json").read_text())
    assert [h["epoch"] for h in log["history"]] == [0.0, 1.0, 2.0, 3.0] and log["best"] == 52.0 and log["best_epoch"] == 1.0


def test_callback_equal_score_does_not_replace_best(tmp_path, tiny_evaluator, monkeypatch):
    with pytest.warns(UserWarning):
        m = _model_b(tmp_path, tiny_evaluator)
    cb = _cb(tmp_path, m, patience=1, scores=[60.0, 60.0], monkeypatch=monkeypatch)
    args = TrainingArguments(output_dir=str(tmp_path / "hf"), report_to=[]); state = TrainerState(); control = TrainerControl()
    cb.on_train_begin(args, state, control); state.epoch = 1.0; cb.on_epoch_end(args, state, control)
    assert cb.best_epoch == 0.0 and control.should_training_stop is True


def test_score_is_mean_over_blocks_and_skips_non_finite(tmp_path, tiny_evaluator, monkeypatch):
    with pytest.warns(UserWarning):
        m = _model_b(tmp_path, tiny_evaluator)
    cb = _cb(tmp_path, m, patience=2, scores=[80.0, 90.0, float("nan"), 70.0, float("nan"), float("nan")],
             monkeypatch=monkeypatch, n_blocks=2)
    assert cb._score() == (85.0, [80.0, 90.0])
    mean, per = cb._score()
    assert mean == 70.0 and np.isnan(per[0]) and per[1] == 70.0
    mean, per = cb._score()
    assert np.isnan(mean) and len(per) == 2


def test_history_records_per_block_scores(tmp_path, tiny_evaluator, monkeypatch):
    with pytest.warns(UserWarning):
        m = _model_b(tmp_path, tiny_evaluator)
    cb = _cb(tmp_path, m, patience=2, scores=[80.0, 90.0, 82.0, 92.0], monkeypatch=monkeypatch, n_blocks=2)
    args = TrainingArguments(output_dir=str(tmp_path / "hf"), report_to=[]); state = TrainerState(); control = TrainerControl()
    cb.on_train_begin(args, state, control); state.epoch = 1.0; cb.on_epoch_end(args, state, control)
    assert cb.history[0]["blocks"] == [80.0, 90.0] and cb.history[0]["nanwang_after_pstar"] == 85.0
    assert cb.history[1]["blocks"] == [82.0, 92.0] and cb.best == 87.0 and cb.best_epoch == 1.0
    log = json.loads((tmp_path / "log.json").read_text())
    assert log["history"][1]["blocks"] == [82.0, 92.0]


def test_callback_rejects_empty_block_list(tmp_path, tiny_evaluator):
    with pytest.warns(UserWarning):
        m = _model_b(tmp_path, tiny_evaluator)
    with pytest.raises(ValueError, match="ds_val_blocks"):
        NanwangEvalCallback(m, None, [], None, m.cfg, best_path=tmp_path / "b.pt", patience=1, log_path=tmp_path / "l.json")


def test_progress_bar_shows_block_label_and_can_be_disabled(tmp_path, tiny_evaluator, capsys):
    ds_cal, ds_val = _datasets()
    with pytest.warns(UserWarning):
        m = _model_b(tmp_path, tiny_evaluator)
    collate = functools.partial(UnifiedDataset.collate_fn, for_hf_trainer=True)
    cb = NanwangEvalCallback(m, None, [ds_val], collate, m.cfg, best_path=tmp_path / "b.pt", patience=1, log_path=tmp_path / "l.json")
    cb._score()
    assert "验证块 1/1" in capsys.readouterr().err and m._progress_desc is None
    off = {**m.cfg, "finetune": {**m.cfg["finetune"], "schedule": {**m.cfg["finetune"]["schedule"], "progress_bar": False}}}
    m.cfg = off
    NanwangEvalCallback(m, None, [ds_val], collate, off, best_path=tmp_path / "b.pt", patience=1, log_path=tmp_path / "l.json")._score()
    assert "验证块" not in capsys.readouterr().err


def _two_val_days():
    """校准段 2 天 + 验证段 2 天（06-10 是 1970-01-01 后第 20249 天，奇数日；06-11 偶数日），2 站，每小时一个起报。"""
    df = make_synthetic_frame(n_days=4, issue_every_min=60, seed=0)
    day = df["timestamp_win"].dt.normalize()
    mk = lambda d: UnifiedDataset(d.reset_index(drop=True), timestamp_cols=["timestamp_win"], entity_cols=[],
                                  label_cols=["observe_power_future"], for_hf_trainer=True)
    return mk(df[day < pd.Timestamp("2025-06-10")]), mk(df[day >= pd.Timestamp("2025-06-10")])


def test_score_split_infers_only_rows_the_metric_uses_and_scores_identically(tmp_path, tiny_evaluator, monkeypatch):
    """南网短期 D+1 每站每天只用一行（当日 <= 10:00 的最晚一行）；只推理这些行，分数与推理全部行逐位相同。"""
    ds_cal, ds_val = _datasets()
    idx = scored_row_index(ds_val, "station_id", "timestamp_win")
    ts = np.asarray(ds_val.column("timestamp_win")).astype("datetime64[ns]")
    assert len(idx) == 2 and all(str(t)[11:16] == "10:00" for t in ts[idx])          # 2 站 × 1 天
    with pytest.warns(UserWarning):
        m = _model_b(tmp_path, tiny_evaluator)
    collate = functools.partial(UnifiedDataset.collate_fn, for_hf_trainer=True)
    seen = []
    real = nc.evaluate_model
    monkeypatch.setattr(nc, "evaluate_model", lambda model, ds, *a, **k: (seen.append(len(ds)), real(model, ds, *a, **k))[1])
    fast = score_split(m, None, ds_val, collate, m.cfg)
    full = score_split(m, None, ds_val, collate, m.cfg, scored_rows_only=False)
    assert seen == [2, len(ds_val)]
    assert fast[0] == full[0] and fast[1].equals(full[1])


def test_scored_row_index_falls_back_to_latest_row_before_issue_time():
    """当天没有 10:00 行时取更早的最近一行（与 short_d1_daily 一致）；10:00 之后的行永远不选。"""
    df = make_synthetic_frame(n_days=2, issue_every_min=60, seed=0)
    df = df[df["timestamp_win"].dt.hour != 10].reset_index(drop=True)
    ds = UnifiedDataset(df, timestamp_cols=["timestamp_win"], entity_cols=[], label_cols=["observe_power_future"],
                        for_hf_trainer=True)
    ts = pd.to_datetime(np.asarray(ds.column("timestamp_win")))[scored_row_index(ds, "station_id", "timestamp_win")]
    assert len(ts) == 4 and all(t.hour == 9 for t in ts)                               # 2 站 × 2 天


def test_score_split_with_no_rows_left_returns_nan(tmp_path, tiny_evaluator):
    df = make_synthetic_frame(n_days=1, issue_every_min=60, seed=0)
    ds_val = UnifiedDataset(df[df["timestamp_win"].dt.hour > 10].reset_index(drop=True), timestamp_cols=["timestamp_win"],
                            entity_cols=[], label_cols=["observe_power_future"], for_hf_trainer=True)   # 当天没有 10:00 前的行
    with pytest.warns(UserWarning):
        m = _model_b(tmp_path, tiny_evaluator)
    collate = functools.partial(UnifiedDataset.collate_fn, for_hf_trainer=True)
    score, table, calib = score_split(m, None, ds_val, collate, m.cfg)
    assert np.isnan(score) and table.empty and calib is None


def test_calibration_is_fitted_once_per_evaluation_and_shared_by_blocks(tmp_path, tiny_evaluator, monkeypatch):
    with pytest.warns(UserWarning):
        m = _model_b(tmp_path, tiny_evaluator)
    fits, seen = [], []
    monkeypatch.setattr(nc, "fit_split_calibrator", lambda *a, **k: fits.append(1) or "CALIB")
    monkeypatch.setattr(nc, "score_split", lambda *a, **k: (seen.append(k["calib"]), (70.0, None, None))[1])
    cb = NanwangEvalCallback(m, "DS_CAL", [None, None, None], None, m.cfg, best_path=tmp_path / "b.pt", patience=1,
                             log_path=tmp_path / "l.json")
    assert cb._score() == (70.0, [70.0, 70.0, 70.0])
    assert fits == [1] and seen == ["CALIB"] * 3


def test_step_mode_evaluates_every_n_steps_and_on_last_step(tmp_path, tiny_evaluator, monkeypatch):
    """eval_every_steps=2、max_steps=5：第 2、4、5 步打分；epoch 末不打；patience 按打分次数计。"""
    with pytest.warns(UserWarning):
        m = _model_b(tmp_path, tiny_evaluator)
    cb = _cb(tmp_path, m, patience=2, scores=[50.0, 55.0, 54.0, 53.0], monkeypatch=monkeypatch, eval_every_steps=2)
    args = TrainingArguments(output_dir=str(tmp_path / "hf"), report_to=[]); state = TrainerState(max_steps=5)
    control = TrainerControl()
    cb.on_train_begin(args, state, control)
    for step in range(1, 6):
        state.global_step, state.epoch = step, step / 5
        cb.on_step_end(args, state, control)
    cb.on_epoch_end(args, state, control)
    assert [h["step"] for h in cb.history] == [0, 2, 4, 5]
    assert cb.best == 55.0 and cb.best_step == 2 and cb.best_epoch == 0.4
    assert control.should_training_stop is True                                        # 第 4、5 步连续两次没超过 55
    log = json.loads((tmp_path / "log.json").read_text())
    assert log["best_step"] == 2 and log["eval_every_steps"] == 2 and log["history"][-1]["bad_evals"] == 2


def _nrmse_cfg(m, **es):
    import copy
    cfg = copy.deepcopy(m.cfg)
    cfg["finetune"]["early_stop"].update({"metric": "nrmse", **es})
    return cfg


def test_nrmse_metric_keeps_lowest_and_stops_on_patience(tmp_path, tiny_evaluator, monkeypatch):
    with pytest.warns(UserWarning):
        m = _model_b(tmp_path, tiny_evaluator)
    it = iter([10.0, 9.0, 9.5, 9.2])
    monkeypatch.setattr(nc, "nrmse_split", lambda *a, **k: next(it))
    monkeypatch.setattr(nc, "score_split", lambda *a, **k: pytest.fail("nrmse 模式不该调南网打分"))
    cb = NanwangEvalCallback(m, None, [None], None, _nrmse_cfg(m), best_path=tmp_path / "b.pt", patience=2,
                             log_path=tmp_path / "l.json")
    args = TrainingArguments(output_dir=str(tmp_path / "hf"), report_to=[]); state = TrainerState(); control = TrainerControl()
    cb.on_train_begin(args, state, control)
    assert cb.best == 10.0
    for e in (1.0, 2.0, 3.0):
        state.epoch = e; cb.on_epoch_end(args, state, control)
    assert cb.best == 9.0 and cb.best_epoch == 1.0 and control.should_training_stop is True
    log = json.loads((tmp_path / "l.json").read_text())
    assert log["metric"] == "nrmse" and log["higher_better"] is False and log["history"][1]["nrmse"] == 9.0


def test_unknown_early_stop_metric_rejected(tmp_path, tiny_evaluator):
    with pytest.warns(UserWarning):
        m = _model_b(tmp_path, tiny_evaluator)
    cfg = _nrmse_cfg(m); cfg["finetune"]["early_stop"]["metric"] = "mape"
    with pytest.raises(ValueError, match="early_stop.metric"):
        NanwangEvalCallback(m, None, [None], None, cfg, best_path=tmp_path / "b.pt", patience=1, log_path=tmp_path / "l.json")


def test_nrmse_split_uses_issue_grid_rows_and_matches_manual(tmp_path, tiny_evaluator, monkeypatch):
    from forecast_backbones.m6.transformer_trainer import recompute_points
    _, ds_val = _datasets()                                                             # 每小时一个起报
    with pytest.warns(UserWarning):
        m = _model_b(tmp_path, tiny_evaluator)
    collate = functools.partial(UnifiedDataset.collate_fn, for_hf_trainer=True)
    seen = []
    real = nc.evaluate_model
    monkeypatch.setattr(nc, "evaluate_model", lambda model, ds, *a, **k: (seen.append(len(ds)), real(model, ds, *a, **k))[1])
    m.train()
    v60 = nc.nrmse_split(m, ds_val, collate, m.cfg, issue_every_min=60)
    nc.nrmse_split(m, ds_val, collate, m.cfg, issue_every_min=120)
    assert m.training is True and seen == [len(ds_val), len(ds_val) // 2]
    df = recompute_points(real(m, ds_val, collate, m.cfg), m.cfg, None, "calibrated_pstar")
    y = np.stack(df["observe_power_future"].to_numpy()); p = np.stack(df["observe_power_predict"].to_numpy())
    r = (y - p) / df["cap"].to_numpy()[:, None]
    assert np.isfinite(v60) and v60 == pytest.approx(100 * np.sqrt(np.nanmean(r ** 2)))


def test_issue_grid_row_index_selects_on_minute_grid():
    from forecast_backbones.m6.rows import issue_grid_row_index
    df = make_synthetic_frame(n_days=1, issue_every_min=30, seed=0)
    ds = UnifiedDataset(df, timestamp_cols=["timestamp_win"], entity_cols=[], label_cols=["observe_power_future"],
                        for_hf_trainer=True)
    ts = pd.to_datetime(np.asarray(ds.column("timestamp_win")))
    assert (ts[issue_grid_row_index(ds, "timestamp_win", 60)].minute == 0).all()
    assert len(issue_grid_row_index(ds, "timestamp_win", 30)) == len(ds)
    assert len(issue_grid_row_index(ds, "timestamp_win", 60)) == len(ds) // 2
