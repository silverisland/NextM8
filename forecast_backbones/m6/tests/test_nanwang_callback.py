import functools
import json
import numpy as np
import pytest
import torch
from transformers import TrainerControl, TrainerState, TrainingArguments

pytest.importorskip("timesfm3")
from pipelines.transformer_dataset import UnifiedDataset
from forecast_backbones.m6 import nanwang_callback as nc
from forecast_backbones.m6.nanwang_callback import score_split, NanwangEvalCallback
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


def test_score_split_rejects_model_with_calibrator(tmp_path, tiny_evaluator):
    ds_cal, ds_val = _datasets()
    with pytest.warns(UserWarning):
        m = _model_b(tmp_path, tiny_evaluator)
    from forecast_backbones.m6.transformer_trainer import fit_calibrator_from_result, evaluate_model
    collate = functools.partial(UnifiedDataset.collate_fn, for_hf_trainer=True)
    m.set_calibrator(fit_calibrator_from_result(evaluate_model(m, ds_cal, collate, m.cfg), m.cfg))
    with pytest.raises(ValueError, match="calibrator"):
        score_split(m, ds_cal, ds_val, collate, m.cfg)


def _cb(tmp_path, m, patience, scores, monkeypatch, n_blocks=1):
    """scores：每次 _score 调用依次消费 n_blocks 个分数。"""
    it = iter(scores)
    monkeypatch.setattr(nc, "score_split", lambda *a, **k: (next(it), None, None))
    return NanwangEvalCallback(m, None, [None] * n_blocks, None, m.cfg, best_path=tmp_path / "best.pt", patience=patience,
                               log_path=tmp_path / "log.json")


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
