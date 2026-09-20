import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("timesfm3")
import forecast_backbones.m6.transformer_models as tm_mod
from forecast_backbones.m6.transformer_models import TransformersAdaptedM6Model
from forecast_backbones.m6.train import make_splits, build_training_args, train_m6
from forecast_backbones.m6.routing import load_routing
from pipelines.transformer_dataset import UnifiedDataset
from .conftest import make_synthetic_frame, write_station_meta, CAP


def _patch_tiny_backbone(monkeypatch, evaluator):
    def _ensure(self, want):
        self._backbone = evaluator; self._backbone_device = "cpu"
        if self.torch_path and self.backbone is None:
            self._attach_torch_path()
    monkeypatch.setattr(TransformersAdaptedM6Model, "_ensure_backbone", _ensure)


def _inputs(tmp_path, n_days=12):
    df = make_synthetic_frame(n_days=n_days, issue_every_min=60, seed=0, start="2025-06-08")   # 06-08 … 06-20
    parquet = tmp_path / "data.parquet"; df.to_parquet(parquet, index=False)
    out = tmp_path / "run"
    cfg = {"dataset": {"station_meta": str(write_station_meta(tmp_path))},
           "run": {"out_dir": str(out)}, "eval": {"batch_size": 64},
           "finetune": {"enabled": True, "inference_path": "torch",
                        "schedule": {"batch_size": 4, "epochs_max": 1, "lr": 1e-3, "precision": "fp32", "logging_steps": 1},
                        # train [06-08,06-10) | purge | val [06-12,06-14) | purge | cal [06-16,06-17) | purge | test [06-19,06-20)
                        "split": {"train_start": None, "val_blocks": [["2025-06-12", "2025-06-14"]],
                                  "test_start": "2025-06-19", "test_end": None,
                                  "cal_days": 1, "purge_days": 2, "embargo_days": 7},
                        "early_stop": {"patience": 1}}}
    return df, parquet, out, cfg


def test_make_splits_boundaries_and_shift(tmp_path):
    df, parquet, _, cfg = _inputs(tmp_path)
    ds = UnifiedDataset(str(parquet), timestamp_cols=["timestamp_win"], entity_cols=[], label_cols=["observe_power_future"],
                        timestamp_output="epoch_s", epoch_unit="s", for_hf_trainer=True)
    per_day = len(df) // 12
    tr, cal, val_all, blocks, plan = make_splits(ds, cfg["finetune"]["split"])
    assert len(tr) == 2 * per_day and len(cal) == per_day and len(val_all) == 2 * per_day
    assert len(blocks) == 1 and len(blocks[0]) == len(val_all)
    assert plan.to_dict()["cal"] == ["2025-06-16", "2025-06-17"]
    ts = lambda d: pd.to_datetime(d._ts_as_datetime64("timestamp_win"))
    assert ts(tr).max() < pd.Timestamp("2025-06-10") and ts(val_all).min() >= pd.Timestamp("2025-06-12")
    tr2, cal2, val2, blocks2, plan2 = make_splits(ds, cfg["finetune"]["split"], shift_days=1)      # 整体后移 1 天
    assert len(tr2) == 3 * per_day and len(cal2) == per_day and len(val2) == 2 * per_day          # train 起点不动，上限后移 1 天
    assert ts(tr2).min() >= pd.Timestamp("2025-06-08") and ts(tr2).max() < pd.Timestamp("2025-06-11")
    assert plan2.test_start == pd.Timestamp("2025-06-20")
    with pytest.raises(ValueError, match="段为空"):
        make_splits(ds, {**cfg["finetune"]["split"], "val_blocks": [["2025-07-01", "2025-07-03"]], "test_start": "2025-07-10"})  # 块与校准段都落在数据之外
    with pytest.raises(ValueError, match="cal_days"):
        make_splits(ds, {**cfg["finetune"]["split"], "cal_days": 0})
    with pytest.raises(ValueError, match="旧键"):
        make_splits(ds, {"train_end": "2025-06-11", "val_start": "2025-06-12", "val_end": "2025-06-14", "cal_days": 1, "test_start": "2025-06-14"})


def test_build_training_args_cpu_defaults(tmp_path):
    _, _, out, cfg = _inputs(tmp_path)
    from forecast_backbones.m6.config import load_config
    args = build_training_args(load_config(cfg), out, max_steps=3)
    assert args.max_steps == 3 and args.per_device_train_batch_size == 4 and args.remove_unused_columns is False
    assert args.evaluation_strategy.value == "no" and args.save_strategy.value == "no" and args.report_to == []
    assert args.bf16 is False and args.max_grad_norm == 1.0 and args.use_cpu is (not torch.cuda.is_available())


def test_train_m6_end_to_end_writes_all_artifacts(tmp_path, tiny_evaluator, monkeypatch):
    _patch_tiny_backbone(monkeypatch, tiny_evaluator)
    df, parquet, out, cfg = _inputs(tmp_path)
    with pytest.warns(UserWarning, match="LoRA 权重|站级路由"):                    # 训练前无权重：构造期 warn 一次
        res = train_m6(cfg, str(parquet), max_steps=2)
    # 产物
    assert (out / "lora_weights.pt").exists() and (out / "station_routing.json").exists()
    assert (out / "calibration.parquet").exists() and (out / "calibration.parquet.json").exists()
    assert (out / "train_log.json").exists() and (out / "result_val_lora.parquet").exists()
    ladder = pd.read_csv(out / "reports" / "ladder_lora.csv")
    assert list(ladder["variant"]) == ["zeroshot_calibrated_pstar", "lora_calibrated_pstar", "routed_calibrated_pstar"]
    assert np.isfinite(ladder["nanwang_short_D+1"]).all()
    st = pd.read_csv(out / "reports" / "station_ladder.csv")
    assert set(st["station_id"]) == set(CAP) and set(st["route"]) <= {"lora", "zeroshot"}
    routing = load_routing(out / "station_routing.json")
    assert routing == dict(zip(st["station_id"].astype(int), st["route"]))
    # 路由后的全体得分 ≥ 两者逐站取优的下界：不低于零样本行（逐站不输零样本）
    z = ladder.set_index("variant")["nanwang_short_D+1"]
    assert z["routed_calibrated_pstar"] >= z["zeroshot_calibrated_pstar"] - 1e-9
    log = json.loads((out / "train_log.json").read_text())
    assert log["history"][0]["epoch"] == 0.0 and len(log["history"]) >= 2 and "best" in log
    assert set(res) >= {"ladder", "routing", "history", "best"}
    assert res["ladder"]["variant"].tolist() == ladder["variant"].tolist() and res["routing"] == routing


def test_trained_artifacts_load_into_inference_model(tmp_path, tiny_evaluator, monkeypatch):
    _patch_tiny_backbone(monkeypatch, tiny_evaluator)
    df, parquet, out, cfg = _inputs(tmp_path)
    with pytest.warns(UserWarning, match="LoRA 权重|站级路由"):
        train_m6(cfg, str(parquet), max_steps=2)
    from forecast_backbones.m6.config import load_config
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                       # 权重、路由、校准表齐全 → 零 warning
        m = TransformersAdaptedM6Model(load_config(cfg))                     # calibration.enabled 默认 True → 读训练产物
    assert m.lora_loaded is True and m.calibrator is not None and m.station_routing is not None
    from .test_model_route_a import _batch
    bd, _ = _batch(df, [len(df) - 4, len(df) - 3, len(df) - 2, len(df) - 1])
    with torch.no_grad():
        o = m(bd)
    assert o.logits.shape == (4, 192) and torch.isfinite(o.logits).all()


def test_train_m6_raises_clear_error_when_no_finite_score_ever_recorded(tmp_path, tiny_evaluator, monkeypatch):
    """南网打分全程非有限（没有一天能凑出完整可评分的一天）→ NanwangEvalCallback 从未记录到
    best_epoch/存过权重；train_m6 必须在 trainer.train() 之后立刻拦下来，报出点名 val 窗口的
    ValueError，而不是让后面的 torch.load(best_path) 摔出 FileNotFoundError。"""
    _patch_tiny_backbone(monkeypatch, tiny_evaluator)
    df, parquet, out, cfg = _inputs(tmp_path)
    import forecast_backbones.m6.nanwang_callback as nc
    monkeypatch.setattr(nc, "score_split", lambda *a, **k: (float("nan"), None, None))
    with pytest.warns(UserWarning, match="LoRA 权重|站级路由"):
        with pytest.raises(ValueError) as exc:
            train_m6(cfg, str(parquet), max_steps=1)
    msg = str(exc.value)
    assert cfg["finetune"]["split"]["val_start"] in msg and cfg["finetune"]["split"]["val_end"] in msg
    assert "非有限" in msg
    assert not (out / "lora_weights.pt").exists()


def test_second_run_in_same_out_dir_starts_fresh(tmp_path, tiny_evaluator, monkeypatch):
    """跨任务约束 1 的回归测试：同一 out_dir 里已经有上一轮训练的 lora_weights.pt / station_routing.json
    时，第二轮训练的模型构造仍须是零样本、无路由起点——不能把上一轮的产物读回来当成本轮的起点。"""
    _patch_tiny_backbone(monkeypatch, tiny_evaluator)
    df, parquet, out, cfg = _inputs(tmp_path)
    with pytest.warns(UserWarning, match="LoRA 权重|站级路由"):
        train_m6(cfg, str(parquet), max_steps=1)
    assert (out / "lora_weights.pt").exists() and (out / "station_routing.json").exists()

    # 第二轮换一份全新底座（同一份 tiny 权重目录重新构造）：避免两次构造共享同一个 evaluator.model 对象，
    # 让第一轮训练留在内存里的 LoRA 权重把"零样本起点"判定悄悄冲掉——真实场景里每轮训练都会从磁盘
    # checkpoint 重新加载一份全新底座，这里也应如此，spy 才测的是 train.py 自己的哨兵路径逻辑。
    from timesfm3 import ModelConfig, TimesFM3Evaluator
    evaluator2 = TimesFM3Evaluator(ModelConfig(checkpoint_path=tiny_evaluator.config.checkpoint_path,
                                               per_core_batch_size=4, device="cpu"))
    _patch_tiny_backbone(monkeypatch, evaluator2)

    RealModel = tm_mod.TransformersAdaptedM6Model
    recorded = []

    class _SpyModel(RealModel):
        def __init__(self, model_config=None, **kw):
            super().__init__(model_config, **kw)
            recorded.append({"lora_path": self.cfg["finetune"]["lora_path"],
                             "routing_path": self.cfg["finetune"]["routing_path"],
                             "lora_loaded": self.lora_loaded, "station_routing": self.station_routing})

    monkeypatch.setattr(tm_mod, "TransformersAdaptedM6Model", _SpyModel)
    with pytest.warns(UserWarning, match="LoRA 权重|站级路由"):
        train_m6(cfg, str(parquet), max_steps=1)

    assert len(recorded) == 1, "第二轮训练模型只应构造一次"
    rec = recorded[0]
    assert not Path(rec["lora_path"]).exists() and not Path(rec["routing_path"]).exists()
    assert rec["lora_loaded"] is False and rec["station_routing"] is None
    log2 = json.loads((out / "train_log.json").read_text())
    assert log2["history"][0]["epoch"] == 0.0
