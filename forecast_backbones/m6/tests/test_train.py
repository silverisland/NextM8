import json
import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("timesfm3")
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


def _inputs(tmp_path, n_days=6):
    df = make_synthetic_frame(n_days=n_days, issue_every_min=60, seed=0, start="2025-06-08")
    parquet = tmp_path / "data.parquet"; df.to_parquet(parquet, index=False)
    out = tmp_path / "run"
    cfg = {"dataset": {"station_meta": str(write_station_meta(tmp_path))},
           "run": {"out_dir": str(out)}, "eval": {"batch_size": 64},
           "finetune": {"enabled": True, "inference_path": "torch",
                        "schedule": {"batch_size": 4, "epochs_max": 1, "lr": 1e-3, "precision": "fp32", "logging_steps": 1},
                        "split": {"train_start": None, "train_end": "2025-06-11", "cal_days": 1,
                                  "val_start": "2025-06-12", "val_end": "2025-06-14", "test_start": "2025-06-14"},
                        "early_stop": {"patience": 1}}}
    return df, parquet, out, cfg


def test_make_splits_boundaries_and_shift(tmp_path):
    df, parquet, _, cfg = _inputs(tmp_path)
    ds = UnifiedDataset(str(parquet), timestamp_cols=["timestamp_win"], entity_cols=[], label_cols=["observe_power_future"],
                        timestamp_output="epoch_s", epoch_unit="s", for_hf_trainer=True)
    tr, cal, val = make_splits(ds, cfg["finetune"]["split"])
    per_day = len(df) // 6
    assert len(tr) == 3 * per_day and len(cal) == per_day and len(val) == 2 * per_day
    tr2, cal2, val2 = make_splits(ds, cfg["finetune"]["split"], shift_days=1)
    assert len(tr2) == 4 * per_day and len(cal2) == per_day and len(val2) == per_day       # 整体后移 1 天，val 被数据末端截短
    with pytest.raises(ValueError, match="val_start"):
        make_splits(ds, {**cfg["finetune"]["split"], "cal_days": 2})                        # cal 撞进 val


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
    with pytest.warns(UserWarning, match="LoRA 权重"):                    # 训练前无权重：构造期 warn 一次
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
    with pytest.warns(UserWarning, match="LoRA 权重"):
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
