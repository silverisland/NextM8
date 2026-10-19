"""中间 checkpoint（train.schedule.save_every / --save-every，2026-10-19）：全量训练没有验证集、不能早停，
每 N 步存一份自包含 checkpoint，训完用线上推理脚本逐个评测、挑最优步（事后早停）。"""
import json

import pytest
import torch
import yaml
from safetensors.torch import load_file

from .conftest import make_frame, write_station_meta

CFG_MODEL = {"d_model": 32, "d_ff": 64, "head_hidden": 32, "n_heads": 2, "blocks": ["conv"], "pos_in_patch_dim": 8}


def _train(tmp_path, ema=None, save_every=2, max_steps=5):
    from forecast_backbones.m8.train import train_m8
    meta = write_station_meta(tmp_path)
    d = tmp_path / "train" / "date=2025-06-08"; d.mkdir(parents=True)
    make_frame(n_days=2, issue_every_min=120).to_parquet(d / "data_0.parquet", index=False)
    cfg = {"dataset": {"station_meta": str(meta)}, "model": CFG_MODEL,
           "train": {"schedule": {"batch_size": 8, "warmup_steps": 1, "logging_steps": 1, "progress_bar": False}, "ema_decay": ema}}
    out = tmp_path / "run"
    train_m8(cfg, out_dir=out, max_steps=max_steps, train_parquet=[str(tmp_path / "train")], save_every=save_every)
    return out


@pytest.mark.parametrize("ema", [None, 0.5])
def test_full_training_saves_loadable_checkpoints(tmp_path, ema):
    from forecast_backbones.m8.transformer_models import TransformersAdaptedM8Model
    out = _train(tmp_path, ema=ema)
    steps = sorted(p.name for p in (out / "checkpoints").iterdir())
    assert steps == ["step_00002", "step_00004", "step_00005"]                     # 每 2 步 + 最后一步
    for s in steps:
        d = out / "checkpoints" / s
        cfg = yaml.safe_load((d / "model_config.yaml").read_text(encoding="utf-8"))
        assert cfg["dataset"]["known_stations"] == [1001, 1002] and cfg["features"]["cov_present"]
        m = TransformersAdaptedM8Model(cfg)                                             # 原地加载：weights_path 指向自己目录
        assert m.weights_loaded
    final, last = load_file(str(out / "model.safetensors")), load_file(str(out / "checkpoints/step_00005/model.safetensors"))
    assert all(torch.equal(final[k], last[k]) for k in final)                           # 最后一步 = 最终权重（开 EMA 时都是 EMA 权重）
    first = load_file(str(out / "checkpoints/step_00002/model.safetensors"))
    assert any(not torch.equal(first[k], last[k]) for k in first)
    log = json.loads((out / "train_log.json").read_text(encoding="utf-8"))
    assert log["checkpoints"] == [f"checkpoints/{s}" for s in steps]


def test_no_checkpoints_by_default(tmp_path):
    out = _train(tmp_path, save_every=None, max_steps=2)
    assert not (out / "checkpoints").exists() and (out / "model.safetensors").exists()
