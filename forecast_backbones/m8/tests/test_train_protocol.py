"""训练协议开关（2026-10-13）：权重 EMA（train.ema_decay）与按最近 k 次验证均值选点（train.early_stop.smooth_k）。"""
import json
import numpy as np
import pandas as pd
import pytest
import torch
from .conftest import write_splits, write_station_meta


def _tiny_cfg(meta, blocks=("conv",), **train_over):
    tr = {"schedule": {"batch_size": 8, "max_steps": 4, "eval_every_steps": 2, "warmup_steps": 1, "logging_steps": 1, "progress_bar": False},
          "early_stop": {"patience": 3}}
    for k, v in train_over.items():
        tr[k] = {**tr[k], **v} if isinstance(v, dict) and isinstance(tr.get(k), dict) else v
    return {"dataset": {"station_meta": str(meta)},
            "model": {"d_model": 32, "d_ff": 64, "head_hidden": 32, "n_heads": 2, "blocks": list(blocks), "pos_in_patch_dim": 8},
            "train": tr, "eval": {"batch_size": 16}}


def test_weight_ema_recursion_and_swap():
    """shadow ← d_n·shadow + (1−d_n)·θ，d_n = min(decay, (1+n)/(10+n))（前几步快速跟上）；applied() 期间模型权重 = shadow，退出后复原；
    整型缓冲不做平均、直接拷贝。"""
    from forecast_backbones.m8.train import WeightEma
    torch.manual_seed(0)
    m = torch.nn.Linear(3, 2)
    m.register_buffer("count", torch.tensor(0))
    w0 = m.weight.detach().clone()
    ema = WeightEma(m, decay=0.9)
    assert torch.equal(ema.shadow["weight"], w0)
    with torch.no_grad():
        m.weight.fill_(1.0); m.count.fill_(5)
    ema.update(m)                                                       # n=1: d = min(0.9, 2/11)
    d1 = min(0.9, 2 / 11)
    assert torch.allclose(ema.shadow["weight"], d1 * w0 + (1 - d1) * torch.ones_like(w0))
    assert int(ema.shadow["count"]) == 5
    with torch.no_grad():
        m.weight.fill_(2.0)
    ema.update(m)                                                       # n=2: d = min(0.9, 3/12)
    d2 = min(0.9, 3 / 12)
    expect = d2 * (d1 * w0 + (1 - d1) * torch.ones_like(w0)) + (1 - d2) * 2 * torch.ones_like(w0)
    assert torch.allclose(ema.shadow["weight"], expect)
    live = m.weight.detach().clone()
    with ema.applied(m):
        assert torch.allclose(m.weight, expect)
    assert torch.equal(m.weight, live)                                  # 复原
    # 步数够多后 d_n 收敛到 decay
    for _ in range(200):
        ema.update(m)
    assert ema.last_decay == pytest.approx(0.9)


def test_train_m8_ema_weights_are_what_gets_scored_and_saved(tmp_path):
    """ema_decay 开：验证与最优权重都用 EMA 权重 → 和不开 EMA 的同种子跑结果不同；evaluate.py 重算存下的权重分数与 summary 一致
    （证明存的就是被打分的 EMA 权重）；train_log.json / summary.csv 记录 ema_decay。"""
    from forecast_backbones.m8.train import train_m8
    from forecast_backbones.m8.evaluate import evaluate
    meta = write_station_meta(tmp_path)
    splits = write_splits(tmp_path, n_train_days=3, n_val_days=2)
    plain = train_m8(_tiny_cfg(meta), splits_dir=splits, out_dir=tmp_path / "plain", tag="plain")
    ema = train_m8(_tiny_cfg(meta, ema_decay=0.5), splits_dir=splits, out_dir=tmp_path / "ema", tag="ema")
    s_plain, s_ema = plain["summary"].iloc[0], ema["summary"].iloc[0]
    assert np.isfinite(s_ema["nanwang_short_D+1"]) and s_ema["ema_decay"] == 0.5 and np.isnan(s_plain["ema_decay"])
    assert s_ema["nanwang_short_D+1"] != s_plain["nanwang_short_D+1"]
    from safetensors.torch import load_file
    w_plain, w_ema = load_file(str(tmp_path / "plain" / "model.safetensors")), load_file(str(tmp_path / "ema" / "model.safetensors"))
    assert any(not torch.equal(w_plain[k], w_ema[k]) for k in w_plain)
    e = evaluate([tmp_path / "ema"], [str(splits / "valid.parquet")], out_dir=tmp_path / "ev", progress=False)
    assert e["summary"]["nanwang_short_D+1"] == pytest.approx(float(s_ema["nanwang_short_D+1"]), abs=1e-6)
    log = json.loads((tmp_path / "ema" / "train_log.json").read_text(encoding="utf-8"))
    assert log["ema_decay"] == 0.5


def test_train_m8_refit_saves_ema_weights(tmp_path):
    """refit（无验证）+ EMA：最终存的是 EMA 权重，与同种子不开 EMA 的 refit 不同。"""
    from forecast_backbones.m8.train import train_m8
    from safetensors.torch import load_file
    meta = write_station_meta(tmp_path)
    splits = write_splits(tmp_path, n_train_days=3, n_val_days=1)
    train_m8(_tiny_cfg(meta), splits_dir=splits, out_dir=tmp_path / "r0", tag="r0", refit=True)
    train_m8(_tiny_cfg(meta, ema_decay=0.5), splits_dir=splits, out_dir=tmp_path / "r1", tag="r1", refit=True)
    a, b = load_file(str(tmp_path / "r0" / "model.safetensors")), load_file(str(tmp_path / "r1" / "model.safetensors"))
    assert any(not torch.equal(a[k], b[k]) for k in a)
    assert json.loads((tmp_path / "r1" / "train_log.json").read_text(encoding="utf-8"))["ema_decay"] == 0.5


def _callback(small_cfg, tmp_path, patience=3, **early_stop):
    from forecast_backbones.m8.config import load_config
    from forecast_backbones.m8.train import EvalCallback
    from forecast_backbones.m8.transformer_models import TransformersAdaptedM8Model
    cfg = load_config({**small_cfg, "train": {"early_stop": {"patience": patience, **early_stop}}})
    model = TransformersAdaptedM8Model(cfg)
    return EvalCallback(model, None, None, cfg, best_path=tmp_path / "best.safetensors", patience=patience, log_path=tmp_path / "log.json")


def test_smooth_k_selects_checkpoint_on_mean_of_last_k_scores(small_cfg, tmp_path):
    """smooth_k=3：best / 早停都按最近 3 次验证分的均值；history 同时记原始分与平滑分。
    原始分 [70, 78, 70, 74, 75, 76]：不平滑选第 2 次（78）；平滑后均值 [70, 74, 72.67, 74, 73, 75] → 选第 6 次。"""
    scores = [70.0, 78.0, 70.0, 74.0, 75.0, 76.0]
    cb1 = _callback(small_cfg, tmp_path / "k1", patience=5)
    for i, s in enumerate(scores, 1):
        cb1._record(i, i / 10, s, None)
    assert cb1.best_step == 2 and cb1.best == 78.0
    cb3 = _callback(small_cfg, tmp_path / "k3", patience=5, smooth_k=3)
    stops = [cb3._record(i, i / 10, s, None) for i, s in enumerate(scores, 1)]
    assert cb3.best_step == 6 and cb3.best == pytest.approx(75.0)
    assert not any(stops)                                                # patience 5：第 6 次刷新前 bad_evals 最多 3
    h = cb3.history
    assert [x["nanwang_after_pstar"] for x in h] == scores
    assert [round(x["nanwang_after_pstar_smooth"], 2) for x in h] == [70.0, 74.0, 72.67, 74.0, 73.0, 75.0]
    assert [x["bad_evals"] for x in h] == [0, 0, 1, 2, 3, 0]
    log = json.loads((tmp_path / "k3" / "log.json").read_text(encoding="utf-8"))
    assert log["best_step"] == 6 and log["smooth_k"] == 3 and (tmp_path / "k3" / "best.safetensors").exists()


def test_smooth_k_early_stop_counts_smoothed_scores(small_cfg, tmp_path):
    """早停按平滑分计：原始分 [60, 80, 70, 70, 70] 不平滑时第 5 次停（80 之后连续 3 次不刷新，best_step 2）；
    smooth_k=3 的平滑分 [60, 70, 70, 73.3, 70] 第 4 次刷新 best → 不停，best_step 4。"""
    seq = [60.0, 80.0, 70.0, 70.0, 70.0]
    raw = _callback(small_cfg, tmp_path / "raw")
    assert [raw._record(i, i / 10, s, None) for i, s in enumerate(seq, 1)] == [False, False, False, False, True] and raw.best_step == 2
    cb = _callback(small_cfg, tmp_path / "sm", smooth_k=3)
    assert not any(cb._record(i, i / 10, s, None) for i, s in enumerate(seq, 1)) and cb.best_step == 4
