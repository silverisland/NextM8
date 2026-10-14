import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

pytest.importorskip("timesfm3")
from .conftest import full_cfg, write_splits


def test_lora_run_end_to_end_with_step0_zero_shot(tmp_path, tiny_ckpt_dir):
    from forecast_backbones.m5.train import train_m5
    splits = write_splits(tmp_path, n_train_days=2, n_val_days=2, train_stations=(1001,), val_stations=(1001, 1002))
    out = tmp_path / "lora"
    train_m5(full_cfg(tmp_path, tiny_ckpt_dir, train={"schedule": {"lr": 1.0e-2}}), splits_dir=splits, out_dir=out, tag="l1")
    log = json.loads((out / "train_log.json").read_text(encoding="utf-8"))
    assert [h["step"] for h in log["history"]][:3] == [0, 2, 4]                 # 第 0 步（零样本）+ 每 2 步
    s = pd.read_csv(out / "reports" / "summary.csv").iloc[0]
    assert s["nanwang_zeroshot"] == pytest.approx(log["history"][0]["nanwang_after_pstar"])
    assert s["gain"] == pytest.approx(s["nanwang_short_D+1"] - s["nanwang_zeroshot"])
    assert s["method"] == "lora" and s["rank"] == 8 and s["n_trainable"] > 0 and s["n_unseen_stations"] == 1
    assert (out / "model.safetensors").exists() and not (out / ".fresh" / "model.safetensors").exists()
    assert (out / "result_val.parquet").exists() and (out / "reports" / "station_table.csv").exists()
    saved = yaml.safe_load((out / "model_config.yaml").read_text(encoding="utf-8"))
    assert saved["dataset"]["known_stations"] == [1001] and saved["finetune"]["enabled"] is True


def test_zero_shot_run_scores_without_training_and_matches_lora_step0(tmp_path, tiny_ckpt_dir):
    from forecast_backbones.m5.train import train_m5
    splits = write_splits(tmp_path, n_train_days=2, n_val_days=2)
    z = train_m5(full_cfg(tmp_path, tiny_ckpt_dir, finetune={"enabled": False}), splits_dir=splits, out_dir=tmp_path / "z0", tag="z0")
    zs = z["summary"].iloc[0]
    assert not (tmp_path / "z0" / "model.safetensors").exists() and (tmp_path / "z0" / "model_config.yaml").exists()
    assert zs["best_step"] == 0 and zs["n_evals"] == 1 and zs["method"] == "zero_shot" and zs["n_trainable"] == 0
    assert zs["gain"] == 0.0 and np.isfinite(zs["nanwang_short_D+1"])
    log = json.loads((tmp_path / "z0" / "train_log.json").read_text(encoding="utf-8"))
    assert log["zero_shot"] is True and log["best_step"] == 0
    l = train_m5(full_cfg(tmp_path, tiny_ckpt_dir), splits_dir=splits, out_dir=tmp_path / "l1", tag="l1")
    assert l["history"][0]["nanwang_after_pstar"] == pytest.approx(zs["nanwang_short_D+1"], abs=1e-6)   # B=0 → 与零样本同分
    with pytest.raises(ValueError, match="零样本"):
        train_m5(full_cfg(tmp_path, tiny_ckpt_dir, finetune={"enabled": False}), splits_dir=splits,
                 out_dir=tmp_path / "zr", refit=True)


def test_refit_with_ema_saves_final_weights(tmp_path, tiny_ckpt_dir):
    from forecast_backbones.m5.train import train_m5
    splits = write_splits(tmp_path, n_train_days=2, n_val_days=2)
    out = tmp_path / "refit"
    train_m5(full_cfg(tmp_path, tiny_ckpt_dir, train={"ema_decay": 0.5}), splits_dir=splits, out_dir=out, tag="rf", refit=True)
    s = pd.read_csv(out / "reports" / "summary.csv").iloc[0]
    assert (out / "model.safetensors").exists() and s["best_step"] == 4 and np.isnan(s["nanwang_short_D+1"])
    assert s["ema_decay"] == 0.5


def test_weight_ema_shadows_only_trainable_params():
    from forecast_backbones.m5.train import WeightEma
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 2))
    m[0].requires_grad_(False)
    frozen = m[0].weight.detach().clone()
    ema = WeightEma(m, decay=0.9)
    assert set(ema.shadow) == {"1.weight", "1.bias"}
    w0 = m[1].weight.detach().clone()
    with torch.no_grad():
        m[1].weight.add_(1.0)
    ema.update(m)
    d1 = min(0.9, 2 / 11)
    assert torch.allclose(ema.shadow["1.weight"], d1 * w0 + (1 - d1) * (w0 + 1.0))
    live = m[1].weight.detach().clone()
    with ema.applied(m):
        assert torch.allclose(m[1].weight, ema.shadow["1.weight"]) and torch.equal(m[0].weight, frozen)
    assert torch.equal(m[1].weight, live)
    ema.load_into(m)
    assert torch.allclose(m[1].weight, ema.shadow["1.weight"])
    with pytest.raises(ValueError, match="可训"):
        WeightEma(torch.nn.Linear(2, 2).requires_grad_(False), 0.9)


def test_collect_runs_lists_zero_shot_and_lora(tmp_path, tiny_ckpt_dir):
    from forecast_backbones.m5.train import train_m5
    from forecast_backbones.m5.collect_runs import collect
    splits = write_splits(tmp_path, n_train_days=2, n_val_days=2)
    train_m5(full_cfg(tmp_path, tiny_ckpt_dir, finetune={"enabled": False}), splits_dir=splits, out_dir=tmp_path / "b" / "z0", tag="z0")
    train_m5(full_cfg(tmp_path, tiny_ckpt_dir), splits_dir=splits, out_dir=tmp_path / "b" / "l1", tag="l1")
    df = collect([tmp_path / "b"])
    assert set(df["tag"]) == {"z0", "l1"} and {"nanwang_zeroshot", "gain", "method"} <= set(df.columns)


def test_seed_controls_lora_init_and_run_is_reproducible(tmp_path, tiny_ckpt_dir):
    from safetensors.torch import load_file
    from forecast_backbones.m5.train import train_m5
    splits = write_splits(tmp_path, n_train_days=2, n_val_days=2)
    def run(name, seed):
        out = tmp_path / name
        train_m5(full_cfg(tmp_path, tiny_ckpt_dir, run={"seeds": [seed]}, train={"schedule": {"max_steps": 2, "eval_every_steps": 2}}),
                 splits_dir=splits, out_dir=out, tag=name)
        return load_file(str(out / "model.safetensors"))
    a, b, c = run("s0a", 0), run("s0b", 0), run("s1", 1)
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)         # 同种子逐位相同
    lora_a = [k for k in a if "lora_A" in k]
    assert lora_a and any(not torch.equal(a[k], c[k]) for k in lora_a)               # 换种子 lora_A 不同
