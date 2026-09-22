import importlib.util
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("timesfm3")
from forecast_backbones.m6.station_eval import load_trained_model, predict_station, scale_history_tail, station_key
from forecast_backbones.m6.train import train_m6
from forecast_backbones.m6.calibration import quantile_cols
from forecast_backbones.m6.decision import MODEL_LEVELS
from forecast_backbones.m6.transformer_models import TransformersAdaptedM6Model
from .conftest import make_synthetic_frame
from .test_train import _patch_tiny_backbone, _inputs

ROOT = Path(__file__).resolve().parents[3]
QCOLS = quantile_cols(MODEL_LEVELS)


def _trained_run(tmp_path, tiny_evaluator, monkeypatch):
    """训一个 2 步的 run，再把产物目录整体搬走、删掉原目录：之后只许从新位置读。
    之后建的模型各自从 checkpoint 重新加载一份底座（与服务器上评测一样），不复用训练时改过的那份。"""
    _patch_tiny_backbone(monkeypatch, tiny_evaluator)
    df, parquet, out, cfg = _inputs(tmp_path)
    with pytest.warns(UserWarning, match="LoRA 权重"):
        train_m6(cfg, str(parquet), max_steps=2)
    moved = tmp_path / "moved_run"
    shutil.copytree(out, moved)
    shutil.rmtree(out)
    from timesfm3 import ModelConfig, TimesFM3Evaluator
    ckpt = tiny_evaluator.config.checkpoint_path

    def _fresh(self, want):
        self._backbone = TimesFM3Evaluator(ModelConfig(checkpoint_path=ckpt, per_core_batch_size=4, device="cpu"))
        self._backbone_device = "cpu"
        self._attach_torch_path()
    monkeypatch.setattr(TransformersAdaptedM6Model, "_ensure_backbone", _fresh)
    return df, moved


def _station_file(df, path, station_id=1001, name="S1001", days=("2025-06-12", "2025-06-14")):
    """单站测试 parquet：只有站名列（没有 station_id），逼推理走 station_meta 映射。"""
    sub = df[(df["station_id"] == station_id) & (df["timestamp_win"] >= days[0]) & (df["timestamp_win"] < days[1])]
    sub = sub.drop(columns=["station_id"]).assign(station=name).reset_index(drop=True)
    sub.to_parquet(path, index=False)
    return sub


def test_trained_run_dir_is_self_contained_and_matches_training(tmp_path, tiny_evaluator, monkeypatch):
    df, run = _trained_run(tmp_path, tiny_evaluator, monkeypatch)
    text = (run / "model_config.yaml").read_text(encoding="utf-8")
    assert "${run.out_dir}/model.safetensors" in text                       # 占位符留着，目录能整体搬走
    from safetensors import safe_open
    with safe_open(str(run / "model.safetensors"), framework="pt") as f:
        keys = list(f.keys())
        assert f.metadata()["root"] == "backbone.model"
    assert keys and all(k.endswith((".lora_A", ".lora_B")) or k.startswith("output_head.") for k in keys)

    (run / "station_routing.json").write_text('{"1001": "zeroshot", "1002": "zeroshot"}')   # 旧 run 的路由表：不再读
    model, cfg = load_trained_model(run, device="cpu", batch_size=8)
    assert model.lora_loaded and model.calibrator is not None
    assert cfg["finetune"]["lora_path"] == str(run / "model.safetensors")
    assert cfg["calibration"]["table_path"] == str(run / "calibration.parquet") and cfg["eval"]["batch_size"] == 8

    src = _station_file(df, tmp_path / "station=S1001.parquet")
    st, res = predict_station(model, tmp_path / "station=S1001.parquet", scratch_dir=tmp_path / "scratch")
    assert st == "S1001" and set(res["station"]) == {"S1001"} and set(res["station_id"]) == {1001}
    assert len(res) == len(src) and (res["timestamp_win"].to_numpy() == src["timestamp_win"].to_numpy()).all()
    assert not any((tmp_path / "scratch").iterdir())                       # 临时 parquet 推理完即删
    # 与训练收尾时同一个模型（最优 LoRA）在验证集上的原生分位逐行一致：全部站都走 LoRA
    ref = pd.read_parquet(run / "result_val_lora.parquet")
    ref = ref[ref["station_id"] == 1001].sort_values("timestamp_win").reset_index(drop=True)
    assert len(ref) == len(res)
    for c in QCOLS:
        np.testing.assert_allclose(np.stack(res[c]), np.stack(ref[c]), atol=1e-5)
    truth = np.stack(src["observe_power_future"])
    np.testing.assert_allclose(np.stack(res["observe_power_future"]), truth[:, :192], atol=1e-6)


def test_load_trained_model_refuses_incomplete_run(tmp_path, tiny_evaluator, monkeypatch):
    _, run = _trained_run(tmp_path, tiny_evaluator, monkeypatch)
    (run / "calibration.parquet").rename(run / "calibration.bak")
    with pytest.raises(FileNotFoundError, match="校准表"):
        load_trained_model(run, device="cpu")
    (run / "calibration.bak").rename(run / "calibration.parquet")
    (run / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        load_trained_model(run, device="cpu")
    with pytest.raises(FileNotFoundError, match="--config"):
        load_trained_model(tmp_path / "nowhere", device="cpu")


def test_loaded_model_is_fully_frozen_and_inference_changes_nothing(tmp_path, tiny_evaluator, monkeypatch):
    df, run = _trained_run(tmp_path, tiny_evaluator, monkeypatch)
    model, _ = load_trained_model(run, device="cpu")
    params = list(model.parameters())
    assert params and not any(p.requires_grad for p in params)
    assert not any(m.training for m in model.modules())                     # LoRA dropout 等全部关闭
    before = {k: v.clone() for k, v in model.state_dict().items()}
    _station_file(df, tmp_path / "station=S1001.parquet")
    _station_file(df, tmp_path / "station=S1002.parquet", station_id=1002, name="S1002")
    _, a = predict_station(model, tmp_path / "station=S1001.parquet")
    predict_station(model, tmp_path / "station=S1002.parquet")
    _, b = predict_station(model, tmp_path / "station=S1001.parquet")
    after = model.state_dict()
    assert before.keys() == after.keys() and all(torch.equal(before[k], after[k]) for k in before)   # 权重与 buffer 都没动
    for c in QCOLS:
        np.testing.assert_array_equal(np.stack(a[c]), np.stack(b[c]))     # 中间推理别的站，结果逐位不变
    assert not any(p.requires_grad for p in model.parameters()) and not any(m.training for m in model.modules())


def test_old_run_with_pt_weights_and_training_yaml(tmp_path, tiny_evaluator, monkeypatch):
    """旧 run：权重是 lora_weights.pt、没有 model_config.yaml → 读 .pt，配置用 --config 给的训练 yaml。"""
    import yaml
    from forecast_backbones.m6.lora import load_lora_weights, save_lora_weights
    df, run = _trained_run(tmp_path, tiny_evaluator, monkeypatch)
    sd = load_lora_weights(run / "model.safetensors")
    save_lora_weights(run / "lora_weights.pt", sd)
    (run / "model.safetensors").unlink()
    raw = yaml.safe_load((run / "model_config.yaml").read_text(encoding="utf-8"))
    (run / "model_config.yaml").unlink()
    (tmp_path / "train.yaml").write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    model, cfg = load_trained_model(run, config=tmp_path / "train.yaml", device="cpu")
    assert model.lora_loaded and cfg["finetune"]["lora_path"] == str(run / "lora_weights.pt")


def test_station_key_and_tail_scale(tmp_path):
    df = make_synthetic_frame(n_days=1, issue_every_min=360)
    df[df["station_id"] == 1001].assign(station=1001.0).to_parquet(tmp_path / "a.parquet", index=False)
    assert station_key(tmp_path / "a.parquet") == "1001"                    # 1001.0 这种浮点站号归一成 '1001'
    df.to_parquet(tmp_path / "two.parquet", index=False)
    with pytest.raises(ValueError, match="2 个站"):
        station_key(tmp_path / "two.parquet")
    scaled = scale_history_tail(df, "observe_power", tail_len=4, factor=0.1)
    a, b = np.stack(df["observe_power"]), np.stack(scaled["observe_power"])
    np.testing.assert_allclose(b[:, -4:], a[:, -4:] * 0.1, rtol=1e-6)
    np.testing.assert_array_equal(b[:, :-4], a[:, :-4])
    assert scale_history_tail(df, "observe_power", factor=None) is df


def _load_station_script(monkeypatch):
    """eval_station_script 只在服务器上有：换成替身，只验证 m6_station_script 的装配与落盘。"""
    stub = types.ModuleType("eval_station_script")

    class Enc(json.JSONEncoder):
        def default(self, o):
            return o.item() if isinstance(o, np.generic) else str(o)

    stub.NumpyJSONEncoder = Enc
    stub.extract_summary = lambda report, tables: {"nanwang_stations": sorted(tables)}
    stub.station_cap_from_df = lambda df, cap: float(df["cap_power_on"].iloc[0]) if "cap_power_on" in df else cap
    monkeypatch.setitem(sys.modules, "eval_station_script", stub)
    spec = importlib.util.spec_from_file_location("m6_station_script", ROOT / "eval_shell" / "m6_station_script.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_m6_station_script_writes_metrics_and_summary(tmp_path, tiny_evaluator, monkeypatch):
    df, run = _trained_run(tmp_path, tiny_evaluator, monkeypatch)
    script = _load_station_script(monkeypatch)
    src = _station_file(df, tmp_path / "station=S1001.parquet")
    st, cap, result_df = script.infer_station_m6(run, tmp_path / "station=S1001.parquet", 465.0, device="cpu",
                                                 scratch_dir=tmp_path / "scratch")
    assert st == "S1001" and cap == 500.0
    assert list(result_df.columns) == ["timestamp_win", "station", "observe_power_predict", "observe_power_future", "cap_power_on"]
    assert (result_df["cap_power_on"].to_numpy() == src["cap_power_on"].to_numpy()).all()
    out = tmp_path / "metrics_ens"
    summary = script.run_station(run, tmp_path / "station=S1001.parquet", out, 465.0, device="cpu")
    assert (out / "S1001-metrics.xlsx").exists()
    saved = json.loads((out / "S1001-summary.json").read_text(encoding="utf-8"))
    assert saved == summary and saved["station"] == "S1001" and saved["cap_power"] == 500.0
    assert saved["rows"] == len(src) and saved["nanwang_stations"] == ["S1001"]
    assert "nanwang" in pd.ExcelFile(out / "S1001-metrics.xlsx").sheet_names
    assert not [p for p in out.iterdir() if p.name.startswith(".m6_eval_")]


FAKE_PYTHON = """#!/bin/bash
while [ $# -gt 0 ]; do
  case "$1" in --station_path) sp="$2"; shift;; --out_dir) od="$2"; shift;; esac
  shift
done
st=$(basename "$sp" .parquet); st=${st#station=}
echo "log line mentioning OK and SKIP"
[ "$st" = "bad" ] && { echo "boom" >&2; exit 3; }
echo '{}' > "$od/$st-summary.json"
"""


def _run_eval_sh(env_over, tmp_path):
    env = {**os.environ, **env_over}
    return subprocess.run(["bash", str(ROOT / "eval_shell" / "eval_m6.sh")], env=env, capture_output=True, text=True,
                          timeout=120)


def test_eval_m6_sh_parallel_workers_skip_and_fail(tmp_path):
    ckpt = tmp_path / "run"; ckpt.mkdir()
    for f in ("model_config.yaml", "model.safetensors"):
        (ckpt / f).write_text("x")
    data = tmp_path / "test"; data.mkdir()
    for st in ("1001", "1002", "bad"):
        (data / f"station={st}.parquet").write_text("x")
    fake = tmp_path / "fake_python"; fake.write_text(FAKE_PYTHON); fake.chmod(0o755)
    env = {"CKPT_DIR": str(ckpt), "STATION_DIR": str(data), "NUM_WORKERS": "2", "PYTHON": str(fake)}

    p = _run_eval_sh({**env, "DRY_RUN": "1"}, tmp_path)
    assert p.returncode == 0 and "worker[0] GPU0 : 1001 1002" in p.stdout and "worker[1] GPU1 : bad" in p.stdout
    assert not (ckpt / "metrics_ens" / "1001-summary.json").exists()

    p = _run_eval_sh(env, tmp_path)
    out = ckpt / "metrics_ens"
    assert p.returncode != 0 and "失败 worker: 1" in p.stdout
    assert (out / "1001-summary.json").exists() and (out / "1002-summary.json").exists()
    w0 = (out / "logs" / "worker_0.log").read_text(encoding="utf-8")
    w1 = (out / "logs" / "worker_1.log").read_text(encoding="utf-8")
    assert "OK=2 FAIL=0 SKIP=0" in w0 and "OK=0 FAIL=1 SKIP=0" in w1 and "FAIL bad (rc=3)" in w1

    (data / "station=bad.parquet").unlink()
    p = _run_eval_sh(env, tmp_path)
    assert p.returncode == 0                                                  # 剩 2 站、2 个 worker 各一站，都已有 summary
    assert all("OK=0 FAIL=0 SKIP=1" in (out / "logs" / f"worker_{i}.log").read_text(encoding="utf-8") for i in (0, 1))

    (ckpt / "model.safetensors").unlink()
    p = _run_eval_sh(env, tmp_path)
    assert p.returncode == 1 and "model.safetensors" in p.stderr
    p = _run_eval_sh({k: v for k, v in env.items() if k != "CKPT_DIR"}, tmp_path)
    assert p.returncode == 1 and "CKPT_DIR" in p.stderr
