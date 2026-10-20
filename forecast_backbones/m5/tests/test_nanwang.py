"""南网 / 线上适配（2026-10-20，同 M8 2026-10-19 / 10-20 那一轮）：线上 info.csv 站表表头自动识别、只有 station 站名时按站表映射站号、
站表相对路径按仓库根目录解析、配置列名大小写不敏感 + 别名、cov_present、南网训练入口 train_shell/m5_train_cli.py
（pipelines.UnifiedDataset + train_model，checkpoint 只存 LoRA，产物原地可加载）。nanwang_online/m5_train_cli.py 是同一份拷贝。"""
import glob
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from forecast_backbones.m5.config import load_config
from forecast_backbones.m5.data import station_ids_from_names
from forecast_backbones.m5.solar import StationMeta
from .conftest import make_frame, write_station_meta

M5 = Path(__file__).resolve().parents[1]
ROOT = M5.parents[1]
CLI = ROOT / "train_shell" / "m5_train_cli.py"
INFO_COLS = {"station_id": "plantid", "station": "plant_pointname", "lat": "LATITUDE", "lon": "LONGITUDE",
             "alt_m": None, "cap_mw": "GCCAPACITY"}
F_SERIES = ["f1_final", "f2_lr5e5", "f3_lr2e4", "f4_drop01", "f5_wd01"]


def test_info_csv_header_is_detected_without_mapping():
    meta = StationMeta.from_csv(M5 / "info.csv")                     # 不给 station_meta_cols：info.csv 表头自动识别
    assert meta.resolve(["plant_guangfu1780"])[0] == {"plant_guangfu1780": 1780} and meta.has_coords(1780)
    assert meta.get(1780)["cap_mw"] == 178.87
    assert StationMeta.from_csv(M5 / "info.csv", INFO_COLS).ids() == meta.ids()
    assert 1001 in StationMeta.from_csv(M5 / "station_meta.csv").ids()  # 标准表头照旧


def test_relative_station_meta_resolves_against_repo_not_cwd(tmp_path, monkeypatch):
    """CLI 在 train_scripts/ 下跑：相对路径 forecast_backbones/m5/info.csv 要对到仓库里那份，不能悄悄换成 station_meta.csv。"""
    from forecast_backbones.m5.transformer_models import load_station_meta
    monkeypatch.chdir(tmp_path)
    assert 1780 in load_station_meta("forecast_backbones/m5/info.csv").ids()
    assert 1001 in load_station_meta("forecast_backbones/m5/station_meta.csv").ids()
    with pytest.raises(FileNotFoundError):
        load_station_meta("forecast_backbones/m5/nope.csv")


def test_station_names_map_to_ids(tmp_path):
    from forecast_backbones.m5.rowctx import RowContextBuilder
    meta = StationMeta.from_csv(write_station_meta(tmp_path))
    assert station_ids_from_names(meta, ["S1002", "S1001", "S1002"]) == [1001, 1002]
    with pytest.raises(ValueError, match="站名映射失败"):
        station_ids_from_names(meta, ["S1001", "nope"])
    rb = RowContextBuilder(load_config({"dataset": {"station_meta": str(write_station_meta(tmp_path))}}), meta)
    assert rb.station_ids({"station": ["S1002", "S1001"]}).tolist() == [1002, 1001]           # 只有站名（线上 / 南网数据）
    assert rb.station_ids({"station_id": np.array([1001]), "station": ["S1002"]}).tolist() == [1001]   # station_id 优先
    with pytest.raises(ValueError, match="站名映射失败"):
        rb.station_ids({"station": ["nope"]})


def test_f_series_yamls_carry_nanwang_settings():
    """每个 f 系列 yaml 自带南网训练设置：数据位置、列名、站表（info.csv + 列名映射）、save_every、p*。"""
    files = sorted(glob.glob(str(M5 / "config" / "f*.yaml")))
    assert [Path(f).stem for f in files] == F_SERIES
    for f in files:
        cfg = load_config(f)
        ds = cfg["dataset"]
        assert ds["src_dir"] == "/home/wangding_hw/datasets/cleaned_processed_dataset_rulev3_with_tsfm", f
        assert ds["station_meta"] == "forecast_backbones/m5/info.csv" and ds["station_meta_cols"] == INFO_COLS, f
        assert (ds["ts_col"], ds["pv_col"], ds["label_col"]) == ("timestamp_win", "observe_power", "observe_power_future")
        assert cfg["features"]["past_future"] == [["general_ghi", "general_ghi_predict"], ["general_temp", "general_temp_predict"]]
        assert cfg["train"]["schedule"]["save_every"] == 500 and cfg["train"]["schedule"]["lr_scheduler"] == "constant", f
        assert cfg["decision"]["point"]["default"] == "p_star" and cfg["finetune"]["unfreeze_output_head"] is False, f
        assert cfg["run"]["tag"] == Path(f).stem


def test_nanwang_online_copy_is_identical():
    assert (ROOT / "nanwang_online" / "m5_train_cli.py").read_bytes() == CLI.read_bytes()


def test_cli_and_online_template_compile():
    for p in (CLI, ROOT / "nanwang_online" / "eval_scripts_template" / "m5" / "single_model_scene_infer_m5.py"):
        compile(p.read_text(encoding="utf-8"), str(p), "exec")


def _cli_unsupported():
    """只取 CLI 里的 unsupported()（不 import 整个脚本：它顶层要 pipelines.transformer_trainer）。"""
    import ast
    src = CLI.read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "unsupported")
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(CLI), "exec"), ns)
    return ns["unsupported"]


def test_cli_rejects_what_train_model_cannot_do():
    unsupported = _cli_unsupported()
    bad = unsupported(load_config({"train": {"ema_decay": 0.999, "optim": {"lora_plus_ratio": 16}}, "finetune": {"init_from": "x"}}))
    assert any("ema_decay" in b for b in bad) and any("train.optim" in b for b in bad) and any("init_from" in b for b in bad)
    assert any("零样本" in b for b in unsupported(load_config(str(M5 / "config" / "z0_zeroshot.yaml"))))
    for f in F_SERIES:
        assert unsupported(load_config(str(M5 / "config" / f"{f}.yaml"))) == [], f


# ------------------------------------------------------------------ 以下要 timesfm3（服务器上跑）
def _nanwang_frame(**kw):
    """线上 / 南网训练数据的列名：大写 general_GHI / general_TEMP、站名 station（没有 station_id）、带 cap_power_on。"""
    df = make_frame(**kw).rename(columns={"general_ghi": "general_GHI", "general_ghi_predict": "general_GHI_predict",
                                          "general_temp": "general_TEMP", "general_temp_predict": "general_TEMP_predict"})
    df.insert(1, "station", "S" + df.pop("station_id").astype(str))
    return df


def test_column_resolution_case_and_aliases(tmp_path, tiny_ckpt_dir):
    from forecast_backbones.m5.data import unified_feature_cols
    from forecast_backbones.m5.transformer_models import TransformersAdaptedM5Model
    from .conftest import full_cfg
    m = TransformersAdaptedM5Model(full_cfg(tmp_path, tiny_ckpt_dir, features={"col_aliases": {"general_temp": "tem2m"}}))
    cols = ["timestamp_win", "station", "observe_power", "observe_power_future", "general_GHI", "general_GHI_predict",
            "tem2m", "general_TEMP_predict", "cap_power_on"]
    assert unified_feature_cols(m, cols) == ["observe_power", "general_GHI", "general_GHI_predict", "tem2m", "general_TEMP_predict"]
    with pytest.raises(KeyError, match="observe_power"):
        unified_feature_cols(m, [c for c in cols if c != "observe_power"])
    with pytest.raises(ValueError, match="大小写变体"):
        m.resolve_col(["general_GHI", "general_ghi_X", "GENERAL_GHI"], "general_ghi")
    assert m.context_length == 672 and m.prediction_length == 192          # rolling_inference 读这两个名字


def _run_cli(tmp_path, ckpt, *extra, schedule=None):
    """子进程跑 CLI（与服务器上 python m5_train_cli.py 一样）。本仓库没有 pipelines/transformer_utils.py（南网仓库有），用桩模块顶上。"""
    stub = tmp_path / "stub"; stub.mkdir()
    (stub / "transformer_utils.py").write_text("def assert_cuda_available():\n    pass\n\n"
                                               "def print_run_fingerprint(trainer):\n    pass\n", encoding="utf-8")
    for i, (start, sub) in enumerate([("2025-06-08", "date=a"), ("2025-06-10", "date=b")]):
        d = tmp_path / "data" / "train" / sub; d.mkdir(parents=True)
        _nanwang_frame(n_days=2, issue_every_min=120, seed=i, start=start).to_parquet(d / "data_0.parquet", index=False)
    cfg_path = tmp_path / "cli_cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "run": {"tag": "cli", "seeds": [3]},
        "dataset": {"src_dir": str(tmp_path / "data"), "station_meta": str(write_station_meta(tmp_path))},
        "timesfm3": {"checkpoint": str(ckpt), "device": "cpu", "per_core_batch_size": 8},
        "finetune": {"unfreeze_output_head": False},
        "train": {"schedule": {"lr": 2.5e-4, "lr_scheduler": "constant", "weight_decay": 0.07, "batch_size": 4, "max_steps": 3,
                               "logging_steps": 1, "progress_bar": False, **(schedule or {})}},
        "decision": {"point": {"default": "p_star"}}}), encoding="utf-8")
    run = tmp_path / "run"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(stub), str(ROOT / "pipelines")])}
    cmd = [sys.executable, str(CLI), "--config", str(cfg_path), "--ckpt-dir", str(run), *extra]
    p = subprocess.run(cmd, env=env, cwd=str(CLI.parent), capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return run, p.stdout


def _bd_nanwang(batch):
    bd = {k: v for k, v in batch["batch_dict"].items() if k != "station_id"}
    bd["station"] = ["S1001"] * len(batch["batch_dict"]["station_id"])
    for a, b in (("general_ghi", "general_GHI"), ("general_ghi_predict", "general_GHI_predict"),
                 ("general_temp", "general_TEMP"), ("general_temp_predict", "general_TEMP_predict")):
        bd[b] = bd.pop(a)
    return bd


def test_cli_full_training_saves_lora_only_and_loads_in_place(tmp_path, tiny_ckpt_dir, batch):
    import torch
    from safetensors.torch import load_file
    from forecast_backbones.m5.transformer_models import TransformersAdaptedM5Model
    run, log = _run_cli(tmp_path, tiny_ckpt_dir, schedule={"save_every": 2})
    out = run / "model_result"
    cfg = yaml.safe_load((out / "model_config.yaml").read_text(encoding="utf-8"))
    assert cfg["dataset"]["known_stations"] == [1001, 1002]
    assert cfg["features"]["cov_present"] == cfg["features"]["past_future"]                  # 大写数据列对上了配置列
    sd = load_file(str(out / "model.safetensors"))
    assert sd and all(k.endswith((".lora_A", ".lora_B")) for k in sd)                       # 只有 LoRA，没有底座
    m = TransformersAdaptedM5Model(cfg)                                                      # weights_path → 同目录权重
    assert m.weights_loaded and str(m.cfg["train"]["weights_path"]) == str((out / "model.safetensors").resolve())
    assert torch.isfinite(m(_bd_nanwang(batch)).logits).all()                                # 站名 + 大写列推理
    ta = torch.load(out / "training_args.bin", weights_only=False)
    assert (ta.learning_rate, ta.weight_decay, ta.max_steps, ta.seed, ta.per_device_train_batch_size) == (2.5e-4, 0.07, 3, 3, 4)
    assert str(ta.lr_scheduler_type).endswith("constant")                                   # 调度跟配置走（m8 写死 cosine）
    ck = run / "hf_trainer_output" / "checkpoint-2"
    assert not (ck / "optimizer.pt").exists() and not (ck / "scheduler.pt").exists()           # save_only_model：只存权重
    assert sorted(p.name for p in out.iterdir() if p.suffix in (".safetensors", ".yaml")) == ["model.safetensors", "model_config.yaml"]
    assert all(k.endswith((".lora_A", ".lora_B")) for k in load_file(str(ck / "model.safetensors")))
    m2 = TransformersAdaptedM5Model(yaml.safe_load((ck / "model_config.yaml").read_text(encoding="utf-8")))
    assert m2.weights_loaded
    a, b = m.state_dict(), m2.state_dict()
    assert any(not torch.equal(a[k], b[k]) for k in a if k.endswith(".lora_B"))
