"""南网训练入口 train_shell/m8_train_cli.py（2026-10-20，结构对齐 m6_b_train_cli.py）：pipelines.UnifiedDataset + train_model 全量训练，
超参取配置 train.schedule，站名列保留给 M8 查站表，产物 model_result/{model.safetensors, model_config.yaml} 原地可加载。
nanwang_online/m8_train_cli.py 是同一份拷贝。"""
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from forecast_backbones.m8.config import load_config
from forecast_backbones.m8.data import unified_feature_cols, station_ids_from_names
from forecast_backbones.m8.features import FeatureBuilder
from forecast_backbones.m8.solar import StationMeta
from .conftest import ROOT, make_frame, write_station_meta

CLI = ROOT / "train_shell" / "m8_train_cli.py"
CFG_MODEL = {"d_model": 32, "d_ff": 64, "head_hidden": 32, "n_heads": 2, "blocks": ["conv", "deltanet"], "pos_in_patch_dim": 8}


def _nanwang_frame(**kw):
    """线上 / 南网训练数据的列名：大写 general_GHI / general_TEMP、站名 station（没有 station_id）、TimesFM 列 timesfm3、带 cap_power_on。"""
    df = make_frame(**kw).rename(columns={"general_ghi": "general_GHI", "general_ghi_predict": "general_GHI_predict",
                                          "general_temp": "general_TEMP", "general_temp_predict": "general_TEMP_predict",
                                          "timesfm_predict": "timesfm3"})
    df.insert(1, "station", "S" + df.pop("station_id").astype(str))
    return df


def test_feature_cols_follow_config_case_and_aliases(small_cfg):
    cfg = load_config({**small_cfg, "features": {"hint_col": "timesfm3"}})
    fe = FeatureBuilder(cfg, StationMeta.from_csv(cfg["dataset"]["station_meta"]))
    cols = ["timestamp_win", "station", "observe_power", "observe_power_future", "general_GHI", "general_GHI_predict",
            "general_TEMP", "general_TEMP_predict", "tsfm_predict", "cap_power_on"]
    assert unified_feature_cols(fe, cols) == ["observe_power", "general_GHI", "general_GHI_predict", "general_TEMP",
                                              "general_TEMP_predict", "tsfm_predict"]       # 大写列名对上，timesfm3 ← 别名
    with pytest.raises(KeyError, match="observe_power"):
        unified_feature_cols(fe, [c for c in cols if c != "observe_power"])


def test_station_ids_from_names(tmp_path):
    meta = StationMeta.from_csv(write_station_meta(tmp_path))
    assert station_ids_from_names(meta, ["S1002", "S1001", "S1002"]) == [1001, 1002]
    with pytest.raises(ValueError, match="站名映射失败"):
        station_ids_from_names(meta, ["S1001", "nope"])


SITECUSTOMIZE = """
import sys
class _Block:                      # 全量训练不验证、不切分：禁掉 nanwang_metric 与 time_splits，CLI 照样要能跑
    def find_spec(self, name, path=None, target=None):
        if name.endswith(('nanwang_metric', 'time_splits')):
            raise ImportError('blocked: ' + name)
        return None
sys.meta_path.insert(0, _Block())

import torch
if not torch.cuda.is_available():  # 用真实 pipelines/transformer_utils.py；它的 print_run_fingerprint 断言设备是 cuda，无卡机器上只把设备名报成 cuda
    import pipelines.transformer_utils as tu
    _real = tu.print_run_fingerprint
    class _Cuda:
        def __str__(self):
            return "cuda (cpu test)"
    class _Proxy:
        def __init__(self, obj, **over):
            self._obj, self._over = obj, over
        def __getattr__(self, k):
            return self._over[k] if k in self._over else getattr(self._obj, k)
    def print_run_fingerprint(trainer):
        _real(_Proxy(trainer, args=_Proxy(trainer.args, device=_Cuda()), accelerator=_Proxy(trainer.accelerator, device=_Cuda())))
    tu.print_run_fingerprint = print_run_fingerprint
"""


def _cli_env(tmp_path) -> dict:
    """CLI 子进程环境：真实 pipelines/transformer_utils.py（南网仓库那份）+ 屏蔽 nanwang_metric / time_splits + 无卡时的设备名替身。"""
    assert (ROOT / "pipelines" / "transformer_utils.py").exists(), "缺 pipelines/transformer_utils.py（从南网仓库拷来）"
    stub = tmp_path / "stub"; stub.mkdir(exist_ok=True)
    (stub / "sitecustomize.py").write_text(SITECUSTOMIZE, encoding="utf-8")
    return {**os.environ, "PYTHONPATH": os.pathsep.join([str(stub), str(ROOT), str(ROOT / "pipelines")])}


def _run_cli(tmp_path, *extra, schedule=None):
    """子进程跑 CLI（与服务器上 python m8_train_cli.py 一样）。"""
    for i, (start, sub) in enumerate([("2025-06-08", "a"), ("2025-06-10", "b")]):
        d = tmp_path / "data" / "train" / sub; d.mkdir(parents=True)
        _nanwang_frame(n_days=2, issue_every_min=120, seed=i, start=start).to_parquet(d / "data_0.parquet", index=False)
    cfg_path = tmp_path / "cli_cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({                               # 数据位置 / 站表 / save_every 都在配置里，命令行只给模板那几个参数
        "run": {"tag": "cli", "seeds": [3]}, "model": CFG_MODEL,
        "dataset": {"src_dir": str(tmp_path / "data"), "station_meta": str(write_station_meta(tmp_path))},
        "features": {"hint_col": "timesfm3", "derived": {"clear_sky_ratio": {"enabled": True}, "history_envelope": {"enabled": True}}},
        "train": {"schedule": {"lr": 2.5e-4, "weight_decay": 0.07, "batch_size": 8, "max_steps": 3, "warmup_steps": 1,
                               "logging_steps": 1, "progress_bar": False, **(schedule or {})}}}), encoding="utf-8")
    run = tmp_path / "run"
    env = _cli_env(tmp_path)
    cmd = [sys.executable, str(CLI), "--config", str(cfg_path), "--ckpt-dir", str(run), *extra]
    p = subprocess.run(cmd, env=env, cwd=str(CLI.parent), capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert "RUN FINGERPRINT" in p.stdout                                # 真实 transformer_utils 的训练指纹确实跑了
    return run, p.stdout


def test_cli_full_training_produces_loadable_checkpoint(tmp_path, batch):
    from forecast_backbones.m8.transformer_models import TransformersAdaptedM8Model
    run, log = _run_cli(tmp_path, schedule={"save_every": 2})
    out = run / "model_result"
    cfg = yaml.safe_load((out / "model_config.yaml").read_text(encoding="utf-8"))
    assert cfg["dataset"]["known_stations"] == [1001, 1002]
    assert ["general_ghi", "general_ghi_predict"] in cfg["features"]["cov_present"]                  # 大写数据列对上了配置列
    assert cfg["features"]["past_future"] == cfg["features"]["cov_present"]                       # 默认两对 NWP 都对上
    assert cfg["features"]["hint_col"] == "timesfm3" and cfg["model"]["blocks"] == CFG_MODEL["blocks"]
    m = TransformersAdaptedM8Model(cfg)                                                              # weights_path → 同目录权重
    assert m.weights_loaded and str(m.cfg["train"]["weights_path"]) == str((out / "model.safetensors").resolve())
    bd = {k: v for k, v in batch["batch_dict"].items() if k != "station_id"}
    bd["station"] = ["S1001"] * len(batch["batch_dict"]["station_id"])
    bd["timesfm3"] = bd.pop("timesfm_predict")
    assert torch.isfinite(m(bd).logits).all()
    # 超参来自配置 train.schedule（不是 m6_b 模板的 optimization 段 / 写死的 wd 0.01、warmup_ratio 0.05）
    ta = torch.load(out / "training_args.bin", weights_only=False)
    assert (ta.learning_rate, ta.weight_decay, ta.warmup_steps, ta.max_steps, ta.seed, ta.per_device_train_batch_size) == \
           (2.5e-4, 0.07, 1, 3, 3, 8)
    # save_every 2：checkpoint-2 带 model_config.yaml，原地可加载，且与最终权重不同
    ck = run / "hf_trainer_output" / "checkpoint-2"
    m2 = TransformersAdaptedM8Model(yaml.safe_load((ck / "model_config.yaml").read_text(encoding="utf-8")))
    assert m2.weights_loaded
    assert any(not torch.equal(a, b) for a, b in zip(m.state_dict().values(), m2.state_dict().values()))


def test_cli_max_steps_flag_and_fresh_start(tmp_path):
    run, _ = _run_cli(tmp_path, "--max-steps", "2")
    ta = torch.load(run / "model_result" / "training_args.bin", weights_only=False)
    assert ta.max_steps == 2 and not (run / ".fresh").exists()                                   # 哨兵路径从不写入


def test_nanwang_online_copy_is_identical():
    assert (ROOT / "nanwang_online" / "m8_train_cli.py").read_bytes() == CLI.read_bytes()


def test_variant_yamls_carry_nanwang_settings():
    """每个变体 yaml 自带南网训练设置：数据位置、列名、站表（info.csv + 列名映射）、save_every。"""
    import glob
    from pathlib import Path
    from forecast_backbones.m8.transformer_models import load_station_meta
    files = sorted(glob.glob(str(ROOT / "forecast_backbones/m8/config/*.yaml")))
    assert len(files) == 7
    for f in files:
        cfg = load_config(f)
        ds = cfg["dataset"]
        assert ds["src_dir"] == "/home/wangding_hw/datasets/cleaned_processed_dataset_rulev3_with_tsfm", f
        assert ds["station_meta"] == "forecast_backbones/m8/info.csv", f
        assert ds["station_meta_cols"] == {"station_id": "plantid", "station": "plant_pointname", "lat": "LATITUDE",
                                           "lon": "LONGITUDE", "alt_m": None, "cap_mw": "GCCAPACITY"}, f
        assert (ds["ts_col"], ds["pv_col"], ds["label_col"]) == ("timestamp_win", "observe_power", "observe_power_future")
        assert cfg["features"]["past_future"] == [["general_ghi", "general_ghi_predict"], ["general_temp", "general_temp_predict"]]
        assert cfg["features"]["hint_col"] == "timesfm3" and cfg["train"]["schedule"]["save_every"] == 500, f
    meta = load_station_meta(ds["station_meta"], ds["station_meta_cols"])
    assert meta.resolve(["plant_guangfu1780"])[0] == {"plant_guangfu1780": 1780} and meta.has_coords(1780)
    assert meta.get(1780)["cap_mw"] == 178.87


def test_relative_station_meta_resolves_against_repo_not_cwd(tmp_path, monkeypatch):
    """CLI 在 train_scripts/ 下跑：相对路径 forecast_backbones/m8/info.csv 要对到仓库里那份，不能悄悄换成 station_meta.csv。"""
    from forecast_backbones.m8.transformer_models import load_station_meta
    monkeypatch.chdir(tmp_path)
    assert 1780 in load_station_meta("forecast_backbones/m8/info.csv").ids()
    assert 1001 in load_station_meta("forecast_backbones/m8/station_meta.csv").ids()
    with pytest.raises(FileNotFoundError):
        load_station_meta("forecast_backbones/m8/nope.csv")


NAMES = {1001: "plant_guangfu1358", 1002: "plant_guangfu6089"}            # 南网站名；6089 > 4096 个槽位，验证取模后的槽位两边一致


def _write_info_csv(path):
    """线上 info.csv 表头（plantid / plant_pointname / GCCAPACITY / LONGITUDE / LATITUDE），容量与合成数据一致。"""
    from .conftest import CAP, COORDS
    rows = ["plantid,plantname,plant_pointname,province,province_pointname,city,country,GCCAPACITY,LONGITUDE,LATITUDE"]
    for s, name in NAMES.items():
        pid = int(name.removeprefix("plant_guangfu"))
        rows.append(f"{pid},光伏{pid},{name},广西,province_guangxi,南宁,横州市,{CAP[s]},{COORDS[s][1]},{COORDS[s][0]}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_cli_checkpoint_runs_in_online_inference_template(tmp_path, monkeypatch):
    """（超短期 unified_inference + 短期 rolling_inference 两条路径）m8_train_cli.py 训出的 model_result/ 按 nanwang_online/eval_scripts_template/m8/single_model_scene_infer_m8.py 的方式加载、
    喂线上格式的数据（大写 general_GHI、tsfm_predict、station_id 实体列、多出来的 *_SOLARGIS 列）走 unified_inference：
    每行都有预测（不是批量失败后的 None），且与训练格式数据上的前向逐位一致。"""
    from functools import partial
    from forecast_backbones.m8.transformer_models import TransformersAdaptedM8Model
    from pipelines.pipeline_wrapper_inference import unified_inference
    from pipelines.transformer_dataset import UnifiedDataset
    from forecast_backbones.m8 import features
    monkeypatch.setattr(features, "_WARNED_MISSING", set())                      # “只提示一次”的登记不漏到别的测试

    # 1) 南网格式训练数据 + 真实 j1_final2.yaml（含 info.csv 表头映射），只把模型缩小、数据 / 站表指到临时目录
    train_df = make_frame(n_days=2, issue_every_min=120, seed=0)
    train_df.insert(1, "station", train_df.pop("station_id").map(NAMES))
    train_df = train_df.rename(columns={"timesfm_predict": "timesfm3"})
    d = tmp_path / "data" / "train" / "date=2025-06-08"; d.mkdir(parents=True)
    train_df.to_parquet(d / "data_0.parquet", index=False)
    cfg = yaml.safe_load((ROOT / "forecast_backbones/m8/config/j1_final2.yaml").read_text(encoding="utf-8"))
    cfg["model"] = CFG_MODEL
    cfg["dataset"].update({"src_dir": str(tmp_path / "data"), "station_meta": str(_write_info_csv(tmp_path / "train_info.csv"))})
    cfg["train"]["schedule"].update({"batch_size": 8, "max_steps": 3, "warmup_steps": 1, "save_every": 0, "progress_bar": False})
    (tmp_path / "j1.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    run = tmp_path / "run"
    env = _cli_env(tmp_path)
    p = subprocess.run([sys.executable, str(CLI), "--config", str(tmp_path / "j1.yaml"), "--ckpt-dir", str(run)],
                       env=env, cwd=str(CLI.parent), capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]

    # 2) 线上模板第 74–85 行：读 model_config.yaml，覆盖权重路径与站表（线上那份 info.csv，表头自动识别）
    ck = tmp_path / "algo/checkpoints/j1_final2"; ck.mkdir(parents=True)
    for f in ("model_config.yaml", "model.safetensors"):
        (ck / f).write_bytes((run / "model_result" / f).read_bytes())
    config_m8 = yaml.safe_load(open(ck / "model_config.yaml", "r", encoding="utf-8"))
    config_m8["train"]["weights_path"] = str(ck / "model.safetensors")
    config_m8["dataset"]["station_meta"] = str(_write_info_csv(ck / "info.csv"))
    config_m8["dataset"]["station_meta_cols"] = None
    model_m8 = TransformersAdaptedM8Model(config_m8).eval()
    assert model_m8.weights_loaded and config_m8["dataset"]["known_stations"] == [1358, 6089]

    # 3) 线上模板第 159–191 行：tsfm_predict 改名、站名 → station_id、大写列名、unified_inference 同样的参数
    online = train_df.drop(columns=["observe_power_future"]).rename(columns={
        "general_ghi": "general_GHI", "general_ghi_predict": "general_GHI_predict", "general_temp": "general_TEMP",
        "general_temp_predict": "general_TEMP_predict", "timesfm3": "tsfm_predict"})
    for c in ("GHI", "TEMP"):                                                  # multi_infer.py 把 general_* 抄成 *_SOLARGIS
        online[f"{c}_SOLARGIS"], online[f"{c}_SOLARGIS_predict"] = online[f"general_{c}"], online[f"general_{c}_predict"]
    online["station_id"] = online["station"].map(model_m8.station_meta.id_of)
    online = online[online["station_id"].notna()].astype({"station_id": "int64"})
    feats = ["observe_power", "general_GHI", "general_GHI_predict", "general_TEMP", "general_TEMP_predict", "tsfm_predict"]
    ds_out = unified_inference(model_m8, online, timestamp_cols=("timestamp_win",),
                               entity_cols=("timestamp_win", "station", "station_id"), feature_cols=feats)
    assert ds_out["TransformersAdaptedM8Model"].notna().all() and len(ds_out) == len(train_df)
    pred_online = np.stack(ds_out["TransformersAdaptedM8Model"].to_numpy())
    assert pred_online.shape == (len(train_df), 192) and np.isfinite(pred_online).all()

    # 4) 与训练格式数据（小写列、timesfm3、只有站名）上的前向一致：列名 / 别名 / 站号 / 槽位两边对得上
    ds = UnifiedDataset(train_df, timestamp_cols=["timestamp_win"], entity_cols=["timestamp_win", "station"],
                        feature_cols=["observe_power", "general_ghi", "general_ghi_predict", "general_temp", "general_temp_predict", "timesfm3"],
                        label_cols=["observe_power_future"], for_hf_trainer=True)
    collate = partial(UnifiedDataset.collate_fn, timestamp_output="epoch_s", drop_str_type=False, for_hf_trainer=True)
    with torch.no_grad():
        pred_train = model_m8(collate([ds[i] for i in range(len(ds))])["batch_dict"]).logits.numpy()
    np.testing.assert_allclose(pred_online, pred_train, rtol=0, atol=1e-5)
    assert (pred_online.max(1) > 0).any()                                       # 不是全 0 的空预测

    # 5) 短期分支（predict_type='short'，模板第 192–212 行）：未来列 480 步，rolling_inference 按 192 + 192 + 96 滚三次。
    #    未来列用 192 步首尾拼成 480（与 rolling_wrapper_inference.py 自带自测同法），只验证这条路能跑通、输出完整
    from pipelines.rolling_wrapper_inference import rolling_inference
    short = online.copy()
    for c in ("general_GHI_predict", "general_TEMP_predict", "tsfm_predict"):
        short[c] = short[c].map(lambda a: np.concatenate([a, a, a[:96]]))
    rolling_config = {"past_covariate": ["general_GHI", "general_TEMP"], "future_covariate": ["general_GHI_predict", "general_TEMP_predict"],
                      "past_target": ["observe_power"], "future_target": ["tsfm_predict"], "future_len": 480}
    ds_short = rolling_inference(model_m8, short, config=rolling_config, timestamp_cols=("timestamp_win",),
                                 entity_cols=("timestamp_win", "station", "station_id"),
                                 feature_cols=["general_GHI", "general_GHI_predict", "general_TEMP", "general_TEMP_predict", "observe_power", "tsfm_predict"])
    assert len(ds_short) == len(short) and ds_short["TransformersAdaptedM8Model"].notna().all()
    pred_short = np.stack(ds_short["TransformersAdaptedM8Model"].map(np.asarray).to_numpy())
    assert pred_short.shape == (len(short), 480) and np.isfinite(pred_short).all()
    caps = short["station_id"].map(lambda s: model_m8.station_meta.get(int(s))["cap_mw"]).to_numpy()
    assert (pred_short >= -1e-6).all() and (pred_short <= caps[:, None] + 1e-4).all()  # 截在 [0, 容量]
    np.testing.assert_allclose(pred_short[:, :192], pred_online, rtol=0, atol=1e-4)     # 第一窗 = 超短期同一次前向


def test_station_slot_collision_is_an_error(small_cfg):
    """槽位 = 站号 % n_station_slots：南网 plantid 超过 4096，相差 4096 的两站会悄悄共用一个站点嵌入，训练开始前直接报错。"""
    from forecast_backbones.m8.transformer_models import TransformersAdaptedM8Model
    m = TransformersAdaptedM8Model(small_cfg)
    assert m.set_known_stations([1358, 6089, 1001]) == [1001, 1358, 6089]          # 6089 % 4096 = 1993，不冲突
    with pytest.raises(ValueError, match="n_station_slots"):
        m.set_known_stations([1358, 1358 + 4096])


def test_check_hint_units(tmp_path, small_cfg):
    """插值列与历史功率须同单位（MW）：容量归一的 timesfm3（0–1）对 MW 功率要报出来，同单位不报。"""
    from forecast_backbones.m8.data import check_hint_units
    cfg = load_config({**small_cfg, "features": {"hint_col": "timesfm3"}})
    fe = FeatureBuilder(cfg, StationMeta.from_csv(cfg["dataset"]["station_meta"]))
    df = make_frame(n_days=1, issue_every_min=240).rename(columns={"timesfm_predict": "timesfm3"})
    df.to_parquet(tmp_path / "mw.parquet", index=False)
    assert check_hint_units(fe, tmp_path / "mw.parquet") is None
    df["timesfm3"] = [a / 500.0 for a in df["timesfm3"]]                              # 容量归一（cap 500 MW）
    df.to_parquet(tmp_path / "pu.parquet", index=False)
    msg = check_hint_units(fe, tmp_path / "pu.parquet")
    assert msg and "同单位" in msg


def test_cli_does_not_import_nanwang_metric():
    src = CLI.read_text(encoding="utf-8")
    assert "nanwang_metric" not in src and "forecast_backbones.m8.train" not in src


def test_province_row_trains_and_infers_by_name(tmp_path, monkeypatch):
    """省级综合"站"（station = province_guangxi_solar）：训练数据只有站名、没有 station_id；info.csv 加一行
    plant_pointname = province_guangxi_solar（plantid 9001，经纬度留空 → 历史日形包络）。训练不报错、它进 known_stations，
    线上模板方式推理每行都有预测。"""
    from forecast_backbones.m8 import features
    from forecast_backbones.m8.transformer_models import TransformersAdaptedM8Model
    from pipelines.pipeline_wrapper_inference import unified_inference
    from .conftest import CAP, COORDS
    monkeypatch.setattr(features, "_WARNED_MISSING", set())
    info = tmp_path / "info.csv"
    info.write_text("plantid,plantname,plant_pointname,province,province_pointname,city,country,GCCAPACITY,LONGITUDE,LATITUDE\n"
                    f"1358,光伏1358,plant_guangfu1358,广西,province_guangxi,南宁,横州市,{CAP[1001]},{COORDS[1001][1]},{COORDS[1001][0]}\n"
                    f"9001,广西光伏合计,province_guangxi_solar,广西,province_guangxi,,,{CAP[1002]},,\n", encoding="utf-8")
    names = {1001: "plant_guangfu1358", 1002: "province_guangxi_solar"}
    df = make_frame(n_days=2, issue_every_min=120, seed=0)
    df.insert(1, "station", df.pop("station_id").map(names))                       # 只有站名，没有 station_id
    df = df.rename(columns={"timesfm_predict": "timesfm3"})
    d = tmp_path / "data" / "train" / "date=2025-06-08"; d.mkdir(parents=True)
    df.to_parquet(d / "data_0.parquet", index=False)
    cfg = yaml.safe_load((ROOT / "forecast_backbones/m8/config/j1_final2.yaml").read_text(encoding="utf-8"))
    cfg["model"] = CFG_MODEL
    cfg["dataset"].update({"src_dir": str(tmp_path / "data"), "station_meta": str(info)})
    cfg["train"]["schedule"].update({"batch_size": 8, "max_steps": 3, "warmup_steps": 1, "save_every": 0, "progress_bar": False})
    (tmp_path / "j1.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    run = tmp_path / "run"
    env = _cli_env(tmp_path)
    p = subprocess.run([sys.executable, str(CLI), "--config", str(tmp_path / "j1.yaml"), "--ckpt-dir", str(run)],
                       env=env, cwd=str(CLI.parent), capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]

    config_m8 = yaml.safe_load((run / "model_result" / "model_config.yaml").read_text(encoding="utf-8"))
    assert config_m8["dataset"]["known_stations"] == [1358, 9001]                 # 省级行按站名映射到 plantid 9001
    config_m8["train"]["weights_path"] = str(run / "model_result" / "model.safetensors")
    config_m8["dataset"]["station_meta"], config_m8["dataset"]["station_meta_cols"] = str(info), None
    m = TransformersAdaptedM8Model(config_m8).eval()
    assert m.weights_loaded and not m.station_meta.has_coords(9001)
    online = df[df["station"] == "province_guangxi_solar"].drop(columns=["observe_power_future"]).rename(
        columns={"timesfm3": "tsfm_predict", "general_ghi": "general_GHI", "general_ghi_predict": "general_GHI_predict",
                 "general_temp": "general_TEMP", "general_temp_predict": "general_TEMP_predict"})
    online["station_id"] = online["station"].map(m.station_meta.id_of)
    assert online["station_id"].notna().all()                                      # 线上模板第 165–169 行不会把它剔掉
    online = online.astype({"station_id": "int64"})
    out = unified_inference(m, online, timestamp_cols=("timestamp_win",), entity_cols=("timestamp_win", "station", "station_id"),
                            feature_cols=["observe_power", "general_GHI", "general_GHI_predict", "general_TEMP", "general_TEMP_predict", "tsfm_predict"])
    pred = np.stack(out["TransformersAdaptedM8Model"].to_numpy())
    assert out["TransformersAdaptedM8Model"].notna().all() and np.isfinite(pred).all() and pred.max() <= CAP[1002] + 1e-6
