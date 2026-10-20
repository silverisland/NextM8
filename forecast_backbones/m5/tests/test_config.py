from pathlib import Path

import pytest
import yaml

M5 = Path(__file__).resolve().parents[1]
ROOT = M5.parents[1]


def test_nanwang_keys_default_off():
    """2026-10-20 新键（同 M8）：默认全关，家里的旧配置行为不变。"""
    from forecast_backbones.m5.config import load_config
    c = load_config(None)
    assert c["dataset"]["src_dir"] is None and c["features"]["col_aliases"] == {} and c["features"]["cov_present"] is None
    assert c["train"]["schedule"]["save_every"] is None
    assert not (M5 / "config" / "finetune.yaml").exists() and not (M5 / "config" / "variants").exists()   # 配置平铺在 config/ 下


def test_defaults_match_spec():
    from forecast_backbones.m5.config import load_config
    c = load_config(None)
    assert c["features"]["derived"] == {"solar_elevation": False, "clear_sky_power": False,
                                        "history_envelope": False, "envelope_nwp": False}
    assert c["features"]["past_future"] == [["general_ghi", "general_ghi_predict"], ["general_temp", "general_temp_predict"]]
    assert c["decision"]["physics"] == {"night_zero": False, "clear_sky_cap_ratio": None, "capacity_cap": True, "clip_negative": True}
    assert c["decision"]["point"]["default"] == "median"
    assert c["train"]["loss"] == {"night_weight": 1.0, "normalize_by_cap": True}
    assert c["train"]["schedule"]["precision"] == "fp32" and c["train"]["schedule"]["lr"] == 1.0e-4
    assert c["finetune"]["enabled"] is True and c["finetune"]["method"] == "lora"
    assert c["finetune"]["lora"] == {"rank": 8, "alpha": 16, "dropout": 0.05,
                                     "target_modules": ["query_proj", "key_proj", "value_proj", "out_proj"],
                                     "experts": 1, "gate": False}
    # LoRA 之外的适配器 / 协议旋钮默认全关
    ft = c["finetune"]
    assert ft["ia3"]["target_modules"] == [] and ft["prompt"]["n_tokens"] == 0 and ft["static_embed"]["enabled"] is False
    assert ft["resblock_adapter"]["rank"] == 0 and ft["init_from"] is None
    assert c["train"]["schedule"]["lr_scheduler"] == "cosine"
    assert c["train"]["optim"] == {"lora_plus_ratio": None, "head_lr": None, "adapter_lr": None}
    assert c["train"]["weights_path"] == "runs/m5_baseline/model.safetensors"
    sm = Path(c["dataset"]["station_meta"])
    assert sm.parts[:2] == ("forecast_backbones", "m5") and (ROOT / sm).exists()


@pytest.mark.parametrize("path,hint", [
    (("finetune", "schedule"), "train.schedule"),
    (("finetune", "lora_path"), "train.weights_path"),
    (("timesfm3", "context_len"), "dataset.context_len"),
    (("timesfm3", "past_future_cols"), "features.past_future"),
    (("calibration",), "conformal"),
])
def test_removed_keys_raise_with_new_location(path, hint):
    from forecast_backbones.m5.config import load_config
    over = {}
    d = over
    for k in path[:-1]:
        d = d.setdefault(k, {})
    d[path[-1]] = {"enabled": True} if path == ("calibration",) else 1
    with pytest.raises(ValueError, match=hint):
        load_config(over)


def test_validation_errors():
    from forecast_backbones.m5.config import load_config
    with pytest.raises(ValueError, match="method"):
        load_config({"finetune": {"method": "dora"}})
    with pytest.raises(ValueError, match="inference_path"):
        load_config({"finetune": {"inference_path": "jax"}})
    with pytest.raises(ValueError, match="rank"):
        load_config({"finetune": {"lora": {"rank": 0}}})
    with pytest.raises(ValueError, match="二选一"):
        load_config({"finetune": {"lora": {"gate": True, "experts": 2}}})
    with pytest.raises(ValueError, match="experts"):
        load_config({"finetune": {"lora": {"experts": 0}}})
    with pytest.raises(ValueError, match="lr_scheduler"):
        load_config({"train": {"schedule": {"lr_scheduler": "onecycle"}}})
    with pytest.raises(ValueError, match="clear_sky_cap_ratio"):
        load_config({"decision": {"physics": {"clear_sky_cap_ratio": 0}}})
    with pytest.raises(ValueError, match="station_meta"):
        load_config({"dataset": {"station_meta": None}})


F_SERIES = ["f1_final", "f2_lr5e5", "f3_lr2e4", "f4_drop01", "f5_wd01"]


@pytest.mark.parametrize("name,check", [
    ("z0_zeroshot", lambda c: c["finetune"]["enabled"] is False and c["finetune"]["inference_path"] == "torch"),
    # f 系列：s7_fh_const（冻结 head + 常数 lr）+ p*，每个只差一处超参
    ("f1_final", lambda c: c["finetune"]["unfreeze_output_head"] is False and c["train"]["schedule"]["lr_scheduler"] == "constant"
                           and c["train"]["schedule"]["lr"] == 1e-4 and c["train"]["ema_decay"] is None
                           and c["decision"]["point"]["default"] == "p_star"
                           and c["finetune"]["lora"]["target_modules"] == ["query_proj", "key_proj", "value_proj", "out_proj"]),
    ("f2_lr5e5", lambda c: c["train"]["schedule"]["lr"] == 5e-5),
    ("f3_lr2e4", lambda c: c["train"]["schedule"]["lr"] == 2e-4),
    ("f4_drop01", lambda c: c["finetune"]["lora"]["dropout"] == 0.1),
    ("f5_wd01", lambda c: c["train"]["schedule"]["weight_decay"] == 0.1),
    ("s3_c1_envelope", lambda c: c["features"]["derived"]["history_envelope"] and not c["features"]["derived"]["envelope_nwp"]
                                 and c["finetune"]["enabled"] and c["train"]["schedule"]["lr_scheduler"] == "constant"),
    ("s3_c1_envelope_zs", lambda c: c["features"]["derived"]["history_envelope"] and c["finetune"]["enabled"] is False),
])

def test_variants_load(name, check):
    from forecast_backbones.m5.config import load_config
    assert check(load_config(str(M5 / "config" / f"{name}.yaml")))


def _diff(a, b, path=()):
    out = []
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        if isinstance(x, dict) and isinstance(y, dict):
            out += _diff(x, y, path + (k,))
        elif x != y:
            out.append(".".join(path + (k,)))
    return out


@pytest.mark.parametrize("name,changed", [("f2_lr5e5", ["train.schedule.lr"]), ("f3_lr2e4", ["train.schedule.lr"]),
                                          ("f4_drop01", ["finetune.lora.dropout"]), ("f5_wd01", ["train.schedule.weight_decay"])])
def test_f_series_is_same_model_one_hyperparam_apart(name, changed):
    """f2–f5 与 f1_final 是同一个模型：除 run 段外只差一处超参。"""
    from forecast_backbones.m5.config import load_config
    a, b = load_config(str(M5 / "config" / "f1_final.yaml")), load_config(str(M5 / "config" / f"{name}.yaml"))
    assert [d for d in _diff(a, b) if not d.startswith(("run.", "eval.", "train.weights_path"))] == changed


def test_decision_yaml_only_touches_decision():
    from forecast_backbones.m5.config import load_config
    raw = yaml.safe_load((M5 / "config" / "decision" / "d2_pstar_point.yaml").read_text(encoding="utf-8"))
    assert list(raw) == ["decision"] and load_config(raw)["decision"]["point"]["default"] == "p_star"


def test_configs_yaml_stays_zero_shot():
    raw = yaml.safe_load((M5 / "config" / "configs.yaml").read_text(encoding="utf-8"))
    assert "station_meta" not in (raw.get("dataset") or {}) and "finetune" not in raw


def test_every_variant_yaml_is_covered_by_a_test():
    names = {p.stem for p in (M5 / "config").glob("*.yaml")} - {"configs"}       # configs.yaml = 线上零样本（legacy）配置
    covered = {"z0_zeroshot", *F_SERIES, "s3_c1_envelope", "s3_c1_envelope_zs"}
    assert names == covered, names ^ covered
