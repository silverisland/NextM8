from pathlib import Path

import pytest
import yaml

M5 = Path(__file__).resolve().parents[1]
ROOT = M5.parents[1]


def test_finetune_yaml_is_default_cfg_on_disk():
    from forecast_backbones.m5.config import load_config
    assert load_config(str(M5 / "config" / "finetune.yaml")) == load_config(None)


def test_defaults_match_spec():
    from forecast_backbones.m5.config import load_config
    c = load_config(None)
    assert c["features"]["derived"] == {"solar_elevation": False, "clear_sky_power": False}
    assert c["features"]["past_future"] == [["general_ghi", "general_ghi_predict"], ["general_temp", "general_temp_predict"]]
    assert c["decision"]["physics"] == {"night_zero": False, "clear_sky_cap_ratio": None, "capacity_cap": True, "clip_negative": True}
    assert c["decision"]["point"]["default"] == "median"
    assert c["train"]["loss"] == {"night_weight": 1.0, "normalize_by_cap": True}
    assert c["train"]["schedule"]["precision"] == "fp32" and c["train"]["schedule"]["lr"] == 1.0e-4
    assert c["finetune"]["enabled"] is True and c["finetune"]["method"] == "lora"
    assert c["finetune"]["lora"] == {"rank": 8, "alpha": 16, "dropout": 0.05,
                                     "target_modules": ["query_proj", "key_proj", "value_proj", "out_proj"]}
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
    with pytest.raises(ValueError, match="clear_sky_cap_ratio"):
        load_config({"decision": {"physics": {"clear_sky_cap_ratio": 0}}})
    with pytest.raises(ValueError, match="station_meta"):
        load_config({"dataset": {"station_meta": None}})


@pytest.mark.parametrize("name,check", [
    ("z0_zeroshot", lambda c: c["finetune"]["enabled"] is False and c["finetune"]["inference_path"] == "torch"),
    ("l1_lora_r8", lambda c: c["finetune"]["enabled"] is True and c["finetune"]["lora"]["rank"] == 8),
    ("d2_pstar_point", lambda c: c["decision"]["point"]["default"] == "p_star"),
    ("g1_solar_cov", lambda c: c["features"]["derived"] == {"solar_elevation": True, "clear_sky_power": True}),
    ("g2_cs_ceiling", lambda c: c["decision"]["physics"]["clear_sky_cap_ratio"] == 1.05),
    # 第 2 阶段：训练协议 + 收益定位（与 l1 同协议，只差一处）
    ("s2_lr3e5_ema", lambda c: c["train"]["schedule"]["lr"] == 3e-5 and c["train"]["ema_decay"] == 0.998
                               and c["train"]["early_stop"]["smooth_k"] == 3),
    ("s2_lr1e4_ema", lambda c: c["train"]["schedule"]["lr"] == 1e-4 and c["train"]["ema_decay"] == 0.998
                               and c["train"]["early_stop"]["smooth_k"] == 3),
    ("s2_head_only", lambda c: c["finetune"]["lora"]["target_modules"] == [] and c["finetune"]["unfreeze_output_head"]),
    ("s2_frozen_head", lambda c: c["finetune"]["unfreeze_output_head"] is False
                                 and c["finetune"]["lora"]["target_modules"] == ["query_proj", "key_proj", "value_proj", "out_proj"]),
    ("s2_lora_resblock", lambda c: c["finetune"]["lora"]["target_modules"][4:] == ["hidden_layer", "output_layer", "residual_layer"]),
    ("s2_probe_resblock", lambda c: c["finetune"]["lora"]["target_modules"] == []
                                    and c["finetune"]["unfreeze_modules"] == ["pre_transformer_resblock"]),
    ("s2_lora_var_resblock", lambda c: c["finetune"]["lora"]["target_modules"][0].startswith("re:")
                                       and "var_attn" in c["finetune"]["lora"]["target_modules"][0]),
    ("s2_lora_var_only", lambda c: c["finetune"]["lora"]["target_modules"] == [r"re:.*\.var_attn\.(query|key|value|out)_proj"]),
    ("s2_lora_seq_only", lambda c: c["finetune"]["lora"]["target_modules"] == [r"re:.*\.seq_attn\.(query|key|value|out)_proj"]),
    ("s2_lr1e5_ema", lambda c: c["train"]["schedule"]["lr"] == 1e-5 and c["train"]["ema_decay"] == 0.998),
    ("s6_all_linear", lambda c: {"ff0", "ff1", "hidden_layer"} <= set(c["finetune"]["lora"]["target_modules"])),
    ("s6_lora_vo_only", lambda c: c["finetune"]["lora"]["target_modules"] == ["value_proj", "out_proj"]),
    ("s6_full_ft_lr1e6", lambda c: c["finetune"]["lora"]["target_modules"] == [] and c["train"]["schedule"]["lr"] == 1e-6
                                   and c["finetune"]["unfreeze_modules"] == ["pre_transformer_resblock", "transformer_stack"]),
    ("s6_full_ft_lr1e5", lambda c: c["train"]["schedule"]["lr"] == 1e-5 and "transformer_stack" in c["finetune"]["unfreeze_modules"]),
    ("s6_partial_ends", lambda c: "transformer_stack.layers.19" in c["finetune"]["unfreeze_modules"]
                                  and "transformer_stack" not in c["finetune"]["unfreeze_modules"]),
    ("s3_solargis_cov", lambda c: c["features"]["past_future"][2:] == [["GHI_solargis", "ghi_solargis_predict"],
                                                                        ["temp_solargis", "temp_solargis_predict"]]
                                  and c["finetune"]["lora"]["target_modules"] == ["query_proj", "key_proj", "value_proj", "out_proj"]),
    ("s3_cov_all_var", lambda c: len(c["features"]["past_future"]) == 4 and c["features"]["derived"]["clear_sky_power"]
                                 and c["finetune"]["lora"]["target_modules"][0].startswith("re:")),
    ("s5_rank4", lambda c: (c["finetune"]["lora"]["rank"], c["finetune"]["lora"]["alpha"]) == (4, 8)),
    ("s5_rank16", lambda c: (c["finetune"]["lora"]["rank"], c["finetune"]["lora"]["alpha"]) == (16, 32)),
    ("s5_rank32", lambda c: (c["finetune"]["lora"]["rank"], c["finetune"]["lora"]["alpha"]) == (32, 64)),
])

def test_variants_load(name, check):
    from forecast_backbones.m5.config import load_config
    assert check(load_config(str(M5 / "config" / "variants" / f"{name}.yaml")))


def test_configs_yaml_stays_zero_shot():
    raw = yaml.safe_load((M5 / "config" / "configs.yaml").read_text(encoding="utf-8"))
    assert "station_meta" not in (raw.get("dataset") or {}) and "finetune" not in raw


def test_every_variant_yaml_is_covered_by_a_test():
    names = {p.stem for p in (M5 / "config" / "variants").glob("*.yaml")}
    covered = {"z0_zeroshot", "l1_lora_r8", "d2_pstar_point", "g1_solar_cov", "g2_cs_ceiling", "s2_lr3e5_ema", "s2_lr1e4_ema",
               "s2_lr1e5_ema", "s2_head_only", "s2_frozen_head", "s2_lora_resblock", "s2_probe_resblock", "s2_lora_var_resblock",
               "s2_lora_var_only", "s2_lora_seq_only", "s6_all_linear", "s6_lora_vo_only",
               "s6_full_ft_lr1e6", "s6_full_ft_lr1e5", "s6_partial_ends",
               "s3_solargis_cov", "s3_cov_all_var", "s5_rank4", "s5_rank16", "s5_rank32"}
    assert names == covered, names ^ covered
