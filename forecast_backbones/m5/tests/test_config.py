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
])
def test_variants_load(name, check):
    from forecast_backbones.m5.config import load_config
    assert check(load_config(str(M5 / "config" / "variants" / f"{name}.yaml")))


def test_configs_yaml_stays_zero_shot():
    raw = yaml.safe_load((M5 / "config" / "configs.yaml").read_text(encoding="utf-8"))
    assert "station_meta" not in (raw.get("dataset") or {}) and "finetune" not in raw
