from pathlib import Path

import pytest

from forecast_backbones.m6.config import DEFAULT_CFG, deep_merge, interpolate, load_config


def test_deep_merge_only_overrides_present_keys():
    out = deep_merge({"a": {"x": 1, "y": 2}, "b": 3}, {"a": {"y": 9}})
    assert out == {"a": {"x": 1, "y": 9}, "b": 3}

def test_deep_merge_replaces_lists_whole():
    out = deep_merge({"a": [1, 2, 3]}, {"a": [9]})
    assert out["a"] == [9]

def test_interpolate_out_dir():
    cfg = {"run": {"out_dir": "runs/x"}, "eval": {"report_dir": "${run.out_dir}/reports"},
           "calibration": {"table_path": "${run.out_dir}/calibration.parquet"}}
    out = interpolate(cfg)
    assert out["eval"]["report_dir"] == "runs/x/reports"
    assert out["calibration"]["table_path"] == "runs/x/calibration.parquet"

def test_load_config_none_gives_defaults():
    cfg = load_config(None)
    assert cfg["timesfm3"]["context_len"] == 672
    assert cfg["decision"]["point"]["default"] == "p_star"
    assert cfg["calibration"]["levels"][0] == 0.01

def test_load_config_yaml_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("timesfm3:\n  context_len: 96\nrun:\n  out_dir: runs/t\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg["timesfm3"]["context_len"] == 96
    assert cfg["timesfm3"]["pred_len"] == 192
    assert cfg["eval"]["report_dir"] == "runs/t/reports"

def test_yaml_matches_defaults():
    p = Path(__file__).resolve().parents[1] / "config" / "configs.yaml"
    cfg = load_config(p)
    assert cfg == interpolate(DEFAULT_CFG)

def test_load_config_accepts_path_object_and_str_equally(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("timesfm3:\n  context_len: 96\nrun:\n  out_dir: runs/t\n", encoding="utf-8")
    cfg_from_path = load_config(p)
    cfg_from_str = load_config(str(p))
    assert cfg_from_path == cfg_from_str

def test_finetune_defaults_use_real_timesfm3_module_names():
    from forecast_backbones.m6.config import load_config
    ft = load_config(None)["finetune"]
    assert ft["lora"]["target_modules"] == ["query_proj", "key_proj", "value_proj", "out_proj"]
    assert ft["lora_path"].endswith("/lora_weights.pt") and "${" not in ft["lora_path"]
    assert ft["routing_path"].endswith("/station_routing.json")
    assert ft["split"]["train_start"] is None
    assert ft["schedule"]["logging_steps"] == 20
    assert ft["guards"]["min_gain_pct"] == 0.0
    assert ft["inference_path"] == "evaluator" and ft["enabled"] is False


def test_split_defaults_are_multi_block_with_purge_and_embargo():
    sp = load_config(None)["finetune"]["split"]
    assert sp["train_start"] is None and sp["test_end"] is None
    assert sp["val_blocks"] == [["2025-01-10", "2025-01-31"], ["2025-04-10", "2025-05-01"], ["2025-07-16", "2025-08-06"]]
    assert sp["test_start"] == "2025-10-01" and sp["cal_days"] == 45
    assert sp["purge_days"] == 2 and sp["embargo_days"] == 7
    assert not {"train_end", "val_start", "val_end", "rolling_folds"} & set(sp)


def test_load_config_rejects_legacy_split_keys():
    with pytest.raises(ValueError) as e:
        load_config({"finetune": {"split": {"train_end": "2025-05-31", "val_start": "2025-07-16", "val_end": "2025-08-31"}}})
    msg = str(e.value)
    assert "train_end" in msg and "val_start" in msg and "val_blocks" in msg


def test_yaml_split_section_matches_defaults():
    from pathlib import Path
    import yaml
    y = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "configs.yaml").read_text(encoding="utf-8"))
    assert y["finetune"]["split"] == load_config(None)["finetune"]["split"]


def test_default_split_is_self_consistent():
    from pipelines.time_splits import SplitPlan
    plan = SplitPlan.from_dict(load_config(None)["finetune"]["split"])
    assert plan.to_dict()["cal"] == ["2025-08-15", "2025-09-29"]
    assert plan.coverage_note is None
