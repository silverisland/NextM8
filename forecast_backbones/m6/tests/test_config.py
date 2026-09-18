from pathlib import Path

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
    p = Path(__file__).resolve().parents[1] / "model_config.yaml"
    cfg = load_config(p)
    assert cfg == interpolate(DEFAULT_CFG)

def test_load_config_accepts_path_object_and_str_equally(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("timesfm3:\n  context_len: 96\nrun:\n  out_dir: runs/t\n", encoding="utf-8")
    cfg_from_path = load_config(p)
    cfg_from_str = load_config(str(p))
    assert cfg_from_path == cfg_from_str
