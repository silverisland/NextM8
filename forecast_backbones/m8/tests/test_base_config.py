"""仓库默认配置 config/configs.yaml 必须能直接加载（train_m8.sh 不给 --config 时用它）。"""
from forecast_backbones.m8.config import load_config
from forecast_backbones.m8.tests.conftest import ROOT


def test_base_configs_yaml_loads():
    cfg = load_config(str(ROOT / "forecast_backbones/m8/config/configs.yaml"))
    assert cfg["features"]["hint_col"] is None and cfg["features"]["future_only"] == []
