"""插值列 features.hint_col（2026-10-15）：TimesFM 之类只有未来段的单变量预测单独声明，列名随数据改，与 NWP 协变量分开。"""
import numpy as np
import pytest
import torch
from forecast_backbones.m8.config import load_config
from forecast_backbones.m8.features import FeatureBuilder
from forecast_backbones.m8.solar import StationMeta
from forecast_backbones.m8.transformer_models import TransformersAdaptedM8Model
from forecast_backbones.m8.tests.conftest import ROOT

V = ROOT / "forecast_backbones/m8/config/variants"


def _fb(small_cfg, features):
    cfg = load_config({**small_cfg, "features": features})
    return cfg, FeatureBuilder(cfg, StationMeta.from_csv(cfg["dataset"]["station_meta"]))


def _renamed(batch, new):
    bd = dict(batch["batch_dict"]); bd[new] = bd.pop("timesfm_predict")
    return bd


def test_hint_col_any_name_same_input_as_future_only(small_cfg, batch):
    """hint_col 写什么列名都行；喂进模型的通道与旧写法 future_only=[该列] 逐值相同（换名不换数）。"""
    _, old = _fb(small_cfg, {"future_only": [{"col": "timesfm_predict", "scale": "target"}], "residual_base_col": "timesfm_predict"})
    _, new = _fb(small_cfg, {"hint_col": "我的插值列", "hint_residual": True})
    assert new.resid_col == "我的插值列" and len(new.channel_names) == len(old.channel_names)
    a = old.build(batch["batch_dict"], training=False)
    b = new.build(_renamed(batch, "我的插值列"), training=False)
    np.testing.assert_array_equal(a["x"], b["x"]); np.testing.assert_array_equal(a["base"], b["base"])
    assert (b["x"][:, :, new.idx["hint_missing"]] == 0).all()


def test_hint_col_is_not_a_covariate(small_cfg):
    _, fb = _fb(small_cfg, {"hint_col": "tfm"})
    assert fb.idx["fo:tfm"] not in fb.nwp_channels                       # 不随 NWP dropout 清零，走 hint_dropout


def test_hint_col_missing_in_training_raises(small_cfg, batch):
    """训练数据里没有插值列 → 直接报错（以前只 warning 一次、通道置 0，模型白训）。"""
    _, fb = _fb(small_cfg, {"hint_col": "tfm"})
    with pytest.raises(ValueError, match="tfm"):
        fb.build(batch["batch_dict"], training=True, y_future=batch["labels"]["observe_power_future"].numpy(),
                 rng=np.random.default_rng(0))


def test_hint_col_missing_at_inference_falls_back(small_cfg, batch):
    """推理时缺列仍按设计兜底：通道 0、hint_missing=1（训练时 hint_dropout 就是为这个准备的）。"""
    _, fb = _fb(small_cfg, {"hint_col": "tfm", "hint_residual": True})
    with pytest.warns(UserWarning, match="tfm"):
        inp = fb.build(batch["batch_dict"], training=False)
    assert (inp["x"][:, :, fb.idx["hint_missing"]] == 1).all() and (inp["base"] == 0).all()


@pytest.mark.parametrize("features", [
    {"hint_residual": True},                                                            # 残差但没给插值列
    {"hint_col": "tfm", "hint_residual": True, "future_only": ["x_pred"], "residual_base_col": "x_pred"},  # 两种残差写法同时给
    {"hint_col": "tfm", "future_only": ["tfm"]},                                        # 插值列又写进 future_only
    {"hint_col": "general_ghi_predict"},                                                # 插值列与 NWP 协变量同名
])
def test_hint_col_bad_config_rejected(small_cfg, features):
    with pytest.raises(ValueError):
        load_config({**small_cfg, "features": features})


@pytest.mark.parametrize("name", ["j1_final2", "k1_hint01", "k2_hint00",
                                  "k3_resid", "k4_lr2e4", "k5_lr5e4", "k6_regular"])
def test_variants_declare_hint_col(name):
    fe = load_config(str(V / f"{name}.yaml"))["features"]
    assert fe["hint_col"] == "timesfm3" and fe["future_only"] == [] and fe["residual_base_col"] is None
    assert fe["hint_residual"] == (name == "k3_resid")


def test_old_run_config_loads_into_new_style_model(small_cfg, batch):
    """旧 run 目录的 model_config.yaml（future_only + residual_base_col）照常能用；新旧写法权重形状一致、可互相加载、输出相同。"""
    old_cfg = {**small_cfg, "features": {"future_only": [{"col": "timesfm_predict", "scale": "target"}], "residual_base_col": "timesfm_predict"}}
    new_cfg = {**small_cfg, "features": {"hint_col": "tfm", "hint_residual": True}}
    torch.manual_seed(0)
    m_old = TransformersAdaptedM8Model(old_cfg); m_old.eval()
    m_new = TransformersAdaptedM8Model(new_cfg); m_new.load_state_dict(m_old.state_dict()); m_new.eval()
    with torch.no_grad():
        a = m_old(batch["batch_dict"]).logits
        b = m_new(_renamed(batch, "tfm")).logits
    assert torch.allclose(a, b)
