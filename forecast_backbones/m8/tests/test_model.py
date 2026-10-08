import numpy as np
import pytest
import torch
from pipelines.model_registry import build_model
from forecast_backbones.m8.config import load_config
from forecast_backbones.m8.transformer_models import TransformersAdaptedM8Model, save_weights, load_weights
from forecast_backbones.m8.transformer_trainer import evaluate_model


def _model(small_cfg, **model_over):
    cfg = {**small_cfg, "model": {**small_cfg["model"], **model_over}}
    return build_model(model_name="m8", model_config=cfg)


def test_registry_discovers_m8(small_cfg):
    m = _model(small_cfg)
    assert isinstance(m, TransformersAdaptedM8Model) and isinstance(m.config, dict)
    assert m.compute_metrics(None) == {}


VARIANTS = [
    dict(), dict(blocks=["slstm"]), dict(blocks=["attention"]), dict(blocks=[]),
    dict(layout="split"), dict(layout="split", bidirectional=True, blocks=["conv", "slstm"]),
    dict(head_mode="token"), dict(patch_len=32), dict(patch_len=8, blocks=["deltanet"]),
]


@pytest.mark.parametrize("over", VARIANTS)
def test_forward_backward_variants(small_cfg, batch, over):
    torch.manual_seed(0)
    m = _model(small_cfg, **over); m.train()
    out = m(batch["batch_dict"], labels=batch["labels"])
    assert out.loss is not None and torch.isfinite(out.loss)
    out.loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    B = batch["labels"]["observe_power_future"].shape[0]
    assert out.logits.shape == (B, 192) and out.aux["quantiles"].shape == (B, 192, 9)
    assert out.aux["quantiles"].requires_grad


def test_compute_loss_path_matches_forward_loss(small_cfg, batch):
    m = _model(small_cfg); m.eval()                      # eval：无随机增强，两条路径应逐位一致
    with torch.no_grad():
        l1 = m(batch["batch_dict"], labels=batch["labels"]).loss
        out = m(batch["batch_dict"]); l2 = m.compute_loss(out, batch["labels"])
    assert out.loss is None
    assert torch.allclose(l1, l2, atol=1e-6)


def test_night_zero_and_physics_clip(small_cfg, batch):
    m = _model(small_cfg); m.eval()
    with torch.no_grad():
        out = m(batch["batch_dict"])
    night = out.aux["night"]; pred = out.logits.numpy()
    assert (pred[night] == 0).all()
    assert (pred >= 0).all() and (pred <= out.aux["cap"][:, None] * 1.0001).all()
    assert (pred <= 1.05 * out.aux["p_clear"] * (1 + 1e-5) + 1e-3).all()


def test_eval_does_not_read_future_truth(small_cfg, batch):
    m = _model(small_cfg); m.eval()
    bd = dict(batch["batch_dict"])
    with torch.no_grad():
        a = m(bd).logits
        bd2 = dict(bd); bd2["observe_power_future"] = torch.full((a.shape[0], 192), 1e6)
        b = m(bd2).logits
    assert torch.equal(a, b)


def test_save_load_roundtrip(small_cfg, batch, tmp_path):
    m = _model(small_cfg); m.eval()
    p = tmp_path / "w.safetensors"; save_weights(m, p)
    m2 = _model(small_cfg); load_weights(m2, p); m2.eval()
    with torch.no_grad():
        assert torch.allclose(m(batch["batch_dict"]).logits, m2(batch["batch_dict"]).logits)
    cfg = {**small_cfg, "train": {"weights_path": str(p)}}
    m3 = TransformersAdaptedM8Model(cfg)
    assert m3.weights_loaded


def test_evaluate_model_schema(small_cfg, dataset, collate):
    m = _model(small_cfg)
    res = evaluate_model(m, dataset, collate, m.cfg)
    assert len(res) == len(dataset)
    for c in ["timestamp_win", "station", "station_id", "cap", "observe_power_predict", "observe_power_future",
              "p_star", "median", "night", "elev_deg", "p_clear", "var_bin", "var_signal", "degraded", "q_0.1", "q_0.9"]:
        assert c in res.columns, c
    assert res["observe_power_predict"].iloc[0].shape == (192,)


def test_bad_config_rejected(small_cfg):
    with pytest.raises(ValueError):
        load_config({**small_cfg, "model": {**small_cfg["model"], "patch_len": 10}})
    with pytest.raises(ValueError):
        load_config({**small_cfg, "model": {**small_cfg["model"], "blocks": ["mamba"]}})


def test_residual_variant_is_identity_shift_of_hint(small_cfg, batch):
    """残差模式与提示模式共用权重时，分位输出恰好相差 TimesFM 提示值（真实单位）。"""
    fo = {"future_only": [{"col": "timesfm_predict", "scale": "target"}]}
    plain_cfg = {**small_cfg, "features": fo, "train": {"hint_dropout": 0.0}}
    torch.manual_seed(0)
    m_plain = build_model(model_name="m8", model_config=plain_cfg); m_plain.eval()
    m_res = build_model(model_name="m8", model_config={**plain_cfg, "features": {**fo, "residual_base_col": "timesfm_predict"}})
    m_res.load_state_dict(m_plain.state_dict()); m_res.eval()
    with torch.no_grad():
        a = m_plain(batch["batch_dict"]).aux["quantiles"]; b = m_res(batch["batch_dict"]).aux["quantiles"]
    tf = batch["batch_dict"]["timesfm_predict"][:, :192].float()
    assert torch.allclose(b, a + tf[:, :, None], atol=1e-3)
    m_res.train()
    out = m_res(batch["batch_dict"], labels=batch["labels"]); out.loss.backward()
    assert torch.isfinite(out.loss)


@pytest.mark.parametrize("name", ["d3_timesfm_hint", "d4_timesfm_resid"])
def test_timesfm_variant_files_train_step(small_cfg, batch, name):
    from forecast_backbones.m8.tests.conftest import ROOT
    cfg = load_config(str(ROOT / "forecast_backbones/m8/config/variants" / f"{name}.yaml"))
    cfg["dataset"]["station_meta"] = small_cfg["dataset"]["station_meta"]
    cfg["model"].update(small_cfg["model"])
    m = build_model(model_name="m8", model_config=cfg); m.train()
    out = m(batch["batch_dict"], labels=batch["labels"]); out.loss.backward()
    assert torch.isfinite(out.loss) and out.logits.shape == (batch["labels"]["observe_power_future"].shape[0], 192)


def test_all_variant_configs_load(small_cfg):
    import glob
    from forecast_backbones.m8.tests.conftest import ROOT
    files = sorted(glob.glob(str(ROOT / "forecast_backbones/m8/config/variants/*.yaml")))
    assert len(files) >= 21
    tags = [load_config(f)["run"]["tag"] for f in files]
    assert len(set(tags)) == len(tags)
