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


VARIANTS = [dict(), dict(blocks=["attention"]), dict(blocks=[]), dict(patch_len=32), dict(patch_len=8, blocks=["deltanet"])]


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


def test_removed_switches_are_rejected_not_ignored(small_cfg):
    """瘦身（2026-10-14）删掉的开关：旧 yaml 再写它们必须报错，不能被 deep_merge 静默收下。"""
    for over in ({"model": {"mean_head": True}}, {"model": {"layout": "split"}}, {"model": {"head_mode": "token"}},
                 {"model": {"bidirectional": True}}, {"model": {"recent_skip": {"enabled": True}}}, {"model": {"ultra_head": {"enabled": True}}},
                 {"features": {"nwp_members": {"enabled": True}}}, {"train": {"loss": {"horizon_weights": {"ultra_weight": 3}}}},
                 {"train": {"loss": {"metric_aux": {"ultra_l1": 0.5}}}}, {"train": {"early_stop": {"combo_weights": {"short": 1}}}},
                 {"decision": {"point": {"by_step": [{"steps": [55, 151], "method": "mean_head"}]}}},
                 {"model": {"blocks": ["slstm"]}}):
        with pytest.raises(ValueError):
            load_config({**small_cfg, **over, "model": {**small_cfg["model"], **over.get("model", {})}})
    from forecast_backbones.m8.train import EARLY_STOP_METRICS
    assert "nanwang_combo" not in EARLY_STOP_METRICS


def test_compute_loss_path_matches_forward_loss(small_cfg, batch):
    m = _model(small_cfg); m.eval()                      # eval：无随机增强，两条路径应逐位一致
    with torch.no_grad():
        l1 = m(batch["batch_dict"], labels=batch["labels"]).loss
        out = m(batch["batch_dict"]); l2 = m.compute_loss(out, batch["labels"])
    assert out.loss is None
    assert torch.allclose(l1, l2, atol=1e-6)


def test_night_zero_and_physics_clip(small_cfg, batch):
    """night_zero 显式打开时夜间归零；默认关（与其它模型同口径）：aux['night'] 全 False、夜间预测不被强制为 0，
    包括晴空上限也不能在夜间（p_clear=0）把输出钳成 0；白天仍受 1.05·p_clear 与 cap 钳位。"""
    m = _model(small_cfg); m.eval()
    assert m.cfg["decision"]["physics"]["night_zero"] is False and m.cfg["train"]["loss"]["night_weight"] == 1.0
    with torch.no_grad():
        out = m(batch["batch_dict"])
    pred = out.logits.numpy(); night_true = out.aux["night_loss"]; pc = out.aux["p_clear"]
    assert not out.aux["night"].any() and night_true.any() and (pc[night_true] == 0).all()
    assert (pred >= 0).all() and (pred <= out.aux["cap"][:, None] * 1.0001).all()
    day = ~night_true
    assert (pred[day] <= 1.05 * pc[day] * (1 + 1e-5) + 1e-3).all()
    # 决策层直接喂一个夜间为正的分位阵：默认不归零（只受 cap 钳位），night_zero=True 才归零
    (B, H), K = pred.shape, len(m.levels)
    q_syn = np.full((B, H, K), 0.5, float) * out.aux["cap"][:, None, None]
    pt, *_ = m.decide(q_syn, night_true, pc, out.aux["cap"])
    assert np.allclose(pt[night_true], 0.5 * np.broadcast_to(out.aux["cap"][:, None], pt.shape)[night_true])
    assert (pt[day] <= np.minimum(1.05 * pc[day], np.broadcast_to(out.aux["cap"][:, None], pt.shape)[day]) + 1e-6).all()
    m.cfg["decision"]["physics"]["night_zero"] = True
    with torch.no_grad():
        out2 = m(batch["batch_dict"])
    assert out2.aux["night"].any() and (out2.logits.numpy()[out2.aux["night"]] == 0).all()
    assert np.allclose(out2.logits.numpy()[~night_true], pred[~night_true])          # 白天不受影响
    pt2, *_ = m.decide(q_syn, night_true, pc, out.aux["cap"])
    assert (pt2[night_true] == 0).all() and np.allclose(pt2[day], pt[day])
    # clip_negative：默认把负值截到 0；decision yaml 设 false 时负值原样进提交曲线（严格裸对比）
    import copy
    dec = copy.deepcopy(m.cfg["decision"]); dec["physics"]["night_zero"] = False
    q_neg = np.full((B, H, K), -0.01, float) * out.aux["cap"][:, None, None]
    p0, *_ = m.decide(q_neg, night_true, pc, out.aux["cap"], dec)
    assert (p0 == 0).all()
    dec["physics"]["clip_negative"] = False
    p1, *_ = m.decide(q_neg, night_true, pc, out.aux["cap"], dec)
    assert (p1 < 0).all()


def test_night_steps_enter_loss_by_default(small_cfg, batch):
    """night_weight=1（默认）时夜间步参与监督：改成 0（旧行为）损失变化；夜间全 0 的目标在归零后两者差异来自夜间项。"""
    import copy
    m = _model(small_cfg); m.eval()
    bd, labels = batch["batch_dict"], batch["labels"]
    with torch.no_grad():
        l_full = m(bd, labels=labels).loss
        m.cfg["train"]["loss"]["night_weight"] = 0.0
        l_day = m(bd, labels=labels).loss
        out = m(bd); l_day2 = m.compute_loss(out, labels)
    assert torch.isfinite(l_full) and torch.isfinite(l_day) and not torch.allclose(l_full, l_day)
    assert torch.allclose(l_day, l_day2, atol=1e-6)


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


@pytest.mark.parametrize("name", ["d3_timesfm_hint", "d4_timesfm_resid", "j1_final2"])
def test_timesfm_variant_files_train_step(small_cfg, batch, name):
    from forecast_backbones.m8.tests.conftest import ROOT
    cfg = load_config(str(ROOT / "forecast_backbones/m8/config/variants" / f"{name}.yaml"))
    cfg["dataset"]["station_meta"] = small_cfg["dataset"]["station_meta"]
    cfg["model"].update(small_cfg["model"])
    m = build_model(model_name="m8", model_config=cfg); m.train()
    bd = {**batch["batch_dict"], cfg["features"]["hint_col"]: batch["batch_dict"]["timesfm_predict"]}   # 合成数据的提示列改成变体声明的插值列名
    out = m(bd, labels=batch["labels"]); out.loss.backward()
    assert torch.isfinite(out.loss) and out.logits.shape == (batch["labels"]["observe_power_future"].shape[0], 192)


def test_j1_final2_stacks_d3_and_h4_on_e1_final():
    """j1_final2 = e1_final 调度 / 决策 + h4（晴空出力比 + 历史包络）+ d3（timesfm3 未来提示 + hint_dropout）。"""
    from forecast_backbones.m8.tests.conftest import ROOT
    v = ROOT / "forecast_backbones/m8/config/variants"
    j1, e1 = load_config(str(v / "j1_final2.yaml")), load_config(str(v / "e1_final.yaml"))
    assert j1["features"]["derived"]["clear_sky_ratio"]["enabled"] and j1["features"]["derived"]["history_envelope"]["enabled"]
    assert j1["features"]["hint_col"] == "timesfm3" and j1["train"]["hint_dropout"] == 0.3
    assert j1["train"]["schedule"] == e1["train"]["schedule"] and j1["train"]["early_stop"] == e1["train"]["early_stop"]
    assert j1["decision"]["point"] == e1["decision"]["point"] and j1["model"] == e1["model"]


def test_all_variant_configs_load(small_cfg):
    import glob
    from forecast_backbones.m8.tests.conftest import ROOT
    files = sorted(glob.glob(str(ROOT / "forecast_backbones/m8/config/variants/*.yaml")))
    assert len(files) >= 10
    tags = [load_config(f)["run"]["tag"] for f in files]
    assert len(set(tags)) == len(tags)


def test_set_known_stations_changes_unseen_prediction_only(small_cfg, batch):
    m = _model(small_cfg); m.eval()
    with torch.no_grad():
        a = m(batch["batch_dict"]).logits
        ids = m.set_known_stations([1002])
        b = m(batch["batch_dict"]).logits
    assert ids == [1002] and m.config["dataset"]["known_stations"] == [1002]
    sid = batch["batch_dict"]["station_id"].numpy().astype(int)
    assert torch.equal(a[sid == 1002], b[sid == 1002])                        # 已见站不变
    assert not torch.equal(a[sid == 1001], b[sid == 1001])                    # 未见站改走未知槽


def test_no_station_emb_variant_ignores_known_list(small_cfg, batch):
    cfg = {**small_cfg, "features": {"static": {"station_embedding": False}}}
    m = build_model(model_name="m8", model_config=cfg); m.eval()
    with torch.no_grad():
        a = m(batch["batch_dict"]).logits; m.set_known_stations([1002]); b = m(batch["batch_dict"]).logits
    assert torch.equal(a, b)
