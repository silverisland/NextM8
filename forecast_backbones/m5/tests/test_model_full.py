import numpy as np
import pytest
import torch

pytest.importorskip("timesfm3")
from forecast_backbones.m5.transformer_models import TransformersAdaptedM5Model, save_weights
from forecast_backbones.m5.lora import LoRALinear
from .conftest import full_cfg, write_station_meta

L, H = 672, 192


def _model(tmp_path, ckpt, **over):
    return TransformersAdaptedM5Model(full_cfg(tmp_path, ckpt, **over))


def test_lora_at_init_equals_zero_shot(tmp_path, tiny_ckpt_dir, batch):
    bd = batch["batch_dict"]
    lo = _model(tmp_path, tiny_ckpt_dir)
    zs = _model(tmp_path, tiny_ckpt_dir, finetune={"enabled": False})
    assert any(isinstance(x, LoRALinear) for x in lo.backbone.modules())
    assert not any(isinstance(x, LoRALinear) for x in zs.backbone.modules())
    with torch.no_grad():
        a, b = lo(bd), zs(bd)
    assert torch.allclose(a.aux["quantiles"], b.aux["quantiles"], atol=1e-6)
    assert torch.allclose(a.logits, b.logits, atol=1e-5)
    assert zs.n_params() == 0 and 0 < lo.n_params() < lo.n_total_params()


def test_torch_and_evaluator_zero_shot_agree(tmp_path, tiny_ckpt_dir, batch):
    bd = batch["batch_dict"]
    a = _model(tmp_path, tiny_ckpt_dir, finetune={"enabled": False, "inference_path": "evaluator"})
    b = _model(tmp_path, tiny_ckpt_dir, finetune={"enabled": False, "inference_path": "torch"})
    assert a.backbone is None and b.backbone is not None
    with torch.no_grad():
        assert torch.allclose(a(bd).aux["quantiles"], b(bd).aux["quantiles"], atol=1e-4)


def test_default_inputs_are_two_nwp_channels_and_g1_adds_geometry(tmp_path, tiny_ckpt_dir, batch):
    bd = batch["batch_dict"]
    seen = {}
    for name, over in (("base", {}), ("g1", {"features": {"derived": {"solar_elevation": True, "clear_sky_power": True}}})):
        m = _model(tmp_path, tiny_ckpt_dir, **over)
        orig = m.backbone.predict_quantiles

        def spy(ctx, pf, po, _n=name, _o=orig, **kw):
            seen[_n] = pf
            return _o(ctx, pf, po, **kw)

        m.backbone.predict_quantiles = spy
        with torch.no_grad():
            m(bd)
    B = len(bd["timestamp_win"])
    assert seen["base"].shape == (B, 2, L + H)
    assert seen["g1"].shape == (B, 4, L + H)
    assert np.array_equal(seen["g1"][:, :2], seen["base"]) and (seen["g1"][:, 2] < 0).any()   # 第 3 通道 = 太阳高度角（夜间为负）


def test_loss_grad_only_on_lora_and_head_and_cap_normalisation(tmp_path, tiny_ckpt_dir, batch):
    bd, labels = batch["batch_dict"], batch["labels"]
    sid = np.asarray(bd["station_id"]).reshape(-1)
    keep = torch.as_tensor(np.flatnonzero(sid == sid[0]))                      # 单站：cap 是一个数
    one = {k: v[keep] for k, v in bd.items()}
    lab = {k: v[keep] for k, v in labels.items()}
    m = _model(tmp_path, tiny_ckpt_dir, finetune={"lora": {"dropout": 0.0}})
    m.train()
    out = m(one, labels=lab)
    assert out.loss.requires_grad and torch.isfinite(out.loss)
    out.loss.backward()
    grads = {n for n, p in m.named_parameters() if p.grad is not None}
    assert grads and all(".lora_" in n or ".output_head." in n for n in grads)
    raw = _model(tmp_path, tiny_ckpt_dir, finetune={"lora": {"dropout": 0.0}}, train={"loss": {"normalize_by_cap": False}})
    raw.train()
    cap = float(out.aux["cap"][0])
    assert float(raw(one, labels=lab).loss) == pytest.approx(float(out.loss) * cap, rel=1e-4)


def test_nan_history_and_missing_labels(tmp_path, tiny_ckpt_dir, batch):
    bd = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch["batch_dict"].items()}
    labels = {k: v.clone() for k, v in batch["labels"].items()}
    bd["observe_power"][0, 100:300] = float("nan")
    bd["general_ghi"][1, :] = float("nan")                                     # 整行缺测 → 置 0
    labels["observe_power_future"][0, :50] = float("nan")
    m = _model(tmp_path, tiny_ckpt_dir)
    m.train()
    out = m(bd, labels=labels)
    assert torch.isfinite(out.loss) and torch.isfinite(out.aux["quantiles"]).all() and torch.isfinite(out.logits).all()
    out.loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_decide_default_has_no_clear_sky_ceiling_and_g2_restores_it(tmp_path):
    m = TransformersAdaptedM5Model({"dataset": {"station_meta": str(write_station_meta(tmp_path))},
                                    "timesfm3": {"eager_init": False}})
    cap = np.array([100.0]); q = np.full((1, 4, 9), 30.0)
    night = np.array([[True, False, False, False]]); pc = np.array([[0.0, 20.0, 80.0, 80.0]])
    point, *_ = m.decide(q, night, pc, cap)
    assert np.allclose(point, 30.0)                                            # 默认：夜间不归零、无晴空上限
    dec = {**m.cfg["decision"], "physics": {**m.cfg["decision"]["physics"], "clear_sky_cap_ratio": 1.05}}
    point2, *_ = m.decide(q, night, pc, cap, dec)
    assert np.allclose(point2, [[30.0, 21.0, 30.0, 30.0]])                     # g2：min(1.05·p_clear, cap)


def test_predict_raw_matches_forward(tmp_path, tiny_ckpt_dir, batch):
    m = _model(tmp_path, tiny_ckpt_dir)
    bd = batch["batch_dict"]
    q, ctx = m.predict_raw(bd)
    with torch.no_grad():
        out = m(bd)
    np.testing.assert_allclose(q, out.aux["quantiles"].numpy(), atol=1e-6)
    assert np.array_equal(ctx.station_id, out.aux["station_id"])
    point, *_ = m.decide(q, ctx.night[:, L:], ctx.p_clear[:, L:], ctx.cap)
    np.testing.assert_allclose(point, out.logits.numpy(), atol=1e-5)
    assert set(out.aux) == {"quantiles", "quantiles_cal", "levels_cal", "p_star", "median", "night", "night_loss",
                            "elev_deg", "p_clear", "var_bin", "var_signal", "station_id", "cap", "degraded"}


def test_weights_round_trip_only_lora_and_head(tmp_path, tiny_ckpt_dir, batch):
    from safetensors.torch import load_file
    m = _model(tmp_path, tiny_ckpt_dir)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("lora_B"):
                p.normal_(0, 0.1)
    w = tmp_path / "w" / "model.safetensors"
    save_weights(m, w)
    keys = set(load_file(str(w)))
    assert keys and all(k.endswith((".lora_A", ".lora_B")) or k.startswith("output_head.") for k in keys)
    m2 = _model(tmp_path, tiny_ckpt_dir, train={"weights_path": str(w)})
    assert m2.weights_loaded and not m.weights_loaded
    bd = batch["batch_dict"]
    m0 = _model(tmp_path, tiny_ckpt_dir)
    with torch.no_grad():
        q1, q2, q0 = m(bd).aux["quantiles"], m2(bd).aux["quantiles"], m0(bd).aux["quantiles"]
    assert torch.allclose(q1, q2, atol=1e-6) and not torch.allclose(q0, q2, atol=1e-6)


def test_predict_raw_restores_per_module_training_flags(tmp_path, tiny_ckpt_dir, batch):
    m = _model(tmp_path, tiny_ckpt_dir)
    before = {n: mod.training for n, mod in m.named_modules()}
    m.predict_raw(batch["batch_dict"])
    assert {n: mod.training for n, mod in m.named_modules()} == before
