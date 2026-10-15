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


def _perturb(m, seed=0):
    """把 LoRA B 与 output_head 随机挪开，模拟训练过的 run。"""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for x in m.backbone.modules():
            if isinstance(x, LoRALinear):
                x.lora_B.copy_(torch.randn(x.lora_B.shape, generator=g) * 0.05)
        for p in m.backbone.model.output_head.parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.05)


def test_wise_alpha_interpolates_between_zero_shot_and_finetuned(tmp_path, tiny_ckpt_dir, batch):
    """W2 WiSE-FT：alpha=0 与零样本逐位一致，alpha=1 回到微调结果，中间值两头都不是；可反复调。"""
    bd = batch["batch_dict"]
    lo, zs = _model(tmp_path, tiny_ckpt_dir), _model(tmp_path, tiny_ckpt_dir, finetune={"enabled": False})
    _perturb(lo)
    with torch.no_grad():
        ft, z = lo(bd).aux["quantiles"].clone(), zs(bd).aux["quantiles"]
        assert not torch.allclose(ft, z, atol=1e-4)
        lo.wise(0.0)
        assert torch.allclose(lo(bd).aux["quantiles"], z, atol=1e-5)
        lo.wise(0.5)
        mid = lo(bd).aux["quantiles"]
        assert not torch.allclose(mid, z, atol=1e-5) and not torch.allclose(mid, ft, atol=1e-5)
        lo.wise(1.0)
        assert torch.allclose(lo(bd).aux["quantiles"], ft, atol=1e-6)
    with pytest.raises(ValueError, match="alpha"):
        lo.wise(1.5)
    with pytest.raises(ValueError, match="LoRA"):
        zs.wise(0.5)


def test_decide_applies_spread_and_linear_tail_from_config(tmp_path, tiny_ckpt_dir):
    import copy
    m = _model(tmp_path, tiny_ckpt_dir)
    B = 2
    q = np.tile(np.linspace(10.0, 90.0, 9), (B, H, 1))
    night, pc, cap = np.zeros((B, H), bool), np.full((B, H), 100.0), np.full(B, 100.0)
    base = m.cfg["decision"]
    _, q0, med0, ps0, _ = m.decide(q, night, pc, cap)
    dec = copy.deepcopy(base); dec["point"]["spread_lo"] = 1.5
    _, q1, med1, _, _ = m.decide(q, night, pc, cap, dec)
    assert np.allclose(med1, med0) and np.allclose(q1[..., 5:], q0[..., 5:]) and (q1[..., :4] < q0[..., :4]).all()
    dec = copy.deepcopy(base); dec["point"]["tail"] = "linear"
    _, _, _, ps2, _ = m.decide(q, night, pc, cap, dec)
    assert (ps2 < ps0).all()


def _lora_paths(m):
    return sorted(n for n, x in m.backbone.model.named_modules() if isinstance(x, LoRALinear))


def test_var_attn_regex_targets_only_cross_variate_attention(tmp_path, tiny_ckpt_dir):
    """W4：re: 靶点只挂 var_attn 的 q/k/v/o（加输入残差块的三个层），seq_attn / FFN 不挂。"""
    m = _model(tmp_path, tiny_ckpt_dir, finetune={"lora": {"target_modules": [
        r"re:.*\.var_attn\.(query|key|value|out)_proj", "hidden_layer", "output_layer", "residual_layer"]}})
    paths = _lora_paths(m)
    assert paths and all(".var_attn." in p or p.startswith("pre_transformer_resblock.") for p in paths)
    assert sum(p.startswith("pre_transformer_resblock.") for p in paths) == 3
    assert not any(".seq_attn." in p for p in paths)


def test_head_only_and_resblock_probe_train_the_right_params(tmp_path, tiny_ckpt_dir, batch):
    """W3a：靶点为空 = 只训 output_head；W3d：再整块解冻输入残差块。两者都不挂 LoRA，能存能读、读回输出一致。"""
    head = _model(tmp_path, tiny_ckpt_dir, finetune={"lora": {"target_modules": []}})
    assert _lora_paths(head) == []
    n_head = sum(p.numel() for p in head.backbone.model.output_head.parameters())
    assert head.n_params() == n_head
    probe = _model(tmp_path, tiny_ckpt_dir, finetune={"lora": {"target_modules": []},
                                                        "unfreeze_modules": ["pre_transformer_resblock"]})
    n_res = sum(p.numel() for p in probe.backbone.model.pre_transformer_resblock.parameters())
    assert probe.n_params() == n_head + n_res
    with torch.no_grad():
        for p in probe.backbone.model.pre_transformer_resblock.parameters():
            p.add_(0.01)
        want = probe(batch["batch_dict"]).aux["quantiles"]
    w = tmp_path / "probe.safetensors"
    save_weights(probe, w)
    back = _model(tmp_path, tiny_ckpt_dir, finetune={"lora": {"target_modules": []}, "unfreeze_modules": ["pre_transformer_resblock"]},
                  train={"weights_path": str(w)})
    assert back.weights_loaded
    with torch.no_grad():
        assert torch.allclose(back(batch["batch_dict"]).aux["quantiles"], want, atol=1e-6)
    with pytest.raises(ValueError, match="可训"):
        _model(tmp_path, tiny_ckpt_dir, finetune={"lora": {"target_modules": []}, "unfreeze_output_head": False})
    with pytest.raises(ValueError, match="unfreeze_modules"):
        _model(tmp_path, tiny_ckpt_dir, finetune={"unfreeze_modules": ["no_such_block"]})


def test_wise_also_blends_unfrozen_modules(tmp_path, tiny_ckpt_dir, batch):
    bd = batch["batch_dict"]
    probe = _model(tmp_path, tiny_ckpt_dir, finetune={"lora": {"target_modules": []}, "unfreeze_modules": ["pre_transformer_resblock"]})
    zs = _model(tmp_path, tiny_ckpt_dir, finetune={"enabled": False})
    with torch.no_grad():
        for p in probe.backbone.model.pre_transformer_resblock.parameters():
            p.add_(0.05)
        probe.wise(0.0)
        assert torch.allclose(probe(bd).aux["quantiles"], zs(bd).aux["quantiles"], atol=1e-5)
