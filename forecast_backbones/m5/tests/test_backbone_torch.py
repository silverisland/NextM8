import numpy as np
import pytest
import torch

pytest.importorskip("timesfm3")
from forecast_backbones.m5.backbone import TorchBackbone
from forecast_backbones.m5.lora import inject_lora, freeze_all_but_lora
from forecast_backbones.m5.decision import MODEL_LEVELS

L, H = 672, 192


def _inputs(B=3, C=2, seed=0, nan_at=None):
    rng = np.random.default_rng(seed)
    ctx = np.abs(rng.normal(50, 10, (B, L))).astype(np.float32)
    pf = rng.normal(0, 1, (B, C, L + H)).astype(np.float32)
    po = rng.normal(0, 1, (B, 1, L)).astype(np.float32)
    if nan_at is not None:
        ctx[0, nan_at] = np.nan
    return ctx, pf, po


def _evaluator_quantiles(e, ctx, pf, po, **flags):
    outs = list(e.predict_batch([c[None, :] for c in ctx], horizon=H, return_quantiles=True,
                                past_future_covariates=list(pf), past_only_covariates=list(po), **flags))
    return np.stack([np.asarray(o.quantiles)[0] for o in outs])          # [B,H,9]


@pytest.mark.parametrize("flags", [
    dict(use_symmetric_averaging=True, use_znorm=False, make_positive=True),
    dict(use_symmetric_averaging=False, use_znorm=False, make_positive=True),
    dict(use_symmetric_averaging=True, use_znorm=True, make_positive=True),
    dict(use_symmetric_averaging=True, use_znorm=False, make_positive=False),
])
def test_torch_path_matches_evaluator(tiny_evaluator, flags):
    bb = TorchBackbone.from_evaluator(tiny_evaluator)
    ctx, pf, po = _inputs(nan_at=[10, 11, 300])
    ref = _evaluator_quantiles(tiny_evaluator, ctx, pf, po, **flags)
    with torch.no_grad():
        got = bb.predict_quantiles(ctx, pf, po, horizon=H, **flags).cpu().numpy()
    assert got.shape == (3, H, 9)
    assert np.allclose(got, ref, atol=1e-4), float(np.abs(got - ref).max())


def test_torch_path_without_covariates_matches_evaluator(tiny_evaluator):
    bb = TorchBackbone.from_evaluator(tiny_evaluator)
    ctx, _, _ = _inputs()
    outs = list(tiny_evaluator.predict_batch([c[None, :] for c in ctx], horizon=H, return_quantiles=True))
    ref = np.stack([np.asarray(o.quantiles)[0] for o in outs])
    with torch.no_grad():
        got = bb.predict_quantiles(ctx, None, None, horizon=H).cpu().numpy()
    assert np.allclose(got, ref, atol=1e-4)


def test_make_positive_only_clamps_nonnegative_rows(tiny_evaluator):
    bb = TorchBackbone.from_evaluator(tiny_evaluator)
    ctx, pf, po = _inputs(B=2)
    ctx[1, 5] = -3.0                                                     # 第 1 行含负值 → 不钳
    with torch.no_grad():
        q = bb.predict_quantiles(ctx, pf, po, horizon=H, make_positive=True).cpu().numpy()
        q_raw = bb.predict_quantiles(ctx, pf, po, horizon=H, make_positive=False).cpu().numpy()
    assert (q[0] >= 0).all() and np.allclose(q[0], np.maximum(q_raw[0], 0)) and np.allclose(q[1], q_raw[1])


def test_gradient_reaches_lora_only():
    bb = TorchBackbone.tiny_for_tests(seed=1)
    names = inject_lora(bb.model, ["query_proj", "key_proj", "value_proj", "out_proj"], 4, 8, 0.0)
    assert len(names) == 2 * 2 * 4                                       # 2 层 × (seq_attn, var_attn) × 4
    freeze_all_but_lora(bb.model, unfreeze_output_head=True)
    bb.train()
    ctx, pf, po = _inputs(B=2)
    q = bb.predict_quantiles(ctx, pf, po, horizon=H)
    assert q.requires_grad
    q[..., 4].mean().backward()
    grads = {n: p.grad for n, p in bb.model.named_parameters() if p.requires_grad}
    assert all(g is not None and torch.isfinite(g).all() for g in grads.values())
    assert any(float(g.abs().sum()) > 0 for n, g in grads.items() if n.endswith("lora_A") or n.startswith("output_head"))
    assert all(p.grad is None for n, p in bb.model.named_parameters() if not p.requires_grad)


def test_batch_rows_are_independent(tiny_evaluator):
    bb = TorchBackbone.from_evaluator(tiny_evaluator)
    ctx, pf, po = _inputs(B=3)
    with torch.no_grad():
        full = bb.predict_quantiles(ctx, pf, po, horizon=H).cpu().numpy()
        one = bb.predict_quantiles(ctx[1:2], pf[1:2], po[1:2], horizon=H).cpu().numpy()
    assert np.allclose(full[1], one[0], atol=1e-5)


def test_rejects_bad_shapes_and_horizon(tiny_evaluator):
    bb = TorchBackbone.from_evaluator(tiny_evaluator)
    ctx, pf, po = _inputs(B=2)
    with pytest.raises(ValueError, match="past_future"):
        bb.predict_quantiles(ctx, pf[..., :-1], po, horizon=H)
    with pytest.raises(ValueError, match="output_patch_len"):
        bb.predict_quantiles(ctx, None, None, horizon=H + 1)
    with pytest.raises(ValueError, match="quantiles"):
        TorchBackbone(_wrong_levels_model())


def test_backbone_import_raises_if_finite_running_stats_shim_fails_to_attach(monkeypatch):
    """timesfm3 build 把 update_running_stats 改名/删掉时，ensure_finite_running_stats() 会静默
    返回 False（不打补丁），但训练要等到某次夜间 batch 恰好命中零方差分支的反向传播才 NaN——必须在
    backbone.py 的 import 期就用 RuntimeError 拦下来。用 monkeypatch 删掉该函数、重新 import
    backbone.py 复现；测试结束务必把状态修复回去，不能让这个模块留在半初始化状态污染其他测试。"""
    import importlib
    import forecast_backbones.m5.backbone as backbone_mod
    from timesfm3.torch import util as tfm_util
    monkeypatch.delattr(tfm_util, "update_running_stats")
    try:
        with pytest.raises(RuntimeError, match="update_running_stats"):
            importlib.reload(backbone_mod)
    finally:
        monkeypatch.undo()                     # 立刻恢复被删的属性（已打过补丁的原函数）
        importlib.reload(backbone_mod)         # 重新正常初始化一遍，不让后续测试踩到半初始化的模块


def _wrong_levels_model():
    from timesfm3.torch import configs
    from timesfm3.torch.model import TimesFM3Torch
    rb = configs.ResidualBlockConfig(hidden_dims=32, output_dims=32, use_bias=False, activation="relu", dropout=0.0)
    tc = configs.StackedTransformersConfig(num_layers=1, use_remat=False, transformer=configs.TransformerConfig(
        model_dims=32, hidden_dims=32, num_heads=4, attention_norm="rms", feedforward_norm="rms", qk_norm="rms",
        use_rope_seq=True, use_rope_var=True, use_bias=False, ff_activation="relu", deterministic=True, use_sdpa=True))
    return TimesFM3Torch(input_patch_len=8, output_patch_len=16, quantiles=[0.1, 0.5, 0.9], residual_block_config=rb,
                         transformer_config=tc, use_stitching=True, use_linear_detrending=True,
                         use_iterative_cpm_revin=True, use_frozen_running_stats=False)
