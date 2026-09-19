import numpy as np
import pytest
import torch

pytest.importorskip("timesfm3")
from forecast_backbones.m6.transformer_models import TransformersAdaptedM6Model
from forecast_backbones.m6.lora import lora_state_dict, LoRALinear
from .conftest import make_synthetic_frame, write_station_meta
from .test_model_route_a import _batch


def _model_b(tmp_path, evaluator, *, finetune=True, inference_path="torch", lora_path=None, routing_path=None, **over):
    cfg = {"timesfm3": {"eager_init": False},
           "dataset": {"station_meta": str(write_station_meta(tmp_path))},
           "calibration": {"enabled": False},
           "finetune": {"enabled": finetune, "inference_path": inference_path,
                        "lora_path": lora_path or str(tmp_path / "no_such_lora.pt"),
                        "routing_path": routing_path or str(tmp_path / "no_such_routing.json")}}
    for k, v in over.items():
        cfg.setdefault(k, {}).update(v)
    m = TransformersAdaptedM6Model(cfg)
    m._backbone = evaluator; m._backbone_device = "cpu"
    if m.torch_path:
        m._attach_torch_path()
    return m


@pytest.fixture
def frame():
    return make_synthetic_frame(n_days=1, issue_every_min=360)


def test_torch_path_zero_shot_matches_evaluator_path(tmp_path, tiny_evaluator, frame):
    bd, _ = _batch(frame, [0, 1, 2])
    m_eval = _model_b(tmp_path, tiny_evaluator, finetune=False, inference_path="evaluator")
    m_torch = _model_b(tmp_path, tiny_evaluator, finetune=False, inference_path="torch")
    assert m_eval.torch_path is False and m_torch.torch_path is True and m_torch.backbone is not None
    with torch.no_grad():
        a, b = m_eval(bd), m_torch(bd)
    assert np.allclose(a.aux["quantiles"].numpy(), b.aux["quantiles"].numpy(), atol=1e-4)
    assert np.allclose(a.logits.numpy(), b.logits.numpy(), atol=1e-3)
    assert not any(isinstance(x, LoRALinear) for x in m_torch.backbone.model.modules())   # 未开 finetune 不挂 LoRA


def test_forward_handles_bf16_quantiles_without_numpy_typeerror(tmp_path, tiny_evaluator, frame, monkeypatch):
    """_quantiles_torch 在 bf16 autocast（GPU 训练）下可能返回 bf16 张量：forward 必须先转 fp32 再
    .numpy()（bf16 张量直接 .numpy() 会 TypeError），aux["quantiles"]/logits 也必须落回 fp32，
    不随底座计算精度漂移。用 monkeypatch 把真实结果强制转 bf16 来复现，不依赖 GPU。"""
    bd, _ = _batch(frame, [0, 1])
    with pytest.warns(UserWarning):
        m = _model_b(tmp_path, tiny_evaluator)
    orig = m._quantiles_torch
    monkeypatch.setattr(m, "_quantiles_torch", lambda *a, **k: orig(*a, **k).bfloat16())
    with torch.no_grad():
        out = m(bd)
    assert out.aux["quantiles"].dtype == torch.float32
    assert out.logits.dtype == torch.float32


def test_finetune_enabled_warns_when_lora_missing_and_matches_zero_shot(tmp_path, tiny_evaluator, frame):
    bd, _ = _batch(frame, [0, 1])
    with pytest.warns(UserWarning) as record:
        m = _model_b(tmp_path, tiny_evaluator)
    msgs = [str(w.message) for w in record.list]
    assert any("LoRA 权重" in s for s in msgs)
    assert any("站级路由" in s for s in msgs)                            # routing_path 同样缺失，对称告警
    assert m.lora_loaded is False
    assert any(isinstance(x, LoRALinear) for x in m.backbone.model.modules())
    m_zs = _model_b(tmp_path, tiny_evaluator, finetune=False, inference_path="torch")
    with torch.no_grad():
        assert np.allclose(m(bd).aux["quantiles"].numpy(), m_zs(bd).aux["quantiles"].numpy(), atol=1e-5)   # B=0


def test_loss_has_gradient_only_on_lora_and_head(tmp_path, tiny_evaluator, frame):
    bd, labels = _batch(frame, [0, 1])
    with pytest.warns(UserWarning):
        m = _model_b(tmp_path, tiny_evaluator)
    m.train()
    out = m(bd, labels=labels)
    assert out.loss is not None and out.loss.requires_grad and torch.isfinite(out.loss)
    out.loss.backward()
    for n, p in m.named_parameters():
        if p.requires_grad:
            assert n.endswith("lora_A") or n.endswith("lora_B") or ".output_head." in n, n
            assert p.grad is not None, n
            assert torch.isfinite(p.grad).all(), n
        else:
            assert p.grad is None, n
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert any(n.endswith("lora_B") for n in trainable) and any(".output_head." in n for n in trainable)


def test_three_adam_steps_reduce_loss(tmp_path, tiny_evaluator, frame):
    bd, labels = _batch(frame, [0, 1])
    with pytest.warns(UserWarning):
        m = _model_b(tmp_path, tiny_evaluator)
    m.train()
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    l0 = float(m(bd, labels=labels).loss)
    for _ in range(3):
        opt.zero_grad(); loss = m(bd, labels=labels).loss; loss.backward(); opt.step()
    l1 = float(m(bd, labels=labels).loss)
    assert l1 < l0


def test_lora_weights_load_from_lora_path(tmp_path, tiny_evaluator, frame):
    bd, _ = _batch(frame, [0])
    with pytest.warns(UserWarning):
        m = _model_b(tmp_path, tiny_evaluator)
    with torch.no_grad():
        for x in m.backbone.model.modules():
            if isinstance(x, LoRALinear):
                x.lora_B.normal_()
    p = tmp_path / "lora_weights.pt"
    torch.save(lora_state_dict(m.backbone.model), p)
    with torch.no_grad():
        ref = m(bd).aux["quantiles"].numpy()
    # 新 evaluator 副本：同一权重目录再加载一次，避免共享已改的 module
    from timesfm3 import ModelConfig, TimesFM3Evaluator
    e2 = TimesFM3Evaluator(ModelConfig(checkpoint_path=tiny_evaluator.config.checkpoint_path, per_core_batch_size=4, device="cpu"))
    from forecast_backbones.m6.routing import save_routing
    rp = tmp_path / "routing.json"; save_routing(rp, {})                 # 路由表也齐全：与 lora_path 一起零 warning
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                   # 权重、路由都齐全就不许 warn
        m2 = _model_b(tmp_path, e2, lora_path=str(p), routing_path=str(rp))
    assert m2.lora_loaded is True
    with torch.no_grad():
        assert np.allclose(m2(bd).aux["quantiles"].numpy(), ref, atol=1e-5)


def test_first_local_module_still_m6_model():
    import inspect, torch.nn as nn, forecast_backbones.m6.transformer_models as mod
    first = next(o for _, o in mod.__dict__.items() if inspect.isclass(o) and o.__module__ == mod.__name__ and issubclass(o, nn.Module))
    assert first is TransformersAdaptedM6Model
