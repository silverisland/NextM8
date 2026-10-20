"""2026-10-18 的 LoRA 之外的适配器与协议旋钮：门控 / 多专家 LoRA、(IA)³、虚拟变量 prompt、静态嵌入、残差块适配器、
C1 / C2 包络通道、优化器参数分组、LP-FT 热启动。"""
import json

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("timesfm3")
from forecast_backbones.m5.transformer_models import TransformersAdaptedM5Model, save_weights, load_weights
from forecast_backbones.m5.lora import LoRALinear, IA3Linear, ADAPTER_ROOT
from .conftest import full_cfg, write_splits

L, H = 672, 192
ALL_IDENTITY = {   # 初始化即与零样本逐位相同的适配器
    "gate": {"finetune": {"lora": {"gate": True}}},
    "experts": {"finetune": {"lora": {"experts": 3}}},
    "ia3": {"finetune": {"lora": {"target_modules": ["query_proj", "out_proj"]}, "ia3": {"target_modules": ["key_proj", "value_proj", "ff0"]}}},
    "static": {"finetune": {"static_embed": {"enabled": True}}},
    "rb_adapter": {"finetune": {"resblock_adapter": {"rank": 4}}},
}


def _model(tmp_path, ckpt, **over):
    return TransformersAdaptedM5Model(full_cfg(tmp_path, ckpt, **over))


def _adapters(m):
    return getattr(m.backbone.model, ADAPTER_ROOT, None)


def _perturb(m, scale=0.05, seed=1):
    """把所有可训参数加随机扰动（让零初始化的增量 / 门 / 嵌入都不再是恒等）。"""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, p in m.backbone.model.named_parameters():
            if p.requires_grad:
                p.add_(torch.randn(p.shape, generator=g) * scale)


def _single_station(batch):
    bd, labels = batch["batch_dict"], batch["labels"]
    sid = np.asarray(bd["station_id"]).reshape(-1)
    keep = torch.as_tensor(np.flatnonzero(sid == sid[0]))
    return {k: v[keep] for k, v in bd.items()}, {k: v[keep] for k, v in labels.items()}


def _rows(bd, idx):
    idx = torch.as_tensor(idx)
    return {k: v[idx] for k, v in bd.items()}


@pytest.mark.parametrize("name", sorted(ALL_IDENTITY))
def test_adapters_at_init_equal_zero_shot(tmp_path, tiny_ckpt_dir, batch, name):
    bd = batch["batch_dict"]
    m = _model(tmp_path, tiny_ckpt_dir, **ALL_IDENTITY[name])
    zs = _model(tmp_path, tiny_ckpt_dir, finetune={"enabled": False})
    with torch.no_grad():
        assert torch.allclose(m(bd).aux["quantiles"], zs(bd).aux["quantiles"], atol=1e-5)


def test_default_config_attaches_nothing_extra(tmp_path, tiny_ckpt_dir):
    m = _model(tmp_path, tiny_ckpt_dir)
    assert _adapters(m) is None and m._cond is None
    assert not any(isinstance(x, IA3Linear) for x in m.backbone.modules())
    assert all(x.cond is None and x.experts == 1 for x in m.backbone.modules() if isinstance(x, LoRALinear))


def test_prompt_adds_variates_but_head_sees_only_originals(tmp_path, tiny_ckpt_dir, batch):
    bd = batch["batch_dict"]
    m = _model(tmp_path, tiny_ckpt_dir, finetune={"prompt": {"n_tokens": 2}})
    zs = _model(tmp_path, tiny_ckpt_dir, finetune={"enabled": False})
    seen = {}
    m.backbone.model.output_head.register_forward_hook(lambda _m, i, o: seen.update(v=i[0].shape[1]))   # update 返回 None：不改输出
    with torch.no_grad():
        q, z = m(bd).aux["quantiles"], zs(bd).aux["quantiles"]
    assert q.shape == z.shape and torch.isfinite(q).all()
    assert seen["v"] == 3                                   # 目标 + 2 路 NWP；prompt 的 2 个变量在 head 前已切掉
    assert not torch.allclose(q, z, atol=1e-6)              # 多出的 token 进 softmax：初始就不是零样本（文档里写明）


@pytest.mark.parametrize("over", [
    {"finetune": {"lora": {"gate": True}}},
    {"finetune": {"lora": {"experts": 2}}},
    {"finetune": {"prompt": {"n_tokens": 2, "per_station": True}, "static_embed": {"enabled": True}, "resblock_adapter": {"rank": 4}}},
], ids=["gate", "experts", "prompt_station_static_rb"])
def test_per_sample_conditioning_is_batch_consistent(tmp_path, tiny_ckpt_dir, batch, over):
    """逐样本量（门 / 路由 / 静态嵌入 / 逐站 prompt）必须按样本广播对：整批前向 = 每行单独前向。"""
    bd = batch["batch_dict"]
    m = _model(tmp_path, tiny_ckpt_dir, **over)
    m.set_known_stations([1001, 1002])
    _perturb(m, scale=0.2)
    m.eval()
    with torch.no_grad():
        full = m(bd).aux["quantiles"]
        for i in (0, len(bd["timestamp_win"]) - 1):
            one = m(_rows(bd, [i])).aux["quantiles"]
            assert torch.allclose(full[i:i + 1], one, rtol=1e-5, atol=1e-3), f"row {i} differs"


def test_conditioning_really_varies_per_sample(tmp_path, tiny_ckpt_dir, batch):
    """门控 LoRA 的门是逐样本的：不同形势特征 → 不同的门（同一模块）。"""
    bd = batch["batch_dict"]
    m = _model(tmp_path, tiny_ckpt_dir, finetune={"lora": {"gate": True}})
    _perturb(m, scale=0.5)
    with torch.no_grad():
        m(bd)
    w = m._cond.lora_weights(0)
    assert w.shape == (len(bd["timestamp_win"]), 1) and float(w.std()) > 0
    assert m._cond.z.shape[1] == 4


def test_gradients_reach_every_adapter(tmp_path, tiny_ckpt_dir, batch):
    one, lab = _single_station(batch)
    for over, prefix in (({"finetune": {"lora": {"gate": True, "dropout": 0.0}}}, "lora_cond"),
                         ({"finetune": {"lora": {"experts": 2, "dropout": 0.0}}}, "lora_cond"),
                         ({"finetune": {"prompt": {"n_tokens": 2, "per_station": True, "station_dropout": 0.0}}}, "prompt"),
                         ({"finetune": {"static_embed": {"enabled": True, "dropout": 0.0}}}, "static_embed"),
                         ({"finetune": {"resblock_adapter": {"rank": 4}}}, "resblock_adapter")):
        m = _model(tmp_path, tiny_ckpt_dir, **over)
        m.set_known_stations([1001, 1002])
        _perturb(m, scale=0.1)                              # B / up / W 离开零点，门与路由才有梯度
        m.train()
        m(one, labels=lab).loss.backward()
        got = {n for n, p in m.backbone.model.named_parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0}
        assert any(n.startswith(f"{ADAPTER_ROOT}.{prefix}") for n in got), (prefix, sorted(got)[:5])
    m = _model(tmp_path, tiny_ckpt_dir, **ALL_IDENTITY["ia3"])
    m.train()
    m(one, labels=lab).loss.backward()
    assert any(n.endswith(".ia3_scale") and p.grad is not None for n, p in m.backbone.model.named_parameters())


def test_unknown_station_uses_global_prompt_only(tmp_path, tiny_ckpt_dir, batch):
    bd = batch["batch_dict"]
    m = _model(tmp_path, tiny_ckpt_dir, finetune={"prompt": {"n_tokens": 1, "per_station": True}})
    m.set_known_stations([1001])
    with torch.no_grad():
        m(bd)
    sid = np.asarray(bd["station_id"]).reshape(-1)
    slot = m._cond.slot.cpu().numpy()
    assert (slot[sid == 1001] > 0).all() and (slot[sid == 1002] == 0).all()


def test_all_adapters_save_load_roundtrip(tmp_path, tiny_ckpt_dir, batch):
    bd = batch["batch_dict"]
    over = {"finetune": {"lora": {"experts": 2}, "ia3": {"target_modules": ["ff0"]}, "prompt": {"n_tokens": 2, "per_station": True},
                         "static_embed": {"enabled": True}, "resblock_adapter": {"rank": 4}}}
    a = _model(tmp_path, tiny_ckpt_dir, **over)
    a.set_known_stations([1001, 1002])
    _perturb(a)
    a.eval()
    path = tmp_path / "w.safetensors"
    save_weights(a, path)
    b = _model(tmp_path, tiny_ckpt_dir, **over, train={"weights_path": str(path)})
    b.set_known_stations([1001, 1002])
    assert b.weights_loaded
    with torch.no_grad():
        assert torch.allclose(a(bd).aux["quantiles"], b(bd).aux["quantiles"], atol=1e-6)
    plain = _model(tmp_path, tiny_ckpt_dir, finetune={"lora": {"experts": 2}, "ia3": {"target_modules": ["ff0"]}})
    with pytest.raises(ValueError, match="unexpected"):           # 存档有 m5_adapters、模型没开 → 拒绝
        load_weights(plain, path)


def test_wise_zero_is_zero_shot_with_identity_initialised_adapters(tmp_path, tiny_ckpt_dir, batch):
    bd = batch["batch_dict"]
    over = {"finetune": {"lora": {"gate": True}, "ia3": {"target_modules": ["ff0"]}, "static_embed": {"enabled": True},
                         "resblock_adapter": {"rank": 4}}}
    m, zs = _model(tmp_path, tiny_ckpt_dir, **over), _model(tmp_path, tiny_ckpt_dir, finetune={"enabled": False})
    _perturb(m)
    with torch.no_grad():
        m.wise(0.0)
        assert torch.allclose(m(bd).aux["quantiles"], zs(bd).aux["quantiles"], atol=1e-5)


# ------------------------------------------------------------------ 包络（C1 / C2）
def test_history_envelope_slots_and_quantile():
    from forecast_backbones.m5.envelope import history_envelope, envelope_rows
    n_day, D = 4, 3
    y = np.array([[1.0, 2, 3, 4] * D])                      # 槽 0..3 的值恒为 1..4
    env, days = history_envelope(y, np.array([0]), n_day, 0.9, 1)
    assert np.allclose(env[0], [1, 2, 3, 4]) and days[0] == D
    env2, _ = history_envelope(np.roll(y, -1, axis=1), np.array([1]), n_day, 0.9, 1)   # 起点在槽 1：按槽对齐后不变
    assert np.allclose(env2[0], [1, 2, 3, 4])
    slot = np.array([[1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2]])                        # L=12 + H=2
    rows = envelope_rows(np.roll(y, -1, axis=1), slot, n_day, 0.9, 1)
    assert np.allclose(rows[0, -2:], [2, 3]) and rows.shape == (1, 14)


def test_nwp_ratio_and_regime_features():
    from forecast_backbones.m5.envelope import nwp_ratio, regime_features
    r = nwp_ratio(np.array([[0.0, 500, 2000, np.nan]]), np.array([[0.0, 1000, 1000, 1000]]), 20.0, 1.2)
    assert np.allclose(r, [[0.0, 0.5, 1.2, 0.0]])
    z = regime_features(np.array([[0.5, 1.0, 0.0]]), np.array([[True, True, False]]), np.array([1.0]))
    assert z.shape == (1, 4) and z[0, 0] == pytest.approx(0.75) and z[0, 2] == pytest.approx(0.5) and z[0, 3] == 1.0
    assert (regime_features(np.zeros((1, 3)), np.zeros((1, 3), bool), np.array([0.0])) == 0).all()


def test_envelope_channels_are_appended_after_nwp(tmp_path, tiny_ckpt_dir, batch):
    bd = batch["batch_dict"]
    seen = {}
    for name, d in (("c1", {"history_envelope": True}), ("c2", {"history_envelope": True, "envelope_nwp": True})):
        m = _model(tmp_path, tiny_ckpt_dir, features={"derived": d})
        orig = m.backbone.predict_quantiles

        def spy(ctx, pf, po, _n=name, _o=orig, **kw):
            seen[_n] = pf
            return _o(ctx, pf, po, **kw)

        m.backbone.predict_quantiles = spy
        with torch.no_grad():
            assert torch.isfinite(m(bd).aux["quantiles"]).all()
    B = len(bd["timestamp_win"])
    assert seen["c1"].shape == (B, 3, L + H) and seen["c2"].shape == (B, 4, L + H)
    env, p_nwp = seen["c2"][:, 2], seen["c2"][:, 3]
    assert (env >= 0).all() and env.max() > 0
    assert np.allclose(env[:, :96], env[:, 96:192])                          # 包络按时刻槽铺：逐日重复
    assert ((p_nwp <= env * 1.2 + 1e-4) & (p_nwp >= 0)).all()                # C2 = 包络 × clip(比值, 0, 1.2)


def test_envelope_needs_ghi_columns(tmp_path, tiny_ckpt_dir, batch):
    bd = {k: v for k, v in batch["batch_dict"].items() if k != "general_ghi"}
    m = _model(tmp_path, tiny_ckpt_dir, features={"derived": {"envelope_nwp": True}, "past_future": [["general_temp", "general_temp_predict"]]})
    with pytest.raises(KeyError, match="features.envelope.ghi"):
        with torch.no_grad():
            m(bd)


# ------------------------------------------------------------------ 训练协议
def test_param_groups_split_lora_plus_head_and_adapters(tmp_path, tiny_ckpt_dir):
    from forecast_backbones.m5.train import param_groups
    m = _model(tmp_path, tiny_ckpt_dir, finetune={"resblock_adapter": {"rank": 4}})
    assert param_groups(m, {}, 1e-4, 0.01) is None
    groups = param_groups(m, {"lora_plus_ratio": 16, "head_lr": 1e-5, "adapter_lr": 3e-4}, 1e-4, 0.01)
    by_param = {id(p): (g["lr"], g["weight_decay"]) for g in groups for p in g["params"]}
    named = dict(m.named_parameters())
    a = next(p for n, p in named.items() if n.endswith(".lora_A"))
    b = next(p for n, p in named.items() if n.endswith(".lora_B"))
    head_w, head_b = named["backbone.model.output_head.weight"], named["backbone.model.output_head.bias"]
    rb = next(p for n, p in named.items() if ".resblock_adapter.down.weight" in n)
    assert by_param[id(a)] == (1e-4, 0.01) and by_param[id(b)] == (1.6e-3, 0.01)
    assert by_param[id(head_w)] == (1e-5, 0.01) and by_param[id(head_b)] == (1e-5, 0.0)
    assert by_param[id(rb)] == (3e-4, 0.01)
    assert sum(len(g["params"]) for g in groups) == sum(p.requires_grad for p in m.parameters())
    ia3 = _model(tmp_path, tiny_ckpt_dir, **ALL_IDENTITY["ia3"])         # (IA)³ 即使 optim 全 null 也自建分组：缩放向量不 decay
    g3 = param_groups(ia3, {}, 1e-4, 0.01)
    assert g3 is not None and all(g["weight_decay"] == 0.0 for g in g3 for p in g["params"] if p.ndim == 1)


def test_warm_start_copies_head_and_rejects_wrong_files(tmp_path, tiny_ckpt_dir):
    from forecast_backbones.m5.train import warm_start
    from forecast_backbones.m5.lora import save_lora_weights
    m = _model(tmp_path, tiny_ckpt_dir)
    w = {n: p.detach().clone() + 0.5 for n, p in m.backbone.model.named_parameters() if n.startswith("output_head.")}
    save_lora_weights(tmp_path / "head_s3.safetensors", w)
    keys = warm_start(m, tmp_path / "head_s{seed}.safetensors", seed=3)
    assert keys == sorted(w) and torch.allclose(m.backbone.model.output_head.weight, w["output_head.weight"])
    with pytest.raises(FileNotFoundError):
        warm_start(m, tmp_path / "head_s{seed}.safetensors", seed=4)
    save_lora_weights(tmp_path / "bad.safetensors", {"pre_transformer_resblock.hidden_layer.weight": torch.zeros(2, 2)})
    with pytest.raises(ValueError, match="不可训"):
        warm_start(m, tmp_path / "bad.safetensors", seed=0)


def test_train_run_with_adapters_lpft_groups_and_constant_lr(tmp_path, tiny_ckpt_dir):
    """端到端：门控 LoRA + prompt + 静态嵌入 + 包络通道 + LoRA+ 分组 + 常数 lr + LP-FT 热启动跑通，summary 记下配置。"""
    from forecast_backbones.m5.train import train_m5
    from forecast_backbones.m5.lora import save_lora_weights
    splits = write_splits(tmp_path, n_train_days=2, n_val_days=2, train_stations=(1001,), val_stations=(1001, 1002))
    probe = _model(tmp_path, tiny_ckpt_dir)
    save_lora_weights(tmp_path / "head_s0.safetensors",
                      {n: p.detach().clone() for n, p in probe.backbone.model.named_parameters() if n.startswith("output_head.")})
    out = tmp_path / "adapters"
    cfg = full_cfg(tmp_path, tiny_ckpt_dir,
                   features={"derived": {"history_envelope": True, "envelope_nwp": True}},
                   finetune={"lora": {"gate": True}, "prompt": {"n_tokens": 1, "per_station": True}, "static_embed": {"enabled": True},
                             "init_from": str(tmp_path / "head_s{seed}.safetensors")},
                   train={"schedule": {"lr": 1e-2, "lr_scheduler": "constant"}, "optim": {"lora_plus_ratio": 4}})
    train_m5(cfg, splits_dir=splits, out_dir=out, tag="ad")
    s = pd.read_csv(out / "reports" / "summary.csv").iloc[0]
    assert s["adapters"] == "gate+prompt1_station+static+lpft" and s["lr_scheduler"] == "constant"
    assert np.isnan(s["nanwang_zeroshot"]) and np.isnan(s["gain"])          # prompt / LP-FT 的第 0 步不是零样本
    assert s["optim"] == "lora_plus_ratio=4" and s["derived"] == "history_envelope+envelope_nwp"
    assert np.isfinite(s["nanwang_short_D+1"]) and (out / "model.safetensors").exists()
    log = json.loads((out / "train_log.json").read_text(encoding="utf-8"))
    assert len(log["history"]) >= 2


@pytest.mark.parametrize("name", ["f1_final", "f4_drop01", "s3_c1_envelope", "s3_c1_envelope_zs"])
def test_new_variant_yamls_build_forward_and_backward(tmp_path, tiny_ckpt_dir, batch, name):
    """变体 yaml 叠在最小配置上：能构造、前向有限；可训的变体反向有梯度。"""
    from pathlib import Path
    from forecast_backbones.m5.config import raw_override, deep_merge
    over = raw_override(str(Path(__file__).resolve().parents[1] / "config" / f"{name}.yaml"))
    over = {k: v for k, v in over.items() if k not in ("run", "dataset", "timesfm3")}   # 站表 / 底座用测试的小模型与两站表
    m = TransformersAdaptedM5Model(deep_merge(full_cfg(tmp_path, tiny_ckpt_dir), over))
    m.set_known_stations([1001])
    one, lab = _single_station(batch)
    if not m.cfg["finetune"]["enabled"]:
        with torch.no_grad():
            assert torch.isfinite(m(one).aux["quantiles"]).all()
        return
    m.train()
    out = m(one, labels=lab)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert any(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_experts_mixture_equals_explicit_weighted_sum():
    from forecast_backbones.m5.lora import LoRALinear
    torch.manual_seed(0)
    m = LoRALinear(torch.nn.Linear(6, 5), rank=3, alpha=6, dropout=0.0, experts=4)
    torch.nn.init.normal_(m.lora_B)

    class _C:
        def __init__(self, w): self.w = w
        def lora_weights(self, idx): return self.w
    w = torch.softmax(torch.randn(2, 4), -1)
    m.cond, m.cond_idx = _C(w), 0
    x = torch.randn(2 * 7, 3, 6)                                       # 样本为主序：每样本 7 行
    got = m(x)
    h = x @ m.lora_A.t()
    ws = w.repeat_interleave(7, 0)
    ref = m.base(x) + sum(ws[:, k, None, None] * (h @ m.lora_B[k].t()) for k in range(4)) * m.scaling
    assert torch.allclose(got, ref, atol=1e-5)
