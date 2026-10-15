import pytest
import torch
import torch.nn as nn
from forecast_backbones.m5.lora import (LoRALinear, inject_lora, freeze_all_but_lora,
                                        lora_state_dict, load_lora_state_dict, trainable_summary)


class _Attn(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.query_proj = nn.Linear(d, d); self.key_proj = nn.Linear(d, d); self.ff0 = nn.Linear(d, 2 * d)

    def forward(self, x):
        return self.ff0(self.query_proj(x) + self.key_proj(x))


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Attn(), _Attn()])
        self.output_head = nn.Linear(16, 3)

    def forward(self, x):
        for l in self.layers:
            x = l(x)[..., :8]
        return self.output_head(torch.cat([x, x], -1))


def test_inject_matches_only_targets_and_keeps_output_at_init():
    torch.manual_seed(0)
    m = _Toy(); x = torch.randn(4, 8); y0 = m(x).detach()
    names = inject_lora(m, ["query_proj", "key_proj"], rank=2, alpha=4, dropout=0.0)
    assert names == ["layers.0.query_proj", "layers.0.key_proj", "layers.1.query_proj", "layers.1.key_proj"]
    assert isinstance(m.layers[0].query_proj, LoRALinear) and isinstance(m.layers[0].ff0, nn.Linear)
    assert torch.allclose(m(x), y0)                                   # B=0 → 与原模型逐位相同
    assert m.layers[0].query_proj.scaling == 2.0


def test_inject_is_idempotent_and_raises_on_no_match():
    m = _Toy()
    inject_lora(m, ["query_proj"], 2, 4, 0.0)
    assert inject_lora(m, ["query_proj"], 2, 4, 0.0) == []           # 已挂的不重复挂
    with pytest.raises(ValueError, match="q_proj"):
        inject_lora(_Toy(), ["q_proj"], 2, 4, 0.0)                    # spec 里的错名必须报错，不得静默


def test_inject_raises_on_no_match_after_partial_injection():
    """验证：inject_lora 的幂等检查作用域限于当前目标。
    先挂 query_proj，再试图挂不存在的 bogus_name 应该报错，不是因为 query_proj 已挂就沉默。"""
    m = _Toy()
    inject_lora(m, ["query_proj"], 2, 4, 0.0)                        # 先挂 query_proj
    with pytest.raises(ValueError, match="bogus"):
        inject_lora(m, ["bogus"], 2, 4, 0.0)                         # 试图挂不存在的目标→应报错
    assert inject_lora(m, ["query_proj"], 2, 4, 0.0) == []           # 重复挂相同目标→返回 []（幂等）


def test_freeze_and_trainable_summary():
    m = _Toy(); inject_lora(m, ["query_proj", "key_proj"], 2, 4, 0.05)
    freeze_all_but_lora(m, unfreeze_output_head=True)
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert all(("lora_A" in n) or ("lora_B" in n) or n.startswith("output_head.") for n in trainable)
    assert "layers.0.query_proj.base.weight" not in trainable and "layers.0.ff0.weight" not in trainable
    s = trainable_summary(m)
    assert s["n_lora_modules"] == 4 and 0 < s["frac"] < 1 and s["n_trainable"] == sum(p.numel() for n, p in m.named_parameters() if n in trainable)
    freeze_all_but_lora(m, unfreeze_output_head=False)
    assert not m.output_head.weight.requires_grad


def test_state_dict_round_trip_and_validation():
    torch.manual_seed(0)
    m = _Toy(); inject_lora(m, ["query_proj"], 2, 4, 0.0); freeze_all_but_lora(m, True)
    with torch.no_grad():
        m.layers[1].query_proj.lora_B.normal_(); m.output_head.bias.fill_(7.0)
    sd = lora_state_dict(m)
    assert set(sd) == {"layers.0.query_proj.lora_A", "layers.0.query_proj.lora_B", "layers.1.query_proj.lora_A",
                       "layers.1.query_proj.lora_B", "output_head.weight", "output_head.bias"}
    m2 = _Toy(); inject_lora(m2, ["query_proj"], 2, 4, 0.0); freeze_all_but_lora(m2, True)
    load_lora_state_dict(m2, sd)
    x = torch.randn(3, 8)
    assert torch.allclose(m2.output_head.bias, torch.full((3,), 7.0)) and torch.allclose(m2.layers[1].query_proj.lora_B, m.layers[1].query_proj.lora_B)
    with pytest.raises(ValueError, match="unexpected"):
        load_lora_state_dict(m2, {**sd, "layers.0.ff0.weight": torch.zeros(16, 8)})
    with pytest.raises(ValueError, match="lora_B"):
        load_lora_state_dict(m2, {k: v for k, v in sd.items() if "layers.1.query_proj.lora_B" != k})


def test_dropout_only_affects_lora_branch_in_train_mode():
    torch.manual_seed(0)
    m = _Toy(); inject_lora(m, ["query_proj"], 2, 4, dropout=0.5)
    x = torch.randn(4, 8); m.train()
    y_a, y_b = m(x), m(x)
    assert torch.allclose(y_a, y_b)          # B=0 时 dropout 作用在零分支上，输出仍确定


def test_regex_targets_match_full_path_and_empty_targets_inject_nothing():
    """'re:' 开头的靶点按完整模块路径 fullmatch（可只挂某一层 / 某类注意力）；普通靶点仍按尾段；空列表不挂、不报错。"""
    m = _Toy()
    hits = inject_lora(m, [r"re:layers\.0\.(query|key)_proj"], rank=2, alpha=4, dropout=0.0)
    assert sorted(hits) == ["layers.0.key_proj", "layers.0.query_proj"]
    assert not isinstance(m.layers[1].query_proj, LoRALinear)
    assert inject_lora(_Toy(), [], rank=2, alpha=4, dropout=0.0) == []
    with pytest.raises(ValueError, match="命中"):
        inject_lora(_Toy(), [r"re:nothing\..*"], rank=2, alpha=4, dropout=0.0)


def test_unfrozen_modules_are_trained_saved_and_loaded():
    """unfreeze_modules 里的整块参数可训，lora_state_dict 一起存；读回时按允许前缀校验，缺键 / 多余键都报错。"""
    torch.manual_seed(0)
    m = _Toy()
    inject_lora(m, ["query_proj"], rank=2, alpha=4, dropout=0.0)
    freeze_all_but_lora(m, unfreeze_output_head=True, unfreeze_modules=["layers.1.ff0"])
    tr = {n for n, p in m.named_parameters() if p.requires_grad}
    assert {"layers.1.ff0.weight", "layers.1.ff0.bias", "output_head.weight"} <= tr and "layers.0.ff0.weight" not in tr
    sd = lora_state_dict(m)
    assert "layers.1.ff0.weight" in sd and "layers.0.ff0.weight" not in sd
    with torch.no_grad():
        m.layers[1].ff0.weight.add_(1.0)
    sd = lora_state_dict(m)
    fresh = _Toy(); inject_lora(fresh, ["query_proj"], rank=2, alpha=4, dropout=0.0)
    load_lora_state_dict(fresh, sd, allowed_prefixes=["output_head", "layers.1.ff0"])
    assert torch.equal(fresh.layers[1].ff0.weight, m.layers[1].ff0.weight)
    with pytest.raises(ValueError, match="unexpected"):
        load_lora_state_dict(_with_lora(), sd)                                  # 不允许 layers.1.ff0 前缀
    with pytest.raises(ValueError, match="缺"):
        load_lora_state_dict(_with_lora(), {k: v for k, v in sd.items() if not k.startswith("layers.1.ff0")},
                             allowed_prefixes=["output_head", "layers.1.ff0"])


def _with_lora():
    t = _Toy(); inject_lora(t, ["query_proj"], rank=2, alpha=4, dropout=0.0)
    return t
