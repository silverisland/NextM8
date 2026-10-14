import torch
import pytest
from forecast_backbones.m8.layers import GatedLongConv, GatedDeltaNet, CausalAttention, Block, ResidualMLP

D, H, T, B = 16, 2, 12, 3


@pytest.mark.parametrize("kind", ["conv", "deltanet", "attention"])
def test_mixer_is_causal(kind):
    torch.manual_seed(0)
    blk = Block(kind, D, H, 32, T, dropout=0.0).eval()
    x = torch.randn(B, T, D)
    y = blk(x)
    x2 = x.clone(); x2[:, 7:] += torch.randn(B, T - 7, D)      # 扰动 7 之后
    y2 = blk(x2)
    assert torch.allclose(y[:, :7], y2[:, :7], atol=1e-5), f"{kind}: 过去输出受未来扰动影响"
    assert not torch.allclose(y[:, 7:], y2[:, 7:]), f"{kind}: 未来扰动没有传到输出"
    assert y.shape == (B, T, D)


def test_deltanet_state_retrieval():
    """delta 规则：写入 (k, v) 后用同一个 k 查询应接近 v（β→1 时精确）。"""
    torch.manual_seed(0)
    m = GatedDeltaNet(D, 1, short_kernel=1, gated=False)
    with torch.no_grad():
        m.beta_proj.weight.zero_(); m.beta_proj.bias.fill_(20.0)      # β≈1
    x = torch.randn(1, 2, D)
    x[:, 1] = x[:, 0]                                               # 第二步与第一步同输入 → 同 k, v
    out = m(x)
    assert torch.isfinite(out).all()


def test_removed_mixers_rejected():
    with pytest.raises(ValueError, match="conv / deltanet / attention"):
        Block("slstm", D, H, 32, T)


def test_residual_mlp_shapes():
    m = ResidualMLP(5, 7, 3)
    assert m(torch.randn(4, 9, 5)).shape == (4, 9, 3)


def test_long_conv_short_sequence():
    m = GatedLongConv(D, seq_len=T)
    assert m(torch.randn(B, T - 3, D)).shape == (B, T - 3, D)
