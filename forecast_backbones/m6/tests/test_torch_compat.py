import pytest
import torch
import torch.nn as nn
from forecast_backbones.m6._torch_compat import ensure_rmsnorm, ensure_finite_running_stats


def test_ensure_rmsnorm_provides_working_rmsnorm():
    ensure_rmsnorm()
    assert hasattr(nn, "RMSNorm")
    ln = nn.RMSNorm(8)
    x = torch.randn(3, 5, 8)
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + torch.finfo(x.dtype).eps)
    assert torch.allclose(ln(x), ref, atol=1e-6)
    assert list(ln.state_dict().keys()) == ["weight"] and ln.weight.shape == (8,)


def test_ensure_rmsnorm_is_idempotent():
    first = ensure_rmsnorm()
    assert ensure_rmsnorm() is False          # 第二次一定是空操作
    assert isinstance(first, bool)


def test_ensure_finite_running_stats_gives_finite_grad_on_zero_variance_patch():
    pytest.importorskip("timesfm3")
    from timesfm3.torch import util
    ensure_finite_running_stats()
    n = torch.zeros(1, 1); mu = torch.zeros(1, 1); sigma = torch.zeros(1, 1)
    x = torch.full((1, 1, 4), 3.0, requires_grad=True)          # 恰好零方差、全未掩码的 patch（光伏夜间常态）
    mask = torch.zeros(1, 1, 4, dtype=torch.bool)
    _, _, new_sigma = util.update_running_stats(n, mu, sigma, x, mask)
    new_sigma.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_ensure_finite_running_stats_is_idempotent():
    pytest.importorskip("timesfm3")
    first = ensure_finite_running_stats()
    assert ensure_finite_running_stats() is False          # 第二次一定是空操作
    assert isinstance(first, bool)


def test_ensure_finite_running_stats_forward_bit_identical():
    """垫片只挡梯度，不许挪前向值：零方差 patch 与非零方差 patch 混在同一 batch 里，
    补丁前后逐位相同（route A 用同一个 update_running_stats，前向必须与 master 逐位一致）。"""
    pytest.importorskip("timesfm3")
    from timesfm3.torch import util
    ensure_finite_running_stats()
    patched = util.update_running_stats
    orig = getattr(patched, "_m6_original", None)
    assert orig is not None
    n = torch.zeros(1, 2); mu = torch.zeros(1, 2); sigma = torch.zeros(1, 2)
    x = torch.tensor([[[3.0, 3.0, 3.0, 3.0], [1.0, 2.0, 3.0, 4.0]]])   # 行0恒定（零方差）、行1有方差
    mask = torch.zeros(1, 2, 4, dtype=torch.bool)
    out_patched = patched(n, mu, sigma, x, mask)
    out_orig = orig(n, mu, sigma, x, mask)
    for a, b in zip(out_patched, out_orig):
        assert torch.equal(a, b)


def test_ensure_finite_running_stats_second_sqrt_guarded():
    """第二个 sqrt（new_sigma）作用在一个不同的零：n>0、sigma=0、mu 与新 patch 均值相同时，
    var_sum 的四项各自为 0，new_var 恰好为 0——即便第一个 sqrt（inc_sigma）已经安全，
    这里若不单独守卫仍会在反向炸出 NaN，证明两处守卫缺一不可。"""
    pytest.importorskip("timesfm3")
    from timesfm3.torch import util
    ensure_finite_running_stats()
    n = torch.full((1, 1), 4.0); mu = torch.full((1, 1), 3.0); sigma = torch.zeros(1, 1)
    x = torch.full((1, 1, 4), 3.0, requires_grad=True)          # 与已有 running mean 完全相同的恒定 patch
    mask = torch.zeros(1, 1, 4, dtype=torch.bool)
    _, _, new_sigma = util.update_running_stats(n, mu, sigma, x, mask)
    new_sigma.sum().backward()
    assert torch.isfinite(x.grad).all()
