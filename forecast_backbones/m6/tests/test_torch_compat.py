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
