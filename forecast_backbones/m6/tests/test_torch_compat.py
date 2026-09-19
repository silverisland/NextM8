import torch
import torch.nn as nn
from forecast_backbones.m6._torch_compat import ensure_rmsnorm


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
