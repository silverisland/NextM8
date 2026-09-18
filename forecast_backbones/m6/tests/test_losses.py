import numpy as np
import torch
from forecast_backbones.m6.losses import pinball_loss, nanwang_r2_loss
from forecast_backbones.m6.decision import MODEL_LEVELS

def test_pinball_zero_when_all_quantiles_equal_truth():
    y = torch.full((2, 5), 3.0); q = y[..., None].repeat(1, 1, 9); m = torch.ones(2, 5)
    assert pinball_loss(q, y, MODEL_LEVELS, m).item() == 0.0

def test_pinball_symmetric_levels_sum_to_abs_error():
    # 同一条预测喂 q 与 1-q：ρ_q(e)+ρ_{1-q}(e)=|e|；9 档总和 = 4.5|e|，除以 9 → 0.5|e|
    y = torch.tensor([[10.0]]); q = torch.full((1, 1, 9), 7.0); m = torch.ones(1, 1)
    assert abs(pinball_loss(q, y, MODEL_LEVELS, m).item() - 0.5 * 3.0) < 1e-6

def test_pinball_mask_excludes_points():
    y = torch.tensor([[10.0, 10.0]]); q = torch.zeros(1, 2, 9); q[0, 1] = 10.0
    m = torch.tensor([[0.0, 1.0]])
    assert pinball_loss(q, y, MODEL_LEVELS, m).item() == 0.0
    assert pinball_loss(q, y, MODEL_LEVELS, torch.ones(1, 2)).item() > 0.0

def test_nanwang_r2_matches_reference():
    y = torch.tensor([[0.0, 10.0, 50.0, 100.0]]); p = torch.tensor([[5.0, 20.0, 40.0, 100.0]])
    cap = torch.tensor([[100.0]]); m = torch.ones(1, 4)
    d = np.maximum(y.numpy(), 20.0)
    ref = np.mean(((y.numpy() - p.numpy()) / d) ** 2)
    assert abs(nanwang_r2_loss(p, y, cap, m).item() - ref) < 1e-6
