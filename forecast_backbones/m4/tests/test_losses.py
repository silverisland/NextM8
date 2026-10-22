"""Compare against the original M4 main-loss expression and gradients."""

import pytest
import torch

from forecast_backbones.m4.losses import M4MainLoss


@pytest.mark.parametrize("kind", ["mae", "mse", "huber"])
@pytest.mark.parametrize("capacity_kind", ["scalar", "row", "time", "fallback"])
def test_original_expression_and_gradient(kind, capacity_kind):
    target = torch.tensor([[-2.0, 0.0, 0.5, 40.0], [0.0, 10.0, 80.0, 100.0]])
    prediction = torch.tensor(
        [[1.0, 2.0, 3.0, 35.0], [4.0, 15.0, 70.0, 150.0]], requires_grad=True
    )
    capacity = {
        "scalar": 100.0,
        "row": torch.tensor([50.0, 200.0]),
        "time": torch.tensor([[10.0, 20.0, 30.0, 40.0], [100.0, 200.0, 300.0, 400.0]]),
        "fallback": None,
    }[capacity_kind]
    # Original compute_loss main branch, before lambda_main and the removed nw term.
    cap = capacity
    if cap is None:
        cap = torch.max(target, torch.ones_like(target))
        cap_v = cap.unsqueeze(1)
    elif isinstance(cap, torch.Tensor) and cap.dim() == 1:
        cap_v = cap.view(-1, 1, 1)
    else:
        cap_v = cap
    cap_row = (
        cap.view(-1, 1) if isinstance(cap, torch.Tensor) and cap.dim() == 1 else cap
    )
    criterion = {
        "mae": torch.nn.L1Loss,
        "mse": torch.nn.MSELoss,
        "huber": torch.nn.HuberLoss,
    }[kind]()
    # Per-step capacities use the original [B,1,T] shape for target normalization.
    if isinstance(cap_v, torch.Tensor) and cap_v.dim() == 2:
        cap_v = cap_v.unsqueeze(1)
    expected = criterion(
        (prediction / cap_row).unsqueeze(1), target.unsqueeze(1) / cap_v
    )
    actual = M4MainLoss(kind)(prediction, target, capacity)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected_grad = torch.autograd.grad(expected, prediction, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(actual, prediction)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def test_main_loss_is_not_scaled_or_masked():
    loss = M4MainLoss()
    assert (
        loss(torch.tensor([[0.0, 10.0]]), torch.tensor([[2.0, 6.0]]), 2.0).item() == 1.5
    )
    assert torch.isnan(
        loss(torch.zeros(1, 2), torch.tensor([[float("nan"), 2.0]]), 100.0)
    )
