"""Freeze the user's original expression as the value/gradient reference."""

import numpy as np
import pytest
import torch

from forecast_backbones.m4.losses import M4QuantileLoss, quantile_loss


def original_quantile_loss(y_pred, y_true):
    error = y_true - y_pred
    loss = 0
    for quantile in np.arange(0.1, 1, 0.1):
        loss += torch.mean(torch.max(quantile * error, (quantile - 1) * error))
    return loss


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(1, 192), (3, 192)])
def test_original_loss_and_gradient(dtype, shape):
    generator = torch.Generator().manual_seed(42)
    pred = torch.randn(shape, generator=generator, dtype=dtype).requires_grad_()
    target = torch.randn(shape, generator=generator, dtype=dtype)
    target[0, 0] = pred.detach()[0, 0]  # Also cover the zero-error subgradient.
    expected = original_quantile_loss(pred, target)
    actual = M4QuantileLoss()(pred, target)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grad_expected = torch.autograd.grad(expected, pred, retain_graph=True)[0]
    grad_actual = torch.autograd.grad(actual, pred)[0]
    torch.testing.assert_close(grad_actual, grad_expected, rtol=0, atol=0)


def test_sum_not_mean_no_normalization_and_no_separate_quantile_heads():
    pred = torch.tensor([[1.0, 98.0]])
    target = torch.tensor([[0.0, 100.0]])
    # Errors 1 and 2 give MAE 1.5; summing the nine levels gives 6.75.
    assert quantile_loss(pred, target).item() == pytest.approx(6.75)
    torch.testing.assert_close(
        quantile_loss(100 * pred, 100 * target), 100 * quantile_loss(pred, target)
    )


def test_equal_predictions_and_nan_labels_follow_original():
    target = torch.tensor([[0.0, 100.0]])
    assert quantile_loss(target, target).item() == 0
    target[0, 0] = float("nan")
    assert torch.isnan(quantile_loss(torch.zeros_like(target), target))


def test_reject_accidental_broadcasting():
    with pytest.raises(ValueError, match="same"):
        M4QuantileLoss()(torch.zeros(2, 192), torch.zeros(2, 1))
