"""Original M4 loss supplied by the user; independent of all other models."""

import numpy as np
import torch
from torch import nn


def quantile_loss(y_pred, y_true):
    """Sum nine pinball losses on the same point prediction, without rescaling.

    Each term is averaged over all elements; the nine terms are summed, not
    averaged. The symmetric levels make this mathematically 4.5 * MAE.
    Keep the original expression to preserve its floating-point behavior.
    """
    error = y_true - y_pred
    loss = 0
    for quantile in np.arange(0.1, 1.0, 0.1):
        loss += torch.mean(torch.maximum(quantile * error, (quantile - 1) * error))
    return loss


class M4QuantileLoss(nn.Module):
    def forward(self, prediction, target):
        if prediction.ndim != 2 or prediction.shape != target.shape:
            raise ValueError("Prediction and target must have the same [B,T] shape")
        return quantile_loss(prediction, target)
