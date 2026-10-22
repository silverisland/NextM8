"""M4's original main loss, without auxiliary losses or model dependencies."""

import torch
from torch import nn


class M4MainLoss(nn.Module):
    """Mean criterion(prediction / capacity, target / capacity), in raw units.

    Preserve the original fallback: max(target, 1) at each future position.
    Missing labels retain the criterion's original NaN behavior, not a new
    masked or per-row reduction. No lambda_main multiplier is applied.
    """

    def __init__(self, loss_type="mae"):
        super().__init__()
        criteria = {"mae": nn.L1Loss, "mse": nn.MSELoss, "huber": nn.HuberLoss}
        if loss_type not in criteria:
            raise ValueError("M4 main loss must be mae, mse or huber")
        self.criterion = criteria[loss_type]()

    def forward(self, prediction, target, capacity=None):
        if prediction.ndim != 2 or prediction.shape != target.shape:
            raise ValueError("Prediction and target must have the same [B,T] shape")
        if capacity is None:
            capacity = torch.maximum(target, torch.ones_like(target))
        else:
            capacity = torch.as_tensor(
                capacity, device=prediction.device, dtype=prediction.dtype
            )
            if capacity.ndim == 1:
                capacity = capacity[:, None]
            if capacity.ndim > 2 or (
                capacity.ndim == 2
                and capacity.shape not in ((prediction.shape[0], 1), prediction.shape)
            ):
                raise ValueError("Capacity must be scalar, [B], [B,1] or [B,T]")
        return self.criterion(
            (prediction / capacity).unsqueeze(1),
            (target / capacity).unsqueeze(1),
        )
