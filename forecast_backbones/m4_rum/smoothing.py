"""Row-local PV moving averages; no imputation and no history/label concatenation."""

import torch
from torch.nn import functional as F


def masked_moving_average(values, valid, window, alignment="centered"):
    """Centered W-point arithmetic mean, only for complete valid windows.

    For W=2k+1, the first/last k positions keep their original values. If any
    neighbor is missing, keep the current value instead of changing the divisor.
    Missing positions stay zero with their original mask false; no imputation.
    """
    if values.ndim != 2 or valid.shape != values.shape:
        raise ValueError("Moving average expects values and mask with shape [B,T]")
    if (
        type(window) is not int
        or window < 1
        or window > values.shape[1]
        or window % 2 == 0
    ):
        raise ValueError("Moving average window must be a positive odd integer <= T")
    if alignment != "centered":
        raise ValueError("Moving average alignment must be centered")
    valid = valid.bool() & torch.isfinite(values)
    clean = torch.where(valid, values, 0)
    if window == 1:
        return clean
    left = right = window // 2
    # Keep smoothing in fp32 (or fp64) even under AMP: this is preprocessing,
    # not a learned convolution. Labels and inference inputs use identical math.
    dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=values.device.type, enabled=False):
        kernel = torch.ones(1, 1, window, dtype=dtype, device=values.device)
        sums = F.conv1d(
            F.pad(clean.to(dtype).unsqueeze(1), (left, right)), kernel
        ).squeeze(1)
        counts = F.conv1d(
            F.pad(valid.to(dtype).unsqueeze(1), (left, right)), kernel
        ).squeeze(1)
        mean = sums / window
    # Padding contributes no valid points, so boundary windows cannot qualify.
    return torch.where(counts == window, mean.to(values.dtype), clean)
