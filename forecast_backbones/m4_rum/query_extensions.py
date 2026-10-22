"""Optional QueryRum branches, using only inputs available at issue time."""

import torch
from torch import nn
from torch.nn import functional as F


class WeatherDayReference(nn.Module):
    """Match each future phase to observed historical days by their weather.

    Power values are aggregated, never used to select the matching day. If no
    weather-matched day is available, use the mean of valid same-phase powers.
    All-missing history produces four zeros, including the coverage indicators.
    """

    def __init__(self, n_weather, hidden):
        super().__init__()
        self.weather_weights = nn.Parameter(torch.zeros(n_weather))
        # Future/past weather, absolute difference, three masks, and relative age.
        self.score = nn.Sequential(
            nn.Linear(6 * n_weather + 1, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, history, mask, future, future_mask):
        batch, length, channels = history.shape
        days, horizon = length // 96, future.shape[1]
        phase = torch.arange(horizon, device=history.device) % 96
        # Phase 0 is S+15 minutes (history L-96); no midnight assumption.
        hist = history.reshape(batch, days, 96, channels)[:, :, phase]
        valid = mask.reshape(batch, days, 96, channels)[:, :, phase]
        past_weather = torch.where(valid[..., 1:], hist[..., 1:], 0)
        fut = torch.where(future_mask, future, 0)[:, None].expand(-1, days, -1, -1)
        fm = future_mask[:, None].expand_as(past_weather)
        hm = valid[..., 1:]
        common = fm & hm
        delta = torch.where(common, fut - past_weather, 0)
        age = (
            torch.arange(days, 0, -1, device=history.device, dtype=history.dtype) / days
        )
        age = age[None, :, None, None].expand(batch, -1, horizon, -1)
        score_inputs = torch.cat(
            [
                fut,
                past_weather,
                delta.abs(),
                fm.float(),
                hm.float(),
                common.float(),
                age,
            ],
            -1,
        )
        distance = (delta.square() * F.softplus(self.weather_weights)).sum(-1)
        distance = distance / common.sum(-1).clamp_min(1)
        logits = self.score(score_inputs).squeeze(-1) - distance
        candidates = valid[..., 0] & common.any(-1)
        has_match = candidates.any(1, keepdim=True)
        eligible = torch.where(has_match, candidates, valid[..., 0])
        logits = torch.where(has_match, logits, torch.zeros_like(logits))
        # Finite masking and normalization avoid softmax(-inf, ...,-inf) NaNs.
        weights = logits.masked_fill(~eligible, torch.finfo(logits.dtype).min).softmax(
            1
        )
        weights = weights * eligible
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-6)
        power = torch.where(valid[..., 0], hist[..., 0], 0)
        mean = (weights * power).sum(1)
        variance = (weights * (power - mean[:, None]).square()).sum(1)
        return torch.stack(
            [mean, variance, valid[..., 0].float().mean(1), candidates.float().mean(1)],
            -1,
        )


class LocalWeatherEncoder(nn.Module):
    """Parallel depthwise convolutions over known future weather and clock inputs."""

    def __init__(self, input_dim, width, kernels, dropout):
        super().__init__()
        self.project = nn.Linear(input_dim, width)
        self.convs = nn.ModuleList(
            [nn.Conv1d(width, width, k, padding=k // 2, groups=width) for k in kernels]
        )
        self.fuse = nn.Sequential(
            nn.Linear(width * len(kernels), width), nn.GELU(), nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, known):
        hidden = self.project(known).transpose(1, 2)
        local = torch.cat(
            [F.gelu(conv(hidden)).transpose(1, 2) for conv in self.convs], -1
        )
        return self.norm(self.fuse(local))
