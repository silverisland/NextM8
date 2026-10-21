"""Joint history patches + per-step weather/hint features -> direct point head."""

from __future__ import annotations

import torch
from torch import nn


class Rum(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        self.length = cfg["dataset"]["context_len"]
        self.horizon = cfg["dataset"]["pred_len"]
        self.patch = m["patch_len"]
        self.history_dropout = cfg["train"]["history_dropout"]
        self.hint_dropout = cfg["train"]["hint_dropout"]
        weather = len(cfg["features"]["past_future"])
        channels, width = weather + 1, m["d_model"]
        self.patch_projection = nn.Linear(2 * self.patch, width)
        self.variable = nn.Embedding(channels, width)
        self.time_projection = nn.Linear(5, width)
        self.context_token = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.normal_(self.context_token, std=0.02)
        self.encoder = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    width,
                    m["n_heads"],
                    m["d_ff"],
                    m["dropout"],
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(m["n_layers"])
            ]
        )
        self.context_projection = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, m["context_dim"])
        )
        known_dim = 2 * weather + 5
        self.weather = nn.Sequential(
            nn.Linear(known_dim, m["weather_dim"]),
            nn.GELU(),
            nn.Linear(m["weather_dim"], m["weather_dim"]),
            nn.GELU(),
        )
        head_dim = m["weather_dim"] + known_dim + m["context_dim"] + channels + 1 + 2
        self.head = nn.Sequential(
            nn.Linear(head_dim, m["head_hidden"]),
            nn.GELU(),
            nn.Dropout(m["dropout"]),
            nn.Linear(m["head_hidden"], m["head_dim"]),
            nn.GELU(),
            nn.Linear(m["head_dim"], 1),
        )

    def forward(self, f):
        size = f.history.shape[0]
        mask = f.history_mask
        available = mask[:, :, 0].any(dim=1)
        if self.training:
            available = available & (
                torch.rand(size, device=mask.device) >= self.history_dropout
            )
        # A disabled context is identical to the no-history inference path.
        mask = mask & available[:, None, None]
        values = (
            torch.where(mask, f.history, 0)
            .transpose(1, 2)
            .unfold(-1, self.patch, self.patch)
        )
        patch_mask = mask.transpose(1, 2).unfold(-1, self.patch, self.patch)
        tokens = self.patch_projection(torch.cat([values, patch_mask.float()], -1))
        ends = torch.arange(self.patch - 1, self.length, self.patch, device=mask.device)
        position = ((ends + 1 - self.length) / self.length)[None, :, None].expand(
            size, -1, -1
        )
        time = self.time_projection(torch.cat([f.history_time[:, ends], position], -1))
        tokens = tokens + self.variable.weight[None, :, None] + time[:, None]
        tokens = torch.cat(
            [self.context_token.expand(size, -1, -1), tokens.flatten(1, 2)], 1
        )
        padding = torch.cat(
            [
                torch.zeros(size, 1, dtype=torch.bool, device=mask.device),
                ~patch_mask.any(-1).flatten(1, 2),
            ],
            1,
        )
        for layer in self.encoder:
            tokens = layer(tokens, src_key_padding_mask=padding)
        context = self.context_projection(tokens[:, 0]) * available[:, None]
        quality = torch.cat([mask.float().mean(1), available[:, None].float()], -1)
        hint_mask = f.hint_mask
        if self.training:
            keep = torch.rand(size, 1, device=mask.device) >= self.hint_dropout
            hint_mask = hint_mask & keep
        hint = torch.where(hint_mask, f.hint, 0)
        # Fixed scale of 48 hours; timestamps start at issue + 15 minutes.
        lead = (
            torch.arange(1, self.horizon + 1, device=mask.device, dtype=torch.float32)
            / 192
        )
        known = torch.cat(
            [
                f.future,
                f.future_mask.float(),
                f.future_time,
                lead[None, :, None].expand(size, -1, -1),
            ],
            -1,
        )
        inputs = torch.cat(
            [
                self.weather(known),
                known,
                context[:, None].expand(-1, self.horizon, -1),
                quality[:, None].expand(-1, self.horizon, -1),
                hint[:, :, None],
                hint_mask[:, :, None].float(),
            ],
            -1,
        )
        return self.head(inputs).squeeze(-1)
