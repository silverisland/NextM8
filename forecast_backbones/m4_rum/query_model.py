"""Future queries attend to joint historical patches and the known future sequence.

All future inputs are forecasts available at issue time, never future observations.
Legacy Rum remains separate so its checkpoints and numerical behavior stay stable.
"""

import torch
from torch import nn


class QueryRum(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m, ds = cfg["model"], cfg["dataset"]
        self.length, self.horizon = ds["context_len"], ds["pred_len"]
        self.patch = m["patch_len"]
        self.history_dropout = cfg["train"]["history_dropout"]
        self.hint_dropout = cfg["train"]["hint_dropout"]
        self.is_quantile = cfg["train"]["loss"] == "pinball"
        channels = len(cfg["features"]["past_future"]) + 1
        width = m["d_model"]
        # PV and weather within the same patch are encoded together.
        self.patch_projection = nn.Linear((2 * channels + 4) * self.patch, width)
        self.position = nn.Parameter(
            torch.randn(1, self.length // self.patch, width) * 0.02
        )
        self.empty_token = nn.Parameter(torch.randn(1, 1, width) * 0.02)
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
        # Weather + masks, clock, lead, hint + mask, last-day/mean-day PV + masks.
        self.input_dim = 2 * (channels - 1) + 5 + 2 + 4
        self.query_projection = nn.Linear(self.input_dim, width)
        self.decoder = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    width,
                    m["n_heads"],
                    m["d_ff"],
                    m["dropout"],
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(m["decoder_layers"])
            ]
        )
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(
            nn.Linear(width + self.input_dim + channels, m["head_hidden"]),
            nn.GELU(),
            nn.Dropout(m["dropout"]),
            nn.Linear(m["head_hidden"], m["head_dim"]),
            nn.GELU(),
            nn.Linear(m["head_dim"], len(m["quantiles"]) if self.is_quantile else 1),
        )

    def daily_profile(self, history, mask):
        """Align observed daily profiles to S+15min, repeat for both future days."""
        power = torch.where(mask[..., 0], history[..., 0], 0)
        days = power.reshape(len(power), -1, 96)
        valid = mask[..., 0].reshape(len(power), -1, 96)
        count = valid.sum(1)
        mean = days.sum(1) / count.clamp_min(1)
        profile = torch.stack(
            [days[:, -1], valid[:, -1].float(), mean, count.float() / days.shape[1]], -1
        )
        return profile.repeat(1, self.horizon // 96, 1)

    def forward(self, f):
        size = f.history.shape[0]
        available = f.history_mask[..., 0].any(1)
        if self.training:
            available = available & (
                torch.rand(size, device=f.history.device) >= self.history_dropout
            )
        mask = f.history_mask & available[:, None, None]
        history = torch.where(mask, f.history, 0)
        history_input = torch.cat([history, mask.float(), f.history_time], -1)
        patches = history_input.reshape(size, self.length // self.patch, -1)
        memory = self.patch_projection(patches) + self.position
        memory = torch.cat([self.empty_token.expand(size, -1, -1), memory], 1)
        padding = ~mask.reshape(size, self.length // self.patch, -1).any(-1)
        padding = torch.cat(
            [torch.zeros(size, 1, dtype=torch.bool, device=mask.device), padding], 1
        )
        for layer in self.encoder:
            memory = layer(memory, src_key_padding_mask=padding)
        hint_mask = f.hint_mask
        if self.training:
            hint_mask = hint_mask & (
                torch.rand(size, 1, device=mask.device) >= self.hint_dropout
            )
        hint = torch.where(hint_mask, f.hint, 0)
        lead = (
            torch.arange(1, self.horizon + 1, device=mask.device, dtype=torch.float32)
            / 192
        )
        inputs = torch.cat(
            [
                f.future,
                f.future_mask.float(),
                f.future_time,
                lead[None, :, None].expand(size, -1, -1),
                hint[..., None],
                hint_mask[..., None].float(),
                self.daily_profile(history, mask),
            ],
            -1,
        )
        query = self.query_projection(inputs)
        # Bidirectional self-attention is safe here: all 192 NWP/hint steps are
        # known at issue time. No labels or observed future PV enter the decoder.
        for layer in self.decoder:
            query = layer(query, memory, memory_key_padding_mask=padding)
        quality = mask.float().mean(1)[:, None].expand(-1, self.horizon, -1)
        output = self.head(torch.cat([self.norm(query), inputs, quality], -1))
        return output if self.is_quantile else output.squeeze(-1)
