"""Future queries attend to joint historical patches and the known future sequence.

All future inputs are forecasts available at issue time, never future observations.
Legacy Rum remains separate so its checkpoints and numerical behavior stay stable.
"""

import numpy as np
import torch
from torch import nn

from .query_extensions import LocalWeatherEncoder, WeatherDayReference


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
        # Construct last: enabling this option preserves the initial base weights
        # for matched-seed ablations. Disabled models retain the old state_dict.
        self.station_embedding = None
        self.dataset_cfg = ds  # Training fills known_stations after loading data.
        self.slot_dropout = cfg["train"]["slot_dropout"]
        if m["station_embedding"]:
            self.station_embedding = nn.Embedding(m["n_station_slots"] + 1, width)
            nn.init.normal_(self.station_embedding.weight, std=0.02)
        # Disabled branches add neither parameters nor random draws. Old runs
        # keep their exact state_dict keys and initial base weights.
        ext = m["extensions"]
        output_dim = len(m["quantiles"]) if self.is_quantile else 1
        head_input = width + self.input_dim + channels
        self.head_skip = None
        self.day_reference = None
        self.local_weather = None
        self.hint_gate = None
        if ext["residual_head"]:
            self.head_skip = nn.Linear(head_input, output_dim)
            nn.init.zeros_(self.head_skip.weight)
            nn.init.zeros_(self.head_skip.bias)
        if ext["weather_day_reference"]:
            self.day_reference = WeatherDayReference(channels - 1, ext["day_hidden"])
            self.day_query = nn.Linear(4, width)
            self.day_output = nn.Linear(4, output_dim, bias=False)
        if ext["local_weather"]:
            self.local_weather = LocalWeatherEncoder(
                2 * (channels - 1) + 5, width, ext["local_kernels"], m["dropout"]
            )
        if ext["hint_gate"]:
            self.hint_gate = nn.Sequential(
                nn.Linear(head_input, ext["hint_gate_hidden"]),
                nn.GELU(),
                nn.Linear(ext["hint_gate_hidden"], 1),
                nn.Sigmoid(),
            )

    def station_slots(self, station_id, device):
        """M8-compatible hashed slots; only training stations get a named slot."""
        if self.station_embedding is None:
            raise RuntimeError("station_embedding is disabled")
        ids = np.asarray(station_id, dtype=np.int64).reshape(-1)
        known = self.dataset_cfg["known_stations"]
        slots = np.where(
            np.isin(ids, known if known is not None else []),
            ids % (self.station_embedding.num_embeddings - 1) + 1,
            0,
        )
        slots = torch.as_tensor(slots, dtype=torch.long, device=device)
        if self.training and self.slot_dropout > 0:
            slots = slots.masked_fill(
                torch.rand(slots.shape, device=device) < self.slot_dropout, 0
            )
        return slots

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
        station = None
        if self.station_embedding is not None:
            # Draw dropout once per sample, shared by history and future paths.
            slots = self.station_slots(f.station_id, memory.device)
            station = self.station_embedding(slots)[:, None, :]
            memory = memory + station
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
        if station is not None:
            query = query + station
        day = None
        if self.day_reference is not None:
            day = self.day_reference(history, mask, f.future, f.future_mask)
            query = query + self.day_query(day)
        if self.local_weather is not None:
            known = torch.cat(
                [
                    f.future,
                    f.future_mask.float(),
                    f.future_time,
                    lead[None, :, None].expand(size, -1, -1),
                ],
                -1,
            )
            query = query + self.local_weather(known)
        # Bidirectional self-attention is safe here: all 192 NWP/hint steps are
        # known at issue time. No labels or observed future PV enter the decoder.
        for layer in self.decoder:
            query = layer(query, memory, memory_key_padding_mask=padding)
        quality = mask.float().mean(1)[:, None].expand(-1, self.horizon, -1)
        head_inputs = torch.cat([self.norm(query), inputs, quality], -1)
        output = self.head(head_inputs)
        if self.head_skip is not None:
            output = output + self.head_skip(head_inputs)
        if day is not None:
            output = output + self.day_output(day)
        if self.hint_gate is not None:
            # A shared scalar gate shifts all quantiles; the main head learns
            # the quantile-specific correction. Mask/dropout is already in hint.
            output = output + self.hint_gate(head_inputs) * hint[..., None]
        return output if self.is_quantile else output.squeeze(-1)
