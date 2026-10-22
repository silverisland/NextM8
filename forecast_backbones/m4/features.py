"""As-of features with M8 time/capacity conventions; labels never enter here."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from pipelines.forecasting.stations import (
    norm_station_name,
    station_cap_mw,
)


def as_numpy(value):
    return (
        value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
    )


@dataclass
class Features:
    history: torch.Tensor
    history_mask: torch.Tensor
    future: torch.Tensor
    future_mask: torch.Tensor
    hint: torch.Tensor
    hint_mask: torch.Tensor
    capacity: torch.Tensor
    station_id: np.ndarray


class FeatureBuilder:
    def __init__(self, cfg, station_meta):
        self.cfg, self.meta = cfg, station_meta
        self.ds, self.fe = cfg["dataset"], cfg["features"]

    def resolve_col(self, columns, name):
        candidates = [name] + self.fe["col_aliases"].get(name, [])
        for candidate in candidates:
            if candidate in columns:
                return candidate
        for candidate in candidates:
            hits = [c for c in columns if c.lower() == candidate.lower()]
            if len(hits) > 1:
                raise ValueError(f"Ambiguous column {name}: {hits}")
            if hits:
                return hits[0]
        return None

    def check_columns(self, columns):
        """Require power/hint; preserve training weather availability at inference."""
        for col in (self.ds["pv_col"], self.fe["hint_col"]):
            if self.resolve_col(columns, col) is None:
                raise ValueError(f"Training data missing required column {col!r}")
        present = []
        for pair in self.fe["past_future"]:
            resolved = [self.resolve_col(columns, c) for c in pair]
            print(f"[m4] weather {pair} -> {resolved}")
            if all(resolved):
                present.append(pair)
        if not present:
            raise ValueError(
                "Training data has no complete history/future weather pair"
            )
        hint = self.fe["hint_col"]
        print(f"[m4] hint {hint} -> {self.resolve_col(columns, hint)}")
        self.fe["cov_present"] = present
        return present

    def build(self, batch, device, training=False):
        ds = self.ds
        raw_ts = as_numpy(batch[ds["ts_col"]])
        if raw_ts.ndim == 2 and raw_ts.shape[1] == 1:
            raw_ts = raw_ts[:, 0]
        if raw_ts.ndim != 1 or raw_ts.dtype.kind not in "iu":
            raise ValueError("timestamp_win must contain integer epoch seconds [B]")
        if ((raw_ts < 946684800) | (raw_ts >= 4102444800)).any():
            raise ValueError("timestamp_win must use seconds (2000–2099), not ms/ns")
        size, length, horizon = len(raw_ts), ds["context_len"], ds["pred_len"]
        if ds["station_id_col"] in batch:
            raw_id = as_numpy(batch[ds["station_id_col"]]).reshape(-1)
            ids = raw_id.astype(np.int64)
            if not np.array_equal(raw_id.astype(float), ids):
                raise ValueError("station_id must contain integers")
        elif "station" in batch:
            names = as_numpy(batch["station"]).reshape(-1)
            mapping, unknown, conflicts = self.meta.resolve(names)
            if unknown or conflicts:
                raise ValueError(
                    f"Unknown/conflicting station names: {unknown}, {conflicts}"
                )
            ids = np.array(
                [mapping[norm_station_name(n)] for n in names], dtype=np.int64
            )
        else:
            raise ValueError(
                "station_id or station is required to look up fixed capacity"
            )
        if len(ids) != size:
            raise ValueError("Station and timestamp batch sizes differ")
        cap = torch.tensor(
            [station_cap_mw(self.meta, sid) for sid in ids],
            device=device,
            dtype=torch.float32,
        )[:, None]

        def window(col, future=False):
            count = horizon if future else length
            key = self.resolve_col(batch.keys(), col)
            if key is None:
                return torch.full((size, count), float("nan"), device=device)
            value = batch[key]
            if not isinstance(value, torch.Tensor):
                value = np.asarray(value)
                if value.dtype == object:
                    value = np.stack(value)
            x = torch.as_tensor(value, device=device, dtype=torch.float32)
            if x.ndim != 2 or x.shape[0] != size or x.shape[1] == 0:
                raise ValueError(f"{col} must be a nonempty [B,T] array")
            if future:
                if x.shape[1] < horizon:
                    raise ValueError(f"{col} needs at least {horizon} future steps")
                return x[:, :horizon]
            x = x[:, -length:]
            return F.pad(x, (length - x.shape[1], 0), value=float("nan"))

        pv = window(ds["pv_col"]) / cap
        past, future = [pv], []
        for pair, offset, scale in zip(
            self.fe["past_future"],
            self.fe["weather_offsets"],
            self.fe["weather_scales"],
        ):
            allowed = self.fe["cov_present"] is None or pair in self.fe["cov_present"]
            available = allowed and all(self.resolve_col(batch.keys(), c) for c in pair)
            past.append(
                (window(pair[0]) - offset) / scale
                if available
                else torch.full_like(past[0], float("nan"))
            )
            future.append(
                (window(pair[1], True) - offset) / scale
                if available
                else torch.full((size, horizon), float("nan"), device=device)
            )
        hint_key = self.resolve_col(batch.keys(), self.fe["hint_col"])
        if hint_key is None:
            raise ValueError(f"M4 requires hint column {self.fe['hint_col']!r}")
        hint = window(self.fe["hint_col"], True) / cap
        history, weather = torch.stack(past, -1), torch.stack(future, -1)
        hist_mask, weather_mask, hint_mask = (
            torch.isfinite(history),
            torch.isfinite(weather),
            torch.isfinite(hint),
        )
        return Features(
            torch.where(hist_mask, history, 0),
            hist_mask,
            torch.where(weather_mask, weather, 0),
            weather_mask,
            torch.where(hint_mask, hint, 0),
            hint_mask,
            cap,
            ids,
        )
