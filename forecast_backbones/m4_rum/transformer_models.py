"""Registry / HF / unified inference adapter for M4-Rum."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from interfaces.modeloutput import TransformersAdaptedModelOutput
from pipelines.forecasting.checkpoints import load_weights
from pipelines.forecasting.stations import load_station_meta

from . import model as architecture
from .config import load_config
from .decision import decide
from .features import FeatureBuilder
from .query_model import QueryRum
from .smoothing import masked_moving_average


class TransformersAdaptedM4RumModel(nn.Module):
    def __init__(self, model_config=None):
        super().__init__()
        self.cfg = load_config(model_config)
        ds = self.cfg["dataset"]
        self.context_len, self.pred_len = ds["context_len"], ds["pred_len"]
        self.station_meta = load_station_meta(
            ds["station_meta"], ds["station_meta_cols"]
        )
        self.fe = FeatureBuilder(self.cfg, self.station_meta)
        core = (
            QueryRum
            if self.cfg["model"]["architecture"] == "query_decoder"
            else architecture.Rum
        )
        self.core = core(self.cfg)
        self.levels = tuple(self.cfg["model"]["quantiles"])
        self.weights_loaded = False
        weights = self.cfg["train"]["weights_path"]
        if weights and Path(weights).is_file():
            load_weights(self, weights)
            self.weights_loaded = True
        self.eval()

    @property
    def config(self):
        return self.cfg

    @property
    def context_length(self):
        return self.context_len

    @property
    def prediction_length(self):
        return self.pred_len

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def compute_metrics(eval_pred):
        return {}

    def forward(self, batch_dict, labels=None):
        device = next(self.parameters()).device
        features = self.fe.build(batch_dict, device, training=self.training)
        prediction = self.core(features).float()
        loss = None
        if labels is not None:
            target = (
                labels[self.cfg["dataset"]["label_col"]]
                if isinstance(labels, dict)
                else labels
            )
            target = torch.as_tensor(target, device=device, dtype=torch.float32)
            if (
                target.ndim != 2
                or target.shape[0] != prediction.shape[0]
                or target.shape[1] < self.pred_len
            ):
                raise ValueError("Labels must cover [B,192] future positions")
            target = target[:, : self.pred_len] / features.capacity
            valid = torch.isfinite(target)
            rows = valid.any(dim=1)
            if not rows.any():
                raise ValueError("Batch contains no finite future labels")
            smoothing = self.cfg["preprocess"]["pv_smoothing"]
            if self.training and smoothing["target_window"] > 1:
                target = masked_moving_average(
                    target, valid, smoothing["target_window"], smoothing["alignment"]
                )
            target = torch.where(valid, target, 0)
            if self.core.is_quantile:
                # Train raw quantiles; sorting and p_star belong only to reporting.
                error = target.unsqueeze(-1) - prediction
                levels = prediction.new_tensor(self.levels)
                error = torch.maximum(levels * error, (levels - 1) * error).mean(-1)
            else:
                error = prediction - target
                error = (
                    error.square()
                    if self.cfg["train"]["loss"] == "mse"
                    else error.abs()
                )
            loss = ((error * valid).sum(1) / valid.sum(1).clamp_min(1))[rows].mean()
        aux = {
            "cap": features.capacity[:, 0],
            "station_id": features.station_id,
            "history_unavailable": ~features.history_mask[:, :, 0].any(1),
            "hint_unavailable": ~features.hint_mask.any(1),
            "future_weather_unavailable": ~features.future_mask.flatten(1).any(1),
        }
        if self.core.is_quantile:
            quantiles = prediction * features.capacity.unsqueeze(-1)
            with torch.no_grad():
                selected = decide(
                    quantiles,
                    self.levels,
                    features.capacity,
                    self.cfg["decision"]["point"],
                )
            logits = selected.pop("point")
            aux.update(selected, quantiles=quantiles, levels_cal=list(self.levels))
        else:
            logits = prediction * features.capacity
        return TransformersAdaptedModelOutput(
            loss=loss,
            logits=logits,
            aux=aux,
        )
