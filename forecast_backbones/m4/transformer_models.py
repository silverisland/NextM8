"""M4 PatchTST adapter for training, strict reload and unified inference."""

from pathlib import Path

import torch
from torch import nn

from interfaces.modeloutput import TransformersAdaptedModelOutput
from pipelines.forecasting.checkpoints import load_weights
from pipelines.forecasting.stations import load_station_meta

from .config import load_config
from .features import FeatureBuilder
from .layers.PatchTST_backbone import PatchTST_backbone
from .layers.RevIN import RevIN
from .losses import M4QuantileLoss


class TransformersAdaptedM4Model(nn.Module):
    def __init__(self, model_config=None):
        super().__init__()
        self.cfg = load_config(model_config)
        ds, m = self.cfg["dataset"], self.cfg["model"]
        self.context_length, self.prediction_length = ds["context_len"], ds["pred_len"]
        self.labels_name = [ds["label_col"]]
        self.main_loss = M4QuantileLoss()
        self.station_meta = load_station_meta(
            ds["station_meta"], ds["station_meta_cols"]
        )
        self.fe = FeatureBuilder(self.cfg, self.station_meta)
        channels = len(self.cfg["features"]["past_future"]) + 1
        self.revin = m["revin"]
        if self.revin:
            self.weather_revin_layer = RevIN(
                channels - 1, affine=m["affine"], subtract_last=m["subtract_last"]
            )
            self.pv_revin_layer = RevIN(
                1, affine=m["affine"], subtract_last=m["subtract_last"]
            )
        self.model = PatchTST_backbone(
            c_in=channels,
            context_window=self.context_length + self.prediction_length,
            target_window=self.prediction_length,
            patch_len=m["patch_len"],
            stride=m["stride"],
            n_layers=m["n_layers"],
            d_model=m["d_model"],
            n_heads=m["n_heads"],
            d_ff=m["d_ff"],
            norm=m["norm"],
            dropout=m["dropout"],
            act=m["activation"],
            fc_dropout=m["fc_dropout"],
            head_dropout=m["head_dropout"],
            padding_patch=m["padding_patch"],
            embedding_dim=m["embedding_dim"],
        )
        self.weights_loaded = False
        weights = self.cfg["train"]["weights_path"]
        if weights is not None:
            if not Path(weights).is_file():
                raise FileNotFoundError(
                    f"M4 weights not found: {weights}; set weights_path=null for a new model"
                )
            load_weights(self, weights)
            self.weights_loaded = True
        self.eval()

    @property
    def config(self):
        return self.cfg

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def compute_metrics(eval_pred):
        return {}

    def compute_loss(self, y_pred, y_true):
        return self.main_loss(y_pred, y_true)

    def forward(self, batch_dict, labels=None):
        device = next(self.parameters()).device
        features = self.fe.build(batch_dict, device, training=self.training)
        # M4's concatenation, RevIN, PatchTST and embedding head stay local.
        weather = torch.cat((features.history[:, :, 1:], features.future), dim=1)
        pv = torch.cat((features.history[:, :, :1], features.hint.unsqueeze(-1)), dim=1)
        if self.revin:
            weather = self.weather_revin_layer(weather, "norm")
            pv = self.pv_revin_layer(pv, "norm")
        prediction, embedding = self.model(weather.transpose(1, 2), pv.transpose(1, 2))
        if self.revin:
            prediction = self.pv_revin_layer(
                prediction.transpose(1, 2), "denorm"
            ).transpose(1, 2)
        prediction = prediction[:, 0].float()
        logits = prediction * features.capacity
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
                or target.shape[1] < self.prediction_length
            ):
                raise ValueError("Labels must cover [B,192] future positions")
            target = target[:, : self.prediction_length]
            loss = self.compute_loss(logits, target)
        return TransformersAdaptedModelOutput(
            loss=loss,
            logits=logits,
            predictions=logits,
            embeddings=embedding,
            aux={
                "cap": features.capacity[:, 0],
                "station_id": features.station_id,
                "history_unavailable": ~features.history_mask[:, :, 0].any(1),
                "hint_unavailable": ~features.hint_mask.any(1),
                "future_weather_unavailable": ~features.future_mask.flatten(1).any(1),
            },
        )
