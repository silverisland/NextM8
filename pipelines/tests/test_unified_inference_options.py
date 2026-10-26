"""Backward-compatible fallback and strict, single-forward reporting hooks."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import Subset

from pipelines.pipeline_wrapper_inference import unified_inference
from pipelines.transformer_dataset import UnifiedDataset


class Probe(torch.nn.Module):
    prediction_length = 2

    def __init__(self, bad=None):
        super().__init__()
        self.calls = 0
        self.bad = bad

    def forward(self, batch):
        self.calls += 1
        assert "truth" not in batch
        values = batch["value"].reshape(-1, 1).float()
        if self.bad == "error" and (values == 2).any():
            raise ValueError("bad sample")
        out = values.repeat(1, 2)
        if self.bad == "nan":
            out[:] = float("nan")
        if self.bad == "shape":
            out = out[:, :1]
        return SimpleNamespace(logits=out, aux={"value": values})


def frame():
    return pd.DataFrame(
        {
            "timestamp_win": pd.date_range("2025-01-01", periods=3, freq="h"),
            "value": [1, 2, 3],
            "truth": [9, 9, 9],
        }
    )


def test_subset_hook_and_no_second_forward():
    model = Probe()
    ds = UnifiedDataset(frame(), timestamp_cols=["timestamp_win"], for_hf_trainer=True)
    seen = []

    def hook(batch, output):
        assert "truth" in batch["batch_dict"]
        seen.extend(output.aux["value"].flatten().tolist())

    result = unified_inference(
        model,
        Subset(ds, [2, 0, 1]),
        entity_cols=["timestamp_win"],
        batch_size=2,
        exclude_input_cols=["truth"],
        error_policy="raise",
        on_batch_output=hook,
        progress=False,
    )
    assert model.calls == 2 and seen == [3, 1, 2]
    np.testing.assert_array_equal(np.stack(result.Probe), [[3, 3], [1, 1], [2, 2]])


def test_default_fallback_preserves_rows_and_none():
    with pytest.warns(UserWarning, match="bad sample"):
        result = unified_inference(
            Probe("error"),
            frame(),
            entity_cols=["timestamp_win"],
            exclude_input_cols=["truth"],
            progress=False,
        )
    assert len(result) == 3 and result.Probe.iloc[1] is None
    np.testing.assert_array_equal(result.Probe.iloc[2], [3, 3])


@pytest.mark.parametrize(
    "bad,match", [("error", "bad sample"), ("nan", "non-finite"), ("shape", "horizon")]
)
def test_strict_policy_never_silently_drops_invalid_predictions(bad, match):
    model = Probe(bad)
    with pytest.raises(ValueError, match=match):
        unified_inference(
            model,
            frame(),
            entity_cols=["timestamp_win"],
            exclude_input_cols=["truth"],
            error_policy="raise",
            progress=False,
        )
    assert model.calls == 1


def test_hook_errors_propagate_and_do_not_retry():
    model = Probe()

    def fail(*_):
        raise RuntimeError("report assembly failed")

    with pytest.raises(RuntimeError, match="assembly"):
        unified_inference(
            model,
            frame(),
            entity_cols=["timestamp_win"],
            exclude_input_cols=["truth"],
            error_policy="raise",
            on_batch_output=fail,
            progress=False,
        )
    assert model.calls == 1
