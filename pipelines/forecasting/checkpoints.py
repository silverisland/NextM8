"""Strict, model-agnostic safetensors checkpoints."""

from pathlib import Path

from torch import nn


def save_weights(model: nn.Module, path) -> None:
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
        str(path),
    )


def load_weights(model: nn.Module, path) -> None:
    from safetensors.torch import load_file

    model.load_state_dict(load_file(str(path)), strict=True)
