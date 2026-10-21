"""Distributed helpers use local CUDA devices without requiring a GPU in CI."""

from unittest.mock import Mock

import pytest
import torch

from forecast_backbones.m4_rum.config import load_config
from pipelines.forecasting import distributed, training


@pytest.mark.parametrize(
    "initialized,backend,current",
    [(False, "nccl", 0), (True, "gloo", 0), (True, "nccl", 0), (True, "nccl", 1)],
)
def test_barrier_uses_current_logical_device(
    monkeypatch, initialized, backend, current
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,7")
    monkeypatch.setenv("RANK", "9")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(distributed.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: initialized)
    get_backend, current_device = Mock(return_value=backend), Mock(return_value=current)
    barrier = Mock()
    monkeypatch.setattr(distributed.dist, "get_backend", get_backend)
    monkeypatch.setattr(distributed.dist, "barrier", barrier)
    monkeypatch.setattr(torch.cuda, "current_device", current_device)

    distributed.barrier()

    if not initialized:
        barrier.assert_not_called()
        get_backend.assert_not_called()
    elif backend == "nccl":
        barrier.assert_called_once_with(device_ids=[current])
        current_device.assert_called_once_with()
    else:
        barrier.assert_called_once_with()
    if not initialized or backend != "nccl":
        current_device.assert_not_called()


@pytest.fixture
def cuda_stubs(monkeypatch):
    events = []

    class TrainingArguments:
        def __init__(self, **kwargs):
            events.append("args")
            self.__dict__.update(kwargs)

    def probe_bf16():
        events.append("bf16")
        return True

    monkeypatch.setattr(training, "TrainingArguments", TrainingArguments)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda device: events.append(("bind", device))
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", probe_bf16)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,7")
    monkeypatch.setenv("RANK", "9")
    return events


@pytest.mark.parametrize(
    "cuda,local_rank",
    [(True, "0"), (True, "1"), (True, None), (True, "-1"), (False, "1")],
)
def test_bind_before_precision_probe(
    cfg, tmp_path, monkeypatch, cuda_stubs, cuda, local_rank
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    if local_rank is not None:
        monkeypatch.setenv("LOCAL_RANK", local_rank)
    cfg["train"]["schedule"]["precision"] = "bf16"

    args = training.build_training_args(load_config(cfg), tmp_path)

    expected = []
    if cuda and local_rank not in (None, "-1"):
        expected.append(("bind", int(local_rank)))
    assert cuda_stubs == expected + (["bf16"] if cuda else []) + ["args"]
    assert args.bf16 is cuda
    assert args.use_cpu is not cuda


@pytest.mark.parametrize("local_rank", ["2", "5"])
def test_out_of_range_local_rank_fails_before_cuda_work(
    cfg, tmp_path, monkeypatch, cuda_stubs, local_rank
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("LOCAL_RANK", local_rank)
    cfg["train"]["schedule"]["precision"] = "bf16"

    with pytest.raises(ValueError, match="LOCAL_RANK"):
        training.build_training_args(load_config(cfg), tmp_path)
    assert cuda_stubs == []
