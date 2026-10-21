"""
framework.py —— 框架工程师维护，在这里优化训练流程和训练框架。
模型设计 员工不碰这个文件。

提供：
  print_run_fingerprint(Trainer)  — 打印训练指纹信息，trainer 相关的信息
  assert_cuda_available()         检查设备配置信息
"""
import os
import json
from pathlib import Path
from typing import Any, Dict, Optional

import accelerate
import torch
import torch.nn as nn

from transformers import Trainer
from transformers.trainer_utils import enable_full_determinism, set_seed

def set_training_seed(seed: int = 42, full_determinism: bool = False,) -> None:
    """
    设置 Python、NumPy、PyTorch 和 CUDA 随机种子。

    Args:
        seed:
            全局随机种子。

        full_determinism:
            False:
                设置随机种子，速度影响较小。

            True:
                启用 PyTorch 确定性算法，
                用于严格复现和回归测试，
                可能明显降低训练速度。
    """

    seed = int(seed)

    if full_determinism:
        enable_full_determinism(seed)
    else:
        set_seed(seed)

        # 避免 cuDNN 每次重新选择卷积算法
        torch.backends.cudnn.benchmark = False

def print_run_fingerprint(trainer:Trainer):

    accelerator     = trainer.accelerator
    train_loader    = trainer.get_train_dataloader()
    val_loader      = trainer.get_eval_dataloader()
    cfg             = trainer.args

    # 输出并记录当前的运行指纹
    assert hasattr(trainer, "accelerator"), "Trainer 内部没有 accelerator"
    assert str(trainer.args.device).startswith("cuda"), f"Trainer device 异常, 非cuda: {trainer.args.device}"
    assert str(trainer.accelerator.device).startswith("cuda"), f"Accelerator device 异常, 非cuda: {trainer.accelerator.device}"
    
    accelerator.print("=" * 80)
    accelerator.print("RUN FINGERPRINT")
    accelerator.print(f"pid={os.getpid()}")
    accelerator.print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    visible_gpu_count = torch.cuda.device_count()
    accelerator.print(f"检测到 {visible_gpu_count} 块可见GPU")
    for i in range(visible_gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    accelerator.print(f"accelerate_version={accelerate.__version__}")
    accelerator.print(f"torch_version={torch.__version__}")

    accelerator.print(f"num_processes={accelerator.num_processes}")
    accelerator.print(f"process_index={accelerator.process_index}")
    accelerator.print(f"local_process_index={accelerator.local_process_index}")
    accelerator.print(f"distributed_type={accelerator.distributed_type}")
    accelerator.print(f"mixed_precision={accelerator.mixed_precision}")
    accelerator.print(f"device={accelerator.device}")

    accelerator.print(f"seed={getattr(cfg, 'seed', None)}")
    accelerator.print(f"batch_size={getattr(cfg, 'batch_size', None)}")
    accelerator.print(f"grad_accum={getattr(cfg, 'gradient_accumulation_steps', None)}")
    accelerator.print(f"lr={getattr(cfg, 'learning_rate', None)}")
    accelerator.print(f"scheduler={getattr(cfg, 'scheduler', None)}")
    accelerator.print(f"grad_clip={getattr(cfg, 'max_grad_norm', None)}")

    # accelerator.print(f"len(train_loader)={len(train_loader)}")
    # accelerator.print(f"len(val_loader)={len(val_loader) if val_loader is not None else None}")

    accelerator.print(f"cudnn.benchmark={torch.backends.cudnn.benchmark}")
    accelerator.print(f"cudnn.deterministic={torch.backends.cudnn.deterministic}")
    accelerator.print("=" * 80)


def assert_cuda_available():
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        print("current cuda device:", torch.cuda.current_device())
        print("gpu name:", torch.cuda.get_device_name(torch.cuda.current_device()))

def save_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_dir: str,
    model_name: str,
    model_config: Dict[str, Any],
    accelerator=None,
) -> None:
    """
    保存用于评估/推理的模型 checkpoint。

    只保存 model_state_dict，不保存 optimizer/scheduler。
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if accelerator is not None:
        model_to_save = accelerator.unwrap_model(model)
    else:
        model_to_save = model

    payload = {
        "model_state_dict": model_to_save.state_dict(),
        "model_name": model_name,
        "model_config": model_config,
        # "trainer_config": resolved_config["trainer"],
    }

    torch.save(payload, checkpoint_dir / "model.pt")

    # with open(checkpoint_dir / "metadata.json", "w", encoding="utf-8") as f:
    #     json.dump(metadata, f, ensure_ascii=False, indent=2)

def save_training_state(
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    checkpoint_dir: str,
    epoch: int,
    global_step: int,
) -> None:
    """
    保存用于恢复训练的 Trainer 状态。
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
    }

    torch.save(payload, checkpoint_dir / "trainer_state.pt")

def load_model_checkpoint(
    checkpoint_dir: str,
    model_builder,
) -> torch.nn.Module:
    """
    加载模型 checkpoint。

    model_builder 是一个函数：
        model = model_builder(model_config)

    这里使用 strict=True。
    如果模型结构和 checkpoint 不匹配，应直接报错。
    """
    checkpoint_dir = Path(checkpoint_dir)
    assert (checkpoint_dir / "model.pt").exists()
    payload = torch.load((checkpoint_dir / "model.pt"), map_location="cpu")

    model_config = payload["model_config"]
    model = model_builder(model_config)

    model.load_state_dict(payload["model_state_dict"], strict=True)

    # model.to(device)
    model.eval()

    return model, model_config