# fourier_moba_config.py
# -*- coding: utf-8 -*-

"""
Transformer 相关的独立配置文件。

适配：
    transformers==4.40.1

设计：
    1. DatasetConfig：数据列、特征聚合、collator 配置。
    2. FourierMobaTransformerConfig：模型结构配置，继承 PretrainedConfig。
    3. TrainingConfig：loss、optimizer、trainer、路径、设备、断点续训配置。
    4. ExperimentConfig：汇总完整实验配置。

说明：
    - 只有 FourierMobaTransformerConfig 继承 PretrainedConfig。
    - DatasetConfig / TrainingConfig / ExperimentConfig 使用 dataclass。
    - 支持 dict / YAML / JSON 加载。
    - 支持 to_dict() 导出。
    - 不保留 HPO 原始值字段。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from transformers import PretrainedConfig, TrainingArguments


# ---------------------------------------------------------------------
# IO utils
# ---------------------------------------------------------------------


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("需要安装 PyYAML：pip install pyyaml") from exc

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}


def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("需要安装 PyYAML：pip install pyyaml") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False,
        )


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ---------------------------------------------------------------------
# DatasetConfig
# ---------------------------------------------------------------------


@dataclass
class DatasetConfig:
    """
    数据相关配置。

    负责：
        - 原始特征聚合规则；
        - 输入列、预测列、标签列、实体列；
        - context_length / prediction_length / batch_size；
        - feature collator 的字段映射。
    """

    feature_aggregations: List[Dict[str, Any]] = field(default_factory=list)

    conv_cols: List[str] = field(default_factory=list)
    conv_predict_cols: List[str] = field(default_factory=list)

    timestamp_col: List[str] = field(default_factory=list)
    pv_col: List[str] = field(default_factory=list)
    tsfm_col: List[str] = field(default_factory=list)
    target_col: List[str] = field(default_factory=list)

    input_length: int = 672
    pred_length: int = 192
    capacity: int = 465

    @property
    def train_batch_size(self) -> int:
        return self.batch_size

    @property
    def eval_batch_size(self) -> int:
        return self.batch_size

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "DatasetConfig":
        data = data or {}

        feature_aggregations: List[Dict[str, Any]] = []
        for item in data.get("feature_aggregations", []):
            feature_aggregations.append(
                {
                    "agg_output_col": item.get("agg_output_col"),
                    "raw_input_cols": list(item.get("raw_input_cols", [])),
                }
            )

        return cls(
            feature_aggregations=feature_aggregations,
            conv_cols=list(data.get("conv_cols", [])),
            conv_predict_cols=list(data.get("conv_predict_cols", [])),
            timestamp_col=list(data.get("timestamp_col", [])),
            pv_col=list(data.get("pv_col", [])),
            tsfm_col=list(data.get("tsfm_col", [])),
            target_col=list(data.get("target_col", [])),
            input_length=int(float(data.get("input_length", 672))),
            pred_length=int(float(data.get("pred_length", 192))),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_aggregations": [
                {
                    "agg_output_col": item.get("agg_output_col"),
                    "raw_input_cols": list(item.get("raw_input_cols", [])),
                }
                for item in self.feature_aggregations
            ],
            "conv_cols": list(self.conv_cols),
            "conv_predict_cols": list(self.conv_predict_cols),
            "timestamp_col": list(self.timestamp_col),
            "pv_col": list(self.pv_col),
            "tsfm_col": list(self.tsfm_col),
            "target_col": list(self.target_col),
            "input_length": int(self.input_length),
            "pred_length": int(self.pred_length),
        }


# ---------------------------------------------------------------------
# FourierMobaTransformerConfig
# ---------------------------------------------------------------------


class ModelConfig(PretrainedConfig):
    """
    FourierMoBA Transformer 模型结构配置。

    这个类会随 HuggingFace 模型一起 save_pretrained / from_pretrained。
    因此这里只放模型结构相关字段。
    """

    model_type = "fourier_moba_transformer"

    def __init__(
        self,
        selected_model: str = "chronos",
        patch_len: int = 96,
        stride: int = 96,
        token_dim: int = 128,
        output_seq_crop_range: Optional[Any] = None,
        ts_embed: bool = True,
        use_partial_revin: bool = True,
        partial_revin_dims: int = 1,
        partial_revin_affine: bool = True,
        n_layers: int = 5,
        d_model: int = 128,
        n_heads: int = 16,
        d_ff: int = 512,
        norm: str = "LayerNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.1,
        res_attention: bool = True,
        pre_norm: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        head_dropout: float = 0.0,
        moba_experts: int = 64,
        moba_drop_ratio: float = 0.3,
        moba_heads: int = 4,
        moba_pe_norm: float = 1.0,
        fourier_requires_a0: bool = True,
        lambda_main: float = 0.01,
        lambda_ortho: float = 1.0,
        lambda_vicreg: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.selected_model = str(selected_model)

        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.token_dim = int(token_dim)
        self.output_seq_crop_range = output_seq_crop_range

        self.ts_embed = bool(ts_embed)
        self.use_partial_revin = bool(use_partial_revin)
        self.partial_revin_dims = int(partial_revin_dims)
        self.partial_revin_affine = bool(partial_revin_affine)

        self.n_layers = int(n_layers)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_ff = int(d_ff)

        # HuggingFace 生态里常见字段。
        self.hidden_size = int(d_model)
        self.num_hidden_layers = int(n_layers)
        self.num_attention_heads = int(n_heads)

        self.norm = str(norm)
        self.attn_dropout = float(attn_dropout)
        self.dropout = float(dropout)
        self.res_attention = bool(res_attention)
        self.pre_norm = bool(pre_norm)

        self.pe = str(pe)
        self.learn_pe = bool(learn_pe)
        self.head_dropout = float(head_dropout)

        self.moba_experts = int(moba_experts)
        self.moba_drop_ratio = float(moba_drop_ratio)
        self.moba_heads = int(moba_heads)
        self.moba_pe_norm = float(moba_pe_norm)

        self.fourier_requires_a0 = bool(fourier_requires_a0)

        self.lambda_main = float(lambda_main)
        self.lambda_ortho = float(lambda_ortho)
        self.lambda_vicreg = float(lambda_vicreg)


        super().__init__(**kwargs)

    @classmethod
    def from_project_dict(
        cls,
        tsfm_config: Optional[Dict[str, Any]] = None,
        model_hyperparameter: Optional[Dict[str, Any]] = None,
    ) -> "FourierMobaTransformerConfig":
        tsfm_config = tsfm_config or {}
        model_hyperparameter = model_hyperparameter or {}

        return cls(
            selected_model=tsfm_config.get("selected_model", "chronos"),
            patch_len=int(float(model_hyperparameter.get("patch_len", 96))),
            stride=int(float(model_hyperparameter.get("stride", 96))),
            token_dim=int(float(model_hyperparameter.get("token_dim", 128))),
            output_seq_crop_range=model_hyperparameter.get("output_seq_crop_range"),
            ts_embed=bool(model_hyperparameter.get("ts_embed", True)),
            use_partial_revin=bool(model_hyperparameter.get("use_partial_revin", True)),
            partial_revin_dims=int(float(model_hyperparameter.get("partial_revin_dims", 1))),
            partial_revin_affine=bool(model_hyperparameter.get("partial_revin_affine", True)),
            n_layers=int(float(model_hyperparameter.get("n_layers", 5))),
            d_model=int(float(model_hyperparameter.get("d_model", 128))),
            n_heads=int(float(model_hyperparameter.get("n_heads", 16))),
            d_ff=int(float(model_hyperparameter.get("d_ff", 512))),
            norm=model_hyperparameter.get("norm", "LayerNorm"),
            attn_dropout=float(model_hyperparameter.get("attn_dropout", 0.0)),
            dropout=float(model_hyperparameter.get("dropout", 0.1)),
            res_attention=bool(model_hyperparameter.get("res_attention", True)),
            pre_norm=bool(model_hyperparameter.get("pre_norm", False)),
            pe=model_hyperparameter.get("pe", "zeros"),
            learn_pe=bool(model_hyperparameter.get("learn_pe", True)),
            head_dropout=float(model_hyperparameter.get("head_dropout", 0.0)),
            moba_experts=int(float(model_hyperparameter.get("moba_experts", 64))),
            moba_drop_ratio=float(model_hyperparameter.get("moba_drop_ratio", 0.3)),
            moba_heads=int(float(model_hyperparameter.get("moba_heads", 4))),
            moba_pe_norm=float(model_hyperparameter.get("moba_pe_norm", 1.0)),
            fourier_requires_a0=bool(model_hyperparameter.get("fourier_requires_a0", True)),
            lambda_main=bool(model_hyperparameter.get("lambda_main", 0.01)),
            lambda_ortho=bool(model_hyperparameter.get("lambda_ortho", 1.0)),
            lambda_vicreg=bool(model_hyperparameter.get("lambda_vicreg", 1.0)),
        )

    def to_project_dict(self) -> Dict[str, Any]:
        """
        导出项目配置视角下的模型结构字段。
        这个方法用于 ExperimentConfig.to_dict()，避免把 HuggingFace 通用字段全部混进去。
        """

        return {
            "selected_model": self.selected_model,
            "patch_len": int(self.patch_len),
            "stride": int(self.stride),
            "token_dim": int(self.token_dim),
            "output_seq_crop_range": self.output_seq_crop_range,
            "ts_embed": bool(self.ts_embed),
            "use_partial_revin": bool(self.use_partial_revin),
            "partial_revin_dims": int(self.partial_revin_dims),
            "partial_revin_affine": bool(self.partial_revin_affine),
            "n_layers": int(self.n_layers),
            "d_model": int(self.d_model),
            "n_heads": int(self.n_heads),
            "d_ff": int(self.d_ff),
            "norm": self.norm,
            "attn_dropout": float(self.attn_dropout),
            "dropout": float(self.dropout),
            "res_attention": bool(self.res_attention),
            "pre_norm": bool(self.pre_norm),
            "pe": self.pe,
            "learn_pe": bool(self.learn_pe),
            "head_dropout": float(self.head_dropout),
            "moba_experts": int(self.moba_experts),
            "moba_drop_ratio": float(self.moba_drop_ratio),
            "moba_heads": int(self.moba_heads),
            "moba_pe_norm": float(self.moba_pe_norm),
            "fourier_requires_a0": bool(self.fourier_requires_a0),
        }


# ---------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """
    训练与运行配置。

    负责：
        - loss 权重；
        - optimizer 参数；
        - HuggingFace Trainer 参数；
        - output_dir / resume_from_checkpoint / device 等运行字段。
    """

    # Loss weights


    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Trainer loop
    max_epochs: int = 100
    patience: int = 3
    seed: int = 42

    # Runtime-like fields, aligned with HuggingFace TrainingArguments style
    output_dir: str = "outputs/fourier_moba"
    resume_from_checkpoint: Optional[str] = None
    resume_training: bool = False
    device: str = "cuda"

    # HuggingFace Trainer details
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_strategy: str = "steps"
    logging_steps: int = 50
    save_total_limit: int = 3

    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    remove_unused_columns: bool = False

    fp16: bool = False
    bf16: bool = False

    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = False

    disable_tqdm: bool = True
    report_to: Union[str, List[str]] = "none"

    lr_scheduler_type: str = "constant"
    warmup_steps: int = 0
    max_grad_norm: float = 0.0
    gradient_accumulation_steps: int = 1

    overwrite_output_dir: bool = True
    early_stopping: bool = True

    @property
    def use_cpu(self) -> bool:
        return self.device.lower() == "cpu"

    def get_resume_from_checkpoint(self) -> Optional[str]:
        if self.resume_training:
            return self.resume_from_checkpoint
        return None

    @classmethod
    def from_project_dict(
        cls,
        model_hyperparameter: Optional[Dict[str, Any]] = None,
        model_trainer: Optional[Dict[str, Any]] = None,
        root_config: Optional[Dict[str, Any]] = None,
    ) -> "TrainingConfig":
        model_hyperparameter = model_hyperparameter or {}
        model_trainer = model_trainer or {}
        root_config = root_config or {}

        output_dir = (
            root_config.get("output_dir")
            or root_config.get("model_saving_path")
            or "outputs/fourier_moba"
        )

        resume_from_checkpoint = (
            root_config.get("resume_from_checkpoint")
            or root_config.get("model_loading_path")
        )

        return cls(
            # lambda_main=float(model_hyperparameter.get("lambda_main", 1.0)),
            # lambda_ortho=float(model_hyperparameter.get("lambda_ortho", 0.0)),
            # lambda_vicreg=float(model_hyperparameter.get("lambda_vicreg", 0.0)),
            learning_rate=float(model_hyperparameter.get("learning_rate", 1e-4)),
            weight_decay=float(model_hyperparameter.get("weight_decay", 1e-5)),
            max_epochs=float(model_trainer.get("max_epochs", 10.0)),
            patience=int(float(model_trainer.get("patience", 3))),
            seed=int(float(model_hyperparameter.get("seed", 42))),
            output_dir=str(output_dir),
            resume_from_checkpoint=resume_from_checkpoint,
            resume_training=bool(root_config.get("resume_training", False)),
            device=str(root_config.get("device", "cuda")),
        )

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "TrainingConfig":
        data = data or {}

        return cls(
            # lambda_main=float(data.get("lambda_main", 1.0)),
            # lambda_ortho=float(data.get("lambda_ortho", 0.0)),
            # lambda_vicreg=float(data.get("lambda_vicreg", 0.0)),
            learning_rate=float(data.get("learning_rate", 1e-4)),
            weight_decay=float(data.get("weight_decay", 1e-5)),
            max_epochs=float(data.get("max_epochs", 10.0)),
            patience=int(float(data.get("patience", 3))),
            seed=int(float(data.get("seed", 42))),
            output_dir=str(data.get("output_dir", "outputs/fourier_moba")),
            resume_from_checkpoint=data.get("resume_from_checkpoint"),
            resume_training=bool(data.get("resume_training", False)),
            device=str(data.get("device", "cuda")),
            evaluation_strategy=str(data.get("evaluation_strategy", "epoch")),
            save_strategy=str(data.get("save_strategy", "epoch")),
            logging_strategy=str(data.get("logging_strategy", "steps")),
            logging_steps=int(float(data.get("logging_steps", 50))),
            save_total_limit=int(float(data.get("save_total_limit", 3))),
            load_best_model_at_end=bool(data.get("load_best_model_at_end", True)),
            metric_for_best_model=str(data.get("metric_for_best_model", "eval_loss")),
            greater_is_better=bool(data.get("greater_is_better", False)),
            remove_unused_columns=bool(data.get("remove_unused_columns", False)),
            fp16=bool(data.get("fp16", False)),
            bf16=bool(data.get("bf16", False)),
            dataloader_num_workers=int(float(data.get("dataloader_num_workers", 0))),
            dataloader_pin_memory=bool(data.get("dataloader_pin_memory", False)),
            disable_tqdm=bool(data.get("disable_tqdm", True)),
            report_to=data.get("report_to", "none"),
            lr_scheduler_type=str(data.get("lr_scheduler_type", "constant")),
            warmup_steps=int(float(data.get("warmup_steps", 0))),
            max_grad_norm=float(data.get("max_grad_norm", 0.0)),
            gradient_accumulation_steps=int(float(data.get("gradient_accumulation_steps", 1))),
            overwrite_output_dir=bool(data.get("overwrite_output_dir", True)),
            early_stopping=bool(data.get("early_stopping", True)),
        )

    def to_training_arguments(
        self,
        train_batch_size: int,
        eval_batch_size: Optional[int] = None,
        output_dir: Optional[str] = None,
        **overrides: Any,
    ) -> TrainingArguments:
        """
        转换为 HuggingFace TrainingArguments。

        Args:
            train_batch_size:
                per_device_train_batch_size。
            eval_batch_size:
                per_device_eval_batch_size，默认等于 train_batch_size。
            output_dir:
                覆盖 self.output_dir。
            **overrides:
                临时覆盖 TrainingArguments 字段。
        """

        eval_batch_size = train_batch_size if eval_batch_size is None else eval_batch_size
        resolved_output_dir = self.output_dir if output_dir is None else output_dir

        args_dict: Dict[str, Any] = {
            "output_dir": str(resolved_output_dir),
            "overwrite_output_dir": bool(self.overwrite_output_dir),
            "do_train": True,
            "do_eval": self.evaluation_strategy != "no",
            "per_device_train_batch_size": int(train_batch_size),
            "per_device_eval_batch_size": int(eval_batch_size),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "num_train_epochs": float(self.max_epochs),
            "evaluation_strategy": self.evaluation_strategy,
            "save_strategy": self.save_strategy,
            "logging_strategy": self.logging_strategy,
            "logging_steps": int(self.logging_steps),
            "save_total_limit": int(self.save_total_limit),
            "load_best_model_at_end": bool(self.load_best_model_at_end),
            "metric_for_best_model": self.metric_for_best_model,
            "greater_is_better": bool(self.greater_is_better),
            "seed": int(self.seed),
            "data_seed": int(self.seed) + 11,
            "remove_unused_columns": bool(self.remove_unused_columns),
            "fp16": bool(self.fp16),
            "bf16": bool(self.bf16),
            "dataloader_num_workers": int(self.dataloader_num_workers),
            "dataloader_pin_memory": bool(self.dataloader_pin_memory),
            "disable_tqdm": bool(self.disable_tqdm),
            "report_to": self.report_to,
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_steps": int(self.warmup_steps),
            "max_grad_norm": float(self.max_grad_norm),
            "gradient_accumulation_steps": int(self.gradient_accumulation_steps),
            "use_cpu": bool(self.use_cpu),
        }

        args_dict.update(overrides)

        return TrainingArguments(**args_dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            # "lambda_main": float(self.lambda_main),
            # "lambda_ortho": float(self.lambda_ortho),
            # "lambda_vicreg": float(self.lambda_vicreg),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "max_epochs": float(self.max_epochs),
            "patience": int(self.patience),
            "seed": int(self.seed),
            "output_dir": self.output_dir,
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "resume_training": bool(self.resume_training),
            "device": self.device,
            "evaluation_strategy": self.evaluation_strategy,
            "save_strategy": self.save_strategy,
            "logging_strategy": self.logging_strategy,
            "logging_steps": int(self.logging_steps),
            "save_total_limit": int(self.save_total_limit),
            "load_best_model_at_end": bool(self.load_best_model_at_end),
            "metric_for_best_model": self.metric_for_best_model,
            "greater_is_better": bool(self.greater_is_better),
            "remove_unused_columns": bool(self.remove_unused_columns),
            "fp16": bool(self.fp16),
            "bf16": bool(self.bf16),
            "dataloader_num_workers": int(self.dataloader_num_workers),
            "dataloader_pin_memory": bool(self.dataloader_pin_memory),
            "disable_tqdm": bool(self.disable_tqdm),
            "report_to": self.report_to,
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_steps": int(self.warmup_steps),
            "max_grad_norm": float(self.max_grad_norm),
            "gradient_accumulation_steps": int(self.gradient_accumulation_steps),
            "overwrite_output_dir": bool(self.overwrite_output_dir),
            "early_stopping": bool(self.early_stopping),
        }


# ---------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """
    完整实验配置。

    3 + 1 结构：
        - DatasetConfig
        - FourierMobaTransformerConfig
        - TrainingConfig
        - ExperimentConfig
    """

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: FourierMobaTransformerConfig = field(default_factory=FourierMobaTransformerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_project_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """
        从原始项目 YAML 结构加载。

        原始结构：
            dataset:
              ...
            tsfm:
              selected_model: chronos
            model_hyperparameter:
              ...
            model_trainer:
              ...
            model_saving_path:
              ...
            model_loading_path:
              ...
            device:
              ...
            resume_training:
              ...
        """

        data = data or {}

        return cls(
            dataset=DatasetConfig.from_dict(data.get("dataset")),
            model=FourierMobaTransformerConfig.from_project_dict(
                tsfm_config=data.get("tsfm"),
                model_hyperparameter=data.get("model_hyperparameter"),
            ),
            training=TrainingConfig.from_project_dict(
                model_hyperparameter=data.get("model_hyperparameter"),
                model_trainer=data.get("model_trainer"),
                root_config=data,
            ),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """
        从标准化结构加载。

        标准结构：
            dataset:
              ...
            model:
              ...
            training:
              ...
        """

        data = data or {}

        if "model_hyperparameter" in data or "model_trainer" in data or "tsfm" in data:
            return cls.from_project_dict(data)

        return cls(
            dataset=DatasetConfig.from_dict(data.get("dataset")),
            model=FourierMobaTransformerConfig(**dict(data.get("model", {}))),
            training=TrainingConfig.from_dict(data.get("training")),
        )

    @classmethod
    def from_yaml(
        cls,
        path: Union[str, Path],
        project_format: bool = True,
    ) -> "ExperimentConfig":
        data = load_yaml(path)
        if project_format:
            return cls.from_project_dict(data)
        return cls.from_dict(data)

    @classmethod
    def from_json(
        cls,
        path: Union[str, Path],
        project_format: bool = True,
    ) -> "ExperimentConfig":
        data = load_json(path)
        if project_format:
            return cls.from_project_dict(data)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset.to_dict(),
            "model": self.model.to_project_dict(),
            "training": self.training.to_dict(),
        }

    def to_training_arguments(self, **overrides: Any) -> TrainingArguments:
        return self.training.to_training_arguments(
            train_batch_size=self.dataset.train_batch_size,
            eval_batch_size=self.dataset.eval_batch_size,
            **overrides,
        )

    def save_yaml(self, path: Union[str, Path]) -> None:
        save_yaml(self.to_dict(), path)

    def save_json(self, path: Union[str, Path]) -> None:
        save_json(self.to_dict(), path)

    def save_model_config_pretrained(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        resolved_output_dir = self.training.output_dir if output_dir is None else str(output_dir)
        self.model.save_pretrained(resolved_output_dir)



if __name__ == "__main__":

    exp_config = ExperimentConfig.from_yaml(
        "train_config.yaml",
        project_format=True,
    )

    print(exp_config.dataset.input_cols)
    # print(exp_config.model.tsfm)
    print(exp_config.model.d_model)
    print(exp_config.training.learning_rate)
    print(exp_config.training.output_dir)

    exp_config.save_json("tester.json")