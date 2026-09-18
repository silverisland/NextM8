from __future__ import annotations

import copy
import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Type

import pandas as pd
import torch.nn as nn


@dataclass(frozen=True)
class ModelSpec:
    """
    一个模型的注册信息。

    module_path:
        模型类所在的 Python 模块路径。

    class_name:
        模型类名。

    description:
        简短说明，方便报错和 Agent 读取。

    model_type:
        模型类型。

    output_type:
        模型输出类型。
    """

    module_path: str
    class_name: str
    description: str = ""
    model_type: Literal["expert", "weather", "fusion", "overall"] = "expert"
    output_type: Literal["Number", "DataFrame"] = "DataFrame"


@dataclass(frozen=True)
class EvaluatorSpec:
    """
    评估函数注册信息。

    module_path:
        evaluate_model 所在模块路径。

    function_name:
        评估函数名称。
    """

    module_path: str
    function_name: str = "evaluate_model"


# ============================================================================
# 1. 自动发现目录
# ============================================================================
#
# 假定当前文件位于：
#
#   project_root/common/model_registry.py
#
# 三类模型目录位于：
#
#   project_root/forecast_backbones/
#   project_root/hetero_fusion/
#   project_root/weather_fusion/
#
# 查找顺序固定为：
#
#   expert -> fusion -> weather -> legacy registry
#
PROJECT_ROOT = Path(__file__).resolve().parent.parent

_AUTO_SEARCH_ROOTS = (
    ("forecast_backbones", "expert", "DataFrame"),
    ("hetero_fusion", "fusion", "DataFrame"),
    ("weather_fusion", "weather", "Number"),
)


# ============================================================================
# 2. 旧模型注册表
# ============================================================================
#
# 旧名称继续保留，用于兼容 ensemble / 历史配置。
# 新目录结构优先于这里的注册信息。
#
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "FourierMOBATransformer": ModelSpec(
        module_path="forecast_backbones.m1.transformer_models",
        class_name="TransformersAdaptedM1Model",
        description="M1 single model with original name",
    ),
    "PatchRegForcast": ModelSpec(
        module_path="forecast_backbones.m2.transformer_models",
        class_name="TransformersAdaptedM2Model",
        description="M2 single model with original name",
    ),
    "MoiraiPvForecaster": ModelSpec(
        module_path="forecast_backbones.m3.transformer_models",
        class_name="TransformersAdaptedM3Model",
        description="M3 single model with original name",
    ),
    "PatchTSTPvForecaster": ModelSpec(
        module_path="forecast_backbones.m4.transformer_models",
        class_name="TransformersAdaptedM4Model",
        description="M4 single model with original name",
    ),

    "model1": ModelSpec(
        module_path="forecast_backbones.m1.transformer_models",
        class_name="TransformersAdaptedM1Model",
        description="M1 single model with original name",
    ),
    "model2": ModelSpec(
        module_path="forecast_backbones.m2.transformer_models",
        class_name="TransformersAdaptedM2Model",
        description="M2 single model with original name",
    ),
    "model3": ModelSpec(
        module_path="forecast_backbones.m3.transformer_models",
        class_name="TransformersAdaptedM3Model",
        description="M3 single model with original name",
    ),
    "model4": ModelSpec(
        module_path="forecast_backbones.m4.transformer_models",
        class_name="TransformersAdaptedM4Model",
        description="M4 single model with original name",
    ),

    "m1": ModelSpec(
        module_path="forecast_backbones.m1.transformer_models",
        class_name="TransformersAdaptedM1Model",
        description="M1 single model",
    ),
    "m2": ModelSpec(
        module_path="forecast_backbones.m2.transformer_models",
        class_name="TransformersAdaptedM2Model",
        description="M2 single model",
    ),
    "m2_new": ModelSpec(
        module_path="forecast_backbones.m2_new.transformer_models",
        class_name="TransformersAdaptedNewM2Model",
        description="M2 single model",
    ),
    "m3": ModelSpec(
        module_path="forecast_backbones.m3.transformer_models",
        class_name="TransformersAdaptedM3Model",
        description="M3 single model",
    ),
    "m4": ModelSpec(
        module_path="forecast_backbones.m4.transformer_models",
        class_name="TransformersAdaptedM4Model",
        description="M4 single model",
    ),

    "mW": ModelSpec(
        module_path="weather_fusion.mW.transformer_models",
        class_name="TransformersAdaptedPvWFResidule",
        description="weather model for the overall pipeline",
        model_type="weather",
        output_type="Number",
    ),
}


# ============================================================================
# 3. 旧 evaluator 注册表
# ============================================================================
EVALUATOR_REGISTRY: Dict[str, EvaluatorSpec] = {
    "FourierMOBATransformer": EvaluatorSpec(
        module_path="forecast_backbones.m1.transformer_trainer",
    ),
    "PatchRegForcast": EvaluatorSpec(
        module_path="forecast_backbones.m2.transformer_trainer",
    ),
    "MoiraiPvForecaster": EvaluatorSpec(
        module_path="forecast_backbones.m3.transformer_trainer",
    ),
    "PatchTSTPvForecaster": EvaluatorSpec(
        module_path="forecast_backbones.m4.transformer_trainer",
    ),

    "model1": EvaluatorSpec(
        module_path="forecast_backbones.m1.transformer_trainer",
    ),
    "model2": EvaluatorSpec(
        module_path="forecast_backbones.m2.transformer_trainer",
    ),
    "model3": EvaluatorSpec(
        module_path="forecast_backbones.m3.transformer_trainer",
    ),
    "model4": EvaluatorSpec(
        module_path="forecast_backbones.m4.transformer_trainer",
    ),

    "m1": EvaluatorSpec(
        module_path="forecast_backbones.m1.transformer_trainer",
    ),
    "m2": EvaluatorSpec(
        module_path="forecast_backbones.m2.transformer_trainer",
    ),
    "m2_new": EvaluatorSpec(
        module_path="forecast_backbones.m2_new.transformer_trainer",
    ),
    "m3": EvaluatorSpec(
        module_path="forecast_backbones.m3.transformer_trainer",
    ),
    "m4": EvaluatorSpec(
        module_path="forecast_backbones.m4.transformer_trainer",
    ),

    "mW": EvaluatorSpec(
        module_path="weather_fusion.mW.transformer_trainer",
    ),
    "mF": EvaluatorSpec(
        module_path="hetero_fusion.mF.transformer_trainer",
    ),
    "mF_mean": EvaluatorSpec(
        module_path="hetero_fusion.mF_mean.transformer_trainer",
    ),
}


# 已经按名称解析过的自动发现结果。
# 这里只缓存实际访问过的模型，不会在 import 本文件时扫描目录。
_AUTO_MODEL_CACHE: Dict[str, ModelSpec] = {}
_AUTO_EVALUATOR_CACHE: Dict[str, EvaluatorSpec] = {}


# ============================================================================
# 4. 内部辅助函数
# ============================================================================
def _validate_model_name(model_name: str) -> None:
    """
    自动目录查找只接受一级文件夹名称，避免路径穿越。
    """
    if (
        not isinstance(model_name, str)
        or not model_name
        or model_name in {".", ".."}
        or "/" in model_name
        or "\\" in model_name
    ):
        raise ValueError(
            f"Invalid model_name: {model_name!r}. "
            "model_name must be a direct model folder name or a registered alias."
        )


def _find_auto_model_location(
    model_name: str,
) -> tuple[Path, str, str, str] | None:
    """
    按 expert -> fusion -> weather 的顺序寻找同名模型目录。

    返回：
        (
            model_dir,
            package_root,
            model_type,
            output_type,
        )

    注意：
        只检查给定 model_name 对应的三个固定路径，
        不遍历任何模型目录。
    """
    _validate_model_name(model_name)

    for package_root, model_type, output_type in _AUTO_SEARCH_ROOTS:
        model_dir = PROJECT_ROOT / package_root / model_name
        if model_dir.is_dir():
            return model_dir, package_root, model_type, output_type

    return None


def _import_module(module_path: str, *, owner: str, kind: str):
    """
    统一模块导入和错误信息。
    """
    try:
        return importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Failed to import {kind} module for '{owner}'. "
            f"module_path='{module_path}'. "
            "Please check package structure, __init__.py, launch path, "
            "and module dependencies."
        ) from exc


def _find_first_local_nn_module_class(
    module,
    *,
    model_name: str,
) -> Type[nn.Module]:
    """
    返回 transformer_models.py 中第一个在本模块定义、
    且继承 torch.nn.Module 的类。

    Python module.__dict__ 保留模块执行时的插入顺序。
    类定义按源码顺序执行，因此这里能够得到源码中最先定义的候选类。

    obj.__module__ == module.__name__ 用于排除从其他模块 import 进来的类。
    """
    for _, obj in module.__dict__.items():
        if not inspect.isclass(obj):
            continue

        if obj is nn.Module:
            continue

        if obj.__module__ != module.__name__:
            continue

        if issubclass(obj, nn.Module):
            return obj

    raise TypeError(
        f"No torch.nn.Module subclass defined in "
        f"'{module.__name__}' for model '{model_name}'. "
        "Expected the first locally defined nn.Module subclass "
        "to be the top-level model class."
    )


def _build_auto_model_spec(model_name: str) -> ModelSpec | None:
    """
    只为指定 model_name 解析自动模型信息。
    """
    if model_name in _AUTO_MODEL_CACHE:
        return _AUTO_MODEL_CACHE[model_name]

    location = _find_auto_model_location(model_name)
    if location is None:
        return None

    model_dir, package_root, model_type, output_type = location
    model_file = model_dir / "transformer_models.py"

    # 已命中模型目录后，内部结构错误直接失败。
    # 不继续查后续目录或旧注册表。
    if not model_file.is_file():
        raise ModuleNotFoundError(
            f"Found model directory for '{model_name}' at '{model_dir}', "
            "but 'transformer_models.py' does not exist."
        )

    module_path = f"{package_root}.{model_name}.transformer_models"
    module = _import_module(
        module_path,
        owner=model_name,
        kind="model",
    )
    model_cls = _find_first_local_nn_module_class(
        module,
        model_name=model_name,
    )

    spec = ModelSpec(
        module_path=module_path,
        class_name=model_cls.__name__,
        description=(
            f"Auto-discovered model from "
            f"{package_root}/{model_name}/transformer_models.py"
        ),
        model_type=model_type,  # type: ignore[arg-type]
        output_type=output_type,  # type: ignore[arg-type]
    )

    _AUTO_MODEL_CACHE[model_name] = spec
    return spec


def _build_auto_evaluator_spec(
    model_name: str,
) -> EvaluatorSpec | None:
    """
    只为指定 model_name 解析自动 evaluator 信息。
    """
    if model_name in _AUTO_EVALUATOR_CACHE:
        return _AUTO_EVALUATOR_CACHE[model_name]

    location = _find_auto_model_location(model_name)
    if location is None:
        return None

    model_dir, package_root, _, _ = location
    trainer_file = model_dir / "transformer_trainer.py"

    # 已命中模型目录后，内部结构错误直接失败。
    # 不继续查后续目录或旧注册表。
    if not trainer_file.is_file():
        raise ModuleNotFoundError(
            f"Found model directory for '{model_name}' at '{model_dir}', "
            "but 'transformer_trainer.py' does not exist."
        )

    module_path = f"{package_root}.{model_name}.transformer_trainer"
    spec = EvaluatorSpec(
        module_path=module_path,
        function_name="evaluate_model",
    )

    _AUTO_EVALUATOR_CACHE[model_name] = spec
    return spec


# ============================================================================
# 5. Model API
# ============================================================================
def list_models() -> Dict[str, ModelSpec]:
    """
    返回当前可见的模型注册信息。

    为保持懒加载：
        - 始终返回旧 MODEL_REGISTRY；
        - 自动发现模型只有被实际访问过后才会出现在返回结果中；
        - 本函数不会扫描 forecast_backbones / hetero_fusion / weather_fusion。
    """
    result = dict(MODEL_REGISTRY)
    result.update(_AUTO_MODEL_CACHE)
    return result


def get_model_spec(model_name: str) -> ModelSpec:
    """
    根据模型名获取注册信息。

    查找顺序：
        1. forecast_backbones
        2. hetero_fusion
        3. weather_fusion
        4. MODEL_REGISTRY
    """
    auto_spec = _build_auto_model_spec(model_name)
    if auto_spec is not None:
        return auto_spec

    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]

    available = ", ".join(sorted(MODEL_REGISTRY.keys()))
    raise ValueError(
        f"Unknown model_name: {model_name}. "
        "No matching model directory was found in "
        "forecast_backbones, hetero_fusion, or weather_fusion. "
        f"Registered aliases: {available}"
    )


def import_model_class(model_name: str) -> Type[nn.Module]:
    """
    动态导入模型类。

    示例：
        model_cls = import_model_class("m4")
    """
    spec = get_model_spec(model_name)

    module = _import_module(
        spec.module_path,
        owner=model_name,
        kind="model",
    )

    try:
        model_cls = getattr(module, spec.class_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Module '{spec.module_path}' does not contain class "
            f"'{spec.class_name}' for model '{model_name}'."
        ) from exc

    if not inspect.isclass(model_cls) or not issubclass(model_cls, nn.Module):
        raise TypeError(
            "Registered model class must inherit torch.nn.Module. "
            f"model_name='{model_name}', class='{spec.class_name}'"
        )

    return model_cls


def _normalize_m3_data_schema(model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
    """把 M3（m3 / mF_mean 内嵌）旧版 checkpoint config 的 data 段平移到新版 schema。

    旧版训练管线（如 config_d1，2025 前）写出的 model_config.yaml 的 data 段用
        target_variates / dynamic_covariates / future_target_postfix
    而新版 TransformersAdaptedM3Model 从 data 段读取
        past_target / past_covariate / future_covariate / label
    （见 forecast_backbones/m3/transformer_models.py:61-66）。
    平移映射（与 m3/pretrain_100m_v1 新版字段逐一对齐）：
        past_target       <- target_variates
        past_covariate    <- dynamic_covariates
        future_covariate  <- [c + postfix + "_predict"  for c in dynamic_covariates]
                                 （实际约定 general_ghi -> general_ghi_predict）
        label             <- [target_variates[0] + postfix]   （observe_power -> observe_power_future）
    只作用于 deepcopy，绝不改写磁盘上的 checkpoint config。
    """
    if model_name not in ("m3", "mF_mean"):
        return model_config
    data = model_config.get("data")
    if not isinstance(data, dict):
        return model_config
    # 已经是新 schema（有 past_target）就不动
    if "past_target" in data:
        return model_config

    out = copy.deepcopy(model_config)
    d = out["data"]
    postfix = d.get("future_target_postfix", "_future")
    predict_postfix = postfix.replace("future", "predict")
    targets = d.get("target_variates") or []
    covars = d.get("dynamic_covariates") or []
    d["past_target"] = list(targets)
    d["past_covariate"] = list(covars)
    d["future_covariate"] = [
        c if c.endswith("_predict") else f"{c}{predict_postfix}"
        for c in covars
    ]
    d["label"] = [
        t if t.endswith(postfix) else f"{t}{postfix}"
        for t in targets
    ]
    d["past_dynamic_covariates"] = []
    return out


def build_model(
    model_name: str,
    model_config: Dict[str, Any],
) -> nn.Module:
    """
    根据 model_name 和 model_config 构建模型。

    约定：
        每个模型类都支持：
            model = ModelClass(model_config)
    """
    model_cls = import_model_class(model_name)

    try:
        cfg = _normalize_m3_data_schema(model_name, model_config)
        return model_cls(cfg)
    except TypeError as exc:
        raise TypeError(
            f"Failed to build model '{model_name}' with model_config. "
            "Expected constructor signature like: "
            f"{model_cls.__name__}(model_config)."
        ) from exc


def build_model_from_resolved_config(
    resolved_config: Dict[str, Any],
) -> nn.Module:
    """
    从完整 resolved_config 构建模型。

    期望 resolved_config 至少包含：

        run:
          model_name: m4

        model:
          ...
    """
    try:
        model_name = resolved_config["run"]["model_name"]
    except KeyError as exc:
        raise KeyError(
            "resolved_config must contain resolved_config['run']['model_name']"
        ) from exc

    try:
        model_config = resolved_config["model"]
    except KeyError as exc:
        raise KeyError(
            "resolved_config must contain resolved_config['model']"
        ) from exc

    return build_model(
        model_name=model_name,
        model_config=model_config,
    )


# ============================================================================
# 6. Evaluator API
# ============================================================================
def build_evaluator(
    model_name: str,
) -> Callable[..., pd.DataFrame]:
    """
    根据 model_name 返回对应模型的 evaluate_model 函数。

    查找顺序：
        1. forecast_backbones
        2. hetero_fusion
        3. weather_fusion
        4. EVALUATOR_REGISTRY

    返回函数的调用方式保持：
        evaluate_model(
            model,
            test_dataset,
            HF_collate_fn,
            config,
        ) -> pd.DataFrame
    """
    auto_spec = _build_auto_evaluator_spec(model_name)

    if auto_spec is not None:
        spec = auto_spec
    elif model_name in EVALUATOR_REGISTRY:
        spec = EVALUATOR_REGISTRY[model_name]
    else:
        available = ", ".join(sorted(EVALUATOR_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model_name: {model_name}. "
            "No matching evaluator directory was found in "
            "forecast_backbones, hetero_fusion, or weather_fusion. "
            f"Registered evaluator aliases: {available}"
        )

    module = _import_module(
        spec.module_path,
        owner=model_name,
        kind="evaluator",
    )

    try:
        evaluate_model = getattr(module, spec.function_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Module '{spec.module_path}' does not contain function "
            f"'{spec.function_name}' for model '{model_name}'."
        ) from exc

    if not callable(evaluate_model):
        raise TypeError(
            "Registered evaluator is not callable. "
            f"model_name='{model_name}', "
            f"module_path='{spec.module_path}', "
            f"function_name='{spec.function_name}'"
        )

    return evaluate_model


def list_evaluators() -> Dict[str, EvaluatorSpec]:
    """
    返回当前可见的 evaluator 注册信息。

    为保持懒加载：
        - 始终返回旧 EVALUATOR_REGISTRY；
        - 自动发现 evaluator 只有被实际访问过后才会出现在返回结果中；
        - 本函数不会扫描三个模型根目录。
    """
    result = dict(EVALUATOR_REGISTRY)
    result.update(_AUTO_EVALUATOR_CACHE)
    return result