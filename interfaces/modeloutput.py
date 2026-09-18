import torch 

from dataclasses import dataclass
from transformers.utils import ModelOutput
from typing import List, Optional, Dict, Optional, Any, Tuple





@dataclass
class TransformersAdaptedModelOutput(ModelOutput):
    r"""
    TransformedAdaptedMXModel 模型的输出基类。

    Args:
        loss (`torch.Tensor`, *optional*):
            当提供 `labels` 时计算的损失值。对于监督学习训练，这是主要的训练目标。
        logits (`torch.Tensor`, *optional*):
            模型的原始输出 logits，通常经过线性变换但未经过激活函数。
            形状为 `(batch_size, num_classes)` 或根据任务不同的输出形状。
        prediction (`torch.Tensor`, *optional*):
            模型的实际预测输出，通常是经过后处理（如激活函数、归一化等）
            后的最终预测结果。形状与具体预测任务相关。 一般认为该值具有特定语义信息，可以直接作为最终结果返回给用户。
        hidden_states (`torch.Tensor`, *optional*):
            模型各层的隐藏状态。
            这是一个元组，第一个元素是嵌入层的输出，其余元素是每层 Transformer
            的输出。每个元素的形状为 `(batch_size, sequence_length, hidden_size)`。 
        embeddings (`torch.Tensor`, *optional*):
            模型在骨干网络中间产出的embedding。
        aux (`dict[str, Any]`, *optional*):
            辅助信息的字典，用于存储额外的输出信息。常见的键值包括：
            - `"loss_dict"`: 各分量损失的字典，用于多任务学习或损失分解分析。
            - `"attention_weights"`: 注意力权重矩阵。
            - `"representation"`: 学习到的中间表示。
            - `"metrics"`: 评估指标字典。

    Attributes:
        loss (`torch.Tensor`, *optional*):
            模型计算的损失值。
        logits (`torch.Tensor`, *optional*):
            原始输出 logits。
        prediction (`torch.Tensor`, *optional*):
            处理后的预测结果。
        hidden_states (`torch.Tensor`, *optional*):
            各层的隐藏状态。
        labels (`torch.Tensor`, *optional*):
            通过featurecollator生产出来的标签数据。
        aux (`dict[str, Any]`, *optional*):
            包含辅助信息的字典。

    Example:
        ```python
        >>> from transformers import PreTrainedModel
        >>> from dataclasses import dataclass

        >>> # 模拟模型输出
        >>> output = TransformersAdaptedM1ModelOutput(
        ...     loss=torch.tensor(0.5),
        ...     logits=torch.randn(32, 10),
        ...     prediction=torch.randn(32, 10),
        ...     hidden_states=(torch.randn(32, 100, 128),),
        ...     aux={"loss_dict": {"main": 0.3, "ortho": 0.1, "vicreg": 0.1}},
        ... )

        >>> # 访问各输出分量
        >>> print(f"Loss: {output.loss.item()}")
        Loss: 0.5
        >>> print(f"Logits shape: {output.logits.shape}")
        Logits shape: torch.Size([32, 10])
        >>> print(f"Auxiliary info keys: {output.aux.keys()}")
        Auxiliary info keys: dict_keys(['loss_dict'])
        ```

    ```python
        >>> # 在模型前向传播中的使用示例
        >>> model = YourModel(config)
        >>> inputs = {"input_ids": input_ids, "labels": labels}

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # 检查输出
        >>> print(outputs.loss)       # 训练损失
        >>> print(outputs.logits)     # 原始输出
        >>> print(outputs.predictions) # 预测结果
        >>> print(outputs.hidden_states)  # 隐藏状态
        >>> print(outputs.aux)        # 辅助信息
    ```

    Note:
        - 所有字段都定义为 `*optional*`，因为不同的前向传播配置可能返回不同的输出。
        - 当 `return_dict=True` 时，模型的前向传播将返回此类的实例。
        - `aux` 字段提供了一个灵活的方式来传递额外信息，避免不断修改输出类的结构。
    """

    loss: Optional[torch.Tensor] = None 
    logits: Optional[torch.Tensor] = None 
    predictions: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None 
    embeddings: Optional[torch.Tensor] = None 
    aux: Optional[dict[str, Any]] = None 