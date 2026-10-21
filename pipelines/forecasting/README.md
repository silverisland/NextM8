# 公共预测基础设施

本包提供与具体模型架构无关的公共接口，不能导入 `forecast_backbones`。
M4-Rum 直接使用本包；M8 原有入口保留兼容导出，避免复制两份训练/评估实现。

| 模块 | 公共能力 |
|---|---|
| `config` | 配置合并、YAML 读写、输出目录插值 |
| `stations` | 站表、固定容量、站名映射、起报时间网格 |
| `parquet` / `data` | 文件展开、schema 检查、数据读取、预切分与 refit |
| `rows` | 10:00 前最新起报行、整点网格、站日采样 |
| `distributed` | 进程身份、同步与早停广播 |
| `checkpoints` | 严格读写 safetensors 权重 |
| `training` | HF 参数、采样器、EMA、checkpoint、早停基类 |
| `metrics` / `reports` | 南网 D+1、容量 NRMSE、逐站工作簿和 JSON |

`EvaluationCallback` 的子类负责实现 `_score()`；公共层不构造或运行具体模型。
输入特征、网络、损失、模型推理和验证编排由各模型自行实现。
容量以显式配置的站表为准；`assets/` 提供迁移时的默认表，部署时可替换路径。

时间、评分及 checkpoint 选择算法沿用原有实现。公共包不执行数据切分、
不生成基模预测，也不替调用方判断数据是否满足起报时可用的要求。
