# M5：TimesFM-3（零样本 + LoRA 微调）

一个模型类 `TransformersAdaptedM5Model`（旧名 `TransformersAdaptedTimesFM3Model`），两种模式：

| 模式 | 触发 | 谁在用 | 行为 |
|---|---|---|---|
| legacy | 配置里**没有** `dataset.station_meta` | 线上 `nanwang_online`、`eval_shell/eval_pretrain.py`、`experiments/nwp_covariate` | 与 2026-10-14 前逐位相同（`tests/legacy_m5_forward.py` 冻结对照） |
| full | 给了 `dataset.station_meta` | `train.py` / `evaluate.py` / `station_report.py` | 站表容量 + LoRA 可微路径 + M8 决策层 |

配置：`config/configs.yaml` = 线上零样本（不要拿来训练）；`config/finetune.yaml` = 训练默认（= `config.py` 的 `DEFAULT_CFG`）。

基线（D8：站点几何默认关）：输入只有 ghi、温度；决策层只钳容量、不加晴空上限；点预测取中位数。
零样本 `z0_zeroshot` 与 LoRA `l1_lora_r8` 只差 LoRA 权重，summary 里 `gain = nanwang_short_D+1 − nanwang_zeroshot`。
几何留作变体：`g1_solar_cov`（两个几何协变量，需重训）、`g2_cs_ceiling`（晴空上限 1.05，只动决策层）。

## 跑基线（服务器）

    bash train_shell/train_m5_variants.sh --splits_dir data/splits/m8_v1 --variants z0_zeroshot,l1_lora_r8 --seeds 0,1,2 --batch_tag m5_base
    python -m forecast_backbones.m5.evaluate --runs runs/m5/m5_base/l1_lora_r8_s0 --test data/splits/m8_v1/valid.parquet \
           --decision-yaml forecast_backbones/m5/config/variants/d2_pstar_point.yaml        # p*，不重训

GPU 自动检测、按任务数平分：上面 4 个任务（z0 与种子无关只跑一次 + l1 × 3）在 8 卡上各占 2 张卡同时跑；1 个任务占满全部卡；
任务比卡多就每任务 1 张卡排队。加 `--dry_run` 只看分卡计划。`train.schedule.batch_size` 是**全局** batch（所有卡合计，默认 16），
每卡 = 全局 / 卡数，所以卡数不同的 run 配方相同、分数可比；summary 里 `batch_global` / `n_gpus` 两列记着。
一次调用默认占满所有卡，不要同时开两个（要并发就用 `--gpus` 把卡分开）。

## 实验变体（`config/variants/`，每个只和 l1 差一处，3 种子起跑）

| 变体 | 改什么 | 回答的问题 |
|---|---|---|
| `s2_lr3e5_ema` / `s2_lr1e5_ema` / `s2_lr1e4_ema` | lr 3e-5 / 1e-5 / 1e-4 + EMA 0.998 + smooth_k 3 | 训练协议：学习率与 EMA 值多少 |
| `s2_head_only` | 不挂 LoRA，只训 output_head | 收益是不是只是输出再校准 |
| `s2_frozen_head` | LoRA 同 l1，head 冻结 | head 要不要训 |
| `s2_lora_resblock` | l1 + 输入残差块三个层 | NWP 前瞻入口要不要适配 |
| `s2_probe_resblock` | 不挂 LoRA，整块解冻输入残差块 + head | TimesFM 论文的「LP」够不够 |
| `s2_lora_var_resblock` | LoRA 只挂 var_attn + 输入残差块 + head | 新主候选：只改 NWP→功率通路、护住日形 |
| `s2_lora_var_only` / `s2_lora_seq_only` | 只挂跨变量 / 只挂时间注意力 | 收益在哪条通路 |
| `s3_solargis_cov` | 协变量 2 → 4 路（加 Solargis GHI / 温度，与 M8 j1 同） | 多一套 NWP 有没有用（测试 / 线上也要有这两对列） |
| `s3_cov_all_var` | 四路 NWP + 几何协变量 + var_attn/残差块 LoRA | 组合上限探针（非单因素） |
| `s5_rank4` / `s5_rank16` / `s5_rank32` | LoRA rank 4 / 16 / 32（alpha = 2·rank） | 容量 |
| `s6_all_linear` / `s6_lora_vo_only` | LoRA 挂全部线性层 / 只挂 v、o | 容量 |
| `s6_full_ft_lr1e6` / `s6_full_ft_lr1e5` | 全量微调（整块解冻 transformer_stack + 残差块 + head） | 全量微调在极小 lr 下能否赢 LoRA |
| `s6_partial_ends` | 只解冻首尾各两层 + 残差块 + head，lr 1e-5 | 部分微调 |

LoRA 靶点：普通名字按模块名尾段匹配，`re:` 开头按完整路径匹配（如 `re:.*\.var_attn\.(query|key|value|out)_proj`），空列表 = 不挂 LoRA。
`finetune.unfreeze_modules` 整块解冻（全参数可训，权重文件一起存）。只含 `decision:` 段的变体（d2 / g2）调度脚本自动跳过，评估时用。

## 免训练旋钮（不重训，同一批 checkpoint 上比较）

    bash eval_shell/eval_m5.sh --runs <run> --test <parquet> --point p_star \
         --set point.tail=linear        # O0：p* 网格在 q0.1/q0.9 之外线性外推（默认 flat 平延）
         --set point.spread_lo=1.25     # T3：中位数以下离中位数的距离 ×1.25（spread_hi 同理）
         --wise_alpha 0.5               # W2 WiSE-FT：LoRA 增量与 head 向预训练插值（0 = 零样本）

## 站表

`station_meta.csv` 从 M8 复制，M5 / M6 / M8 各一份、各自维护；改站表要三处一起改。容量只取 `cap_mw`（每站固定值）。

## 权重

`model.safetensors` 只含 LoRA（`*.lora_A / *.lora_B`）与 `output_head.*`，键名相对 TimesFM 的 `model`；底座从 `timesfm3.checkpoint` 加载。
零样本 run 没有权重文件。
