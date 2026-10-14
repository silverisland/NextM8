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

    bash train_shell/train_m5_variants.sh --splits_dir data/splits/m8_v1 --variants z0_zeroshot --gpus 0 --batch_tag m5_base
    bash train_shell/train_m5_variants.sh --splits_dir data/splits/m8_v1 --variants l1_lora_r8 --seeds 0,1,2 --gpus 1,2,3 --batch_tag m5_base
    python -m forecast_backbones.m5.evaluate --runs runs/m5/m5_base/l1_lora_r8_s0 --test data/splits/m8_v1/valid.parquet \
           --decision-yaml forecast_backbones/m5/config/variants/d2_pstar_point.yaml        # p*，不重训

## 站表

`station_meta.csv` 从 M8 复制，M5 / M6 / M8 各一份、各自维护；改站表要三处一起改。容量只取 `cap_mw`（每站固定值）。

## 权重

`model.safetensors` 只含 LoRA（`*.lora_A / *.lora_B`）与 `output_head.*`，键名相对 TimesFM 的 `model`；底座从 `timesfm3.checkpoint` 加载。
零样本 run 没有权重文件。
