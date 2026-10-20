# M5：TimesFM-3（零样本 + LoRA 微调）

一个模型类 `TransformersAdaptedM5Model`，两种模式。旧名 `TransformersAdaptedTimesFM3Model` 是行为相同的子类（不是别名）：
`unified_inference` 的预测列名 = 类名，线上 `multi_infer.py` / M8 模板的 TSFM 预推理按 `TransformersAdaptedTimesFM3Model` 取列。


| 模式 | 触发 | 谁在用 | 行为 |
|---|---|---|---|
| legacy | 配置里**没有** `dataset.station_meta` | 线上 `nanwang_online`、`eval_shell/eval_pretrain.py`、`experiments/nwp_covariate` | 与 2026-10-14 前逐位相同（`tests/legacy_m5_forward.py` 冻结对照） |
| full | 给了 `dataset.station_meta` | `train.py` / `evaluate.py` / `station_report.py` | 站表容量 + LoRA 可微路径 + M8 决策层 |

配置（2026-10-20 起同 M8 平铺）：`config/*.yaml` = 各配方，只写与 `config.py` 的 `DEFAULT_CFG` 不同的字段；`config/decision/` = 只动决策层的 yaml；
`config/configs.yaml` = 线上零样本（legacy，不要拿来训练；`train_shell/train_script.py`、`eval_pretrain.py` 按这个路径找）。

基线（D8：站点几何默认关）：输入只有 ghi、温度；决策层只钳容量、不加晴空上限；点预测取中位数。
零样本 `z0_zeroshot` 与 LoRA run 只差 LoRA 权重，summary 里 `gain = nanwang_short_D+1 − nanwang_zeroshot`。

## 配方（`config/`，2026-10-20 定架构）

| 配置 | 内容 | 用途 |
|---|---|---|
| `f1_final` | LoRA r8（q/k/v/o）+ 冻结 head + 常数 lr 1e-4 + p* 选点 | 最终配方（= 家里 `s7_fh_const` + `d2_pstar_point`，valid D+1 76.3） |
| `f2_lr5e5` / `f3_lr2e4` | f1 + lr 5e-5 / 2e-4 | 同一模型，调 lr |
| `f4_drop01` | f1 + LoRA dropout 0.1 | 同一模型，调正则 |
| `f5_wd01` | f1 + weight_decay 0.1 | 同一模型，调正则 |
| `s3_c1_envelope` / `_zs` | f1 + 功率历史日形包络通道（训练 / 零样本） | 输入多一条通道 = 另一个模型，家里先与 f1 成对比 |
| `z0_zeroshot` | 不训练 | 零样本参照 |
| `decision/d2_pstar_point` | 只改点预测为 p* | 给 p* 之前训的 run 不重训改决策 |

f 系列每个 yaml 都带南网训练设置：`dataset.src_dir`、列名（`timestamp_win / observe_power / observe_power_future /
general_ghi(_predict) / general_temp(_predict)`，大写 `general_GHI` 也能对上）、站表 `forecast_backbones/m5/info.csv` + `station_meta_cols`
（`plantid / plant_pointname / GCCAPACITY / LONGITUDE / LATITUDE`）、`train.schedule.save_every: 500`、`timesfm3.checkpoint`
（⚠️ 南网服务器上改成那边的 TimesFM-3 权重目录）。常数 lr 下 `checkpoint-N` 与"N 步就停"的 run 相同，所以全量训练可以事后挑步。

## 家里（切分目录，有验证集）

    bash train_shell/train_m5_variants.sh --splits_dir data/splits/m8_v1 --station_meta forecast_backbones/m5/station_meta.csv \
         --variants f1_final,f2_lr5e5,f3_lr2e4,f4_drop01,f5_wd01 --seeds 0,1,2 --batch_tag m5_f
    bash eval_shell/eval_m5.sh --runs runs/m5/m5_f/f1_final_s0 --test "datasets/test/*.parquet" --full     # 线上同一接口 unified_inference

f 系列的站表是线上 `info.csv`（南网站号 1780…），家里切分的站号是 1001…：家里训练必须 `--station_meta` 换回 `station_meta.csv`
（列名映射随之清空，表头自动识别）。f 系列已经是 p*，评估不用再给 `--decision_yaml`。

GPU 自动检测、按任务数平分：任务数 ≤ 卡数时各分 floor(卡数 / 任务数) 张卡同时跑（z0 与种子无关只跑一次）；1 个任务占满全部卡；
任务比卡多就每任务 1 张卡排队。加 `--dry_run` 只看分卡计划。`train.schedule.batch_size` 是**全局** batch（所有卡合计，默认 16），
每卡 = 全局 / 卡数，所以卡数不同的 run 配方相同、分数可比；summary 里 `batch_global` / `n_gpus` 两列记着。
一次调用默认占满所有卡，不要同时开两个（要并发就用 `--gpus` 把卡分开）。

## 南网服务器全量训练（`train_shell/m5_train_cli.py`，2026-10-20）

结构与 `m8_train_cli.py` / 南网 `train_scripts/m6_b_train_cli.py` 一致（`pipelines.UnifiedDataset` + `pipelines.train_model`，
产物 `runs/m5/<ckpt-dir>/model_result/{model.safetensors, model_config.yaml}`）。`nanwang_online/m5_train_cli.py` 是同一份拷贝
（`tests/test_nanwang.py` 守住两份一致），拷到南网仓库的 `train_scripts/` 下即可用。

```bash
cd <仓库>/train_scripts
python m5_train_cli.py --config f1_final.yaml --ckpt-dir f1_final                         # 读 <src-dir>/train/**/*.parquet
python m5_train_cli.py --config f3_lr2e4.yaml --ckpt-dir f3_lr2e4
python m5_train_cli.py --config f1_final.yaml --ckpt-dir smoke --max-steps 2                # 冒烟
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 -- m5_train_cli.py --config f1_final.yaml --ckpt-dir f1_2gpu
```

- 命令行参数与模板相同（`--config / --ckpt-dir / --src-dir / --train-path / --max-steps`），其余全在 yaml 里；超参来自 `train.schedule`
  （含 `lr_scheduler`：f 系列是 constant）与 `finetune.lora`。每卡 batch = 全局 `batch_size` / 卡数（与家里同口径）。
- `save_every: 500` → `hf_trainer_output/checkpoint-N/`，训完补 `model_config.yaml`，与 `model_result/` 一样原地可加载。
  **checkpoint 只存 LoRA**（`model.save_trainable_only()`，约 13 MB；不开的话 HF 会把 3.3 亿参数的底座每份存一遍，约 1.3 GB，且键名对不上），
  且 `save_only_model`：不存优化器 / 调度器 / RNG 状态（不能断点续训，事后挑步用不到）。每份 = `model.safetensors` + `model_config.yaml`
  （+ HF 自带的 `training_args.bin` / `trainer_state.json`）。`model_config.yaml` 存的是合并后的完整配置：以后改 `DEFAULT_CFG` 不会悄悄改变旧 checkpoint。
- 数据必须有 `station` 站名列（`drop_str_type=False` 保留），站名要在站表 `plant_pointname` 里；列名大小写不敏感；没有的 NWP 对不喂，
  实际用上的对 `cov_present` 与训练站 `known_stations` 写进 `model_config.yaml`。
- 不支持（配置里开了直接报错，要用就走 `train_m5.sh --train_parquet`）：零样本、`train.ema_decay`、`sampling: station_day`、
  `train.optim`（LoRA+ / head_lr / adapter_lr）、`finetune.init_from`。

## 线上推理模板（`nanwang_online/eval_scripts_template/m5/single_model_scene_infer_m5.py`）

照 M8 模板写，差别标 `[M5 适配]`：不做 TSFM 预推理（M5 本身就是 TimesFM；M8 的 `tsfm_predict` 那一路保持零样本 M5，不要换成本模型）；
显式加载 LoRA 权重并断言 `weights_loaded`（否则安静退化成零样本）；TimesFM-3 底座路径从线上零样本 M5 的 `model_config.yaml` 取；
站表指向线上 `info.csv`；超短期 192 步走 `unified_inference`，短期 480 步走 `rolling_inference`（每窗 192 步，D+1 在第一个窗里，
与家里 `eval_m5.sh --full` 同口径；`future_target` 为空）；第一次起报打印输入列核对，warning 全部打出来。
事后挑步：每个 `checkpoint-N` 在**训练期之后**的时段（至少 2–4 周）上跑一遍，挑"配置 × 步数"。

## 已删除的变体（2026-10-20）

m5_s7 / m5_s8 两批之后只有 lr 轴有增益（`s7_fh_const` 74.15 vs `s2_frozen_head` 73.32，p* 下 76.3）。打平或更差而删掉的方向：
LoRA 靶点 / rank / 全量与部分微调（s2_lora_* / s5_rank* / s6_*）、head 训练方式（s2_head_only / s7_head_lr_split / s7_lpft）、
LoRA+ / (IA)³ / 残差块适配器（s7_*）、soft prompt / 静态嵌入（s8_*）、门控 LoRA（s9_gated_lora）、几何协变量（g1_solar_cov）。
没跑过就删的：s9_hydra_k3、s3_c2_envelope_nwp(_zs)、s3_cov_all_var、g2_cs_ceiling。`s3_solargis_cov` 删掉：南网训练数据没有 Solargis 列
（线上 multi_infer 还会把 general_* 抄进 *_SOLARGIS，喂进去的是假值）。都在 git 历史里。

适配器（`adapters.py`，配置在 `finetune.*`，全部默认关，关时与原 LoRA 逐位相同）：`lora.gate` / `lora.experts`、`ia3`、`prompt`、
`static_embed`、`resblock_adapter`、`init_from`；协议旋钮 `train.schedule.lr_scheduler`、`train.optim`（`lora_plus_ratio` / `head_lr` / `adapter_lr`）；
输入 `features.derived.history_envelope` / `envelope_nwp`（参数在 `features.envelope`，GHI 列对默认 `general_ghi` / `general_ghi_predict`）。
除 prompt 外，适配器初始化时与零样本逐位相同（第 0 步 = z0，WiSE α=0 = 零样本）；prompt 多出的 token 进 softmax，第 0 步就偏离零样本。
summary.csv 多三列：`adapters`、`lr_scheduler`、`optim`。逐站 prompt / 静态嵌入看 `nanwang_unseen`：已见站涨、未见站不跌才算数。
prompt 与 LP-FT（`finetune.init_from`）的第 0 步不是零样本，summary 的 `nanwang_zeroshot` / `gain` 记 NaN，直接与 z0 比。
WiSE 对门控 / 路由网络也插值（门向 1、路由向初值），所以门控变体的 WiSE α 不是单纯的"LoRA 强度 × α"。
逐站 prompt 的表按构造时站表的站数建：评估 / 线上换了站数不同的站表会加载失败；训练集外的站一律只用全局 prompt。

LoRA 靶点：普通名字按模块名尾段匹配，`re:` 开头按完整路径匹配（如 `re:.*\.var_attn\.(query|key|value|out)_proj`），空列表 = 不挂 LoRA。
`finetune.unfreeze_modules` 整块解冻（全参数可训，权重文件一起存）。只含 `decision:` 段的 yaml 放在 `config/decision/`，评估时用 `--decision_yaml`。

## 免训练旋钮（不重训，同一批 checkpoint 上比较）

    bash eval_shell/eval_m5.sh --runs <run> --test <parquet> --point p_star \
         --set point.tail=linear        # O0：p* 网格在 q0.1/q0.9 之外线性外推（默认 flat 平延）
         --set point.spread_lo=1.25     # T3：中位数以下离中位数的距离 ×1.25（spread_hi 同理）
         --wise_alpha 0.5               # W2 WiSE-FT：LoRA 增量与 head 向预训练插值（0 = 零样本）

## 站表

`station_meta.csv`（家里切分，站号 1001…）从 M8 复制，M5 / M6 / M8 各一份、各自维护；`info.csv` = 线上站点信息表（南网站号 1780…，
与 `forecast_backbones/m8/info.csv` 同内容）。容量只取 `cap_mw` / `GCCAPACITY`（每站固定值）；f 系列不用经纬度：太阳几何（pvlib 高度角 / 晴空功率 / 夜间）只在 `decision.physics.night_zero`、`clear_sky_cap_ratio`、
`features.derived.solar_elevation / clear_sky_power`、`train.loss.night_weight≠1` 时才算（`geometry_needed()`，2026-10-20）；默认都关，
不算、也不出"缺坐标"warning，结果表的 `elev_deg / p_clear` 列为 NaN。站表缺坐标只影响打开了上面这些开关的配置。

## 权重

`model.safetensors` 只含 LoRA（`*.lora_A / *.lora_B`）、(IA)³（`*.ia3_scale`）、`m5_adapters.*`、`output_head.*` 与 `unfreeze_modules` 的整块参数，
键名相对 TimesFM 的 `model`；底座从 `timesfm3.checkpoint` 加载。
零样本 run 没有权重文件。
