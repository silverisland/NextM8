# M6 —— TimesFM-3 + 分位校准 + 南网口径决策层

路线 A（本目录当前状态，零样本、无训练）：

    # 1. 填站坐标：forecast_backbones/m6/station_meta.csv（station_id 与 parquet 一致）
    # 2. 回填 + 校准 + 阶梯报表：cal 段不得晚于 val 段起点（否则报时间泄漏错误拒绝）；实际参与拟合的
    #    只是 cal 段末尾 calibration.window_days（默认 45）天的滚动窗口——--cal-start/--cal-end 之间
    #    可以给更长的范围（决定评估/落盘覆盖的区间），拟合窗口会自动收窄到最后 45 天。
    #    parquet 路径含通配符时给引号，避免被 shell 展开：
    python -m forecast_backbones.m6.transformer_trainer \
        --config forecast_backbones/m6/config/configs.yaml --parquet 'data/clean/*.parquet' \
        --cal-start 2025-06-01 --cal-end 2025-07-16 --val-start 2025-07-16 --val-end 2025-09-01
    # 产物：
    #   REPORT（人读）  runs/m6_baseline/reports/{ladder.csv, diagnostics.json}
    #     diagnostics.json：calibrator_last_stats（apply 产出的逐 τ 未校准比例等）、degraded_frac、
    #     n_rows_val、n_rows_cal、var_bin_hist、stations_scored、shrink_factor——同样内容也会打印一遍。
    #   STATE（机读）   runs/m6_baseline/{calibration.parquet, calibration.parquet.json,
    #                                     decision_state.json, result_cal.parquet, result_val.parquet}
    # 3. 之后正常推理：config/configs.yaml 的 calibration.table_path 指向上一步的 calibration.parquet，
    #    registry 用 model_name=m6 构建即可，logits 即提交曲线。
    # 注意：--out 覆盖 run.out_dir 时，只有形如 ${run.out_dir}/... 的字面量才会跟着重新插值；
    #       calibration.table_path 若被改写成一个不含 ${run.out_dir} 的字面量路径，不会跟着 --out 走。

阶梯：raw_median → global_shrink（f*）→ physics_median → calibrated_pstar（短期 p* / 超短期 p*_L1）。
先看 ladder.csv 里 calibrated_pstar 与 raw_median 的差：<0.3 点说明当前 σ 已很小，LoRA 收益有限。
raw_median / global_shrink / physics_median 三档用的是中位数（不受下面第 (b) 条 tail 偏置影响），只有
calibrated_pstar 用 p*。

阶梯的 calibrated_pstar 档与打开校准部署的模型 forward 输出数值上是等价的：两者走同一顺序
calibrate → enforce_monotone → physics_clip → select_point，同一套 levels 与 cap 口径
（`tests/test_ladder.py::test_ladder_calibrated_pstar_matches_deployed_model_with_calibration`）。
不是 bit-identical——是 numerically equal（≤1e-5；evaluate_model 落盘时把 p_clear/elev_deg
存成 float32，这一步截断带来的误差量级就是 1e-5）。

测试：`python -m pytest forecast_backbones/m6/tests -q`（不需要 timesfm3；真实底座的冒烟测试见
`tests/test_real_backbone.py`，本机缺 timesfm3 与本地权重时自动跳过，不影响本目录当前测试全绿；
训练机上安装了 timesfm3 且权重路径存在时才会真正运行）。

## 框架契约（与 train_script.py / pipelines.model_registry 对齐）

- `transformer_models.py`：第一个 nn.Module 子类 `TransformersAdaptedM6Model`，构造签名 `Model(model_config)`；
  `model.config` 是合并后的 cfg（train_script 训后落盘用）；`compute_metrics` 是空实现（M6 的早停指标由回调算）。
- `transformer_trainer.py`：`evaluate_model(model, test_dataset, HF_collate_fn, config)` 与 `M6_build_training_args(model_config, result_dir)`。
- 默认配置在 `config/configs.yaml`（train_script 按 `<模型目录>/config/configs.yaml` 自动定位；不再有根目录 `model_config.yaml`）。
- 训练入口：train_script 里 `--model_name m6` 走独立分支直接调 `train.train_m6(config, splits_dir=DATASET_PATH, out_dir=RUN_PATH)`，
  不拆进 `pipelines.train_model`（M6 需要校准段、按块早停、训后校准表拟合）。`DATASET_PATH` 是
  `pipelines.time_splits` 落的目录（`train/valid/cal.parquet + split_manifest.json`，`cal_days ≥ 1`），与 M1–M4 共用同一份切分。
- 站标识：M6 内部按整数 `station_id` 走（校准分层、坐标）。数据只有站名列 `station` 时，训练入口按
  `station_meta.csv` 的 `station` 列自动映射补 `station_id`（查不到的站名直接报错，请先补表），映射后的副本落在 `out_dir/.data/`。
- M5 同规：`TransformersAdaptedM5Model`（旧名 `TransformersAdaptedTimesFM3Model` 保留为别名）+ `transformer_trainer.evaluate_model`
  + `config/configs.yaml`；M5 不训练，没有 build_training_args。

## 备注

(a) `masks.curtail_from_rule` 与 `features.RowContext.k_steps` 已实现并有单元测试
（`tests/test_masks.py`、`tests/test_row_context.py`），但路线 A 没有接入它们：这两者唯一的消费方是
训练损失（弃光掩码进 loss_mask），而路线 A 是零样本 evaluator 推理，没有训练循环。它们要等 Plan 2
（LoRA 微调）打开 `finetune.enabled` 后才会被接进损失计算。

(b) 决策层 `decision.point.tail: flat`（`decision.quantile_grid` 两端平延）会把 p* 系统性抬高，且抬高
幅度依赖分位档位密度：在对称 N(60,15) 预测律、C=100 上实测，用 9 档 `MODEL_LEVELS`（模型原生、未校准）
时 p* 偏高 +4.06；用 15 档 `CAL_LEVELS`（校准后）时只偏高 +0.58——档位越密，两端平延段越短，偏置越小。
因此任何比较「开校准 vs 关校准」对 p* 影响的消融实验，都必须固定同一套分位档位（同为 `MODEL_LEVELS`
或同为 `CAL_LEVELS`），否则档位密度变化本身的偏置会和校准效果混在一起，读不出校准的真实贡献。
`decision.point.tail` 目前只实现了这一种（平延）扩展；`model forward` 与 `recompute_points`
在读到非 `"flat"` 的值时会直接 `raise ValueError`（不会静默按平延跑）。

(c) `calibration.enabled=True` 但 `calibration.table_path` 指向的文件不存在时：本次不校准，p* 会在
未校准的 9 个原生 `MODEL_LEVELS` 上计算——按备注 (b) 的实测，这套分位集合本身系统性偏高。为避免在
这套偏高的分位集合上再跑 p*/p*_L1 进一步放大偏置，这种"缺表退化"场景下 `decision.point.default`
与 `by_step` 里配置成 `p_star`/`p_star_l1` 的方法本次会自动回退为 `median`（`cfg` 本身不改写，只影响
当次 forward 实际生效的方法；后续用 `set_calibrator` 补上真正的 calibrator 后不再回退）。这与主动关闭
`calibration.enabled=False` 是两回事：主动关闭校准时仍按配置跑 p_star/p_star_l1，不触发这个回退。

## 路线 B：LoRA 微调（计划 2）

前置：训练机 torch ≥ 2.4（timesfm3 用 `nn.RMSNorm`；本地 2.2.2 靠 `_torch_compat.ensure_rmsnorm()` 垫片跑测试）、
`transformers==4.40.1`、`accelerate>=0.21`、timesfm3 3.0.2、真权重目录。`finetune.lora.target_modules` 是 timesfm3 的真名
`query_proj/key_proj/value_proj/out_proj`（`seq_attn` 与 `var_attn` 都挂）；spec 初稿里的 `q_proj…` 在包里不存在，挂载零匹配会直接报错。
实测过、确认可跑通的固定版本组合：`transformers==4.40.1`、`accelerate==0.29.3`、`timesfm3==3.0.2`；`build_training_args`
按 `TrainingArguments.__init__` 实际签名探测传 `evaluation_strategy` 还是 `eval_strategy`（transformers ≥ 4.46 删掉了前者改名后者），换更新的 transformers 不需要改代码，但没有实测过。

`_torch_compat.py` 里还有第二个垫片 `ensure_finite_running_stats()`：timesfm3.torch 某些归一化层在方差恒为 0 的输入上反向传播会算出
`sqrt(0)` 的梯度（NaN），这个垫片只改反向路径，前向数值逐位不变。两个垫片都在 `backbone.py` 导入 `timesfm3.torch` 之前打上，但两者
必要性不同：`ensure_rmsnorm()` 只在 torch < 2.4（没有原生 `nn.RMSNorm`）时才需要，torch ≥ 2.4 上是空操作；
`ensure_finite_running_stats()` 与 torch 版本无关，任何 torch 版本都必须打——光伏夜间连续整段恰为 0.0，零方差 patch 是训练期的
常态而不是边角情形，垫片缺失只会在某次夜间 batch 恰好命中该分支的反向传播时才炸出 NaN。`backbone.py` 在打完垫片后会立刻检查
`timesfm3.torch.util.update_running_stats._m6_eps_patched` 确实为真，检查不过直接 `RuntimeError`（导入期就拦下来，不是训练到
一半才发现某个 timesfm3 build 把这个函数改了名字）。

训练（一折），两种喂数方式二选一：
```
# a) 原始行表，按 config 的 finetune.split 在内存里切
python -m forecast_backbones.m6.train --config forecast_backbones/m6/config/configs.yaml \
    --parquet "/path/to/*.parquet" --out runs/m6_lora_f0
# b) pipelines.time_splits 切好的目录（与 M1–M4 同一份切分；train_script.py --model_name m6 走的就是这条）
python -m forecast_backbones.m6.train --config forecast_backbones/m6/config/configs.yaml \
    --splits-dir dataset/<name> --out runs/m6_lora_f0
```
`config/configs.yaml` 里先把 `finetune.enabled: true`、`finetune.split` 改成真实日期；滚动第二折加 `--shift-days N`（只对 a 有效，
b 的整体后移在切分时给），输出目录换 `_f1`。`--max-steps N` 覆盖 `finetune.schedule.max_steps`（0 = 不限步数，按 `epochs_max`）。
服务器上用仓库根目录的 `train_m6.sh`（默认用全部可见 GPU，见下文“训练预算、验证与多卡”）。

`finetune.split` 现在是验证块列表：`val_blocks` 若干个 `[start, end)` 整天区间（分散到不同季节），`test_start`/`test_end` 尾段测试，
`cal_days` 校准段（固定放在 `test_start − purge_days` 之前），`purge_days`（块前剔除训练行）与 `embargo_days`（块后额外剔除）。
两个天数都由 `SplitPlan.from_dict` 当场校验，给不够直接报错：`purge_days` 须 ≥ H 折算天数（默认 2 天），
`purge_days + embargo_days` 须 ≥ L + H 折算天数（默认 9 天）——前 H 步保证块后训练行的标签不与块标签重叠，
再往后的 L 步保证这些训练行 7 天的上下文不含块的标签。早停指标是各块得分（默认南网短期，见下文 `early_stop.metric`）的均值，逐块分数在 `train_log.json` 的 `history[*].blocks`；
切分事实落在 `out_dir/split_manifest.json`（`generated_by: forecast_backbones.m6.train`）。给 M1–M4 生成同一规则的 `train/valid/cal/test.parquet`：
`python -m pipelines.time_splits --parquet RAW.parquet --out-dir dataset/<name> --m6-config forecast_backbones/m6/config/configs.yaml`
（`--m6-config` 时 L/H/step_min 默认取该 yaml 的 `timesfm3.context_len` / `timesfm3.pred_len` / `dataset.step_min`，
显式给的 `--L/--H/--step-min` 仍然胜出；这条路落的 manifest 是 `generated_by: pipelines.time_splits`）。
默认日期是占位示例，训练前改成真实日期；`--shift-days N` 把全部边界整体后移 N 天做第二折。

**按站不同的测试起点**（测试集已按站切好、各站测试时段不一样）：`test_start` 不再是一个日期，而是每站一个。
CLI 直接从现有测试集文件推：
`python -m pipelines.time_splits --parquet TRAIN.parquet --out-dir dataset/<name> --split-yaml split.yaml --test-parquets "data/test/*.parquet"`
（每站 `test_start` = 该站在这些文件里最早起报时刻所在的整天；也可 `--test-start-by-station map.yaml` 给 `{station: 'YYYY-MM-DD'}`；
此时 split.yaml 里的 `test_start` 不写或写 null，写了非空值会报错；站标识列缺省认 `station_id` / `station`，可用 `--station-col` 指定）。
M6 配置里对应写 `finetune.split.test_start: {1001: "2025-06-28", 1002: "2025-06-13"}`，
或 `test_start: null` + `test_parquets: ["data/test/*.parquet"]`。规则：验证块、purge/embargo、cal_days 全站共用；
每站的校准段各自贴在自己的 `test_start − purge` 之前；某站 `test_start` 太早、容不下某个验证块时，该站跳过那个块（其他站照常），
跳过情况记在 manifest 的 `plan.skipped_val_blocks` 与终端输出里；一个站一个块都容不下、或一个块没有任何站容得下、
或训练数据里有站没给 `test_start`，都直接报错。`valid.parquet` 的 `split_block` 与 `train_log.json` 里的块序号一律是全局序号。

**各站跨度不一（有的不到一年）**：切分配置写 `val_blocks: auto` + `n_val_blocks` + `val_block_days`，CLI 按各站实际数据自动放
全站共用的日历块（可用区间等分成 n 段，每段取"块内有数据的站·天"最多的位置）。每次切分都打印逐站覆盖表
（数据起止、测试起点、参加/跳过的块、训练/验证天数与占比，也写进 manifest 的 `station_report`）；有站一行验证数据都没有时报错，
除非配置里写 `allow_stations_without_val: true`。`--dry-run` 只打印不落盘；`--parquet` 直接收通配符（一站一个文件时不必先拼表，
各文件列须一致）。落盘目录里的 `split_resolved.yaml` 是这次实际用到的固定日期（含每站 `test_start` 映射），原样 `--split-yaml` 即可复现；
M6 的 `finetune.split` 不解析 `auto`，要用就抄 `split_resolved.yaml` 里的 `val_blocks`。M1–M4 的模板见仓库根目录 `configs/split_m1_m4.yaml`。

**数据太短的站只进训练**：`min_station_days_for_val: N`（测试起点前有数据的天数 < N 的站）与 `train_only_stations: [站名, ...]`（手动点名）
取并集。这些站不参加验证、没有校准段、不参与自动放块，除测试起点前 `purge_days` 天（防止训练标签伸进它自己的测试段）外全部进 `train.parquet`；
它们落在验证块日期里的行也是训练行。逐站覆盖表的"只进训练"列标出这些站；`split_resolved.yaml` 把最终名单写成 `train_only_stations`（去掉阈值），
原样重跑结果不变。阈值需要数据，M6 的 `finetune.split` 只认 `train_only_stations` 名单（且须按站给 `test_start`）。

**只训 LoRA、不做 conformal**：用 `config/lora_no_conformal.yaml`（`calibration.enabled: false`、`finetune.enabled: true`、
点预测取中位数），可以直接喂 M1–M4 的切分目录，不需要 `cal.parquet`。早停按未校准分位上的中位数打分，
不写 `calibration.parquet`，`ladder_lora.csv` 两行叫 `zeroshot_uncalibrated` / `lora_uncalibrated`。

```bash
python -m forecast_backbones.m6.train --config forecast_backbones/m6/config/lora_no_conformal.yaml \
    --splits-dir data/splits/m1_m4_v1 --out runs/m6_lora_v1
# 服务器：全部可见 GPU，产物在 runs/m6/<run_id>
bash train_m6.sh --splits_dir data/splits/m1_m4_v1 --run_id lora_v1
```

**训练预算、验证与多卡**（`finetune.schedule` / `early_stop`）：
- 预算按优化器步数给：`max_steps`（默认 4000），`epochs_max` 只在 `max_steps: null` 时生效。训练行按 `sampling` 随机抽：`random` 在全部训练行里均匀抽，`station_day` 让每个（站, 起报日）被抽中的概率相同。
- 每 `eval_every_steps` 步（默认 250，训练最后一步必打）在全部验证块上打一次南网短期分，`patience` 按验证次数计。验证只推理指标用到的行（每站每个起报日取 10:00 前最晚的一行），分数与推理全部行相同；做 conformal 时校准段每次验证只推理一遍，各块共用。
- 早停指标 `early_stop.metric`：`nanwang_after_pstar`（默认，南网短期 D+1，越高越好）或 `nrmse`（提交曲线全部 192 步、含夜间的容量归一 RMSE%，越低越好，超短期也算在内）。`nrmse` 每站每天推理起报时刻落在 `nrmse_issue_every_min` 分钟网格上的行（默认 60 → 24 行，验证开销约为南网口径的 24 倍；15 → 全部行）。
- 多卡：`accelerate launch` / `torchrun` 起多个进程即可，`train_m6.sh` 默认用全部可见 GPU。每张卡一个进程、各吃不同的训练批，每步共 卡数 × `batch_size` 行（`batch_size` 是每张卡的）；验证、存最优与报表只在主进程做，其余进程训完即退出。站号需要转换时由主进程写 `out_dir/.data/` 下的临时 parquet，其余进程等它写完再读。

**训练与测试的站对不上**：只在测试集里、训练数据里没有的站，切分时忽略（manifest 的 `ignored_test_only_stations` 记下名单）；
只在训练数据里、没有测试集的站，给占位 `test_start` = 该站最后一天 + 1 + `purge_days`，于是它没有测试行、末尾不剔除任何行，
照常参加训练和验证（写进 `train_only_stations` 则全进训练）。`val_blocks: auto` 的区间终点只看有测试集的站，
没有测试集的站晚于终点的数据不参与放块、照常进训练，免得验证块被拉进别的站的测试时段。这些站列在 `stations_without_test`，覆盖表的测试起点列显示"无测试集"，
`split_resolved.yaml` 连同占位日期一起写下，原样重跑不变。M6 `make_splits` 同样处理。

产物（`run.out_dir`）：`model.safetensors`（LoRA + 可训 output_head 的权重，键名相对 `backbone.model`，不含底座；验证块均值早停指标最优的那次验证；第 0 步是零样本基线，所以在验证集上永远不输零样本；旧 run 的 `lora_weights.pt` 仍能读）、
`model_config.yaml`（训练配置，保留 `${run.out_dir}` 占位符，推理时把 `run.out_dir` 指到产物目录即可整体搬走）、
`calibration.parquet`（在最优 LoRA 上重拟合）、`train_log.json`、`result_val_lora.parquet`；报表（`eval.report_dir`）：
`ladder_lora.csv`（阶梯 4：`zeroshot_calibrated_pstar` / `lora_calibrated_pstar` 两行，全部验证日）、`station_ladder.csv`
（逐站 `acc_zeroshot` / `acc_lora` / `n_days` / `gain`）。推理时全部站都用这份 LoRA，没有逐站回退。

推理：`finetune.enabled: true` 时（`torch_path = bool(finetune.enabled) or inference_path == "torch"`，`enabled=True` 已经短路成
`True`）**永远**走 torch 路径并挂 LoRA，`inference_path` 这个键在 `enabled=True` 时不被读取、不产生任何差异；模型构造期自动加载
`lora_path`（缺失会 warn 并以零样本权重运行）。`inference_path` 只在
`finetune.enabled: false` 时才生效：`evaluator`＝路线 A 原样不变（零样本、无 LoRA、无梯度），`torch`＝同样零样本但走可微的
torch 路径（无 LoRA，仅用于对照 evaluator 与 torch 两条路径本身的数值一致性，见下段 1e-4 门控测试）。
路线 A 的回填 CLI（`transformer_trainer.backfill_and_report`）
不感知 `finetune`：只要 cfg 里 `finetune.enabled=True` 且 `lora_path` 指向的文件存在，模型构造时就会照常经 torch 路径加载
LoRA 权重后再回填——路线 A 与路线 B 的产物在同一套 `run.out_dir` 约定下天然兼容，不需要额外接线（回归测试见
`tests/test_ladder.py::test_backfill_uses_lora_weights_when_finetune_enabled`）。

训练路径只经 `TimesFM3Torch.decode.__wrapped__` 调底座（`decode` 本身被 `@torch.no_grad()` 装饰，`predict_batch` 再套 `inference_mode`，
都不可训）。`backbone.predict_quantiles` 逐条照搬了 `predict_batch` 的前后处理，四种开关组合与 evaluator 对齐到 1e-4（随机小模型测试）；
真权重上的同一测试在 `test_real_backbone.py::test_torch_path_matches_evaluator_on_real_checkpoint`，有 checkpoint 才跑。

真机待办清单：训练机上跑 `python -m pytest forecast_backbones/m6/tests/test_real_backbone.py -q`（本机因无 timesfm3/权重全 skip，
训练机应全绿，含 m5 一致性、torch/evaluator 一致性）；填好 `station_meta.csv`（`station_id` 须与 parquet 一致）；把
`finetune.split` 的 `val_blocks`/`test_start`/`test_end` 改成真实日期（默认值是占位的示例日期，不能直接拿去训练）。

**测试集逐站评测**（输出与 M1–M5 的 `metrics_ens` 同构：`{站}-metrics.xlsx` + `{站}-summary.json`）：

```bash
# 服务器：8 个 worker 各占一张卡，结果默认写 runs/m6/<run_id>/metrics_ens
CKPT_DIR=runs/m6/<run_id> bash eval_shell/eval_m6.sh
# 单站
python eval_shell/m6_station_script.py --checkpoint_dir runs/m6/<run_id> \
    --station_path datasets/test/station=1015.parquet --out_dir runs/m6/<run_id>/metrics_ens --cap_power 465
```

`station_eval.load_trained_model` 从产物目录读 `model_config.yaml`、`model.safetensors` 与校准表，缺任何一个直接报错，
不退回零样本或未校准；返回的模型全部参数 `requires_grad=False`、全部子模块 eval 态（dropout 关闭），推理在 `torch.no_grad()` 下跑；
旧 run 目录里的 `station_routing.json` 不再读，全部站都用 LoRA；旧 run 没有 `model_config.yaml` 时用 `--config`（脚本为 `M6_CONFIG=`）给训练用的 yaml。单站推理与训练期验证同一条路
（`_dataset_with_station_id` → `evaluate_model`），测试文件只有站名列时按 `station_meta` 转整数站号。指标、南网月表与 summary 走
`eval_station_script` 的 `extract_summary` / `NumpyJSONEncoder`，与 `eval_pretrain.py` 相同。**容量只取站表 `cap_mw`，每站一个固定值**
（2026-10-09 起；模型归一化、验证 / 早停的南网分母、逐站评测的南网 / ods 分母都是这一个数；数据里的 `cap_power_on` 是随时间变的开机容量，
一律不用；`--cap_power` 已不使用；站表没有的站直接报错）。之前的 checkpoint 按行 `cap_power_on` 训练，要在新口径下重训。

辅助头（`finetune.loss.aux_nanwang_head`）、阶梯 5 物理协变量消融与消融簇、堆叠 → 计划 3；本计划不读这些键。
