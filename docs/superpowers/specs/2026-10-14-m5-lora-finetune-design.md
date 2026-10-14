# M5 重构：TimesFM-3 + LoRA 微调（自包含，M8 训练框架）—— 设计文档

日期：2026-10-14　状态：用户已审阅（2026-10-14 补 D8：站点几何默认关）

## 0. 一句话

把 M6 里的 LoRA 微调搬进 M5，删掉 M5 原来的 conformal；新 M5 复用 M8 的训练 / 评估框架（EMA、smooth_k、refit、变体目录、
集成评估、`--full` 四指标报表），模型类换成 TimesFM-3 可微路径 + LoRA。线上正在用的零样本类保持同名、同导入路径、同行为。
目的：先在 `m8_v1` 切分上拿到"零样本 vs LoRA"的基线，与 M8 round 6 同口径可比；之后在这个模块上批量测试 PEFT 变体。
基线只差权重：输入与线上 M5 相同（只有 ghi / 温度），决策层不用站点几何，`gain` 就是纯微调收益。

## 1. 目标与成功标准

- 基线：`z0_zeroshot`（1 次，确定性）与 `l1_lora_r8`（×3 seeds）在 `m8_v1` 验证集上的 D+1 / seen / unseen / nrmse / ultra_proxy，
  与 M8 e1_final（75.01 / 76.32 / 70.91 / 11.13 / 77.72）放在同一张表。
- 线上兼容：`nanwang_online/multi_infer.py` 用的配置（无 `dataset.station_meta`）喂进新类，logits / aux 与今天的 M5 逐位相同。
- `eval_shell/eval_pretrain.py` 按 `forecast_backbones/m5/config/configs.yaml` 建出的零样本 M5 行为不变。
- 自包含：`forecast_backbones/m5` 不 import `forecast_backbones.m6` / `forecast_backbones.m8`（测试守住）。
- 成功不预设分数：LoRA 是否优于零样本由服务器结果决定，本文档不下结论。

## 2. 约束与决定（对话中已确认）

| # | 决定 | 理由 |
|---|---|---|
| D1 | 一个模型类 `TransformersAdaptedM5Model`（别名 `TransformersAdaptedTimesFM3Model`），零样本是默认路径 | 线上与 experiments 直接 import 该类；上线微调版只需换配置 + 权重 |
| D2 | 本轮不动 M6；M5 按 M8 方式复制成自包含 | 范围最小；M6 待基线跑通后再决定去留 |
| D3 | 骨架 = M8 框架 + TimesFM-3/LoRA 模型类 | 验证指标、协议、变体机制与 M8 同一份代码，基线直接可比 |
| D4 | 损失在容量归一单位里算（q/cap, y/cap）；`train.loss.normalize_by_cap: false` 退回 M6 真实单位 | 站容量相差 ~60 倍，真实单位下大站主导梯度；M8 也是归一单位 |
| D5 | `config/configs.yaml` 保持零样本原样；训练真源 `config/finetune.yaml` | eval_pretrain 按 configs.yaml 建 M5 做逐站零样本评测，不能悄悄变 |
| D6 | 决策层默认 median；p* 用 `evaluate.py --decision-yaml` 覆盖，不重训 | 与 M8 一致；决策层无可训参数 |
| D7 | 第 0 步（LoRA B=0 = 零样本）打分写进 `train_log.json` 与 summary 的 `nanwang_zeroshot` | 每个 run 自带零样本对照，`gain = best − zeroshot` |
| D8 | 站点几何默认全关：`features.derived` 两项 false、`decision.physics.clear_sky_cap_ratio: null`（只做容量钳位 + 负值截零）、night_zero false、night_weight 1 | 线上 M5 从不用几何；几何协变量与晴空上限是 M8 / M6 给从零训练的小模型的先验，且在 M8 里踩过水平面 GHI 裁峰（1027）、缺坐标钟表几何（1026）、晴空上限隐式夜间归零（86a9c8c 前）的坑。站表只用来取 `cap_mw` 与站号；几何留作变体 g1 / g2 |

## 3. 目录

```
forecast_backbones/m5/
  transformer_models.py   TransformersAdaptedM5Model（第一个 nn.Module 子类）+ 别名 TransformersAdaptedTimesFM3Model（实现取 M5 为主名：unified_inference / eval_pretrain 用 `type(m).__name__` 生成列名，必须与旧输出一致）
                          + save_weights / load_weights / load_station_meta
  backbone.py  lora.py  _torch_compat.py        ← 自 M6 复制（timesfm3 可微路径、LoRA、垫片）
  rowctx.py                                     ← 自 M6 features.py：RowContext / build_row_context / 变率信号
  （masks.py 有意不建：_loss 内用 isfinite 屏蔽非有限标签、内联施加 night_weight，M5 其余处不需要 M6 的 masks.py）
  config.py  train.py  transformer_trainer.py  data.py  rows.py  solar.py  decision.py  losses.py
  evaluate.py  station_report.py  collect_runs.py  _dist.py  _arrays.py  station_meta.csv   ← 自 M8 复制并改名
  config/configs.yaml      不动（线上 / eval_pretrain 的零样本配置）
  config/finetune.yaml     新：DEFAULT_CFG 的落盘
  config/variants/         z0_zeroshot.yaml / l1_lora_r8.yaml / d2_pstar_point.yaml
  tests/
删除：m5/conformal.py、m5/conformal_replay.py、m5/tests/test_conformal.py、eval_shell/conformal_m5.sh
新增：train_shell/train_m5.sh、train_shell/train_m5_variants.sh、eval_shell/eval_m5.sh（由 M8 同名脚本改名适配）
修改：train_shell/train_script.py 加 m5 分支（m6 分支不动）
```

`station_meta.csv` 从 M8 复制（M8 与 M6 两份内容相同）；从此三份各自维护，改站表要三处一起改（README 写明）。

## 4. 模型类

### 4.1 两种模式（按 `dataset.station_meta` 是否给出切换）

- **legacy**（`station_meta` 缺省或为空，线上配置）：forward 逐字沿用当前 M5 代码——`predict_batch` → logits = 点预测或
  `timesfm3.predict_quantile` 档位，`aux = {"quantiles": [B,H,9]}`（无分位时 `aux=None`），不算 loss，无决策层。
  legacy 下 `finetune.enabled: true` 直接 `ValueError`（微调需要站表取容量 / 站号）。
- **full**（给了站表）：
  1. `build_row_context` → cap（站表 `cap_mw`）/ station_id，以及 elev / p_clear / night / degraded（默认配置下只进 aux 供报表，
     不影响预测；站表 lat / lon 可为 null）；
  2. 协变量 = `features.past_future` 配对 + `past_only`；`features.derived`（`solar_elevation`、`clear_sky_power`）默认关，
     开了才追加为 past_future 通道（变体 g1）；
  3. 底座：`finetune.enabled` → torch 路径挂 LoRA + 解冻 output_head；否则按 `finetune.inference_path`（evaluator | torch）零样本；
  4. 决策层 = M8 `decide()`：单调化 → 钳位 → median 或 p*。`clear_sky_cap_ratio: null`（默认）时上界只用 cap，不看 p_clear，
     夜间也不被隐式归零；给数值（变体 g2）时与 M8 相同：min(ratio·p_clear, cap)，night_ceiling 跟随 night_zero；
  5. aux 与 M8 同 schema（quantiles / quantiles_cal / levels_cal / p_star / median / night / night_loss / elev_deg / p_clear /
     var_bin / var_signal / station_id / cap / degraded），`evaluate_model` / `score_nanwang` / `station_report` 原样可用。
- `predict_raw(batch_dict)`：full 模式下返回原始分位（真实单位）+ ctx，供 `evaluate.py` 集成先平均再决策。

### 4.2 损失

pinball 9 档，监督未来 192 步；缺测（非有限真值）掩掉；夜间步按 `train.loss.night_weight` 加权（默认 1，与 M8 同口径）。
`train.loss.normalize_by_cap`（默认 true）：q 与 y 都除以该行 cap 后再算。

### 4.3 框架契约

- `config` 属性返回合并后的 cfg；`compute_metrics` 返回 `{}`。
- `set_known_stations(ids)`：只写进 `cfg["dataset"]["known_stations"]`（seen / unseen 报表用），模型本身不用。
- `reseed(seed)`：空操作（无随机增强）；保留以满足 train.py 调用。
- `n_params()`：可训参数数；`n_total_params()`：全部参数数。
- 权重：`save_weights` / `load_weights` 只存取 LoRA + output_head（`lora.lora_state_dict` / `load_lora_state_dict`，safetensors，
  键名相对 `backbone.model`）。构造时 `train.weights_path` 存在即加载，`weights_loaded=True`。
- timesfm3 仍懒加载（`timesfm3.eager_init`）；legacy 路径不 import `timesfm3.torch`。

## 5. 配置（`config.py`）

```yaml
run:      {model_name: m5, seeds: [0], out_dir: runs/m5_baseline}
dataset:  {pv_col, label_col, ts_col, station_id_col, station_meta: forecast_backbones/m5/station_meta.csv,
           tz, step_min: 15, known_stations: null, context_len: 672, pred_len: 192}
timesfm3: {checkpoint: /data/tangqingang/timesfm-3.0-pytorch, device: auto, per_core_batch_size: 64, nan_policy: interp,
           make_positive: true, use_znorm: false, use_symmetric_averaging: true, eager_init: true,
           return_quantiles: true, predict_quantile: null}        # 后两项只在 legacy 模式生效
features: {past_future: [[general_ghi, general_ghi_predict], [general_temp, general_temp_predict]],
           past_only: [], derived: {solar_elevation: false, clear_sky_power: false}}   # D8：几何默认关
finetune: {enabled: true, inference_path: torch, method: lora,
           lora: {rank: 8, alpha: 16, dropout: 0.05, target_modules: [query_proj, key_proj, value_proj, out_proj]},
           unfreeze_output_head: true}
train:    {schedule: {lr: 1.0e-4, weight_decay: 0.01, batch_size: 16, max_steps: 4000, epochs_max: 10, warmup_steps: 0,
                      grad_clip: 1.0, precision: fp32, logging_steps: 20, eval_every_steps: 250, sampling: random,
                      progress_bar: true, num_workers: 0},
           loss: {night_weight: 1.0, normalize_by_cap: true},
           early_stop: {metric: nanwang_after_pstar, patience: 5, nrmse_issue_every_min: 60, smooth_k: 1},
           ema_decay: null, weights_path: ${run.out_dir}/model.safetensors}
decision: {physics: {night_zero: false, clear_sky_cap_ratio: null, capacity_cap: true, clip_negative: true},   # null = 无晴空上限
           point: {default: median, by_step: [], floor_ratio: 0.2, tau_grid_step: 0.005, tail: flat, shrink_factor: null}}
eval:     {batch_size: 64, report_dir: ${run.out_dir}/reports}
```

- `load_config(src)`：`src` 为 None 或缺 `dataset.station_meta` 的 dict → 只与 legacy 默认（今天的 M5 DEFAULT_CFG）合并，
  不注入站表，保证落到 legacy 模式；给了 `station_meta` → 与上面的 DEFAULT_CFG 深合并。
- `method` 目前只接受 `lora`，其它值报错；它是后续 PEFT 变体的开关位。
- `REMOVED_KEYS`：`calibration`、`stacking`、`variability`、`data`、`finetune.lora_path`、`finetune.loss`、`finetune.masks`、
  `finetune.schedule`、`finetune.split`、`finetune.early_stop`、`features.nwp_members`、`features.target_transform`
  → 出现即报错，并提示新位置（如 `finetune.schedule → train.schedule`）。
- 变体：`z0_zeroshot.yaml`（`finetune.enabled: false`、`inference_path: torch`，与 LoRA 同一评估链）、
  `l1_lora_r8.yaml`（空覆盖 = 默认，基线）、`d2_pstar_point.yaml`（`decision.point.default: p_star`，评估时覆盖用）、
  `g1_solar_cov.yaml`（`features.derived` 两项 true，需重训）、`g2_cs_ceiling.yaml`（`decision.physics.clear_sky_cap_ratio: 1.05`，
  只动决策层，可用 `--decision-yaml` 覆盖已训好的 run；与 M8 同口径比较时用）。g1 / g2 不进本轮基线。

## 6. 训练 / 评估（自 M8 搬，改动如下）

1. `WeightEma` 只影子 `requires_grad` 的参数（底座不复制）；`applied` / `load_into` 只换这些参数。
2. `EvalCallback.on_train_begin` 在第 0 步打一次分并计入 history（`step: 0`），作为初始最优；patience 从它起算。
   零样本 run（`finetune.enabled: false`）不训练：train.py 只做第 0 步打分 + 报表，`best_step = 0`，不写权重文件。
3. `_summary_row` 模型列：`method / rank / alpha / target_modules / unfreeze_head / n_trainable / lr / nanwang_zeroshot / gain`
   （其余列与 M8 相同：D+1、nrmse、ultra_proxy、seen、unseen、best_step、n_evals、train_minutes、ema_decay、smooth_k、行数）。
   `collect_runs` 显示列同步。
4. `evaluate.load_member`：run 目录缺 `model.safetensors` 且 `model_config.yaml` 里 `finetune.enabled: false` → 按零样本成员加载；
   `finetune.enabled: true` 却缺权重 → 报错。
5. `--refit`、seed、smooth_k、集成分位平均、`station_report --full` 原样。
6. `build_training_args`：precision 默认 fp32（服务器 V100 不支持 bf16，HF Trainer 开 bf16 会直接报错）；`fp16` / `bf16` 可选，
   cpu 一律 fp32；`_get_train_sampler` 双签名兼容（沿用）。

## 7. 基线运行方案（服务器）

```bash
bash train_shell/train_m5_variants.sh --splits_dir <m8_v1 切分目录> --variants z0_zeroshot --gpus 0 --batch_tag m5_base
bash train_shell/train_m5_variants.sh --splits_dir <m8_v1 切分目录> --variants l1_lora_r8 --seeds 0,1,2 --gpus 1,2,3 --batch_tag m5_base
python -m forecast_backbones.m5.evaluate --runs runs/m5/m5_base/l1_lora_r8_s0 --test <m8_v1>/valid.parquet \
       --decision-yaml forecast_backbones/m5/config/variants/d2_pstar_point.yaml      # p* 数，不重训
```

上面两条可并发运行并共用同一个 `--batch_tag`：`train_m5_variants.sh` 的暂存目录按调用隔离（`.runs_$$`），先结束的调用只清自己的暂存。

测试集 `eval_m5.sh --full` 等基线确定后再做 refit。

## 8. 测试

- 自 M6 搬（改导入路径）：test_lora、test_backbone_torch、test_torch_compat；路线 B 模型测试改为 full 模式。
- 自 M8 搬：test_decision 相关、losses、test_evaluate、test_station_report、test_train_smoke（CPU，`TorchBackbone.tiny_for_tests`）、
  test_train_protocol（EMA 只影子可训参数、smooth_k）、test_standalone（不引用 m6 / m8）。
- 新增 `test_legacy_contract.py`：把当前 M5 forward 冻结为 `tests/legacy_m5_forward.py`，用 `_FakeBackbone` 对比新类 legacy 模式
  logits / aux 逐位相同；线上式配置（无 station_meta）必须走 legacy；registry `import_model_class("m5")` / `build_evaluator("m5")` 仍可用；
  legacy + `finetune.enabled: true` 报错。
- 新增：第 0 步打分进 history；零样本 run 不写权重；`normalize_by_cap` 两种取值的 loss 关系；`REMOVED_KEYS` 报错信息；
  默认配置下协变量恰为两对 past_future（无几何通道）、`clear_sky_cap_ratio: null` 时 p_clear=0 的夜间步不被钳成 0、
  上界只等于 cap；g1 / g2 打开后分别多出两个通道 / 恢复 M8 钳位结果。
- 删除 test_conformal。现有 test_transformer_trainer 保留（legacy 路径的 evaluate_model 结果表）。

## 9. 不做的事

- M6、`eval_pretrain.py`、线上 release 配置、experiments/nwp_covariate 不动（它们只依赖类名与 legacy 行为）。
- LoRA 以外的 PEFT、conformal、stacking、逐站 LoRA 都不做；PEFT 调研在基线之后。
- 不处理工作树里 `train_shell/` `eval_shell/` 移动等用户未提交改动；新文件直接放新位置，只提交本次相关文件。

## 10. 风险

- **legacy 逐位一致**：靠冻结旧 forward 的对照测试守住；新类 legacy 分支直接复用旧代码而非重写。
- **显存**：TimesFM-3 ~200M 参数，bs 16、fp32、对称平均（实际 2B 行）在 V100 32G 上需实测；OOM 时把 `train.schedule.batch_size` 降到 8。
- **训练速度**：每步过一遍 200M 底座；4000 步 × eval 250 步的耗时以服务器实测为准。
- **框架复制**：M5 / M8 各一份框架，改了不自动同步（与 M8 / M6 现状一致）。
