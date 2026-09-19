# M6 —— TimesFM-3 + 分位校准 + 南网口径决策层

路线 A（本目录当前状态，零样本、无训练）：

    # 1. 填站坐标：forecast_backbones/m6/station_meta.csv（station_id 与 parquet 一致）
    # 2. 回填 + 校准 + 阶梯报表：cal 段不得晚于 val 段起点（否则报时间泄漏错误拒绝）；实际参与拟合的
    #    只是 cal 段末尾 calibration.window_days（默认 45）天的滚动窗口——--cal-start/--cal-end 之间
    #    可以给更长的范围（决定评估/落盘覆盖的区间），拟合窗口会自动收窄到最后 45 天。
    #    parquet 路径含通配符时给引号，避免被 shell 展开：
    python -m forecast_backbones.m6.transformer_trainer \
        --config forecast_backbones/m6/model_config.yaml --parquet 'data/clean/*.parquet' \
        --cal-start 2025-06-01 --cal-end 2025-07-16 --val-start 2025-07-16 --val-end 2025-09-01
    # 产物：
    #   REPORT（人读）  runs/m6_baseline/reports/{ladder.csv, diagnostics.json}
    #     diagnostics.json：calibrator_last_stats（apply 产出的逐 τ 未校准比例等）、degraded_frac、
    #     n_rows_val、n_rows_cal、var_bin_hist、stations_scored、shrink_factor——同样内容也会打印一遍。
    #   STATE（机读）   runs/m6_baseline/{calibration.parquet, calibration.parquet.json,
    #                                     decision_state.json, result_cal.parquet, result_val.parquet}
    # 3. 之后正常推理：model_config.yaml 的 calibration.table_path 指向上一步的 calibration.parquet，
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

`_torch_compat.py` 里还有第二个垫片 `ensure_finite_running_stats()`：timesfm3.torch 某些归一化层在方差恒为 0 的输入上反向传播会算出
`sqrt(0)` 的梯度（NaN），这个垫片只改反向路径，前向数值逐位不变。两个垫片都在 `backbone.py` 导入 `timesfm3.torch` 之前打上，训练机
torch ≥ 2.4 且不遇到零方差输入时其实用不上，留着不影响正确性。

训练（一折）：
```
python -m forecast_backbones.m6.train --config forecast_backbones/m6/model_config.yaml \
    --parquet "/path/to/*.parquet" --out runs/m6_lora_f0
```
`model_config.yaml` 里先把 `finetune.enabled: true`、`finetune.split.*` 改成真实日期；滚动第二折加 `--shift-days N`（全部边界整体后移 N 天），输出目录换 `_f1`。`--max-steps` 只用于冒烟。

产物（`run.out_dir`）：`lora_weights.pt`（val 上南网短期得分最优的 epoch；epoch 0 是零样本基线，所以永远不输零样本）、
`station_routing.json`（逐站 `lora|zeroshot`，输给零样本的站回退；未出现在路由表里的未知站按设计默认 `"lora"`）、
`calibration.parquet`（**在已路由的模型上**重拟合）、`train_log.json`、`result_val_lora.parquet`；报表（`eval.report_dir`）：
`ladder_lora.csv`（阶梯 4：`zeroshot_calibrated_pstar` / `lora_calibrated_pstar` / `routed_calibrated_pstar` 三行）、`station_ladder.csv`
（逐站 `route` 列，取值 `lora|zeroshot`）。

推理：`finetune.enabled: true` 时（`torch_path = bool(finetune.enabled) or inference_path == "torch"`，`enabled=True` 已经短路成
`True`）**永远**走 torch 路径并挂 LoRA，`inference_path` 这个键在 `enabled=True` 时不被读取、不产生任何差异；模型构造期自动加载
`lora_path`（缺失会 warn 并以零样本权重运行）与 `routing_path`（缺失=全站 lora，存在则按表逐站选路）。`inference_path` 只在
`finetune.enabled: false` 时才生效：`evaluator`＝路线 A 原样不变（零样本、无 LoRA、无梯度），`torch`＝同样零样本但走可微的
torch 路径（无 LoRA，仅用于对照 evaluator 与 torch 两条路径本身的数值一致性，见下段 1e-4 门控测试）。
混合站 batch 会跑两次底座按行选路；单一路由的 batch 只跑一次。路线 A 的回填 CLI（`transformer_trainer.backfill_and_report`）
不感知 `finetune`：只要 cfg 里 `finetune.enabled=True` 且 `lora_path` 指向的文件存在，模型构造时就会照常经 torch 路径加载
LoRA 权重后再回填——路线 A 与路线 B 的产物在同一套 `run.out_dir` 约定下天然兼容，不需要额外接线（回归测试见
`tests/test_ladder.py::test_backfill_uses_lora_weights_when_finetune_enabled`）。

训练路径只经 `TimesFM3Torch.decode.__wrapped__` 调底座（`decode` 本身被 `@torch.no_grad()` 装饰，`predict_batch` 再套 `inference_mode`，
都不可训）。`backbone.predict_quantiles` 逐条照搬了 `predict_batch` 的前后处理，四种开关组合与 evaluator 对齐到 1e-4（随机小模型测试）；
真权重上的同一测试在 `test_real_backbone.py::test_torch_path_matches_evaluator_on_real_checkpoint`，有 checkpoint 才跑。

真机待办清单：训练机上跑 `python -m pytest forecast_backbones/m6/tests/test_real_backbone.py -q`（本机因无 timesfm3/权重全 skip，
训练机应全绿，含 m5 一致性、torch/evaluator 一致性）；填好 `station_meta.csv`（`station_id` 须与 parquet 一致）；把
`finetune.split.train_end/val_start/val_end/test_start` 改成真实日期（默认值是占位的示例日期，不能直接拿去训练）。

辅助头（`finetune.loss.aux_nanwang_head`）、阶梯 5 物理协变量消融与消融簇、堆叠 → 计划 3；本计划不读这些键。
