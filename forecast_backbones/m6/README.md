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
    #   REPORT（人读）  runs/m6_baseline/reports/ladder.csv
    #   STATE（机读）   runs/m6_baseline/{calibration.parquet, calibration.parquet.json,
    #                                     decision_state.json, result_cal.parquet, result_val.parquet}
    # 3. 之后正常推理：model_config.yaml 的 calibration.table_path 指向上一步的 calibration.parquet，
    #    registry 用 model_name=m6 构建即可，logits 即提交曲线。

阶梯：raw_median → global_shrink（f*）→ physics_median → calibrated_pstar（短期 p* / 超短期 p*_L1）。
先看 ladder.csv 里 calibrated_pstar 与 raw_median 的差：<0.3 点说明当前 σ 已很小，LoRA 收益有限。
raw_median / global_shrink / physics_median 三档用的是中位数（不受下面第 (b) 条 tail 偏置影响），只有
calibrated_pstar 用 p*。

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
