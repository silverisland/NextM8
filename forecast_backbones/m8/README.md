# M8 "PVRex" —— 带 NWP 协变量的混合递推分位预报模型（从零训练，纯 PyTorch）

设计文档：`docs/superpowers/specs/2026-10-08-m8-pvrex-design.md`；调研综述：`docs/survey_covariate_pv_models.md`。

一句话：TiRex 的「未来 patch 当作掩码 token、一次前向吐出全部分位」+ Reverso 的「门控长卷积 + DeltaNet 交替混合器」，
加上 PV 专属的三件事——**未来 NWP 在预报时刻位置直接可见且在输出头侧二次注入**、**CPM 掩码 + 协变量 dropout**、
**容量归一化 + 晴空/太阳高度角通道 + 夜间掩码**。约 1.5M 参数，V100 fp32 单卡即可训，无 Triton / CUDA 扩展依赖。

## 结构

```
每 15 min 一步，T = 672（历史）+ 192（未来）= 864 步 → patch 16 步 → 42 + 12 = 54 个 token
逐步通道 x[B,864,C]：y/cap、y_mask、is_future、各 NWP 协变量（归一）、k_nwp（NWP 晴空指数）、elev/90、p_clear/cap、
                      is_night、时间编码、nwp_resid（历史白天 y − ghi_pred/1000）、cov_missing、[集合成员均值/离差]、
                      [fo:<col> 只有未来段的提示通道（如 TimesFM 预测，历史段恒 0）+ hint_missing]
静态 static[B,S]：log cap、经纬高、degraded、上下文统计（历史峰值、白天均值、白天 k_nwp 均值）；站点嵌入（station_id）
token = ResidualMLP(patch 展平) + static_proj + station_emb + pos_emb
Block ×4（默认 conv → deltanet → conv → deltanet；pre-RMSNorm + GELU FFN）
输出头（step_cov）：每步拼 [token 隐状态, patch 内位置嵌入, 该步原始通道向量] → ResidualMLP → 9 档分位（归一化单位）× scale
决策层：复用 m6.decision（分位排序 → 夜间归零 / 1.05·晴空 / 装机容量钳位 → 中位数或 p*）
损失：pinball（m6.losses），监督步 = 未来段 + CPM 掩码的历史段，掩夜间与缺测
```

混合器（`layers.py`，全部因果）：`conv` 门控长卷积（Reverso；线性零填充卷积而非循环 FFT）、`deltanet` Gated DeltaNet
（delta 规则 = 在上下文里对「NWP 特征 → 实测功率」做在线回归，未来 token 用自己的 NWP 做 query）、`slstm`（按 xLSTM / TiRex
论文公式自写，未拷贝 NXAI 代码）、`attention`（因果多头注意力，对照）。`model.layout: split` 把目标与协变量分成两条 token 流、
每层后做跨流混合（Chronos-2 group attention 的两流简化版）；`model.head_mode: token` 退回 TiRex 式 token 头（消融输出侧注入）。

## 框架契约（与 m6 相同）

- `transformer_models.py`：第一个 nn.Module 子类 `TransformersAdaptedM8Model(model_config)`；`forward(batch_dict, labels=None)`
  → `TransformersAdaptedModelOutput`（`.logits` 提交曲线，`.aux` 与 m6 同 schema：quantiles / quantiles_cal / p_star / median /
  night / elev_deg / p_clear / var_bin / station_id / cap …，可直接喂 m6 的 `recompute_points` / `ladder_report` / `Calibrator`）。
- `transformer_trainer.py`：`evaluate_model(model, ds, collate, cfg)`（result_df 与 m6 同 schema）、`M8_build_training_args`。
- `train.py`：`train_m8(config, splits_dir, out_dir, max_steps, tag)`；`train_script.py --model_name m8` 走独立入口（同 m6）。
- 配置真源 `config/configs.yaml`（= `config.py` 的 DEFAULT_CFG）；变体 `config/variants/*.yaml` 只写与默认不同的字段。
- 站表：`dataset.station_meta`（默认与 m6 共用 `forecast_backbones/m6/station_meta.csv`，训练前填真实坐标；缺坐标站退化为钟表几何）。

## 训练

```bash
# 单个配置（默认全部可见 GPU；模型小，单卡足够）
bash train_m8.sh --splits_dir data/splits/m1_m4_v1 --gpus 0
bash train_m8.sh --splits_dir data/splits/m1_m4_v1 --config forecast_backbones/m8/config/variants/a2_slstm.yaml --gpus 1
# 全部变体批量跑：每张卡一条队列，跑完汇总 variants_summary.csv
bash train_m8_variants.sh --splits_dir data/splits/m1_m4_v1 --gpus 0,1,2,3,4,5,6,7
bash train_m8_variants.sh --splits_dir ... --variants a1_hybrid,a2_slstm,a3_attention,b4_no_nwp --gpus 0,1
bash train_m8_variants.sh --splits_dir ... --max_steps 50 --gpus 0 --batch_tag smoke       # 先冒烟
# 直接用 python
python -m forecast_backbones.m8.train --config forecast_backbones/m8/config/configs.yaml --splits-dir data/splits/m1_m4_v1 --out runs/m8/x
```

切分目录用 `pipelines.time_splits` 落的 `train/valid.parquet + split_manifest.json`（与 M1–M4 / M6 同一份；不需要 cal.parquet）。
每 `train.schedule.eval_every_steps` 步在验证集打一次分（`nanwang_after_pstar`：每站每天 10:00 前最晚一行 → 南网短期 D+1；
或 `nrmse`：整点起报行 → 192 步容量归一 RMSE%），`patience` 按验证次数计，最优权重存 `model.safetensors`（完整 state_dict）。

产物（`run.out_dir`）：`model.safetensors`、`model_config.yaml`、`train_log.json`（history 每条一次验证）、`result_val.parquet`
（默认整点起报行）、`reports/summary.csv`（一行：tag / 南网 D+1 / nrmse / best_step / 参数量 / 结构字段）、`reports/station_table.csv`。
多个 run 汇总：`python -m forecast_backbones.m8.collect_runs runs/m8/<batch_tag> --out x.csv`。

## 变体矩阵（`config/variants/`，批量测试用）

| 组 | 变体 | 回答的问题 |
|---|---|---|
| A 骨干 | a1_hybrid（基线）/ a2_slstm / a3_attention / a4_deltanet_only / a5_conv_only / a6_no_mixer | 混合器选型；a6 是「没有序列建模」的下界 |
| B 协变量通路 | b1_split_stream / b2_bidirectional / b3_token_head / b4_no_nwp / b5_no_station_emb | 双流 vs 单流；预报时刻之后的 NWP 是否有用；输出侧注入值多少；NWP 总共值多少；站点嵌入对已见 / 未见站的影响 |
| C 配方 | c1_no_cpm / c2_recent_peak / c3_patch32 / c4_patch8 / c5_large / c6_nrmse_stop / c7_station_day | CPM+dropout 的价值；归一化；token 粒度；规模；早停口径；采样 |
| D 数据/决策 | d1_members（需 SSRD_pos_* 列）/ d2_pstar_point / d3_timesfm_hint / d4_timesfm_resid（需 timesfm_predict 列） | 集合成员；p* vs 中位数；TimesFM 单变量预测作为未来提示通道；在其上做残差学习 |

读法：先看 a1 vs b4（协变量价值）与 a1 vs c1（配方价值）——调研表明这两项通常远大于骨干差异（2–5%）；骨干之间差异小于
种子噪声时不要下结论，换 `run.seeds` 多跑几颗。强烈建议补一个 LightGBM 逐步特征基线（竞赛与 15 min PVDAQ 上仍是最强单模型）。

## 测试集评估与集成（`evaluate.py` / `eval_m8.sh`）

训练脚本只在验证集上打分；测试集要单独跑：

```bash
bash eval_m8.sh --runs runs/m8/round2a/a1_hybrid_s0 --test datasets/test                       # 单个产物
bash eval_m8.sh --runs runs/m8/final/e1_final_s0,runs/m8/final/e1_final_s1,runs/m8/final/e1_final_s2 \
     --test "datasets/test/*.parquet" --out runs/m8/final/ens_eval                              # 三种子集成
bash eval_m8.sh --runs runs/m8/round2a/a1_hybrid_s0 --test datasets/test \
     --decision_yaml forecast_backbones/m8/config/variants/d2_pstar_point.yaml                   # 换决策规则，不重训
```

- 多个 run 目录 = 集成：各成员的原始分位（真实单位）逐分位平均后再过一次决策层；`reports/members.csv` 另报每个成员单独的分数，
  看集成收益。同一配置三颗种子的集成通常稳拿 0.3–0.5 分。
- 决策层没有可训练参数，所以 `--point median|p_star|p_star_l1`（清空 by_step）或 `--decision_yaml`（合并某变体的 decision 段）
  可以在评估时换规则。挑主干看 nrmse（不受决策规则影响），挑决策规则看南网分。
- seen / unseen 按第一个成员 `model_config.yaml` 里的 `known_stations` 判定；测试集里训练没见过的站走未知站槽。
- 产物：`result_test.parquet`（与 `result_val.parquet` 同 schema，多成员时多 `predict_m<k>` 列）、`reports/summary.csv`、
  `reports/station_table.csv`、`reports/decision.json`。

第二轮（2026-10-09，m8_v1 切分，三块验证 + 四个整站留出）的结论：所有架构变体（sLSTM、注意力、token 头、双流、大模型、
recent_peak 归一、按站日采样）都在 a1 ±0.6 的种子噪声内；c8 调度略好无副作用；p* 决策高约 1 分（主要来自未见站）；
站点嵌入对已见站 +0.5、对未见站 −0.7；未见站比已见站低 5–6 分且与嵌入无关。最终组合见 `config/variants/e1_final.yaml`。

## 本地测试

`python -m pytest forecast_backbones/m8/tests -q`（CPU，约 15 s；含层因果性、CPM/夜间掩码语义、registry 发现、九种结构变体的
前向/反向、`compute_loss` 两条路径一致、权重读写、`evaluate_model` schema、两种早停指标的端到端训练）。合成数据 400 步 CPU 训练：
有 NWP 时南网 D+1 从 82 升到 87.0、nrmse 6.7%；不喂 NWP 时 80.2、nrmse 10.2%，说明协变量通路在起作用（合成数据，不代表真实水平）。

## 没见过的站（零样本新站）

训练入口把训练集里出现过的站号写进 `model_config.yaml` 的 `dataset.known_stations`；推理时名单外的站走"未知站"槽位 0。
训练时 `train.slot_dropout`（默认 0.3）按概率把站点嵌入换成未知槽，所以槽 0 是训练过的，模型学会只靠静态量
（容量、坐标、上下文统计）和历史窗口给新站预报。报表里 `station_table.csv` 多一列 `seen`，`summary.csv` 多
`nanwang_seen` / `nanwang_unseen` / `n_unseen_stations`，`train_log.json` 每次验证也记 `nanwang_unseen`；早停仍按总分。
要让早停和变体比较也看到零样本表现，切分时用 `val_only_stations` 整站留出一两个短站（见 `pipelines/time_splits.py`），
它们的行全部进 valid、不进训练。对照变体 b5_no_station_emb 关掉站点嵌入。

## 外部提示通道（`features.future_only`）与残差学习

`future_only: [{col: timesfm_predict, scale: target}]` 把一列 **只有未来 192 步** 的数组（每行在起报时刻生成，如 TimesFM-3 的单变量
预测）当协变量：历史段恒 0（CPM 掩掉的历史不会从这里漏回来），未来段除以每行目标尺度并随幅度调制同乘；`train.hint_dropout`
以独立概率整组清零并把 `hint_missing` 置 1，防止模型只抄提示而不学 NWP。`residual_base_col: timesfm_predict` 再加一条从提示到
输出的恒等跳连（输出 = 提示 + 网络输出；提示被 dropout 时基准同步为 0）。训练行与推理行的提示必须用同一套方法生成
（同一模型版本、上下文长度、取点方式），否则分布不一致。缺列时只警告一次，通道置 0、`hint_missing=1`。

## 注意

- 训练 / 验证必须用起报时刻可得的 **业务 NWP**（不能用再分析替代），否则结论会高估协变量价值。
- 评测口径：南网 D+1 只看 10:00 起报行的 `predict[55:151]`；`nrmse` 含夜间，数值会被夜间零值压低；报告时按站与汇总分开看。
- TiRex 代码与权重受 NXAI Community License 约束（年营收 >1 亿欧元需商业授权）；这里的 sLSTM 按论文公式自行实现。Reverso 为 MIT。
