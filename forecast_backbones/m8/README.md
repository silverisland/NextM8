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
| E 组合 | e1_final | a1 + c8 调度 + d2 决策（第二轮结论的组合） |
| F 站点身份 | f1_row_local / f2_no_coords / f3_row_local_final | 去掉能标识站点的输入（坐标、log 容量、站点嵌入）后已见 / 未见站差距是否收窄；f3 = f1 + c8 + d2 |

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

**逐站完整评测**（`--full`，`station_report.py`）：全部起报行推理，产出与 M1–M7 的 `metrics_ens` 同构的
`{站}-metrics.xlsx`（ods_ultra_short / ods_short / ultra_short / short / 48hours / nanwang 六张 sheet）和 `{站}-summary.json`，
外加 `reports/station_metrics.csv`（逐站 ods_short_rmse_acc、ods_ultra_short_rmse_acc、nanwang_short_D+1、nanwang_ultra_short）
与 `reports/summary.csv`（跨站均值、seen/unseen）：

```bash
bash eval_m8.sh --runs runs/m8/refit/e1_final_s0,runs/m8/refit/e1_final_s1,runs/m8/refit/e1_final_s2 \
     --test "datasets/test/*.parquet" --out runs/m8/refit/full_eval --full
```

- 推理默认走线上接口 `pipelines.pipeline_wrapper_inference.unified_inference`（M8 的 `forward(batch_dict)` 与 m6 契约相同，
  `.logits` 就是决策后的提交曲线），每个成员各跑一遍、多成员取点预测平均；`--engine native` 走仓库内
  `evaluate.run_inference`（分位平均后再决策）。单成员时两者结果逐位相同（测试里对过）。
- `summary.json` 的键：仓库根有 `eval_station_script.py`（服务器）时用它的 `extract_summary`，与其它模型逐键一致；
  没有时用回退实现（`<sheet>_<指标>`）。`station_metrics.csv` 的四个主指标直接从表里取，不受键名影响。
- 超短期两项要测试文件里每 15 分钟一行起报（16 个连续 lead）；ods_short 要 09:00 行，nanwang_short_D+1 要 10:00 行。

**测试集诊断**（`diagnose_test.py`，只输出聚合统计，不含原始行，可以直接贴出来）：

```bash
python -m forecast_backbones.m8.diagnose_test --result runs/m8/refit/ens_eval/result_test.parquet \
    --other <M4 的逐站结果.parquet> --test "datasets/test/*.parquet" --station 白马湖光伏电站
```

逐站一行：南网 D+1（提交曲线 / median / p* / 整体乘尺度后）、bias_ratio（预测总量/观测总量）、peak_over_cap、
nrmse、日分数 <40 的占比、与 `--other` 模型在共同起报日上的对比；再加输入侧统计（起报节奏、10:00 行数、
cap_power_on 取值、NWP 未来列缺失率、历史功率缺失/全零率）和指定站的逐小时平均曲线、逐日分数。

第二轮（2026-10-09，m8_v1 切分，三块验证 + 四个整站留出）的结论：所有架构变体（sLSTM、注意力、token 头、双流、大模型、
recent_peak 归一、按站日采样）都在 a1 ±0.6 的种子噪声内；c8 调度略好无副作用；p* 决策高约 1 分（主要来自未见站）；
站点嵌入对已见站 +0.5、对未见站 −0.7；未见站比已见站低 5–6 分且与嵌入无关。最终组合见 `config/variants/e1_final.yaml`。

第三轮（站点身份诊断，2 种子）：去掉全部身份输入（f1：坐标、log 容量、站点嵌入）已见站 −0.3、未见站 −1.0、nrmse +0.5，已见 / 未见差距没有收窄——差距来自留出站本身（冬季块、历史只有几周），不是按站记忆；只去坐标（f2）与 a1 逐位一致，坐标在静态向量里不起作用；f3 = f1 + c8 + p* 得 75.7，等于 d2 + c8 的预期，参数少 1M。e1_final 与 f3 谁做最终模型由测试集集成分数定。

## 全量重训（refit，最终模型）

验证切分是用来选配方和步数的；选定之后最终模型按两段式做：**用全部数据重训**，再在测试集上只评一次。
m8_v1 切分为了量零样本，整站留出了 1006/1016/1022/泗洪，长站还剔掉三个验证块加隔离带；这些站在测试集里是未见站、
会按验证集的差距（p* 决策下约 −3.7 分）压低测试分。全量重训把它们变回已见站、把数据全用上，白马湖仍是唯一的零样本站。

```bash
# 1. 落全量切分：没有验证块、不留出站，每站只剔测试起点前 2 天；valid.parquet 为 0 行
python -m pipelines.time_splits --parquet "<训练文件夹>/*.parquet" --test-parquets "<测试文件夹>/*.parquet" \
    --split-yaml data/splits/m8_refit.yaml --out-dir data/splits/m8_refit
# 2. 用选定的配方训 3 颗种子：--refit 不验证、不早停，固定 --max_steps 步后存最终权重（cosine 在这一步退火到 0）
bash train_m8_variants.sh --splits_dir data/splits/m8_refit --variants e1_final --seeds 0,1,2 --refit --max_steps 5000 --batch_tag refit
# 3. 测试集集成评估（唯一一次）
bash eval_m8.sh --runs runs/m8/refit/e1_final_s0,runs/m8/refit/e1_final_s1,runs/m8/refit/e1_final_s2 --test "<测试文件夹>/*.parquet"
```

- 步数：取该配方在验证切分上各种子 `best_step` 的上限附近（c8 调度 2500–5500 → 5000）；三种子集成本身也抵消单一步数的运气。
- `--refit` 也能直接用在 m8_v1 的切分目录上（train ∪ valid 合并训练），只是拿不回验证块前后的隔离带（约 5% 的行）。
- 产物与普通 run 同构（`model.safetensors` / `model_config.yaml` / `reports/summary.csv`），只是 summary 的验证分数列是 NaN、
  `n_val_rows` 为 0；`known_stations` 含全部训练站，所以 1006/1016/1022 在测试里按已见站判定。

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
