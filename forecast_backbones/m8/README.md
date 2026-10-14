# M8 "PVRex" —— 带 NWP 协变量的混合递推分位预报模型（从零训练，纯 PyTorch）

设计文档：`docs/superpowers/specs/2026-10-08-m8-pvrex-design.md`；调研综述：`docs/survey_covariate_pv_models.md`。

一句话：TiRex 的「未来 patch 当作掩码 token、一次前向吐出全部分位」+ Reverso 的「门控长卷积 + DeltaNet 交替混合器」，
加上 PV 专属的三件事——**未来 NWP 在预报时刻位置直接可见且在输出头侧二次注入**、**CPM 掩码 + 协变量 dropout**、
**容量归一化 + 晴空/太阳高度角通道 + 夜间掩码**。约 1.5M 参数，V100 fp32 单卡即可训，无 Triton / CUDA 扩展依赖。

## 结构

```
每 15 min 一步，T = 672（历史）+ 192（未来）= 864 步 → patch 16 步 → 42 + 12 = 54 个 token
逐步通道 x[B,864,C]：y/cap、y_mask、is_future、各 NWP 协变量（归一）、k_nwp（NWP 晴空指数）、elev/90、p_clear/cap、
                      is_night、时间编码、nwp_resid（历史白天 y − ghi_pred/1000）、cov_missing、
                      [fo:<col> 只有未来段的提示通道（如 TimesFM 预测，历史段恒 0）+ hint_missing]
静态 static[B,S]：log cap、经纬高、degraded、上下文统计（历史峰值、白天均值、白天 k_nwp 均值）；站点嵌入（station_id）
token = ResidualMLP(patch 展平) + static_proj + station_emb + pos_emb
Block ×4（默认 conv → deltanet → conv → deltanet；pre-RMSNorm + GELU FFN）
输出头（step_cov）：每步拼 [token 隐状态, patch 内位置嵌入, 该步原始通道向量] → ResidualMLP → 9 档分位（归一化单位）× scale
决策层：decision.py（分位排序 → 1.05·晴空 / 装机容量钳位 → 中位数或 p*；夜间归零 night_zero 默认关，见下）
损失：pinball（losses.py），监督步 = 未来段 + CPM 掩码的历史段，掩缺测；夜间步按 train.loss.night_weight（默认 1）参与
```

混合器（`layers.py`，全部因果）：`conv` 门控长卷积（Reverso；线性零填充卷积而非循环 FFT）、`deltanet` Gated DeltaNet
（delta 规则 = 在上下文里对「NWP 特征 → 实测功率」做在线回归，未来 token 用自己的 NWP 做 query）、`attention`（因果多头注意力，
只作对照）。sLSTM、双向混合器、split 双流布局、token 头已于 2026-10-14 瘦身删除（见文末「已删除的分支」）。

## 框架契约（与 m6 相同）

- `transformer_models.py`：第一个 nn.Module 子类 `TransformersAdaptedM8Model(model_config)`；`forward(batch_dict, labels=None)`
  → `TransformersAdaptedModelOutput`（`.logits` 提交曲线，`.aux` 与 m6 同 schema：quantiles / quantiles_cal / p_star / median /
  night / elev_deg / p_clear / var_bin / station_id / cap …，可直接喂 m6 的 `recompute_points` / `ladder_report` / `Calibrator`）。
- `transformer_trainer.py`：`evaluate_model(model, ds, collate, cfg)`（result_df 与 m6 同 schema）、`M8_build_training_args`。
- `train.py`：`train_m8(config, splits_dir, out_dir, max_steps, tag)`；`train_script.py --model_name m8` 走独立入口（同 m6）。
- 配置真源 `config/configs.yaml`（= `config.py` 的 DEFAULT_CFG）；变体 `config/variants/*.yaml` 只写与默认不同的字段。
- 站表：`dataset.station_meta`（默认 `forecast_backbones/m8/station_meta.csv`，训练前填真实坐标；缺坐标站退化为钟表几何或历史日形包络）。
- **自包含（2026-10-13 起）**：m8 目录不再引用 `forecast_backbones/m6` 的任何代码或数据。原来从 m6 拿的东西都在本目录自带一份：
  `solar.py`（站表 StationMeta / 太阳几何 / 钟表几何 / 站名映射 / 固定容量）、`decision.py`（决策层 + quantile_cols）、`losses.py`（pinball）、
  `rows.py`（打分行 / 整点行 / 站日采样）、`data.py`（切分目录 → UnifiedDataset）、`_dist.py`、`_arrays.py`、`station_meta.csv`。
  m6 那边改了不会自动同步到这里（站表两份要一起改）。所有模型共享的 `pipelines/`、`interfaces/` 仍然引用，不复制。
  `tests/test_standalone.py` 守住这条：m8 源码里出现 m6 引用、或 import m8 时加载了 m6 模块，测试即失败。
- **容量只来自站表 `cap_mw`，每站一个固定值**（2026-10-09 起）：模型归一化、晴空钳位、南网 / ods 指标的分母都是这一个数；
  数据里的 `cap_power_on` 是随时间变的开机容量（白马湖并网首月 8.88 → 68），一律不用。站表没有的站直接报错。
  之前的 checkpoint 按行 cap 训练，最终模型要在这个口径下重训（refit）。

## 训练

```bash
# 单个配置（默认全部可见 GPU；模型小，单卡足够）
bash train_m8.sh --splits_dir data/splits/m1_m4_v1 --gpus 0
bash train_m8.sh --splits_dir data/splits/m1_m4_v1 --config forecast_backbones/m8/config/variants/a3_attention.yaml --gpus 1
# 全部变体批量跑：每张卡一条队列，跑完汇总 variants_summary.csv
bash train_m8_variants.sh --splits_dir data/splits/m1_m4_v1 --gpus 0,1,2,3,4,5,6,7
bash train_m8_variants.sh --splits_dir ... --variants e1_final,h4_envelope,d3_timesfm_hint,j1_final2 --gpus 0,1
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

| 组 | 变体 | 用途 |
|---|---|---|
| 对照 | a1_hybrid（原始基线）/ a3_attention / a6_no_mixer | 默认骨干；换成注意力；没有序列混合器的下界 |
| 决策 | d2_pstar_point | 只改决策层（p* 分段选点），给 `--decision_yaml` 用，不用重训 |
| 外部提示 | d3_timesfm_hint / d4_timesfm_resid（需 timesfm3 列） | TimesFM-3 预测作未来提示通道；在其上做残差学习（超短期代理 +0.65） |
| 配方 | e1_final / e1_night_old | 第二轮定的配方（a1 + 退火调度 + p* 决策）；同配方的旧夜间口径（A/B 用） |
| 物理先验 | h1_cs_ratio / h4_envelope | 按站晴空出力比；h1 + 缺坐标站历史日形包络 |
| 最终 | j1_final2（需 timesfm3 列） | e1_final + h4 + d3。round6：D+1 75.75（e1 75.01），可加性成立 |
| 调参 | k1–k6（见「参数调参」一节） | 在 j1 上每个只动一处 |

2026-10-14 瘦身前这里有 42 个变体（骨干 / 融合 / 规模 / 站点身份 / 超短期模块 / 训练协议），结论都在种子噪声内或从未跑过，
yaml 与对应代码一起删了；历史结论保留在各节文字里，要复现去 git 历史（commit 203c735 之前）。

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

`--full` 的产物目录下还会落一份 `result_test.parquet`（全部起报行、已过决策层的逐行预测，列与 evaluate.py 的同名文件一致），
可以直接喂 `diagnose_test.py`，不用再跑一次推理：
```bash
python -m forecast_backbones.m8.diagnose_test --result runs/m8/refit_capfix/g4_full/result_test.parquet \
       --test "datasets/test/*.parquet" --station 国能和煦光储电站 华能北方润达光伏电站 --out runs/m8/refit_capfix/g4_full/diag
```

## 晴空出力比（h 系列，2026-10-12）

p_clear 按水平面 GHI 算（1000 W/m² = 满容量），决策层再裁到 1.05·p_clear。对倾斜 / 跟踪 / 双面阵列，冬季高纬的
平面辐照是水平 GHI 的 1.5–2 倍，低温再加效率，电站出力远超这个"晴空功率"。华能北方润达（达拉特旗，40.3N，1108 m，
双面双玻 + 20 MW 平单轴跟踪）12 月：上限 0.51·cap，观测峰 0.78·cap；bias_ratio 0.66；测试集上只把晴空比放大到 3
就从 75.9 升到 77.6，剩下的偏低在网络本身（p_clear 输入通道给了错的先验）。坐标核过是对的，问题在公式不在坐标。

`features.derived.clear_sky_ratio`（默认关）：每行从历史可见段取近正午点（p_clear > near_noon_frac·窗内最大）的
y / p_clear 的 quantile 分位，夹到 [lo, hi]，乘到 p_clear 上（通道 + 决策上限一起变；k_nwp 是辐照比不动；
缺坐标的退化站恒 1）。按窗滚动，季节变化自然跟随，未见站也能用。评估结果表多一列 cs_ratio。

| 变体 | 改动 |
|---|---|
| h1_cs_ratio | e1_final + 晴空出力比 |
| d3_timesfm_hint / d4_timesfm_resid | e1_final + TimesFM-3 零样本预测列 `timesfm3` 作提示 / 作残差基准（列名已按数据改） |
| h4_envelope | h1 + 缺坐标站的历史日形包络（下一节） |

## 缺坐标站的历史日形包络（h4，2026-10-13）

站表没有经纬度的站（国能和煦 1026）原本退化为钟表几何：06:15–20:00 内 p_clear = cap 平顶、elev = ±10。白天过宽、没有日形，
网络只能靠 NWP 猜日出日落，测试集 D+1 比不用几何的 TimesFM 零样本低 12 分。调研（`docs/survey_m8_architecture_review.md` §7.4）
的结论是：几何缺失时用历史推日形，比错的几何好。

`features.derived.history_envelope`（默认关）：对退化站，把窗内 672 步可见历史（CPM 掩掉的不算）按时刻槽排成 7 天 × 96 槽，
每槽取 quantile（0.9）分位、环形平滑 smooth（3）槽得到日包络 env[槽]。p_clear 通道 = env/cap（软先验），elev 通道 = 90·env/峰值
（夜间 −10），夜间 = env < night_frac·cap；k_nwp 的白天窗口随之收窄。**决策层的 p_clear 仍是"白天 = 容量"**，包络不做硬上限
（整周阴天后的晴天不会被裁）。整周无出力（峰值 < peak_frac·cap）或有数据的天数 < min_days 的行回退钟表几何。有坐标的站逐位不变。
评估结果表 ctx 里多一个 envelope_used 标志。

跑 h4_envelope 看国能和煦单站；舰队其它站与 h1 应完全一致。

## 训练协议开关（EMA / 平滑选点，2026-10-13）

三轮消融里所有骨干 / 融合 / 规模变体都落在 a1 ±0.6 的种子噪声内，而验证分在 1,500–2,500 步见顶后来回晃 1.5–3 分：
按单次验证挑 checkpoint 挑的是噪声。调研（`docs/survey_m8_architecture_review.md` §6）对这个数据量级（≈ ETTh1）的处方一致：
小模型、低学习率、权重 EMA、按多次验证均值选点、多种子集成。两个开关：

- `train.ema_decay`（默认 null）：每个优化步后 shadow ← d·shadow + (1−d)·θ，前几步 d = min(decay, (1+n)/(10+n)) 让 shadow 快速
  跟上（decay 0.999 时 n=1000 步 d≈0.991，n=4000 步 d≈0.998，有效窗口数百步）。验证、存最优、refit 的最终权重都用 shadow；
  训练继续用原权重。train_log.json / summary.csv 记 ema_decay。
- `train.early_stop.smooth_k`（默认 1）：best 与早停按最近 k 次验证分的均值判；history 里原始分（`nanwang_after_pstar`）与
  平滑分（`nanwang_after_pstar_smooth`）都记，best_step 指平滑分最高的那次。

round5 结论：i1（小模型 + lr 1e-4 + EMA + 平滑选点）垫底，i2（只开 EMA + 平滑选点）与 e1_final 持平——两个开关保留，变体 yaml 已删。

refit 的三颗种子若在测试集上分数差得很开，可给 refit 配置加 `train.ema_decay: 0.999` 再比一次。

## 夜间口径（2026-10-12 起：不做夜间归零）

M1–M7 都不做夜间后处理，而四个指标都是全点含夜间、分母下限 0.2·容量：夜间报容量 2% 的噪声就扣约 1 分，5% 扣约 5–6 分。
M8 以前在决策层把夜间强制归零，等于比别的模型多了一道后处理，分数不可比。现在：
- `decision.physics.night_zero` 默认 **false**，夜间预测由模型自己给（输入仍有 is_night / elev / p_clear 通道）；
- `train.loss.night_weight` 默认 **1**：夜间步与白天同权进 pinball（以前夜间不监督，所以旧权重的夜间输出是没训过的，
  不能只关 night_zero 不重训）。想回到旧行为：`night_zero: true` + `night_weight: 0`，重训。
- 旧 run 的 model_config.yaml 里存的是 `night_zero: true`，评估旧 run 仍会归零；要看旧 run 不归零的分数用 `--decision_yaml`
  传 `decision: {physics: {night_zero: false}}`，但它夜间没训过，分数只能当下界。
- 晴空 / 容量钳位保留（它们只裁掉物理上不可能的值），但 **晴空上限在夜间不生效**：夜间晴空 GHI 恒为 0，
  上限 1.05·p_clear = 0 会把夜间悄悄钳成 0，等于换个地方做夜间归零。所以 `night_zero: false` 时 p_clear<=0 的步
  只用容量钳位（`physics_clip(..., night_ceiling=False)`），夜间完全由模型给。
  **注意**：这一条是第二次提交才补上的；之前用 `night_zero: false` 评出来的分数夜间其实仍是 0。决策层没有可训练参数，
  拉代码后用同一批 checkpoint 重新 `eval_m8.sh` 即得真正对齐的数字，不用重训。
- 想量化"夜间对齐"值多少分：`config/variants/e1_night_old.yaml`（= e1_final + night_weight 0 + night_zero true，
  即旧口径）与 e1_final 同切分同种子各跑一遍，差值就是夜间强制归零的红利。
- 决策层还剩一道对夜间有一点帮助的处理：负值截到 0（`physics.clip_negative`，默认 true）。略负的夜间输出截成 0 就是全对，
  M1–M7 不截。想严格裸对比在 decision yaml 里设 `clip_negative: false`，不用重训。
- 想完全裸对比可在 decision yaml 里设 `physics: {capacity_cap: false, clear_sky_cap_ratio: 1.0e9, clip_negative: false}`、
  `point: {default: median, by_step: []}`。

## 已删除的分支（2026-10-14 瘦身）

模型架构与输入输出在 j1_final2 上定下来之后，把三轮消融里落在种子噪声内、或从未跑过的分支连代码带 yaml 一起删了，
m8 的 .py 从 3,413 行减到 3,172 行。旧 yaml 再写这些键会在 `load_config` 直接报错（`config.REMOVED_KEYS`），不会被静默忽略。
`tests/test_reference.py` 守住：j1_final2 小尺寸模型加载瘦身前存下的权重，分位 / 提交曲线 / 损失逐位不变。

| 删掉的 | 原变体 | 当时的结论 |
|---|---|---|
| `slstm` 混合器、`model.bidirectional` | a2 / b2 | 与 a1 在 ±0.6 噪声内 |
| `model.layout: split` 双流、`model.head_mode: token` | b1 / b3 | 同上 |
| `features.nwp_members` 集合成员通道 | d1 | 数据没有 SSRD_pos_* 列，没跑过 |
| `model.mean_head` / `recent_skip` / `ultra_head`、`train.loss.horizon_weights` / `metric_aux`、早停 `nanwang_combo` | g1–g7 | 第四轮没跑过；超短期改用 d4 残差连接（+0.65） |
| 配方 / 站点身份 / 训练协议变体 yaml | c1–c8、f1–f3、i1–i2、h2–h3 | 结论见上文各节；开关本身（CPM、EMA、smooth_k、cs_ratio）都还在 |

## 全量重训（refit，最终模型）

两种全量切分，配方都在 m8_v1 上选定后再用：
- `data/splits/m8_refit.yaml`（传 `--test-parquets`）：每站测试起点前 purge 2 天之外全部训练，**不含测试期** → 可信的测试分数。
- `data/splits/m8_all.yaml`（不传 `--test-parquets`）：训练文件夹每一行都训练，**含测试期**，与 M1–M4 "全部数据训练"的口径一致 →
  只用于和 M1–M4 并排的对照数字。同配方同种子两套各跑一遍，差值就是泄漏带来的虚高（M8 有站点嵌入、坐标、日序编码，
  泄漏下比 M4 更会"记住"，这个数字会偏向 M8，不能拿来选配方）。


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
