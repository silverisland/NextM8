# M8 "PVRex"：带 NWP 协变量的混合递推分位预报模型 —— 设计文档

日期：2026-10-08　状态：待用户审阅（实现已按本文档完成并通过本地冒烟测试）

## 0. 一句话

把 TiRex 的「未来 patch 当作掩码 token、一次前向吐出全部分位」与 Reverso 的「门控长卷积 + DeltaNet 交替混合器」
搬到 PV 场景，再加上三件 PV 专属的事：**未来 NWP 在预报时刻位置直接可见且在输出头侧二次注入**、
**CPM 掩码 + 协变量 dropout 逼模型既用 NWP 又保留历史回退**、**容量归一化 + 晴空/太阳高度角通道 + 夜间掩码**。
从零训练，纯 PyTorch，V100 fp32 可跑，接入现有 `train_script.py` / registry 契约。

## 1. 目标与成功标准

- 输入：历史 672 步（7 天 @15 min）实测功率 + 历史/未来 NWP（GHI、温度、可选 Solargis 与 SSRD 集合成员）+ 站点元数据。
- 输出：未来 192 步 9 档分位（0.1…0.9）与提交曲线；aux 与 m6 同 schema，可复用 m6 的决策层、阶梯报表与 conformal 工具。
- 评价：南网短期 D+1 准确率（10:00 起报，`predict[55:151]`）与容量归一 RMSE；按站 / 汇总 / 分 lead 报告。
- 成功标准（需在服务器上验证，本文档不替用户下结论）：验证块上南网 D+1 均值不低于 m6（TimesFM-3 + LoRA）与 m4（PatchTST）；
  NWP 加噪扫描下退化幅度小于 m4。

## 2. 调研结论如何落到设计（证据见 `docs/survey_covariate_pv_models.md`）

| 结论 | 来源 | 设计动作 |
|---|---|---|
| 已知未来协变量必须在预报时刻位置可见，且输出侧注入贡献几乎全部收益 | ChronosX 消融 OIB 0.182 vs IIB 1.035；CITRAS KV-shift；TiDE temporal decoder | 未来 patch 作为带 NWP 的 token 进主干；输出头按步拼接该步协变量向量 |
| 调制 / 门控优于直接相加 | CoRA adaLN 0.278 vs 0.314；TimeXer cross-attn 0.307 vs 0.329 | 混合器全部带门控；头部用 MLP 非线性融合 |
| 没有训练信号逼迫，模型不会用协变量 | COSMIC 无增强时 0.710→0.709；TiRex/PatchTST-FM 的 CPM 贡献 > 骨干 | 历史 patch 连续掩码（CPM）并对掩码处计损 |
| NWP 有误差时协变量模型会崩，整通道 dropout 能恢复 | 2610.07232 λ 扫描；2607.05452 Exogenous Dropout | 训练期按行以 p=0.1 清零全部 NWP 通道并置 `cov_missing` 标志 |
| RevIN 在有夜间零值的 PV 上不稳；晴空残差分支 −5% MAE；夜间必须掩 | CycleNet 附录；PhysTrans；Yang 2020 | 目标按装机容量归一；输入带 `p_clear/cap`、`elev/90`、`k_nwp`；elev ≤ 0 不计损且输出归零 |
| 混合器收益 2–5%，在同参数量级 Conv+DeltaNet 最好，注意力作对照 | Reverso Table 15；TiRex Table 1 | 默认 `[conv, deltanet, conv, deltanet]`；`slstm` / `attention` 可配 |
| 树模型在单站 15 min D+1 上仍是强基线 | HEFTCom2024；2609.15035 | README 建议补 LightGBM 对照（本计划不实现） |

## 3. 数据与 token 布局

- 一行 = 一次起报：历史 `observe_power[672]`，未来真值 `observe_power_future[192]`（只在训练/评测时读），
  协变量对 `(hist_col, fut_col)`，`cap_power_on`，`station_id`（整数），`timestamp_win`（起报时刻，epoch 秒，本地墙钟）。
- patch 长度 P = 16 步（4 h）：历史 42 个 token，未来 12 个 token，共 54 个 token。
  P 可配（须整除 672 与 192：8/16/24/32/48）。

### 3.1 每步通道（numpy 组装，`features.py`）

| 通道 | 定义 |
|---|---|
| `y` | 功率 / cap；掩码处为 0 |
| `y_mask` | 1 = 目标不可见（未来、缺测、CPM 掩码） |
| `is_future` | 未来段为 1 |
| 协变量对 ×N | 按列名规则归一：含 `ghi`/`ssrd` 的 /1000；含 `temp`/`t2m` 的 (x−15)/15；其它用 `features.scales` 配置 |
| `k_nwp` | 第一个 GHI 类协变量 / 晴空 GHI（晴空 > 50 W/m² 时），裁到 [0, 1.5] |
| `elev` | 太阳高度角 / 90 |
| `p_clear` | 晴空功率 / cap（pvlib Ineichen，缺坐标站退化为钟表几何，与 m6 一致） |
| `is_night` | elev ≤ 0 |
| `tod_sin/cos`、`doy_sin/cos` | 时间编码 |
| `nwp_resid` | 观测白天步上的 `y − ghi_pred/1000`，其余 0（显式的 NWP→功率偏差信号） |
| `cov_missing` | 协变量 dropout 或缺列时为 1 |
| 集合成员（可选） | 成员均值 /1000、成员离差 /1000 |

静态向量：`log1p(cap)/7`、lat/90、lon/180、alt/5000、degraded、上下文统计（历史最大 y、白天均值 y、白天均值 k_nwp）。
站点嵌入：`station_id mod n_slots`，0 号槽保留给未知站。

太阳几何按 (station_id, 日期) 缓存 96 步数组，训练吞吐不受 pvlib 拖累。

## 4. 模型（`layers.py` / `model.py`）

```
token_t = ResidualMLP(flatten(P × C)) + static_proj(static) + station_emb + pos_emb[t]
for block in blocks:                       # 默认 [conv, deltanet, conv, deltanet]
    x = x + Mixer(RMSNorm(x)); x = x + FFN(RMSNorm(x))
h = RMSNorm(x)
对每个 token 内的每一步 s：q[t,s,:] = ResidualMLP([h_t, pos_in_patch[s], c_{t,s}]) → K=9 分位（y/cap 单位）
```

混合器（全部纯 torch、因果）：
- `conv`：门控长卷积。`a = SiLU(ShortConv_k4(W_a x))`，`b = CausalDepthwiseConv_{kernel=T}(W_b x)`，`out = W_o(a ⊙ b)`。
  与 Reverso 的区别：线性（零填充）而非循环卷积。
- `deltanet`：Gated DeltaNet。q/k/v = Linear → 因果 ShortConv_k4 → SiLU；q、k 逐头 L2 归一；β = σ(W_β x)；
  α = σ(W_α x + 4)；`S_t = α S_{t−1} + β (v_t − α S_{t−1} k_t) k_tᵀ`；`o_t = RMSNorm_head(S_t q_t) ⊙ SiLU(W_g x)` → W_o。
  机理：delta 规则等价于在上下文里对「NWP 特征 → 实测功率」做在线回归，未来 token 用自己的 NWP 做 query 即得到近期偏差校正后的映射。
- `slstm`：按 TiRex/xLSTM 论文公式自行实现（指数门 + 稳定器 m、f 门裁到 15、逐头块对角递推、组归一化），不拷贝其代码（NXAI 许可证）。
- `attention`：因果多头注意力，作对照。

54 个 token 的递推循环在 fp32 上开销可忽略；无 Triton/CUDA 扩展依赖。

## 5. 损失与训练（`train.py`）

- 损失：`m6.losses.pinball_loss`，监督步 = `y_mask==1 ∧ 真值有限 ∧ 非夜间`（未来段 + CPM 掩码段）。
- 增强（仅 `model.training`）：CPM（p_row 0.5，掩码比例 U(0.05,0.25)，1–3 段连续 token）；
  协变量 dropout p 0.1；幅度调制 p 0.3，u∼U(0.6,1.0) 同乘历史与标签。
- 优化：AdamW lr 5e-4，wd 0.05，warmup 500 步后 cosine，每卡 batch 64，max_steps 20000，grad_clip 1，fp32。
- 验证/早停：复用 m6 的 `rows.scored_row_index`（10:00 行）与 `NanwangMetric.short_d1_daily`，或 `nrmse`（整点行）；
  `patience` 按验证次数；最优权重存 `model.safetensors`（完整 state_dict）。
- 多卡：HF Trainer + `accelerate launch`，主进程打分，`m6._dist` 广播早停。
- 产物：`model.safetensors`、`model_config.yaml`、`train_log.json`、`result_val.parquet`、`reports/summary.csv`、`reports/station_table.csv`。

## 6. 接入契约

- `forecast_backbones/m8/transformer_models.py`：第一个 nn.Module 子类 `TransformersAdaptedM8Model(model_config)`；
  `forward(batch_dict, labels=None) -> TransformersAdaptedModelOutput`；`.config`；`compute_metrics`。
- `forecast_backbones/m8/transformer_trainer.py`：`evaluate_model(model, ds, collate, cfg) -> DataFrame`（与 m6 result_df 同 schema）、
  `M8_build_training_args(cfg, result_dir)`。
- `train_script.py`：`--model_name m8` 与 m6 一样走独立入口 `train.train_m8(config, splits_dir, out_dir)`；`build_train_args` 增加 m8 分支。
- 根目录 `train_m8.sh`：与 `train_m6.sh` 同参数风格。

## 7. 明确假设（用户不在线，按此实现；需确认）

1. 数据切分沿用 `pipelines.time_splits` 落的目录（train/valid.parquet + split_manifest.json），不需要 cal.parquet。
2. 站点坐标来自 `forecast_backbones/m6/station_meta.csv`（现为占位，训练前须填）；缺坐标站退化为钟表几何。
3. 默认不做 conformal 校准，提交点用中位数；校准可直接套 m6 的 Calibrator（aux schema 已对齐）。
4. 训练用全部起报时刻的行；评测口径按南网 10:00 起报。
5. 集合成员 SSRD_pos_1..9 默认关闭（数据里不一定有）。

## 8. 测试

`forecast_backbones/m8/tests/`：层的形状与因果性（扰动未来 token 不改变过去输出）；CPM 与夜间掩码语义；
registry 能发现并构造；合成数据（m6 conftest 的 `make_synthetic_frame`）上前向/反向、3 步训练、`evaluate_model` schema。

## 9. 不做的事（YAGNI）

Reverso 多尺度输入（672 步上下文下收益不明，长卷积已覆盖日周期）、Reverso 注意力读出头（未来 token 已在序列中）、
翻转等变（PV 非负）、mixup、conformal 训练期拟合、LightGBM 基线。

## 10. 变体矩阵（批量测试用，`forecast_backbones/m8/config/variants/`）

每个变体文件只写与默认不同的字段。先看 a1 vs b4（协变量价值）、a1 vs c1（配方价值），再比骨干；骨干差异小于种子噪声时用 `run.seeds` 多跑。

| 组 | 变体 | 回答的问题 |
|---|---|---|
| A 骨干 | a1_hybrid（基线）/ a2_slstm / a3_attention / a4_deltanet_only / a5_conv_only / a6_no_mixer | 混合器选型；a6 为无序列建模下界 |
| B 协变量通路 | b1_split_stream / b2_bidirectional / b3_token_head / b4_no_nwp | 双流 vs 单流；预报时刻后的 NWP 是否有用；输出侧注入价值；NWP 总价值 |
| C 配方 | c1_no_cpm / c2_recent_peak / c3_patch32 / c4_patch8 / c5_large / c6_nrmse_stop / c7_station_day | CPM+dropout；归一化；token 粒度；规模；早停口径；采样 |
| D 数据/决策 | d1_members / d2_pstar_point / d3_timesfm_hint / d4_timesfm_resid | 集合成员；p* vs 中位数；TimesFM 预测作未来提示通道（`features.future_only`，历史段恒 0，独立 `hint_dropout`）；残差学习（`residual_base_col`，输出 = 提示 + 网络输出） |

启动：`bash train_m8_variants.sh --splits_dir <splits> --gpus 0,…,7`，汇总见 `runs/m8/<batch_tag>/variants_summary.csv`。
