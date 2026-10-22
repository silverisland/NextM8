# M4-Rum 架构对照实验（19 组）

这些是可运行的候选架构，没有真实精度提升保证。以 q6（EMA=0.99）为共同基线，
训练数据、验证数据、GPU 数、种子和训练预算必须一致。先比较 a00–a04，
再检验组合收益；a16/a17 为容量与站点信息的组合对照。
所有新结构均限定为 `query_decoder`，不修改 legacy。

## 四种结构

- **R / residual_head**：在原输出 MLP 之外加一条线性输入→分位输出通路。
  跳接权重和偏置以 0 初始化，起始预测与基线相同；训练后学习直接映射。
- **D / weather_day_reference**：保留历史每天同一相位的功率/天气，
  用未来天气与历史天气的距离，以及可学习的匹配网络，为各历史日分配权重。
  仅使用两端都有效的天气维度；存在天气匹配候选时排除缺天气的日子，
  完全没有匹配候选时回退到有效历史功率的同相位均值，完全缺历史时四个统计量为 0。
  输出加权功率均值、方差、功率覆盖率、天气匹配覆盖率，注入 query 并通过独立线性分支进入输出。
  对齐以起报时刻为准，未来第 0 步对应历史 L−96；不是把输入窗口强行当作午夜开始。
  匹配权重不读取历史功率数值，更不读取未来真实功率。
- **L / local_weather**：已知未来天气、掩码、时钟和提前量，经过核长 3/9 的并行深度卷积，
  融合后加入未来 query；与原未来 self-attention、历史 cross-attention 共存。
  卷积可读取左右已知预报点，不对预测结果硬平滑，不读取未来观测。
- **H / hint_gate**：输出为 `原输出 + sigmoid(gate) × 有效 hint`，
  每个时刻一个门值，对全部分位共享，原输出负责分位修正。
  hint 缺失或被 hint_dropout 屏蔽时该加法严格为 0。它不是简单的输出头跳接，
  也不是保证保留 hint；gate 和修正网络共同学习。

开关位于 `model.extensions`；默认全部关闭。所有附加模块在已有模块之后初始化，
关闭时不增加参数、不改变旧权重键；同种子下保留共有基干的初始化。
开启新分支需要重新训练，不应直接修改旧 checkpoint 的配置。
缺测和 history/hint dropout 对新分支同样生效。保留 Pinball 原始分位训练、
p_star 决策和容量/非负裁剪，分支不接触标签。

## 配置清单

全部配置展开了默认值，各自有独立输出目录 `runs/m4_rum/arch_sweep/<配置名>`。
a00–a15 穷举 R/D/L/H 的 16 种组合，不增加站点 embedding、不扩大主干。

| 配置 | 相对共同基线的变化 | 参数量 |
| --- | --- | ---: |
| [a00_ema_baseline.yaml](a00_ema_baseline.yaml) | 与原 q6 等价的 EMA 基线 | 718,601 |
| [a01_residual_head.yaml](a01_residual_head.yaml) | R | 719,924 |
| [a02_weather_days.yaml](a02_weather_days.yaml) | D | 719,760 |
| [a03_local_weather.yaml](a03_local_weather.yaml) | L | 754,825 |
| [a04_hint_gate.yaml](a04_hint_gate.yaml) | H | 723,338 |
| [a05_residual_days.yaml](a05_residual_days.yaml) | R + D | 721,083 |
| [a06_residual_local.yaml](a06_residual_local.yaml) | R + L | 756,148 |
| [a07_residual_hint.yaml](a07_residual_hint.yaml) | R + H | 724,661 |
| [a08_days_local.yaml](a08_days_local.yaml) | D + L | 755,984 |
| [a09_days_hint.yaml](a09_days_hint.yaml) | D + H | 724,497 |
| [a10_local_hint.yaml](a10_local_hint.yaml) | L + H | 759,562 |
| [a11_residual_days_local.yaml](a11_residual_days_local.yaml) | R + D + L | 757,307 |
| [a12_residual_days_hint.yaml](a12_residual_days_hint.yaml) | R + D + H | 725,820 |
| [a13_residual_local_hint.yaml](a13_residual_local_hint.yaml) | R + L + H | 760,885 |
| [a14_days_local_hint.yaml](a14_days_local_hint.yaml) | D + L + H | 760,721 |
| [a15_all_branches.yaml](a15_all_branches.yaml) | R + D + L + H | 762,044 |
| [a16_large_station_ema.yaml](a16_large_station_ema.yaml) | 大模型 + 站点 embedding + EMA，对照组 | 4,836,105 |
| [a17_all_large_station.yaml](a17_all_large_station.yaml) | 大模型 + 站点 embedding + EMA + R/D/L/H | 4,987,196 |
| [a18_residual_wide_head.yaml](a18_residual_wide_head.yaml) | R + 输出头加宽到 256→128 | 763,956 |

共同设置：seed 0、EMA=0.99、batch 8/卡、lr=3e-4、weight_decay=0.05、
普通 dropout=0.1、6000 步上限、warmup 300、每 250 步验证、每 500 步存快照、
early stop patience=8、history/hint dropout=0、Pinball、D+1 p_star。
a16 相当于 q9 加 EMA；a17 相对 a16 只开启四种结构；a18 相对 a01 只加宽输出 MLP。
a00 和原 q6 除 tag/输出目录之外等价，可重新跑以确认实验环境一致。

**本轮没有切换选模口径。** early stop 和最佳权重仍按 `nanwang_short_D+1`
（站日平均），与原 q6 保持一致。完整评估关注 `station_mean_nanwang_short_D+1`
（月报→站均），或 `m8_comparable_summary.csv` 中同值的 `nanwang_short_D+1`。
因此选择出的最佳权重未必也是站均月报口径的最佳步。若研究选模口径，应作为另一轮独立变量，
不能将本轮结果宣称为按站均月报指标选出的最优值。

## 单组训练与评估

在 NextM8 根目录运行，替换实际数据路径：

```bash
VARIANT=a01_residual_head
bash train_shell/train_m4_rum.sh --gpus 0 \
  --config "forecast_backbones/m4_rum/config/arch_sweep/${VARIANT}.yaml" \
  --train_parquet 'datasets/train/*.parquet' \
  --valid_parquet 'datasets/valid/*.parquet'

bash eval_shell/eval_m4_rum.sh \
  --runs "runs/m4_rum/arch_sweep/${VARIANT}" \
  --test 'datasets/test/*.parquet' --full \
  --out "runs/m4_rum/arch_sweep/${VARIANT}/eval_final"
```

必须提供 valid 才会验证、早停并回载最佳权重；仅提供 train 会进入全量 refit。
默认 Python 为 `/home/ma-user/ts3/bin/python`，自定义环境在两个命令上显式传入
`--venv-python /实际环境/bin/python`，或预先显式设置 `PYTHON`。解释器不可用时直接报错，没有自动回退。

重点读取每组 `eval_final/reports/m8_comparable_summary.csv`，或者
`eval_final/reports/summary.csv` 的 `station_mean_nanwang_short_D+1`。
如果要用该指标筛选架构，应在同一验证集上运行上述完整评估（把 --test 换成验证集，
输出到 eval_valid），最终候选再评估固定测试集；不要用测试集选训练步。

## 两张 GPU 并行队列

每张卡串行执行自己的队列，两卡同时运行。以下先跑独立分支和代表性组合；
可在队列末尾补充配置清单中的其他名字，不带 `.yaml`。
不使用 `--gpus auto`，保持每组总 batch 一致。

```bash
TRAIN_PARQUET='datasets/train/*.parquet'
VALID_PARQUET='datasets/valid/*.parquet'
ARCH_ROOT='runs/m4_rum/arch_sweep'
mkdir -p "$ARCH_ROOT/logs"
run_group() {
  local gpu="$1"
  shift
  local variant
  for variant in "$@"; do
    bash train_shell/train_m4_rum.sh --gpus "$gpu" --cpu_threads 4 \
      --config "forecast_backbones/m4_rum/config/arch_sweep/${variant}.yaml" \
      --train_parquet "$TRAIN_PARQUET" --valid_parquet "$VALID_PARQUET" \
      > "$ARCH_ROOT/logs/$variant.log" 2>&1 || return 1
  done
}
run_group 0 a00_ema_baseline a01_residual_head a03_local_weather a05_residual_days a15_all_branches & pid0=$!
run_group 1 a02_weather_days a04_hint_gate a16_large_station_ema a17_all_large_station a18_residual_wide_head & pid1=$!
status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
printf 'All workers finished; status=%s (0 means success)\n' "$status"
```

先确认各卡能容纳当前 batch；如果需要降 batch，比较组统一调整后重跑。
不同种子使用不同输出目录，不覆盖已有结果。初筛后重复 seed 1、2，报告均值和波动。
参数数量仅说明容量，不能据此排序预期精度；不建议默认全开就是最优。
