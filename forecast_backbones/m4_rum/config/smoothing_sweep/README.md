# PV 居中滑动均值对照（8 组）

本组在 q6（query_decoder、EMA=0.99）上仅改变预处理，不与新架构开关同时变化。
平滑可能减少高频噪声，也可能抹掉真实爬坡；平滑标签后，Pinball 学习的是
平滑目标的分位分布，不再是原始目标的分位分布。因此必须以原始功率指标判断效果，
不能用平滑后的标签评分，也不能仅因训练 loss 下降就认定提升。

## 行为与边界

- `history_window`：训练、验证、推理都对历史 PV 使用同一窗口。1 表示不处理。
- `target_window`：只在 `model.train()` 且传入标签时，对训练损失中的 target 平滑。
  验证/推理、评估指标和结果文件中的真实功率始终保留原值。1 表示不处理。
- 只支持 `alignment: centered`：W=2k+1，每个内部点取 t−k 到 t+k 的算术均值，
  分母固定为 W；首尾各 k 个点保持原值，不补零、不复制端点，也不做缩短窗口平均。
  5 点窗口时，首尾各 2 个点不平滑；3/9 点窗口分别保留首尾各 1/4 点。
- 在历史内部可以使用该历史点之后的观测，但所有数据都在起报时刻及之前，
  不读取未来标签。历史和 target 分别处理，二者不拼接，各自保留首尾 k 点。
  target 只在自己的 192 点窗口内平滑；不跨样本、不跨站点。
- 窗口内若有 NaN/Inf 或无效掩码，保留当前点原值，不改用有效邻居的部分均值。
  原本缺测的位置仍保持缺测；全缺测历史走原缺历史分支，缺标签不参与损失。
- 只处理 PV。天气、hint、容量、时间及预测结果都不平滑。历史日曲线和可选天气日匹配分支
  从处理后的历史 PV 构建；各结构的 history/hint dropout 规则保持原样。
- 容量为逐站固定值，因此容量归一化后平滑与原单位平滑后归一化等价。
  平滑计算保持 fp32（输入 fp64 则用 fp64），不随 AMP 降到低精度。

配置示例（与任何 Rum 架构组合均可，以下为历史和训练标签同时做 5 点居中均值（首尾各 2 点不变））：

```yaml
preprocess:
  pv_smoothing:
    history_window: 5
    target_window: 5
    alignment: centered
```

默认两个窗口都为 1，旧配置/旧权重保持兼容，不增加参数。checkpoint 保存预处理配置，
加载后自动应用历史平滑。启用平滑应重新训练，不建议只修改旧权重的推理配置。
应提供原始数据，避免数据已经平滑、模型内部又重复平滑。

## 实验清单

窗口单位是 15 分钟采样点数（3/5/9 点），不是小时数。
全部保持 q6 的训练/选模/决策设置，参数量均为 718,601。

| 配置 | 历史窗口 | 训练 target 窗口 | 对齐 |
| --- | ---: | ---: | --- |
| [s00_baseline.yaml](s00_baseline.yaml) | 1 | 1 | 不平滑 |
| [s01_history3.yaml](s01_history3.yaml) | 3 | 1 | 居中 |
| [s02_history5.yaml](s02_history5.yaml) | 5 | 1 | 居中 |
| [s03_history9.yaml](s03_history9.yaml) | 9 | 1 | 居中 |
| [s04_both3.yaml](s04_both3.yaml) | 3 | 3 | 居中 |
| [s05_both5.yaml](s05_both5.yaml) | 5 | 5 | 居中 |
| [s06_both9.yaml](s06_both9.yaml) | 9 | 9 | 居中 |
| [s07_target5.yaml](s07_target5.yaml) | 1 | 5 | 居中 |

建议先跑 s00/s02/s05/s07，分离历史平滑与标签平滑的效果。
之后比较 3/5/9 点的窗口强度。移除原尾随方案，原 s08/s09 与新 s02/s05 重复，已合并。
每组独立输出到 `runs/m4_rum/smoothing_sweep/<配置名>`。
共同条件为 seed 0、batch 8/卡、lr=3e-4、6000 步上限、每 250 步验证、
patience 8、EMA=0.99、Pinball、D+1 p_star；不改变已有架构实验结果。

## 训练与评估

在 NextM8 根目录运行，替换实际数据路径：

```bash
VARIANT=s05_both5
bash train_shell/train_m4_rum.sh --gpus 0 \
  --config "forecast_backbones/m4_rum/config/smoothing_sweep/${VARIANT}.yaml" \
  --train_parquet 'datasets/train/*.parquet' \
  --valid_parquet 'datasets/valid/*.parquet'

bash eval_shell/eval_m4_rum.sh \
  --runs "runs/m4_rum/smoothing_sweep/${VARIANT}" \
  --test 'datasets/test/*.parquet' --full \
  --out "runs/m4_rum/smoothing_sweep/${VARIANT}/eval_final"
```

无需预先离线平滑数据。训练和评估都可传 `--venv-python /实际环境/bin/python`，
解释器不存在时报错，不自动回退到其他 Python。
多组并行时为每组指定 GPU，保持总 batch 一致；不要多组同时使用 `--gpus auto`。

完整评估关注 `reports/summary.csv` 中的 `station_mean_nanwang_short_D+1`，
或者 `reports/m8_comparable_summary.csv` 中同值的 `nanwang_short_D+1`。
这些分数全部针对原始真实功率。
为与 q6 对照，本组早停仍按原始标签上的站日平均 `nanwang_short_D+1`，未改为站均月报选模。
只给 train_parquet、不提供 valid 时仍是 refit，不进行验证选模。
可将 --test 指向验证集、输出到 eval_valid，用站均月报指标筛选架构/平滑方案；最终再测固定测试集。
