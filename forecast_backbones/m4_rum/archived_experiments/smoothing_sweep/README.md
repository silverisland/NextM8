# PV 平滑实验（已停止，归档记录）

用户决定放弃历史 PV 和训练 target 的平滑处理。此目录仅保存原实验配置及结果，
不再作为后续训练方案。新实验使用 `config/d1_sweep/q6_ema099.yaml` 或 `config/arch_sweep/`；
有效训练配置的 `history_window` 与 `target_window` 均为 1，直接使用原始功率数据。

## 已报告结果

用户标注的指标为 `station_mean_nanwang_ultra_short`，原始数值如下。
未提供 s04 结果；这些结果不作为 D+1 结论。

| 原配置 | 历史窗口 | 训练目标窗口 | 指标 |
| --- | ---: | ---: | ---: |
| [s00_baseline.yaml](s00_baseline.yaml) | 1 | 1 | 0.7688 |
| [s01_history3.yaml](s01_history3.yaml) | 3 | 1 | 0.7690 |
| [s02_history5.yaml](s02_history5.yaml) | 5 | 1 | 0.7688 |
| [s03_history9.yaml](s03_history9.yaml) | 9 | 1 | 0.7686 |
| [s04_both3.yaml](s04_both3.yaml) | 3 | 3 | 未提供 |
| [s05_both5.yaml](s05_both5.yaml) | 5 | 5 | 0.7670 |
| [s06_both9.yaml](s06_both9.yaml) | 9 | 9 | 0.7627 |
| [s07_target5.yaml](s07_target5.yaml) | 1 | 5 | 0.7682 |

仅平滑历史与基线基本持平，同时平滑目标时较大窗口表现下降。
单次实验未证明历史平滑有稳定收益，因此停止这个方向。

## 历史复现与兼容

归档 YAML 的模型、训练、输出目录和预处理数值保持原样，避免混淆已有实验。
平滑方式是中心算术均值：窗口 W=2k+1，首尾各 k 点不处理；
窗口有缺测时保留当前点，不做部分窗口平均。历史与目标分开处理。
历史平滑应用于训练和推理；目标平滑仅用于训练损失；评估真实功率保持原值。

运行时保留旧配置加载支持。评估旧 s 系列权重时，继续读取运行目录中原有的
`model_config.yaml`，不会强制改写其平滑设置。后续不平滑实验使用新运行目录，
不覆盖已有平滑实验产物。EMA 和 early-stop 的指标平滑设置与 PV 平滑无关，保持不变。
