# M4-Rum

面向短期光伏功率预测的 M4 变体。历史上下文、未来气象与基模预测按时刻拼接，
共享 MLP 输出 192 点功率，或输出逐时刻分位并通过独立选点层得到功率。
**没有基模残差相加、门控混合或站点 embedding。**
原结构由 `model.architecture: legacy` 保留；新的 `query_decoder` 见下一节。

## D+1 优化版：query_decoder

针对 M8 K2 约 78%、Rum 约 74% 的 D+1 差距，新增可选架构及配套对照配置。
**当前只完成实现和链路验证，没有真实数据上的提分结论。旧默认配置和旧权重行为保留。**

已核对的实现差异：原 Rum 把全部历史压成 64 维 CTX，各未来时刻只访问本时刻的天气/提示，
缺少未来时刻之间的信息通路；默认 MSE 配置及旧 Pinball 配置均保留 history_dropout=0.2、
hint_dropout=0.3，而 M8 K2 的 hint_dropout=0。旧 Pinball 只有全时段 p_star，没有非负/容量裁剪。
这些是待消融的差异，不是已经证明的 4 分差距归因。M8 还具有太阳几何等特征，不能把本版称作等价复现。

新版数据流：

```text
历史 PV + 气象 + 缺测掩码 + 时间
  → 同一时间片内联合 Patch → 历史 Transformer → 完整历史 token

每个未来时刻：天气/掩码、时间/提前量、TimesFM/掩码、历史日曲线
  → 192 个查询 → 未来已知特征之间的 self-attention
  → 按预测时刻 cross-attention 读取历史 token
  → 拼回原始逐时刻输入与历史覆盖率 → MLP → 原始分位 → Pinball loss
  → 仅验证/推理：分位排序、可选物理裁剪、分段 p_star
```

历史日曲线提供最后 24 小时、历史同一相位均值及可用率；完全由起报时及之前的观测计算。
未来时刻 0 对齐历史 `L-96`，第 95 步对齐 `L-1`，第二天重复相位，不能拿未来真实功率填入。
未来自注意力只使用本次起报已知的 NWP/基模预测，不读取未来标签。
缺历史时仍有可用的空 token，避免全掩码 attention 产生 NaN；日曲线也遵循 history_dropout。
新版没有强制添加 TimesFM 残差，所有分位仍由网络直接预测。

建议至少跑以下两个配置，保持相同切分、种子及 GPU 数：

| 配置 | 架构 | 用途 |
| --- | --- | --- |
| `config/pinball_control.yaml` | 原 legacy | 判断训练/决策调整的收益 |
| `config/query_decoder.yaml` | 新 query_decoder | 在相同训练/决策条件下检验新架构 |

两者均用 Pinball、history/hint dropout=0、batch 8/卡、lr=3e-4、warmup 300、最多 6000 步、
每 250 步验证、D+1 早停 patience=8。保留原普通 dropout=0.1、weight_decay=0.05。
默认前 16 步 p_star_l1，其余 p_star，并启用非负及容量裁剪；对 10:00 起报而言，
D+1 `[55:151]` 全部使用 p_star，前 16 步的规则不直接影响 D+1。
物理裁剪不作用于训练损失，不根据验证或测试标签估计容量。

在 NextM8 根目录依次运行（实际路径自行替换）：

```bash
for VARIANT in pinball_control query_decoder; do
  bash train_shell/train_m4_rum.sh --gpus 0 \
    --config "forecast_backbones/m4_rum/config/${VARIANT}.yaml" \
    --train_parquet 'datasets/train/*.parquet' \
    --valid_parquet 'datasets/valid/*.parquet' \
    --out "runs/m4_rum/${VARIANT}_s0" --max_steps 6000 --save_every 500

  bash eval_shell/eval_m4_rum.sh \
    --runs "runs/m4_rum/${VARIANT}_s0" --test 'datasets/test/*.parquet' \
    --full --out "runs/m4_rum/${VARIANT}_s0/eval_final"
done
```

默认使用 `timesfm3`（兼容 `tsfm_predict`）和 `pipelines/forecasting/assets/info.csv`。
自定义站表/提示列可传 `--station_meta` / `--hint_col`。必须显式提供验证集才能早停；
只给 train_parquet 的旧行为仍是全量 refit，不会按 D+1 挑最佳权重。
多卡时 `batch_size` 是每卡值，对照实验必须固定总批量。换种子修改 YAML 的 `run.seeds: [1]` 等，
不要把多个数字写进同一次配置当成自动重复训练。先按验证分数选方案，再在固定测试集评估。

**注意 D+1 汇总口径**：原 Rum `reports/summary.csv` 的 `nanwang_short_D+1` 是用于选模的站日平均；
M8 `--full` 的同名字段是逐站月报汇总后再站间平均，不能直接混着比较。
现在 Rum `--full` 额外写 `reports/m8_comparable_summary.csv`，其 `nanwang_short_D+1` 与 M8
完整报告使用同一聚合口径；原汇总表仍保留，`station_mean_nanwang_short_D+1` 同样是这个可比值。
缺少足够评估天数时可能是 NaN，不能据此判定模型得分为 0。分数单位是 0–100，0.78 应对应 78。
`reports/members.csv` 增加历史、提示、未来气象整行缺失比例，排查训练/推理输入不一致。

`--point median/p_star/p_star_l1` 会清空分段规则，实现真正的全时段覆盖；保留配置中的物理开关。
未设置 `clip_negative/capacity_cap` 的旧配置仍不裁剪。新权重必须用 query_decoder 配置加载，
不能将旧权重强行加载成新版；加载采用严格形状检查，需要重新训练新版。

本地检查包括：旧结构跨未来步梯度严格为零而新版非零、历史日曲线对齐、缺测/dropout、
标签泄漏、Pinball 梯度、加权中位数、训练/最佳权重/快照/完整报告。
另做了一个专门检验未来跨步信息通路的合成任务：输入信号仅出现在后半段，预测前 16 步，
新版能学习这条关系。这个任务不是光伏 D+1 实验，不代表真实提升幅度。

## D+1 调参配置组（8 组）

配置目录：`config/d1_sweep/`。这些 YAML 将有效参数完整写出，避免后续默认值变化干扰对照。
统一条件：同一训练/验证集、seed 0、单 GPU、batch 8、最多 6000 步、warmup 300、每 250 步验证、
patience 8、Pinball loss、D+1 p_star；历史/提示 dropout 均为 0，验证批量统一为 32。
每个配置有独立 tag 和输出目录。这里提供候选实验，没有预先确定的精度排名。

| 配置 | 相对 q0 的变化 | 要验证的问题 |
| --- | --- | --- |
| `q0_baseline.yaml` | 新版基线：lr=3e-4，weight_decay=0.05，dropout=0.1，patch=16，宽128/FFN256 | 基准结果 |
| `q1_lr2e4.yaml` | lr=2e-4 | 学习率降低后 D+1 是否更稳定 |
| `q2_wd001.yaml` | weight_decay=0.01 | 原权重衰减是否过强 |
| `q3_dropout005.yaml` | model.dropout=0.05 | 普通 Dropout 是否导致欠拟合 |
| `q4_patch8.yaml` | patch_len=8，2 小时一个 Patch | 更细历史时间分辨率是否有帮助 |
| `q5_width192.yaml` | d_model=192、d_ff=384；其他层数/预测头不变 | 增大容量是否有效 |
| `q6_ema099.yaml` | ema_decay=0.99 | 权重平均是否改善验证稳定性 |
| `c0_legacy.yaml` | 换回旧 legacy 架构，其余条件一致 | 新架构是否真的优于旧结构 |

除容量组同时调整宽度与 FFN 外，各调参组只改变一个参数；先观察独立收益，再组合。
EMA 组的验证、最佳权重、快照均使用 EMA 权重。q4/q5 可能占用更多资源，应保持 batch 一致；
若显存不足，统一降低所有组的 batch 后重跑，或先跳过该组。

单组训练（在 NextM8 根目录）：

```bash
bash train_shell/train_m4_rum.sh --gpus 0 \
  --config forecast_backbones/m4_rum/config/d1_sweep/q0_baseline.yaml \
  --train_parquet 'datasets/train/*.parquet' \
  --valid_parquet 'datasets/valid/*.parquet'
```

每组运行中自动按 D+1 验证，无需额外启动验证进程。未提供 valid 时会进入全量 refit，不能用于这轮选模。
不要让每组都用 `--gpus auto` 抢占全部 GPU。以下两卡示例最多同时跑 2 组，每张卡顺序执行自己的队列；
CPU 线程数固定为每进程 4，可按服务器资源统一修改。路径替换成真实切分目录：

```bash
TRAIN_PARQUET='datasets/train/*.parquet'
VALID_PARQUET='datasets/valid/*.parquet'
SWEEP_ROOT='runs/m4_rum/d1_sweep'
mkdir -p "$SWEEP_ROOT/logs"

run_group() {
  local gpu="$1"
  shift
  local variant
  for variant in "$@"; do
    bash train_shell/train_m4_rum.sh --gpus "$gpu" --cpu_threads 4 \
      --config "forecast_backbones/m4_rum/config/d1_sweep/${variant}.yaml" \
      --train_parquet "$TRAIN_PARQUET" --valid_parquet "$VALID_PARQUET" \
      --out "$SWEEP_ROOT/$variant" \
      > "$SWEEP_ROOT/logs/$variant.log" 2>&1 || return 1
  done
}
run_group 0 q0_baseline q1_lr2e4 q2_wd001 q3_dropout005 & pid0=$!
run_group 1 c0_legacy q4_patch8 q5_width192 q6_ema099 & pid1=$!
status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
printf 'All workers finished; status=%s (0 means success)\n' "$status"
```

GPU 编号按服务器修改；`PYTHON=/path/to/python` 可选择环境。自定义站表、提示列时，所有组使用
相同的 `--station_meta`、`--hint_col`。第二轮更换输出根目录，不复用上轮目录覆盖结果。
先比较每组 `train_log.json` 的 `best` 和 `best_step`（越高越好，单位 0–100），
挑前 2–3 组再用 seed 1、2 重复，不能凭单次 seed 下很小的差距认定胜出。
最后对选定模型做 `--full` 测试，用 `reports/m8_comparable_summary.csv` 与 M8 对比；
不要按反复查看测试集的分数挑参数。

## 时间与评估口径

与本仓库 M8 完全相同：十五分钟一步，历史最后一个位置为起报时刻 S，
未来第 i 个位置为 `S + (i+1)*15min`。默认历史 672 点、预测 192 点。
每天 10:00 起报时，输出覆盖当日 10:15 至后天 10:00；次日完整一天是 `prediction[55:151]`。
NWP、基模预测和标签必须使用相同起点；不能传入从次日零点开始的数组。

- 训练使用所提供数据的全部起报行，与 M8 一致，不额外只筛 10:00。
- 南网 D+1：使用公共层 `score_nanwang_result` / `NanwangMetric.short_d1_daily`，
  每站每天选择不晚于 10:00 的最晚起报行，按其实际时间对齐次日，汇总站日平均。
- NRMSE：默认整点起报，全部 192 点、含夜间，使用固定容量归一。
- `--full`：全部起报行推理，使用公共层逐站工作簿生成函数，输出相同六张 sheet。
  完整超短期指标仍需要每十五分钟起报的数据，ODS 短期需要相应 09:00 起报行。
- 旧默认配置不截负值、不做容量/晴空钳位、不夜间归零；新对照配置可启用非负/容量裁剪。
  Pinball 模式的分位插值和 `p_star`
  数学口径与 M8 相同，使用 Rum 自己的实现，不带 M8 的太阳几何或物理后处理。

## 原 legacy 网络

```text
历史功率/固定容量 + 历史气象 + 有效值掩码
  → 分变量 Patch(16点、不重叠)，加入变量和 Patch 结束时间编码
  → 2 层联合 Transformer(宽128，4头，FFN256)
  → CTX Token → LayerNorm → 64维历史上下文

各未来时刻气象 + 掩码 + 时间4维 + 提前量1维
  → 共享气象 MLP(64 → 64)

[气象表示, 原始气象/时间特征, 历史上下文, 历史覆盖率/可用标记,
 基模预测/固定容量, 基模有效值掩码]
  → 共享 MLP(128 → 64 → 1或K) → 192点归一化功率/分位 → 乘固定容量
  → Pinball模式：分位排序 → median或p_star选点
```

默认两种气象变量时，历史含 126 个 Patch Token 和 1 个 CTX Token，点预测模型 310,529 个参数。
各时刻共享回归头，无未来自注意力、无自回归。损失为容量归一 MSE（可设 MAE），
先对每行有限标签平均，再对有标签的行平均；全无标签批次明确报错。
Pinball 模式只将末层从 1 维改为 K 个分位（默认 9 个），原历史编码器和气象融合方式保持不变。
历史上下文整行 dropout 默认 0.2，基模整行 dropout 默认 0.3，仅在训练态启用。
有限值掩码不代表观测可靠性，当前没有自动识别未标记填补值的能力，也未加入填补污染增强。

## 输入与容量

默认列：`timestamp_win`、`station_id`（或 `station`）、`observe_power`、
`observe_power_future`、`general_ghi(_predict)`、`general_temp(_predict)`、`timesfm3`。
`timesfm3` 自动兼容 `tsfm_predict`，天气和基模特征列支持大小写不敏感匹配。
气象按配置中的逐变量 offset/scale 归一；基模预测与功率必须同单位。

这里消费预先生成的基模预测列，不在训练循环或推理 forward 中运行/微调基模。
训练用的预测必须在各历史起报时刻生成，且不使用未来标签；微调过的基模应提供样本外预测。
未来气象也必须使用当次起报时已可得的 NWP 版本。

容量只读配置站表中的固定 `cap_mw`，不使用行内 `cap_power_on`。
默认表位于 `pipelines/forecasting/assets/`，可用 `--station-meta` 指定自己的表。
站点编号仅用于容量查表和分站报告，不送入网络；`known_stations` 仅区分已见/未见站报告。
新站需要补充站表名称、编号、固定容量，不要求训练过的站点 embedding。
经纬度不进入模型；标准站表的 lat/lon/alt_m 列可留空。
省级可将全省视为一个实体配置容量与聚合气象，单独训练权重。

时间沿用 M8 的 **naive 本地墙钟编码为 epoch 秒** 约定（2000–2099），
不会自动加八小时；来自 UTC 的数据须在进入此数据管线前统一时区。
历史不足 672 点时左侧补缺失，超过时取尾部；未来数组至少 192 点，超过时取前 192 点。

训练缺基模列直接报错；推理缺列时警告一次并启用 no-hint 路径。
历史功率全缺失时上下文归零。天气按历史/未来配对读取；训练完全缺失的气象对
写入 `features.cov_present` 后固定禁用，避免推理时突然接入未训练的通道。
`.aux` 包含 `history_unavailable`、`hint_unavailable`、`future_weather_unavailable`；
专用评估保留这些标记。通用 `unified_inference` 仅返回预测，不保留 aux。
所有天气不可用时虽仍可产生有限数值，业务应依据标记决定是否接受。

### 站表映射

训练、验证和推理共用 `pipelines.forecasting.stations` 的站表映射，与 M8 的规则一致，
不调用 M8 私有代码。CSV 支持 UTF-8/BOM、GBK；标准表头为
`station_id,station,lat,lon,alt_m,cap_mw`，坐标和海拔的值可留空。
线上 `info.csv` 的 `plantid / plant_pointname / LATITUDE / LONGITUDE / GCCAPACITY`
也可自动识别，分别映射到编号、站名、纬度、经度、固定容量，海拔默认 0。

- 数据已有整数 `station_id` 时直接查编号；文本数字编号在训练/评估加载时转为整数。
  非数字的文本编号按站名映射；没有编号列时用 `station` 补齐。
- 站名去首尾空格，`1002`、`1002.0`、`"1002.0"` 统一为 `"1002"`。
  按站名查找时先匹配站表的 `station`，找不到且为数字才尝试编号；普通文本精确匹配。
- 编号与站名重复、未知站名、映射冲突都会报错；容量必须是有限正数。
  已有编号时不会再用同时存在的站名列交叉核验，两列须由上游保持一致。

训练可用 `--station-meta /path/to/station_meta.csv`（或 shell 的 `--station_meta`）换表；
它会清空配方中的旧列名映射，自动识别标准表或线上表。自定义中文表头应在 YAML 中一起设置：

```yaml
dataset:
  station_meta: /path/to/stations.csv
  station_meta_cols:
    station_id: 场站编号
    station: 场站名称
    cap_mw: 装机容量
    lat: null
    lon: null
    alt_m: null
```

自定义映射时直接使用该 YAML，不追加会清空映射的 `--station-meta`。
训练保存的 `model_config.yaml` 会保留实际站表路径与映射，推理时须保证该路径可访问。

## 训练

在 NextM8 根目录执行。依赖沿用 M8 的 torch / transformers / accelerate / datasets /
pandas / pyarrow / safetensors / PyYAML；完整报表使用 xlsxwriter。Rum 不需要计算太阳几何。
测试验证环境为 Python 3.12、torch 2.5.1、transformers 4.40.1、accelerate 0.30.1。

```bash
# 独立训练集、验证集路径：无需 split_manifest.json，双卡训练并验证选优
bash train_shell/train_m4_rum.sh \
  --gpus 0,1 \
  --config forecast_backbones/m4_rum/config/configs.yaml \
  --train_parquet /path/to/train \
  --valid_parquet /path/to/valid \
  --out runs/m4_rum/rum_train_valid \
  --max_steps 6000 --save_every 500

# 与 M8 使用同一份 train.parquet、valid.parquet、split_manifest.json
python -m forecast_backbones.m4_rum.train \
  --splits-dir data/splits/m8_v1 --out runs/m4_rum/rum_s0

# 南网数据：configs.yaml 指向公共 assets/info.csv，列名映射与 M8 j1 相同
python -m forecast_backbones.m4_rum.train \
  --config forecast_backbones/m4_rum/config/configs.yaml \
  --train-parquet /path/to/train --out runs/m4_rum/full \
  --max-steps 6000 --save-every 500

# 自定义站表 / 基模列；station-meta 覆盖会清空线上表的列名映射
bash train_shell/train_m4_rum.sh \
  --gpus 0 \
  --splits-dir /path/to/splits --station-meta /path/to/stations.csv \
  --hint-col tsfm_predict --out runs/m4_rum/custom
```

默认值真源为 `config.py`，YAML 只需写覆盖项，未知配置键报错。
默认 AdamW、cosine、lr 3e-4、batch64、6000步、warmup300，
每250步验证，按 `nanwang_short_D+1` 保存最佳，patience8；也可选 `nrmse`。
通过公共层实现 EMA、平滑选点、station_day 采样和分布式停止协议，口径与 M8 一致。

与 M8 一样，通过启动脚本的 `--gpus` 选择单机训练用卡：

```bash
bash train_shell/train_m4_rum.sh \
  --venv-python /path/to/venv/bin/python \
  --gpus 0,1 \
  --config forecast_backbones/m4_rum/config/configs.yaml \
  --splits_dir /path/to/splits \
  --out runs/m4_rum/rum_2gpu
```

`--gpus 0` 为单卡，`--gpus 0,1,2,3` 为四卡；默认 `auto` 使用 PyTorch 当前可见的全部卡，
尊重外部 `CUDA_VISIBLE_DEVICES`。多卡时自动通过 `torch.distributed.run` 启动 DDP，
每张卡一个进程；`--port 29701` 可为并行实验换端口，`--cpu_threads 4` 设置每进程 CPU 线程。
解释器优先级为 `--venv-python` > `$PYTHON` > M8 服务器环境 `/home/ma-user/ts3/bin/python`。
选中的解释器不存在或不可执行时直接报错，禁止自动回退到 PATH 中的 `python`。
`eval_shell/eval_m4_rum.sh` 使用相同优先级，也支持 `--venv-python`，避免训练与评估环境不一致。
本地运行时显式设置 `PYTHON=/path/to/venv/bin/python` 或传 `--venv-python`。
无 GPU 时需显式加 `--allow-cpu`。
同时支持训练参数的连字符写法及上述 M8 风格下划线别名。

环境初始化与 M8 对齐：首次导入 PyTorch 探卡前，将
`/usr/local/cuda-12.1/lib64:/usr/lib/x86_64-linux-gnu` 加到 `LD_LIBRARY_PATH` 前，保留原有路径。
`PRE_CUDA_LD` 与 M8 一样在脚本中写死，不接受外部环境变量覆盖；探卡和训练使用同一解释器与库路径。
未指定 `--cpu_threads` 时沿用 `OMP_NUM_THREADS`，否则按可用 CPU 数除以训练进程数分配，至少为 1。
启动时打印实际解释器、库路径、GPU 和线程设置，便于核对服务器环境。

GPU 选择由启动脚本控制，训练超参数仍由 YAML 的 `train.schedule` 控制。
`batch_size` 是每张卡的大小：64 × 2 卡 = 全局 128，64 × 4 卡 = 全局 256；
如需保持单卡时的全局 batch，应相应减小此值。`precision` 可设 `fp32`、`fp16` 或 `bf16`，
BF16 需要硬件支持。启动逻辑已通过模拟可见卡测试，尚未进行真实 GPU/DDP 实测。

多卡进程在能力检测和 HF 初始化前按 `LOCAL_RANK` 绑定当前逻辑 CUDA 设备，
NCCL 同步显式使用该设备的 `barrier(device_ids=[...])`；CPU/Gloo 同步不传 GPU 编号。
例如 `--gpus 5,7` 对应两个进程内的 `cuda:0`、`cuda:1`，不是 `cuda:5`、`cuda:7`。
各进程会打印 rank、device 和可见卡设置，以及等待主进程准备数据缓存的状态。
若日志停在 `waiting for rank 0 data cache`，先查看 rank 0 是否仍在加载数据或已经报错；
仅凭首次 barrier 的设备未知警告，不能判断 CUDA 动态库配置错误或已经死锁。

`--train-parquet` 与 `--valid-parquet` 都支持递归目录、带引号的通配符和文件列表，
不自动切分或丢弃行。同时提供两者时按验证集指标选优、早停，无需预先生成切分清单；
会记录实际文件路径与行数，并拒绝训练/验证使用同一个文件。不同文件之间的样本重叠需上游保证。
验证路径不能与 `--splits-dir` 或 `--refit` 混用。
只给 `--train-parquet` 时仍为全量训练，不验证、不早停，必须给正的步数预算。
`--splits-dir ... --refit` 合并 train 与 valid。
切分和基模样本外预测由上游负责，Rum 不修复已有数据泄漏。

产物与 M8 布局一致：`model.safetensors`、`model_config.yaml`、`train_log.json`、
`split_manifest.json`、`reports/summary.csv`；有验证时另有 `result_val.parquet`、
`reports/station_table.csv`。每 `save_every` 步及末步输出可独立加载的
`checkpoints/step_XXXXX/`，主目录保存验证最佳权重（refit 保存末步权重）。
这些是推理 checkpoint，不含恢复优化器训练所需的完整状态。

训练入口始终从随机初始化开始；不要用相同输出目录混合不同实验。
权重、训练实际使用的天气对、已见站名单及配置覆盖都会保存。

## Pinball loss 与 p_star

旧配置仍默认 `train.loss: mse`，也支持 `mae`。使用下面的独立配方启用分位训练：

```bash
bash train_shell/train_m4_rum.sh \
  --gpus 0,1 \
  --config forecast_backbones/m4_rum/config/pinball.yaml \
  --train_parquet /path/to/train --valid_parquet /path/to/valid \
  --out runs/m4_rum/rum_pinball --max_steps 6000 --save_every 500
```

关键配置：

```yaml
model:
  quantiles: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
train:
  loss: pinball
decision:
  point: {default: p_star, floor_ratio: 0.2, tau_grid_step: 0.005, tail: flat}
```

训练在容量归一尺度上计算 `max(τ·(y−q), (τ−1)·(y−q))`，先对分位平均，
再对每行的有效标签时间步平均，最后对有标签的行平均，保持 Rum 的逐样本等权口径。
损失使用原始分位；排序和选点不参与训练损失。旧 MSE/MAE 权重继续可用，
但不能直接变成分位模型，切换到 Pinball 需要重新训练新的 K 维输出头。

推理先将分位排序，按等概率中点网格线性插值、两端平延；默认步长 0.005，共 200 点。
随后计算 `p_star = Σ(y·w) / Σw`，其中 `w = 1 / max(y, 0.2·容量)²`，
与 M8 短期 `p_star` 公式一致。这里只对齐选点公式，完整输出仍受两模型的分位预测和物理后处理差异影响。
`floor_ratio` 可配置正有限值，网格步长支持 `[0.0001, 1]`，只支持平延尾部。
该模式的验证和早停使用所配置的选点结果，默认配方为 `p_star`。

同一个 Pinball checkpoint 可在推理时切换选点，无需重训：

```bash
python -m forecast_backbones.m4_rum.evaluate \
  --runs runs/m4_rum/rum_pinball --test /path/to/test \
  --point p_star --out runs/m4_rum/eval_pstar
# 改为 --point median 可对比中位数；省略 --point 使用 checkpoint 保存的规则。
```

`load_member(run, point="p_star")` 也支持覆盖，普通 forward/统一推理按 `model_config.yaml` 中的规则选点。
点预测 checkpoint 不支持 `--point`。`.logits` 始终为 `[B,192]` 原功率单位；
Pinball 的 `.aux` 另含 `quantiles`（原始）、`quantiles_cal`（仅排序）、`levels_cal`、`median`、`p_star`。
结果 Parquet 保存 `q_0.1` 等原始分位列、`median`、`p_star`；评估目录额外记录 `reports/decision.json`。

## 推理与评估

```bash
python -m forecast_backbones.m4_rum.evaluate \
  --runs runs/m4_rum/rum_s0 --test /path/to/test --out runs/m4_rum/eval

bash eval_shell/eval_m4_rum.sh \
  --runs runs/m4_rum/rum_s0 runs/m4_rum/rum_s1 \
  --test '/path/to/test/*.parquet' --out runs/m4_rum/full_eval --full
```

点预测多成员直接平均点预测；Pinball 多成员先平均原始分位，再统一排序和选点。
集成核对行序、站点容量、标签一致；Pinball 成员还必须有相同分位水平和决策配置。
不混合点预测与 Pinball checkpoint。
产出 `result_test.parquet`、`reports/summary.csv`、`reports/station_table.csv`、
`reports/members.csv`。`--full` 额外产出 `metrics_ens/{站}-metrics.xlsx`、
`{站}-summary.json`、`reports/station_metrics.csv`。
汇总中的 `nanwang_short_D+1` 保持站日平均，`station_mean_*` 显式表示逐站等权平均。

注册名为 `m4_rum`，通过现有 `pipelines.model_registry.build_model` 自动发现，
兼容 `forward(batch_dict, labels=None)` 与 `unified_inference`。
训练请使用上述 Rum 专用入口，不用旧的 `train_shell/train_script.py` 分支表。

```bash
python -m pytest forecast_backbones/m4_rum/tests -q
```

## 复用边界

Rum 的网络、特征、训练、评估均不得导入其他模型的私有实现。
配置序列化、站表/时间网格、parquet 读取、行选择、训练调度、checkpoint 回调及指标报表
已抽到公共包 `pipelines/forecasting/`。该公共包也不得反向导入 `forecast_backbones`。
M8 原有模块保留兼容入口，调用同一份公共实现，不维护两份框架逻辑。

部署只需 `forecast_backbones/m4_rum`、`pipelines`、`interfaces` 及配置引用的站表，
**不需要 M8 目录、网络、权重或任何 M8 私有代码**。
`tests/test_dependencies.py` 检查静态导入边界，并在不包含 M8 的临时部署目录里
禁止导入其他模型，实际运行训练、checkpoint 重载、统一推理和完整报表评估。
