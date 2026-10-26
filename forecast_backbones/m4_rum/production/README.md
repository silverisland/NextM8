# M4-Rum 生产二轮复验

本地分数用于筛选候选，不能直接视为生产提分保证。保留所有仍在使用的实验配方，
通过同一个 `pipelines.pipeline_wrapper_inference.unified_inference` 验证实际生产输入。
训练期间验证、独立评估、无标签推理都使用这个引擎。网络、Pinball、p_star、
分位数集成、选模指标和指标聚合规则保持原样，不会自动重训或替换已有权重。

## 保留清单

机器可读清单：[experiments.yaml](experiments.yaml)。引用原 YAML，不维护第二份参数副本。

| 方案 ID | 数量 | 保留用途 |
| --- | ---: | --- |
| `q0`–`q9`、`c0` | 11 | 基础、学习率、正则、Patch、容量、EMA、站点信息对照 |
| `a0`–`a18` | 19 | 四种分支全部组合，以及容量、站点和输出头对照 |
| `a17_epoch` | 1 | a17 按 epoch 验证早停；预算最多 20 轮，非 6000 步等预算对照 |
| `legacy_default`、`legacy_pinball`、`legacy_control`、`query_control` | 4 | 原始配置与结构控制组 |

共 35 份配置；`a0` 与 `q6` 参数等价，仅运行标识不同，通常选一个即可。
平滑实验继续归档，不进入二轮候选。已清理的 a19/a20/a21 不恢复。
所有候选只表示保留复验资格；不承诺其生产收益。

默认执行 `q6 a16 a17`，可用 `--experiments` 自选，也可用 `all` 选择全部。

```bash
# 以下命令均在 NextM8 根目录执行
bash train_shell/compare_m4_rum.sh --mode list --experiments all
```

所有脚本的解释器优先级为 `--venv-python` > `$PYTHON` > `/home/ma-user/ts3/bin/python`。
解释器不可执行即退出，不回退。HF_DATASETS_CACHE 尊重外部值，未设置时使用服务器 work 缓存。
新脚本没有另外的环境切换逻辑；真实 GPU 多卡性能仍需在服务器验证。

## 第一种测试：已有权重直接回放生产数据

用于测试现有训练结果迁移到生产输入后的表现，不修改权重，也不拿候选模型做集成。
复制并编辑 [run_map.example.yaml](run_map.example.yaml)，给每个候选指定完整运行目录，
其中必须有 `model_config.yaml` 和 `model.safetensors`。可指向某个保存的 checkpoint 子目录。
路径建议使用绝对路径；相对路径按仓库根目录解析。

```bash
bash train_shell/compare_m4_rum.sh --mode evaluate \
  --experiments q6 a16 a17 a17_epoch --baseline q6 --seeds 0 \
  --run-map /path/to/production_runs.yaml \
  --test '/path/to/frozen_replay/*.parquet' \
  --station-meta /path/to/station_meta.csv \
  --device cuda:0 --batch-size 32 \
  --out reports/m4_rum/round2_transfer
```

`--station-meta` 可选：跨机器迁移时，覆盖保存配置中的旧站表路径，自动识别标准列名或
info.csv 表头。自定义列名仍需在保存配置中正确配置映射，并使用其原路径而不传此覆盖选项。
应提供每个候选都能使用的同一份站表；未知训练站可走公共 embedding，但容量必须在表中。

推理以 checkpoint 的实际配置为准；清单里的 ID 是实验标签，映射必须对应正确的训练产物。
工具记录实际配置、权重 SHA-256、站表 SHA-256 和保存的种子，并拒绝种子不匹配。
不在评估时更换 hint 语义；上游需提供起报时生成的对应预测，或使用已配置的列名别名。

## 第二种测试：生产训练数据重训后对比

与迁移测试分开建立目录。所有方案使用相同训练/验证切分，保留各自原始训练配方。
同一个命令中的候选按顺序执行；多卡训练由已有 `train_m4_rum.sh` 调用 torchrun。

```bash
bash train_shell/compare_m4_rum.sh --mode train \
  --experiments q0 q6 a4 a7 a9 a14 a15 a16 a17 a17_epoch \
  --seeds 0 --gpus 0 --cpu-threads 4 \
  --train '/path/to/production_train/*.parquet' \
  --valid '/path/to/production_valid/*.parquet' \
  --station-meta /path/to/station_meta.csv \
  --run-root runs/m4_rum/production_round2 \
  --out reports/m4_rum/round2_train

bash train_shell/compare_m4_rum.sh --mode evaluate \
  --experiments q0 q6 a4 a7 a9 a14 a15 a16 a17 a17_epoch \
  --baseline q6 --seeds 0 \
  --run-root runs/m4_rum/production_round2 \
  --test '/path/to/frozen_replay/*.parquet' \
  --device cuda:0 --batch-size 32 \
  --out reports/m4_rum/round2_retrained
```

运行目录为 `<run-root>/<ID>/seed_<N>`。需要不同种子时传 `--seeds 0 1 2`；
基线与候选按相同种子配对。训练必须提供验证集，不将全量 refit 混入本轮对照。
可在训练命令上加 `--hint-col` 改列名；不会生成 hint，需由上游提供。
无站表训练可加 `--no-station-meta --station-key station --capacity-col cap_power_on`。
有站表且按站名匹配则加 `--station-meta /path/to/stations.csv --station-key station`，
CSV 可仅含 `station,cap_mw`。自定义 CSV 容量列在原 YAML 的 `station_meta_cols.cap_mw` 中配置，
不要再用会清空该映射的 `--station-meta` 覆盖。评估从 checkpoint 读取容量来源和匹配规则。
数据容量为浮点数时直接使用，为 192 点列表时严格取首值；不存在的非空 CSV 路径不会自动降级。
每次命令的 `--out` 必须是空目录，训练运行目录非空也会拒绝；不要覆盖前轮产物。
两卡分别运行队列时指定不同 `--gpus` 和 `--out`，多卡队列另用不同 `--port`，
同一 ID/seed 的运行目录不能被两个队列同时使用。
不要把 `eval_m4_rum.sh --runs runA runB` 当成候选对比：该旧用法是模型集成。

## 读取结果

- `comparison.csv`：每个候选/种子的状态、两项站均指标、相对基线 delta、缺失输入比例、
  评估总耗时（含加载、推理和报表，不是纯模型延迟）、产物路径。
- `station_comparison.csv`：每站超短期和 D+1 分数及对应 delta。
- `comparison_manifest.json`：候选、输入文件路径/大小/mtime、代码 commit/工作区状态及源文件 SHA-256。
  输入清单不是全量 parquet 内容哈希；请使用只读且版本固定的回放目录。
  部署包不含 Git 时版本字段为空，仍记录源文件 SHA-256。
- `<ID>/seed_<N>/checkpoint_manifest.json`：实际使用的配置、权重、站表校验值。
- `<ID>/seed_<N>/evaluation/`：原有完整评估产物，包括原始分位数、点预测、逐站 Excel 和指标。
- `<ID>/seed_<N>/status.json`：成功/失败及具体原因；训练另有 `requested_config.yaml`、`command.json`、`train.log`。

分数仍按项目的 **0–100** 口径，`*_delta_pp` 是百分点，不把 0.78 与 78 混用。
重点读取 `station_mean_nanwang_short_D+1` 与 `station_mean_nanwang_ultra_short`。
早停仍用各原配方的 `nanwang_short_D+1`（站日平均），没有偷偷改成站均月报或超短期选模。
无 CSV 且每行容量不同时，D+1 使用起报行容量，超短期使用目标时刻真值来源行的容量。
这种情况下完整报表含南网月报，固定容量版 ODS/DS 指标留空并标注原因；容量固定时保持原报表。
各候选应采用同一容量来源；容量不同的行签名不会计算相对 delta。

仅在同种子、相同行顺序、站号、容量及原始真实值匹配时输出 delta；
否则保留各自分数，`comparable_to_baseline=false`、delta 留空。
站均月报需要足够有效样本；超短期指标还要求连续的 15 分钟起报记录形成 16 个提前量，
不能只提供每小时一次的预测。无有效月报时指标及 delta 保持缺失，并填写 `missing_metrics`。
模型推理异常、非有限输出等使该候选失败，记录后继续其他候选，最终返回非零退出码；
缺少有效站均指标同样返回非零退出码。失败候选不会被静默丢弃后参与排名。
该工具采用严格离线回放，不提供逐失败行的可用性 SLA；线上影子任务应另外统计任务失败率和延迟。

## 无标签生产推理

```bash
bash eval_shell/infer_m4_rum.sh \
  --run /path/to/a17_run \
  --input '/path/to/live_inputs/*.parquet' \
  --station-meta /path/to/station_meta.csv \
  --device cuda:0 --batch-size 32 \
  --out /path/to/new_predictions.parquet
```

不需要 `observe_power_future`；若有也不送入模型。输出含 `observe_power_predict`、原始分位数、
median/p_star、站点/容量及缺失诊断，不计算指标。拒绝覆盖已有输出文件。
生产数据需有起报时间、站号或站名、历史 PV、未来天气和已有 hint；
模型支持缺测兜底，但缺失率应一起观察，不能把训练中完整输入的效果直接外推。
默认 672/192 点、15 分钟粒度。训练未见站点使用公共槽位；站表模式需补站表，
无站表模式由输入数据提供容量。无站表 checkpoint 的推理命令省略 `--station-meta`。
预测为原始功率单位，已执行配置中的选点和裁剪，不要在业务端重复乘容量或执行 p_star。

## 生产 Python 接入 demo：直接调用 unified_inference

可运行代码见 [demo_unified_inference.py](../demo_unified_inference.py)。
它直接调用公共 `unified_inference`，不经过评估脚本，不计算评估指标。
服务初始化时加载一次模型，后续请求传入 DataFrame：

```python
from forecast_backbones.m4_rum.evaluate import load_member
from forecast_backbones.m4_rum.demo_unified_inference import predict_dataframe

# 初始化：加载训练时保存的完整配置和权重，兼容 a17 等各候选。
model = load_member('/path/to/a17_run', device='cuda:0')

# 每次请求：input_df 是业务侧准备好的窗口表，不需要未来真实功率。
pred = predict_dataframe(model, input_df, batch_size=32)
curves = pred['observe_power_predict']  # 每行 ndarray，默认长度 192，单位与训练功率一致。
pred.to_parquet('/path/to/predictions.parquet', index=False)
```

模型目录必须含 `model_config.yaml`、`model.safetensors`，不能只用实验 YAML。
checkpoint 的 `known_stations`、容量来源、列名映射和 decision 配置均保留。
如需跨机器迁移标准列名站表，可给 `load_member` 传 `station_meta='/path/to/stations.csv'`；
该选项会清空自定义 CSV 列名映射，使用自定义映射时应保留保存配置并将 CSV 放在其指定路径。
无 CSV 的 checkpoint 不传此参数，容量来自配置中的 `capacity_col`，列表严格取第一个值。

`predict_dataframe` 中依次完成：去掉标签 → 调用训练侧站点/容量适配器 →
直接 `unified_inference` → 将模型类名预测列改为 `observe_power_predict`。
输出含起报时间、整数站号及输入中存在的站名/自定义站点键，行序与输入一致。
模型已经执行保存的 p_star/median 等选点与裁剪，业务端不用再次处理分位数或乘容量。
示例只返回最终点预测；若要原始分位数和缺失诊断，可用下文的 `predict_run`。

默认输入约定（自定义列名以 checkpoint 配置为准）：

| 字段 | 每行内容 |
| --- | --- |
| `timestamp_win` | 起报时间 S，与训练保持同一时区口径 |
| `station_id` 或 `station` | 按保存的站点匹配规则提供 |
| `observe_power` | 截至 S 的 672 个历史功率值 |
| `general_ghi`、`general_temp` | 对齐历史窗口的 672 个气象值 |
| `general_ghi_predict`、`general_temp_predict` | 起报时已知的未来 192 点气象预测 |
| `timesfm3`（或配置别名 `tsfm_predict`） | 起报时已有的未来 192 点 hint |
| `cap_power_on`（实际以 `capacity_col` 为准） | 无 CSV 时必需；正数或列表，取首值 |

序列使用数值数组/列表，不是 JSON 字符串。默认每点 15 分钟，预测首点为 S+15 分钟。
业务侧需提前构造窗口和 hint；本示例不从原始长表生成它们，也不需要 `observe_power_future`。

也可以直接运行 demo 验证生产 Parquet（在 NextM8 根目录，显式指定实际解释器）：

```bash
/home/ma-user/ts3/bin/python -m forecast_backbones.m4_rum.demo_unified_inference \
  --run /path/to/a17_run \
  --input '/path/to/live_inputs/*.parquet' \
  --device cuda:0 --batch-size 32 \
  --out /path/to/new_predictions.parquet
```

命令行拒绝覆盖已有输出；默认设备是 CPU，GPU 需显式指定，不自动回退。
CLI 会把所选 Parquet 读入内存，大数据建议业务侧分批读取后复用模型调用 `predict_dataframe`。

也可调用 `forecast_backbones.m4_rum.infer.predict_run` 获取与评估相同的标准列和诊断。
正式推理使用 `load_member` / `predict_run` 校验已加载权重，不直接用实验 YAML 构造随机模型。
公共 `unified_inference` 的默认逐样本 fallback 行为保持兼容；M4-Rum 二轮复验明确启用
`error_policy='raise'`，通过 `on_batch_output` 一次前向收集分位数与诊断，无额外模型推理。

## Mock 训练与生产推理接口检查

[smoke_production.py](../smoke_production.py) 可重复生成符合上述约定的生产窗口表，
保持 a17 完整网络、Pinball 和选点规则，以少量 CPU 训练步骤检查接口：

```bash
/home/ma-user/ts3/bin/python -m forecast_backbones.m4_rum.smoke_production \
  --out /path/to/new_mock_check
```

运行时显式使用当前解释器，不切换环境；输出目录必须不存在。
三组案例为 CSV 按 `station` 匹配、无 CSV 标量容量、无 CSV 列表容量。
列表案例使用自定义容量列名，首值与后续值不同；CSV 案例故意提供不一致的数据容量。
历史 672 点、未来 192 点，使用生产天气大写字段和 `tsfm_predict` 别名。
训练包含两个站，验证和推理多一个训练未见站；hint 使用历史日周期生成。

每组实际执行训练 shell、完整评估 shell 和直接调用 `unified_inference` 的 Python demo。
检查权重保存/回载、最佳 checkpoint 一致性、step/epoch 验证、容量来源、未知站公共槽位、
无标签推理、行序、192 点有限输出，以及评估与生产预测的一致性（允许浮点误差）。
`summary.json` 保存结果，各组目录保留 Parquet、CSV、配置、命令、日志、权重、报表及预测。
完整报表需安装 `xlsxwriter`，读取 Excel 的测试还需 `openpyxl`。
该检查只验证 CPU 单进程、单次 192 点预测接口；mock 数据和短训练预算不能用于判断精度，
不包含 GPU/DDP 或 480 点 rolling 链路验收。
