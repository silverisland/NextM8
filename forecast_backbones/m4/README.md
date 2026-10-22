# M4：独立 PatchTST 训练与推理

M4 保留本目录的 PatchTST 主干、历史功率与基模预测拼接、历史/未来气象拼接、
RevIN 和 embedding 预测头。训练入口、配置、特征构造、权重和评估编排均属于 M4，
不导入 m1、m8、m4_rum 或其他模型的私有实现。
仅复用 `pipelines/forecasting` 的数据加载、站表、调度、EMA、checkpoint 和指标报表。
默认站表来自公共 `pipelines/forecasting/assets/station_meta.csv`，不引用 M8 的文件。

## 训练

在 NextM8 根目录执行：

```bash
# 独立训练/验证路径；支持目录、文件列表或加引号的通配符
bash train_shell/train_m4.sh --gpus 0,1 \
  --config forecast_backbones/m4/config/configs.yaml \
  --train_parquet /path/to/train --valid_parquet /path/to/valid \
  --out runs/m4/patchtst_s0 --max_steps 6000 --save_every 500

# 已切分的数据目录：train.parquet、valid.parquet、split_manifest.json
bash train_shell/train_m4.sh --gpus 0 \
  --splits_dir /path/to/splits --out runs/m4/presplit_s0

# 全量训练：只给训练数据，无验证/早停，必须有正的 max_steps
bash train_shell/train_m4.sh --gpus 0 \
  --train_parquet '/path/to/train/*.parquet' \
  --out runs/m4/refit_s0 --max_steps 6000

# 自定义站表和已有基模预测列
bash train_shell/train_m4.sh --gpus 0 \
  --train_parquet /path/to/train --valid_parquet /path/to/valid \
  --station_meta /path/to/info.csv --hint_col timesfm3 \
  --out runs/m4/timesfm_s0
```

也可 `python -m forecast_backbones.m4.train`，Python 入口使用连字符参数名，
shell 同时接受上例中的下划线写法。`--splits-dir ... --refit` 合并 train 与 valid。
训练从随机初始化开始，不自动恢复已有权重；非空输出目录直接拒绝，防止覆盖其他实验。
每次实验指定独立 `--out`；默认是 `runs/m4/patchtst`。

解释器优先级：`--venv-python` > `$PYTHON` > `/home/ma-user/ts3/bin/python`。
不可执行时明确退出，不回退到其他环境。训练与评估相同。
本地测试显式传 `--venv-python /path/to/python`；CPU 训练还要加 `--allow-cpu`。
GPU 默认 `auto`，尊重外部 `CUDA_VISIBLE_DEVICES`；多卡用 torchrun，每卡一个进程。
M4 默认端口 29704；同时启动多个 M4 实验时用 `--port` 分开。
`--cpu_threads` 控制每进程线程数；CUDA 库路径和 HF 缓存设置沿用项目启动约定。

超参数在 YAML 的 `train.schedule` 下设置，`batch_size` 是每卡 batch。
默认 lr=1e-4、batch64、6000步、cosine、warmup300；每250步验证，每500步快照。
默认按 `nanwang_short_D+1` 选最优并早停；可改 `train.early_stop.metric: nrmse`。
支持 `train.ema_decay` 和 `train.schedule.sampling: station_day`。
全部默认值仅定义在本模型 `config.py`，YAML 只写覆盖项，未知字段报错。

## 推理和评估

```bash
bash eval_shell/eval_m4.sh \
  --runs runs/m4/patchtst_s0 --test /path/to/test \
  --out runs/m4/eval_s0 --full

# 多个 M4 checkpoint 直接平均点预测，也可填 checkpoints/step_00500
bash eval_shell/eval_m4.sh \
  --runs runs/m4/patchtst_s0 runs/m4/patchtst_s1 \
  --test '/path/to/test/*.parquet' --out runs/m4/eval_ensemble --full
```

等价入口 `python -m forecast_backbones.m4.evaluate`，支持 `--device cpu`、
`--batch-size`、`--no-progress`。只能加载 M4 配置和权重，不接受 M8/Rum checkpoint。
`load_member(run, device='cpu')` 严格加载权重；移动整个运行目录后可重新定位权重。
模型兼容 `build_model('m4', cfg)`、旧注册别名和 `unified_inference`。
直接构造新模型时 `train.weights_path: null`；显式指定不存在的权重会报错。

训练产物：`model.safetensors`、`model_config.yaml`、`train_log.json`、
`split_manifest.json`、`reports/summary.csv`；有验证时还有 `result_val.parquet` 和
`reports/station_table.csv`。主目录保留验证最佳权重，全量训练保留末步权重。
`checkpoints/step_XXXXX` 可独立推理；包含模型配置/权重，不包含优化器恢复状态。

评估产物：`result_test.parquet`、`reports/summary.csv`、`station_table.csv`、
`members.csv`。`--full` 推理全部起报行，额外输出每站六页 Excel、JSON、
`station_metrics.csv` 和 `m8_comparable_summary.csv`。
后者为逐站报告均值；checkpoint 使用的南网 D+1 为站日均值，二者有明确区分。

## 输入、损失与时间口径

默认历史672点、未来192点、15分钟间隔。历史最后一点是起报 S，未来第一点为 S+15分钟。
10:00 起报的次日全天对应预测 `[55:151]`，与 M8/Rum 一致。
训练使用所有输入起报行；验证用公共 D+1 行选择及整点192点 NRMSE。

默认列：`timestamp_win`、`station_id`（或 `station`）、`observe_power`、
`observe_power_future`、`general_ghi(_predict)`、`general_temp(_predict)`、`chronos`。
使用其他已有基模预测列时传 `--hint-col`；`timesfm3` 支持别名 `tsfm_predict`。
不会运行或微调基模，输入预测须由上游按起报时间生成。
缺少基模列直接报错；有限值缺失位置补0，并在推理 aux/结果中保留缺失标记。
历史短序列左补缺失，长序列取尾部；未来必须至少192点，取前192点。
训练中完全不存在的气象对保存为禁用通道，推理不会自动启用未训练通道。
PatchTST 不额外输入缺失掩码，没有 Rum 的缺失分支机制。

输入预处理和评估容量读站表的固定 `cap_mw`；主损失的容量保持原 M4 规则，见下文。
标准站表及线上 `info.csv` 均通过公共站表接口识别；训练保存实际站表路径，
部署时须携带该站表或更新配置中的路径。
损失只有原 M4 主损失：`criterion(prediction / capacity, target / capacity)`，
默认 MAE，可选 `train.loss: mse` 或 `huber`（Huber 默认 delta=1）。
预测和标签均以原功率单位传入损失。容量优先取 `dataset.label_norm` 的第一列，
默认 `cap_power_on`；列缺失或 `label_norm` 为空时逐位置使用 `max(target, 1)`，
与原主损失一致。容量支持标量、每行容量和逐时间步容量。
直接对所有 batch/时间位置求 mean，不再改成带掩码的逐行平均；缺失标签会保留
原 criterion 的 NaN 行为，训练标签需由上游保证有效。
不乘 `lambda_main`，不添加 `PVNormalizedLoss` 或分位损失。
独立实现位于 `m4/losses.py`，容量仅在有标签的损失计算中使用，不参与推理。
不默认裁剪负值或夜间归零，输出恢复原功率单位。

旧配置备份为 `config/configs_legacy.yaml`，仅作历史参考。当前按主损失训练，
不依赖其他模型的损失实现；旧备份中的复合损失系数不用于新入口。
新配置、固定容量预处理及 checkpoint 格式需要重新训练，旧权重不自动迁移。
训练请使用 `train_m4.sh`，不使用旧 `train_shell/train_script.py`。

## 验证

```bash
/path/to/python -m pytest forecast_backbones/m4/tests -q
```

测试覆盖训练选优、全量训练、EMA、严格重载、迁移运行目录、统一推理、
原主损失数值/梯度、完整报表、解释器优先级和 GPU 启动参数。
独立部署测试仅复制 M4、pipelines、interfaces，并拦截所有其他模型导入，
实际完成训练、推理与完整报表。

本次在显式指定的 Python 3.12 / torch 2.5.1 / transformers 4.40.1 环境验证：
训练、验证选优、EMA 和快照重载由工作流测试覆盖。
GPU 选择与启动参数由模拟测试覆盖，尚未进行真实 CUDA 多卡训练。
