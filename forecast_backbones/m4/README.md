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

## 四套训练方案

四套 YAML 均在本模型 `config/` 下，使用相同网络、原始 quantile 主损失、
默认 seed=2025、每卡 batch64、lr=1e-4、cosine 和 warmup300。
EMA 只在第三套开启；每套默认输出目录独立，不依赖其他模型。

| 配置文件 | 验证与早停 | 训练上限 | 适用场景 |
| --- | --- | --- | --- |
| `epoch_earlystop.yaml` | 每轮验证；连续5轮未改善停止 | 100 epoch | 希望完整遍历训练集后再评估 |
| `step_earlystop.yaml` | 每250 step验证；连续10次未改善停止 | 6000 step | 数据较大，需要在一轮内检查验证效果 |
| `step_ema.yaml` | 同上，使用 decay=0.99 的 EMA 权重验证和选优 | 6000 step | 与普通 step 方案对比权重平均效果 |
| `refit_steps.yaml` | 全量模式不验证、不早停 | 6000 step，可覆盖 | 验证选好训练预算后重训 |

在项目根目录执行，替换解释器及数据路径。前三套必须提供验证数据：

```bash
# 方案一：按 epoch 早停
bash train_shell/train_m4.sh --gpus 0 --venv-python /path/to/python \
  --config forecast_backbones/m4/config/epoch_earlystop.yaml \
  --train-parquet /path/to/train --valid-parquet /path/to/valid \
  --out runs/m4/epoch_earlystop_s0

# 方案二：按 step 早停
bash train_shell/train_m4.sh --gpus 0 --venv-python /path/to/python \
  --config forecast_backbones/m4/config/step_earlystop.yaml \
  --train-parquet /path/to/train --valid-parquet /path/to/valid \
  --out runs/m4/step_earlystop_s0

# 方案三：按 step 早停 + EMA
bash train_shell/train_m4.sh --gpus 0 --venv-python /path/to/python \
  --config forecast_backbones/m4/config/step_ema.yaml \
  --train-parquet /path/to/train --valid-parquet /path/to/valid \
  --out runs/m4/step_ema_s0

# 方案四：固定步数全量训练；不传 valid，数据路径指向准备好的全量数据
bash train_shell/train_m4.sh --gpus 0 --venv-python /path/to/python \
  --config forecast_backbones/m4/config/refit_steps.yaml \
  --train-parquet /path/to/full_train \
  --out runs/m4/refit_steps_s0 --max-steps 6000
```

已有切分目录时，前三套用 `--splits-dir /path/to/splits` 替代 train/valid 参数；
第四套用 `--splits-dir /path/to/splits --refit` 合并训练集和验证集。
全量模式由输入参数决定，单独选择 `refit_steps.yaml` 不会自动合并数据。
全量重训后的最终效果应在独立测试集上评估。

epoch 方案同时设置 `max_steps: null` 和 `eval_every_steps: null`，分别表示
按 `epochs_max` 限制总轮数、在每轮结束验证。不要再传正数 `--max-steps`，
否则会覆盖轮数预算。`save_every: 0` 只关闭定期快照，仍会保存最佳权重。

step 是优化器更新次数；`patience` 始终是连续未改善的**验证次数**。
例如每250步验证、patience=10，通常在最佳点后2500步未改善时停止；
达到总步数上限也会结束，并在末步验证。每次严格改善才重置 patience，
`smooth_k: 1` 表示直接比较本次分数。两类早停均按南网 D+1 分数越高越好，
主目录保存验证最佳权重。可将 `train.early_stop.metric` 改为 `nrmse`，
此时按越低越好选优，训练损失仍保持原公式。

若想调整 epoch 数或 patience，直接改对应 YAML 的 `epochs_max` 和
`train.early_stop.patience`；step 验证间隔改 `eval_every_steps`。
当前每步全局 batch 为每卡 batch × GPU 数，改变卡数会改变每轮 step 数，
也会改变固定 step 预算内看到的样本量。小数据实验需相应缩短 warmup300。
全量训练的6000步只是起始值，应结合验证最佳步数、训练集扩充比例和全局 batch
确定预算；例如保持 batch 不变且希望保持相同数据遍历次数时，按样本数比例放大步数。

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

输入预处理和评估容量读站表的固定 `cap_mw`。
标准站表及线上 `info.csv` 均通过公共站表接口识别；训练保存实际站表路径，
部署时须携带该站表或更新配置中的路径。

损失按用户提供的原始公式实现，配置为 `train.loss: quantile`：

```python
error = y_true - y_pred
loss = 0
for quantile in np.arange(0.1, 1.0, 0.1):
    loss += torch.mean(torch.max(quantile * error, (quantile - 1) * error))
```

这里修正了原示例的拼写 `np.arrange` → `np.arange`。九个分位共用同一组
`[B,192]` 点预测，每项对 batch 和时间点取 mean，再将九项求和，不除以9。
对称分位求和在数学上等价于 `4.5 * MAE`；实现保留原循环和浮点计算顺序，
并未增加九分位输出头。独立实现位于 `m4/losses.py`。

当前 forward 将恢复原功率单位的 `logits` 与原功率标签传入该函数；损失内部
不再除以容量，不读 `cap_power_on`，不使用 `max(target, 1)` 兜底，
不添加南网损失或额外权重。缺失标签仍按原公式传播 NaN，训练标签需由上游保证有效。
不默认裁剪负值或夜间归零。

此前迁移版本将修改后的 MAE/MSE/Huber 主损失误当成原始设计，现已用上述用户
提供的公式替换。旧迁移配置中的 `train.loss: mae/mse/huber` 和
`dataset.label_norm` 不再支持：请改为 `train.loss: quantile` 并删除 `label_norm`。
这次仅改变损失及其配置，不改变网络参数形状和推理前向；已有迁移版权重在更新
配置后仍可推理，但若要得到原公式训练的模型，需要重新训练。
`config/configs_legacy.yaml` 保留为旧格式参考，不由新入口直接加载。

训练请使用 `train_m4.sh`，不使用旧 `train_shell/train_script.py`。

## 验证

```bash
/path/to/python -m pytest forecast_backbones/m4/tests -q
```

测试覆盖训练选优、全量训练、EMA、严格重载、迁移运行目录、统一推理、
用户原始分位求和公式的数值/梯度、完整报表、解释器优先级和 GPU 启动参数。
独立部署测试仅复制 M4、pipelines、interfaces，并拦截所有其他模型导入，
实际完成训练、推理与完整报表。

本次在显式指定的 Python 3.12 / torch 2.5.1 / transformers 4.40.1 环境验证：
训练、验证选优、EMA 和快照重载由工作流测试覆盖。
GPU 选择与启动参数由模拟测试覆盖，尚未进行真实 CUDA 多卡训练。
