# M4-Rum 训练策略对照

## a17_epoch_earlystop

以 `../arch_sweep/a17_all_large_station.yaml` 为基线，模型、数据特征、损失、EMA、
选模指标及其他训练参数不变，仅调整训练预算和验证频率，并使用独立输出目录。

```yaml
train:
  schedule:
    max_steps: null
    epochs_max: 20
    eval_every_steps: null
    save_every: 500
  early_stop:
    metric: nanwang_short_D+1
    patience: 8
    smooth_k: 1
```

每个 epoch 完成后验证一次，以 `nanwang_short_D+1`（站日平均）越高越好选模；
连续 8 次 epoch 末验证无改善则早停，最多训练 20 个 epoch。
验证及最优权重保存继续使用 EMA=0.99，结束后回载最佳权重。
`station_mean_nanwang_short_D+1` 与 `station_mean_nanwang_ultra_short` 可在完整评估中查看，
但本实验没有切换到这两个指标选模。

每 500 步的 checkpoint 快照照常保留，快照保存不触发验证或早停。
`max_steps: null` 取消原来的 6000 步限制，总预算改为 20 个 epoch；
余弦学习率调度随新的总步数变化，warmup 仍为 300 步。
因此本实验同时改变验证频率和总训练预算，并非固定步数的单变量对照。
若需保持原 6000 步预算，命令追加 `--max_steps 6000`；此时按步数预算停止，
最后可能是不完整 epoch，框架仍会在该轮结束时验证。

在 NextM8 根目录运行，替换真实数据路径：

```bash
bash train_shell/train_m4_rum.sh --gpus 0 \
  --config forecast_backbones/m4_rum/config/train_sweep/a17_epoch_earlystop.yaml \
  --train_parquet 'datasets/train/*.parquet' \
  --valid_parquet 'datasets/valid/*.parquet'

bash eval_shell/eval_m4_rum.sh \
  --runs runs/m4_rum/train_sweep/a17_epoch_earlystop \
  --test 'datasets/test/*.parquet' --full \
  --out runs/m4_rum/train_sweep/a17_epoch_earlystop/eval_final
```

必须提供验证集；该按 epoch 配置不用于仅提供训练集的全量 refit。
训练与评估解释器遵循 `--venv-python` > `$PYTHON` > `/home/ma-user/ts3/bin/python`，
没有自动回退。
