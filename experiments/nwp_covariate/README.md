# nwp_covariate —— 用 M5 找最佳 NWP 字段组合

设计文档见 `docs/m5_nwp_covariate_test_plan.md`，本目录是它的可执行实现。所有可变项都在 `config.yaml`。

## 接口选择

不直接调用 `multi_infer.py`（依赖部署机路径、有落盘副作用、无法在训练机 import），而是以它的
`get_feature_store()` 产出的 **DS 轨迹表** `ds_trajectory_input/*.parquet` 为输入。该表已经完成
ODS→DS 的窗口切分（每行一个站一个起报窗，历史 672 点数组 + `_predict` 未来数组），并包含
`observe_power_future` 真值与 `cap_power_on`，是 RULES 聚合**之前**的全字段表。

## 多起报时刻 = 多天回测

`config.report_times` 镜像 `multi_infer.__main__` 的两个循环：短期每天 10:00 一个起报，超短期每 15 分钟一个。
`make_ds.py` 按这张表逐起报时刻调 `DataSetInterface.prepare_infer_data()`，每个起报时刻一个 parquet，
目录结构与线上 `ds_trajectory_input` 一致（`date=/time=/`）。`build_feature_store.py` 把整个时段的所有
起报窗按站合并，所以南网指标天然是逐日打分再月度汇总，几天到几周都可以。

## 两种输入模式（`data.input_mode`）

| 模式 | 输入 | 适用 |
|---|---|---|
| `ds_parquet` | 线上 `get_feature_store()` 已落盘的 `ds_trajectory_input/*.parquet` | 已经有现成 DS 表，直接拷来 |
| `ods` | 本目录 `make_ds.py` 现做 | 想控制字段词典、时段、站点，或线上 DS 表字段不全 |

`ods` 模式需要 ODS→DS 代码：把部署机 `/home/nfdw/nuoya/algo/nuoya_algo/ods_transform/` 整个目录拷到
`experiments/nwp_covariate/vendor/ods_transform/`（连同它 import 的依赖），`make_ds.py` 会优先从 vendor 导入，
找不到再按 `ods.sys_path` 走部署机路径。拿到源码后，把它内部"抽哪些字段"的清单暴露成 `ods.field_dict`。

## DS 格式约定（与 光伏预测/短期分析/station_analysis_short.py 的 input 表一致）

每行一个 (station, timestamp_win)；历史列是长 672 的 list，最后一个元素落在起报时刻；`*_predict` / `*_future`
是长 480（短期）或 192（超短期）的 list，第 k 个元素落在起报时刻 + 15min × (k+1)。feature_store 保持同一格式，
只是多了标准名列，所以同一份 parquet 既能喂 M5 的 `unified_inference`，也能直接给 `station_analysis_short.py`
和 `pvcore` 的反事实工具用。

## 步骤

| 步骤 | 脚本 | 输入 → 输出 | 是否占 GPU |
|---|---|---|---|
| -1 | `make_ds.py`（仅 ods 模式） | 原始气象源 + `config.report_times/ods` → `ds/{predict_type}/date=*/time=*/*.parquet` | 否 |
| 0 | `inspect_columns.py` | DS parquet → `columns_report.csv`（列名、数组长度、NaN 比例、取值范围、来源前缀） | 否 |
| 1 | `build_feature_store.py` | DS parquet + `config.columns/derived/fallback` → `feature_store/station=*.parquet` | 否 |
| 2 | `screen.py` | feature_store → `screening.xlsx`（岭回归样本外 R² / ΔR²，可选 LightGBM） | 否 |
| 3 | `run_experiments.py --ids C0,C1,...`（或 `--all`） | feature_store + `config.experiments` → 每实验 `summary.json` + 逐日指标 csv + 逐站预测 parquet | 是 |
| 4 | `compare.py` | 各实验 summary + 逐日指标 → `report.md` + `comparison.csv`（配对检验、bootstrap 区间、站胜率、次指标否决、消融告警） | 否 |

`run_experiments.py` 对每个实验生成 `<out_dir>/<exp_id>/config/configs.yaml`（只改 `timesfm3.past_future_cols /
past_only_cols`），推理走 `pipelines.model_registry.import_model_class("m5")` + `pipelines.pipeline_wrapper_inference.unified_inference`
（与 `eval_shell/eval_pretrain.py` 同一条路），指标用 `pipelines.nanwang_metric.NanwangMetric` 的逐日表与月度表。
`--backend mock` 不需要 timesfm3，用物理代理当预测，只为本机跑通流程。

实验目录已存在 `summary.json` 即跳过（`--force` 重跑）；带 `ablation` 的实验会在 `feature_store/<pt>/abl_<id>/`
下单独构建特征库，比较时对照其 `extends` 的父实验，其余实验对照 `experiments.baseline`。

## 本机自测

    <venv>/bin/python -m pytest experiments/nwp_covariate/tests -q      # 17 个用例，合成 DS 数据，无需 timesfm3


## 运行环境

- 训练机（有 timesfm3、`pipelines.pipeline_wrapper_inference`、GPU）。
- DS parquet 需从部署机拷到训练机，或把 `data.ds_trajectory_glob` 指到共享盘。
- 步骤 0 的结果用于回填 `config.columns` 里标注 TODO 的前缀、原始列名与时间分辨率。

## 待确认（影响步骤 1 的实现细节）

1. DS 表各源的列名前缀与原始字段名；EC 累积量在 DS 层是否已被差分成 W/m²。
2. GDFS 的时间步长（1 h 还是 15 min）。
3. NWP 的历史 672 段是"当时的预报存档"还是"分析场"。
4. GFS 未接入 `multi_infer.data_path`；本框架已预留 `kind: avg6h`，接入后只需加 `columns.sources.gfs`。
