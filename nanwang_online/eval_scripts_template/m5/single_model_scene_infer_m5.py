# -*- coding: utf-8 -*-
# nanwang_online/eval_scripts_template/m5/single_model_scene_infer_m5.py
# 线上单模型场景推理脚本（站级 / 省级 × 短期 / 超短期）—— M5 LoRA 微调版（f1_final 等，m5_train_cli.py 训出来的 checkpoint）。
# 照 eval_scripts_template/m8/single_model_scene_infer_m8.py 写（读数据、气象聚合、异常检测、选对象、写 DS 表逐行相同），
# 与 M8 模板的差别都标了 [M5 适配]：
#   1. 不需要 TSFM 预推理：M5 本身就是 TimesFM-3（M8 的 tsfm_predict 插值列来自零样本 M5，那一路不要换成本模型——
#      M8 训练时看到的是零样本 TimesFM 的预测）。
#   2. 权重：显式从 checkpoint 目录加载并断言加载成功（weights_loaded）。model_config.yaml 里的 train.weights_path 是训练机上的
#      路径，不覆盖就找不到权重，M5 会安静地退化成零样本。
#   3. TimesFM-3 底座权重目录从线上零样本 M5 的配置（algo/nuoya_algo/forecast_backbones/m5/model_config.yaml）取：
#      checkpoint 里的 timesfm3.checkpoint 是训练机上的路径。
#   4. 站表：dataset.station_meta 指向线上站点信息表（info.csv 表头自动识别，容量只取 GCCAPACITY，与训练同口径）。
#   5. 短期 480 步走 rolling_inference（与 M8 相同：每窗 192 步，D+1 落在第一个窗里，与家里 eval_m5.sh --full 的 10:00 起报同口径）；
#      M5 没有 future_target 列，rolling_config 的 future_target 为空。
#   6. 第一次起报打印"配置列 → 数据列"核对表；推理期间的 warning 全部打出来。
import sys
import glob
import logging
import datetime
import warnings
from pathlib import Path

import duckdb
import yaml
import pandas as pd

algo_home_path = Path("/home/nfdw/nuoya")  # 算法的 home 目录（照 multi_infer.py；以服务器原文件为准）
sys.path.append(str(algo_home_path))
sys.path.append(str(algo_home_path / "algo/nuoya_algo"))
sys.path.append(str(algo_home_path / "algo/nuoya_algo/ensemble_model"))
sys.path.append(str(algo_home_path / "algo/nuoya_algo/individual_models_deploy"))


from weather_aggregation_rules_short_v3 import RULES as RULES_SHORT  # 10.10 新增：短期模型气象聚合配置规则
from weather_aggregation_rules_ultra_short_v3 import RULES as RULES_ULTRA  # 10.10 新增：超短期模型气象聚合配置规则
from feature_aggregation import aggregate_features


from forecast_backbones.m5.transformer_models import TransformersAdaptedM5Model
from pipelines.pipeline_wrapper_inference import unified_inference
from pipelines.rolling_wrapper_inference import rolling_inference


from detect_data_abnormal import check_ds_format_abnormal
from ods_transform.trace_store import save_ds_table, load_ds_table



########## 全局变量 ##########
logger = logging.getLogger('tmp_debug')
CON = duckdb.connect(database=":memory:", config={"threads": "12", "max_memory": "50GB"})
PREDICT_OBJECT = 'station'  # 预测对象为 'station' or 'province'
predict_type = 'ultra_short'  # ultra_short, short
area = 'province_guangxi_solar'
train_data_saved_dir = '/home/wangding_hw/nuoya/data/train'  # DS层输入的目录，子目录就是每个具体的表
infer_out_saved_dir = '/home/wangding_hw/nuoya/data/infer'  # DS临时输出表的目录，子目录就是每个具体的表
# [M5 适配] m5_1 = LoRA 微调版（m5_0 留给线上零样本 M5）；表名按线上约定改
out_table_name = f'ds_out_m5_1_{PREDICT_OBJECT}_{predict_type}'  # 输出表的具体表名

if predict_type == 'ultra_short':
    PRED_LEN = 192
    TIME_INTERVAL = '15min'
elif predict_type == 'short':
    PRED_LEN = 480
    TIME_INTERVAL = '1d'
##########################

########## 模型初始化 ##########
m5_ckpt_dir = algo_home_path / 'algo/checkpoints/f1_final'       # m5_train_cli.py 的 runs/m5/<ckpt-dir>/model_result（或 checkpoint-N）整个拷过来
m5_release_config = str(m5_ckpt_dir / 'model_config.yaml')
m5_ckpt_path = str(m5_ckpt_dir / 'model.safetensors')
# [M5 适配] 线上站点信息表（plantid / plant_pointname / GCCAPACITY / LONGITUDE / LATITUDE，表头自动识别）。
#   station 列按 plant_pointname 对数据里的 plant_guangfu1358；容量只取 GCCAPACITY（与训练同口径：每站固定容量）。换成线上实时那份的路径
m5_station_meta = str(m5_ckpt_dir / 'info.csv')
# [M5 适配] TimesFM-3 底座权重：取线上零样本 M5 配置里的路径（训练机上的路径在这里不存在）
m5_zero_shot_config = str(algo_home_path / 'algo/nuoya_algo/forecast_backbones/m5/model_config.yaml')
timesfm3_ckpt = yaml.safe_load(open(m5_zero_shot_config, 'r', encoding='utf-8'))['timesfm3']['checkpoint']

config_m5 = yaml.safe_load(open(m5_release_config, 'r', encoding='utf-8'))
config_m5['train']['weights_path'] = m5_ckpt_path            # [M5 适配] 构造时从这里加载 LoRA 权重（键不对直接报错）
config_m5['dataset']['station_meta'] = m5_station_meta       # [M5 适配]
config_m5['dataset']['station_meta_cols'] = None             # [M5 适配] info.csv 表头自动识别；自定义表头时在这里写映射
config_m5['timesfm3']['checkpoint'] = timesfm3_ckpt          # [M5 适配]
model_m5 = TransformersAdaptedM5Model(config_m5).eval()
if model_m5.mode != 'full' or not model_m5.weights_loaded:   # [M5 适配] 没加载到 LoRA 权重 = 零样本，直接停
    raise FileNotFoundError(f'M5 LoRA 权重没有加载：{m5_ckpt_path}（mode={model_m5.mode}）')
M5_COL = type(model_m5).__name__                             # unified_inference / rolling_inference 的预测列名
print(f'[M5] 权重 {m5_ckpt_path}；底座 {timesfm3_ckpt}；站表 {m5_station_meta}（{len(model_m5.station_meta.ids())} 个站）；'
      f'context={model_m5.context_length} pred={model_m5.prediction_length}；可训参数 {model_m5.n_params():,}；'
      f'训练站 {len(config_m5["dataset"].get("known_stations") or [])} 个；点预测 {config_m5["decision"]["point"]["default"]}')

######### 按 report_time 时间循环推理 #########
# report_time = "2026-10-01 10:00:00"

time_list = pd.date_range(
    start="2026-10-01 10:00:00",
    end="2026-10-03 10:00:00",
    freq=TIME_INTERVAL,
).strftime("%Y-%m-%d %H:%M:%S").tolist()

_m5_input_reported = False                                   # [M5 适配] 输入列核对只在第一次起报打一次

for report_time in time_list:
    try:
        # 读入轨迹数据，bugfix: 使用原始的 df 后续代码会报错
        rtime = datetime.datetime.strptime(report_time, "%Y-%m-%d %H:%M:%S").strftime("%Y%m%d%H%M")
        # 文件名形如 202610011000_20261016170000.parquet（起报时刻_更新时刻）
        tmp_file_pattern = train_data_saved_dir + f"/ds_trajectory_input/area={area}/predict_type={predict_type}/date={report_time.split()[0]}/{rtime}*.parquet"
        tmp_parquet_file = sorted(glob.glob(tmp_file_pattern))[0]
        df = pd.read_parquet(tmp_parquet_file).drop(columns=['update_time'])
        df['observe_power'] = df['observe_power_repaired']  # bugfix-09.30: 使用算法修补的 observe_power_repaired 列进行替换
        # 读入气象聚合 RULE
        if predict_type == 'ultra_short':
            df = aggregate_features(df, RULES_ULTRA)
        elif predict_type == 'short':
            df = aggregate_features(df, RULES_SHORT)
        else:
            raise ValueError(f'不支持的 predict_type={predict_type}')
        col_his, col_pred = ['general_GHI', 'general_TEMP'], ['general_GHI_predict', 'general_TEMP_predict']
        df_result = check_ds_format_abnormal(df[['timestamp_win', 'station'] + col_his + col_pred],
                                             pred_col_len=PRED_LEN)
        logger.info(f"multi_infer: 数据检查结束")
        ###### 异常检测 #####
        # 异常数据
        df_abnormal = df_result[df_result['is_abnormal'] == True]
        df_abnormal_part = df.merge(df_abnormal, on=['timestamp_win', 'station'], how='inner').drop(
            columns=['is_abnormal'])
        logger.info(
            f"发现异常数据 shape: {df_abnormal_part.shape}, 站点: {df_abnormal_part['station'].tolist()}")
        # 正常数据
        df_normal = df_result[df_result['is_abnormal'] == False]
        df_normal_part = df.merge(df_normal, on=['timestamp_win', 'station'], how='inner').drop(
            columns=['is_abnormal'])
        logger.info(f"正常数据 shape: {df_normal_part.shape} ")
        ###### 选择预测对象省、站 ######
        if PREDICT_OBJECT == 'province':
            ds_input = df_normal_part[df_normal_part['station'] == area]  # 省级
        elif PREDICT_OBJECT == 'station':
            ds_input = df_normal_part[~(df_normal_part['station'] == area)]  # 站级
        else:
            raise ValueError(f'不支持的 PREDICT_OBJECT={PREDICT_OBJECT}')
        ##############################
        # [M5 适配] 站名 → 站号、剔除站表里没有容量的站（M5 按站表容量归一 / 钳位，没容量的站推不了）
        meta = model_m5.station_meta
        ds_input = ds_input.copy()
        ds_input['station_id'] = ds_input['station'].map(meta.id_of)
        no_meta = sorted(ds_input.loc[ds_input['station_id'].isna(), 'station'].unique().tolist())
        ds_input = ds_input[ds_input['station_id'].notna()].astype({'station_id': 'int64'})
        no_cap = sorted(ds_input.loc[~ds_input['station_id'].map(lambda s: ((meta.get(int(s)) or {}).get('cap_mw') or 0) > 0), 'station'].unique().tolist())
        ds_input = ds_input[ds_input['station_id'].map(lambda s: ((meta.get(int(s)) or {}).get('cap_mw') or 0) > 0)]
        if no_meta or no_cap:
            print(f'[M5] {report_time} 剔除站表里找不到的站 {no_meta}、没有容量的站 {no_cap}；剩 {ds_input["station"].nunique()} 个站 {len(ds_input)} 行')
        if ds_input.empty:
            raise ValueError(f'{report_time} 没有可推理的站（站表 {m5_station_meta} 的 plant_pointname 要和数据 station 列一致）')
        feats = ['observe_power', 'general_GHI', 'general_GHI_predict', 'general_TEMP', 'general_TEMP_predict']
        if not _m5_input_reported:                           # [M5 适配] 核对模型要的列在数据里叫什么（✗ = 没对上，该协变量不喂）
            lines, _ = model_m5.input_report(['station', 'station_id'] + feats)
            print('[M5] ======== 输入列核对（配置列 → 推理数据列）========')
            for ln in lines:
                print(f'[M5]   {ln}')
            _m5_input_reported = True

        with warnings.catch_warnings(record=True) as w:      # [M5 适配] 两条路径都收 warning
            warnings.simplefilter('always')
            if predict_type == 'ultra_short':
                ds_out = unified_inference(
                    model_m5,
                    ds_input,
                    timestamp_cols=("timestamp_win",),
                    entity_cols=("timestamp_win", "station", 'station_id'),
                    feature_cols=feats
                )
            elif predict_type == 'short':
                rolling_config = {
                    'past_covariate': ['general_GHI', 'general_TEMP'],
                    'future_covariate': ['general_GHI_predict', 'general_TEMP_predict'],
                    'past_target': ['observe_power'],
                    'future_target': [],                     # [M5 适配] M5 没有插值列（M8 这里是 tsfm_predict）
                    'future_len': PRED_LEN,
                }
                ds_out = rolling_inference(
                    model_m5,
                    ds_input,
                    config=rolling_config,
                    timestamp_cols=("timestamp_win",),
                    entity_cols=("timestamp_win", "station", "station_id"),
                    feature_cols=[
                        "general_GHI",
                        "general_GHI_predict",
                        "general_TEMP",
                        "general_TEMP_predict",
                        "observe_power",
                    ],
                )
        msgs = sorted({f'{x.category.__name__}: {x.message}' for x in w})
        for m in msgs:
            print(f'[M5] {report_time} warning: {m}')
        n_none = int(ds_out[M5_COL].isna().sum())
        if n_none:
            print(f'[M5] {report_time} 有 {n_none}/{len(ds_out)} 行没有预测（批量失败后逐样本兜底也失败），原因见上面的 warning')

        save_ds_table(
            ds_out,
            base_path=infer_out_saved_dir,
            table_name=out_table_name,  # 表名
            area=area,
            predict_type=predict_type,
            report_time=report_time,
            dtype="float16"
        )
        logger.info(f'完成ds层表:{out_table_name} 的写入,base目录:{infer_out_saved_dir}, 预测类型:{predict_type},起报时间:{report_time}')
    except Exception as e:
        print(f'错误:{str(e)}')
        continue
