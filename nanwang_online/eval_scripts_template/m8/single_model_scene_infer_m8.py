# -*- coding: utf-8 -*-
# nanwang_online/eval_scripts_template/m8/single_model_scene_infer_m8.py
# 线上单模型场景推理脚本（站级 / 省级 × 短期 / 超短期），按服务器 nuoya 上 eval_scripts_template/m8/single_model_scene_infer_m8.py
# 的照片转录（2026-10-16 版）。第 1–26 行照片里看不到，按后文用到的名字补齐（algo_home_path 照 multi_infer.py 写），以服务器原文件为准。
# 与原文件的差别都标了 [M8 适配]，其余逐行照抄：
#   1. 权重：显式从 checkpoint 目录加载并断言加载成功。原脚本只构造模型、不 load：M8 构造时按 model_config.yaml 里的
#      train.weights_path（训练机上的 run 目录）找权重，找不到就是随机初始化，没有任何报错。
#   2. 站表：dataset.station_meta 指向线上站点信息表（info.csv 表头自动识别）；原来指向训练机路径，找不到会回退到 m8 自带那份。
#   3. 站号映射 / 剔除没有容量的站提到分支外面，短期（rolling）路径也带 station_id；剔掉了哪些站打出来。
#   4. 第一次起报打印"配置列 → 数据列"核对表；推理期间的 warning 全部打出来（原来只打含"兜底"的，列名对不上的提示被吞掉，于是只看到 set()）。
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
# sys.path.append( str(algo_home_path / "algo/nuoya_algo/individual_models_deploy/ultra_province"))


from weather_aggregation_rules_short_v3 import RULES as RULES_SHORT  # 10.10 新增：短期模型气象聚合配置规则
from weather_aggregation_rules_ultra_short_v3 import RULES as RULES_ULTRA  # 10.10 新增：超短期模型气象聚合配置规则
from feature_aggregation import aggregate_features


from safetensors.torch import load_file

from forecast_backbones.m8.transformer_models import TransformersAdaptedM8Model
from forecast_backbones.m5.transformer_models import TransformersAdaptedTimesFM3Model
from pipelines.pipeline_wrapper_inference import unified_inference
from pipelines.rolling_wrapper_inference import rolling_inference


from detect_data_abnormal import check_ds_format_abnormal
# from inference import multi_station_inference, ds_to_ods, MultiStationInference
from ods_transform.trace_store import save_ds_table, load_ds_table



########## 全局变量 ##########
logger = logging.getLogger('tmp_debug')
CON = duckdb.connect(database=":memory:", config={"threads": "12", "max_memory": "50GB"})
PREDICT_OBJECT = 'station'  # 预测对象为 'station' or 'province'
predict_type = 'ultra_short'  # ultra_short, short
area = 'province_guangxi_solar'
train_data_saved_dir = '/home/wangding_hw/nuoya/data/train'  # DS层输入的目录，子目录就是每个具体的表
infer_out_saved_dir = '/home/wangding_hw/nuoya/data/infer'  # DS临时输出表的目录，子目录就是每个具体的表
out_table_name = f'ds_out_m8_0_{PREDICT_OBJECT}_{predict_type}'  # 输出表的具体表名

if predict_type == 'ultra_short':
    PRED_LEN = 192
    TIME_INTERVAL = '15min'
elif predict_type == 'short':
    PRED_LEN = 480
    TIME_INTERVAL = '1d'
##########################

########## 模型初始化 ##########
# m6_release_config = str(algo_home_path / "algo/checkpoints/m6_1015_0/model_config.yaml")
# m6_ckpt_path = str(algo_home_path / "algo/checkpoints/m6_1015_0/model.safetensors")
# config_m6 = yaml.safe_load(open(m6_release_config, 'r', encoding="utf-8"))
# model_m6 = TransformersAdaptedM6Model(config_m6)
# model_m6.load_state_dict(load_file(m6_ckpt_path), strict=False)

m8_release_config = str(algo_home_path / 'algo/checkpoints/j1_final2/model_config.yaml')
m8_ckpt_path = str(algo_home_path / 'algo/checkpoints/j1_final2/model.safetensors')
# [M8 适配] 线上站点信息表（plantid / plant_pointname / GCCAPACITY / LONGITUDE / LATITUDE，表头自动识别）。
#   station 列按 plant_pointname 对数据里的 plant_guangfu1358；容量只取 GCCAPACITY（与训练同口径：每站固定容量）。换成线上实时那份的路径
m8_station_meta = str(algo_home_path / 'algo/checkpoints/j1_final2/info.csv')
config_m8 = yaml.safe_load(open(m8_release_config, 'r', encoding='utf-8'))
config_m8['train']['weights_path'] = m8_ckpt_path            # [M8 适配] 构造时从这里严格加载权重（strict=True）
config_m8['dataset']['station_meta'] = m8_station_meta       # [M8 适配]
config_m8['dataset']['station_meta_cols'] = None             # [M8 适配] info.csv 表头自动识别；自定义表头时在这里写映射
model_m8 = TransformersAdaptedM8Model(config_m8).eval()
if not model_m8.weights_loaded:                              # [M8 适配] 没加载到权重 = 随机初始化，直接停
    raise FileNotFoundError(f'M8 权重没有加载：{m8_ckpt_path}')
print(f'[M8] 权重 {m8_ckpt_path}；站表 {m8_station_meta}（{len(model_m8.station_meta.ids())} 个站）；'
      f'context={model_m8.context_length} pred={model_m8.prediction_length}；训练站 {len(config_m8["dataset"].get("known_stations") or [])} 个')


if predict_type == 'ultra_short':
    m5_release_config = str(algo_home_path / "algo/nuoya_algo/forecast_backbones/m5/model_config.yaml")
    config_m5 = yaml.safe_load(open(m5_release_config, 'r', encoding="utf-8"))
    model_m5 = TransformersAdaptedTimesFM3Model(config_m5)
elif predict_type == 'short':
    m5_release_config_480 = str(algo_home_path / "algo/nuoya_algo/forecast_backbones/m5/model_config_480.yaml")
    config_m5_480 = yaml.safe_load(open(m5_release_config_480, 'r', encoding="utf-8"))
    model_m5 = TransformersAdaptedTimesFM3Model(config_m5_480)

######### 按 report_time 时间循环推理 #########
# report_time = "2026-10-01 10:00:00"

time_list = pd.date_range(
    start="2026-10-01 10:00:00",
    end="2026-10-03 10:00:00",
    freq=TIME_INTERVAL,
).strftime("%Y-%m-%d %H:%M:%S").tolist()

_m8_input_reported = False                                   # [M8 适配] 输入列核对只在第一次起报打一次

for report_time in time_list:
    try:
        # 读入轨迹数据，bugfix: 使用原始的 df 后续代码会报错
        rtime = datetime.datetime.strptime(report_time, "%Y-%m-%d %H:%M:%S").strftime("%Y%m%d%H%M")
        # 照片里这一行右侧被截断：文件名形如 202610011000_20261016170000.parquet（起报时刻_更新时刻）
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
        # TSFM 预推理
        tsfm_input = unified_inference(
            model_m5,
            ds_input,
            timestamp_cols=("timestamp_win",),
            entity_cols=("timestamp_win", "station"),
            feature_cols=["observe_power"],
        )

        ds_input = ds_input.merge(tsfm_input[['timestamp_win', 'station', 'TransformersAdaptedTimesFM3Model']], on=['timestamp_win', 'station'])
        ds_input = ds_input.rename(columns={"TransformersAdaptedTimesFM3Model": "tsfm_predict"})

        # * end of TSFM 推理
        # [M8 适配] 站名 → 站号、剔除站表里没有容量的站：两条路径都要（原来只在超短期分支里做；短期 rolling 不带 station_id）
        meta = model_m8.station_meta
        ds_input['station_id'] = ds_input['station'].map(meta.id_of)
        no_meta = sorted(ds_input.loc[ds_input['station_id'].isna(), 'station'].unique().tolist())
        ds_input = ds_input[ds_input['station_id'].notna()].astype({'station_id': 'int64'})
        no_cap = sorted(ds_input.loc[~ds_input['station_id'].map(lambda s: ((meta.get(int(s)) or {}).get('cap_mw') or 0) > 0), 'station'].unique().tolist())
        ds_input = ds_input[ds_input['station_id'].map(lambda s: ((meta.get(int(s)) or {}).get('cap_mw') or 0) > 0)]
        if no_meta or no_cap:
            print(f'[M8] {report_time} 剔除站表里找不到的站 {no_meta}、没有容量的站 {no_cap}；剩 {ds_input["station"].nunique()} 个站 {len(ds_input)} 行')
        if ds_input.empty:
            raise ValueError(f'{report_time} 没有可推理的站（站表 {m8_station_meta} 的 plant_pointname 要和数据 station 列一致）')
        feats = ['observe_power', 'general_GHI', 'general_GHI_predict', 'general_TEMP', 'general_TEMP_predict', 'tsfm_predict']
        if not _m8_input_reported:                           # [M8 适配] 核对模型要的列在数据里叫什么（✗ = 没对上，该通道恒 0）
            lines, _ = model_m8.fe.input_report(['station', 'station_id'] + feats)
            print('[M8] ======== 输入列核对（配置列 → 推理数据列）========')
            for ln in lines:
                print(f'[M8]   {ln}')
            _m8_input_reported = True

        with warnings.catch_warnings(record=True) as w:      # [M8 适配] 两条路径都收 warning
            warnings.simplefilter('always')
            if predict_type == 'ultra_short':
                ds_out = unified_inference(
                    model_m8,
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
                    'future_target': ['tsfm_predict'],
                    'future_len': PRED_LEN,
                }
                ds_out = rolling_inference(
                    model_m8,
                    ds_input,
                    config=rolling_config,
                    timestamp_cols=("timestamp_win",),
                    entity_cols=("timestamp_win", "station", "station_id"),   # [M8 适配] 加 station_id
                    feature_cols=[
                        "general_GHI",
                        "general_GHI_predict",
                        "general_TEMP",
                        "general_TEMP_predict",
                        "observe_power",
                        "tsfm_predict",
                    ],
                )
        # [M8 适配] 原来是 print({... if '兜底' in ...})：没有批量失败时打出 set()，列名对不上之类的提示全被过滤掉
        msgs = sorted({f'{x.category.__name__}: {x.message}' for x in w})
        for m in msgs:
            print(f'[M8] {report_time} warning: {m}')
        n_none = int(ds_out['TransformersAdaptedM8Model'].isna().sum())
        if n_none:
            print(f'[M8] {report_time} 有 {n_none}/{len(ds_out)} 行没有预测（批量失败后逐样本兜底也失败），原因见上面的 warning')

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
