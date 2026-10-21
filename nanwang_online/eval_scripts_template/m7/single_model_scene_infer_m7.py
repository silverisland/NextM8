# -*- coding: utf-8 -*-
# nanwang_online/eval_scripts_template/m7/single_model_scene_infer_m7.py
# 线上单模型场景推理脚本（站级 / 省级 × 短期 / 超短期）—— M7 Timestral-209M 零样本（不微调、不做晴空 / 夜间处理），
# 由 M7_CONFIG 一行切换（2026-10-21）：config/z1_zeroshot_pstar.yaml = 零样本 + p*；config/configs.yaml = 原始零样本（0.5 档直出）。
# 照 eval_scripts_template/m5/single_model_scene_infer_m5.py 写（读数据、气象聚合、异常检测、选对象、站表映射、输入核对、
# 两条推理路径、warning 收集、写 DS 表逐行相同），与 M5 模板的差别都标了 [M7 适配]：
#   1. 零样本，没有微调权重：不加载 safetensors。配置取线上部署的仓库文件（M7_CONFIG），
#      Timestral 底座权重目录 timestral.checkpoint 在本脚本里显式覆盖（配置里是家里服务器的路径）。
#   2. 上报点由配置的 decision 段决定（脚本里不改）：z1_zeroshot_pstar 为 p_star（模型全部 21 档 → 200 点等概率网格，
#      权重 1/max(y, 0.2·cap)²，与南网短期考核同口径），分位先排序再钳到 [0, cap]；没有晴空上限、夜间不归零（与 M5 默认 D8 相同）。
#   3. 站表：仍用线上站点信息表 info.csv（p* 要装机容量 cap：权重下限 0.2·cap 与考核分母同一个数；只读容量和站名映射，不用经纬度）。
#   4. 配置列名是小写 general_ghi / general_temp，线上数据是 general_GHI / general_TEMP：M7 按 精确 → timestral.col_aliases →
#      大小写不敏感 解析（第一次起报的输入列核对表里能看到对上了哪一列）。
#   5. 短期 480 步走 rolling_inference（每窗 192 步），与 M5 相同；没有插值列，future_target 为空。
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


from forecast_backbones.m7.transformer_models import TransformersAdaptedM7Model  # [M7 适配]
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
# [M7 适配] 选配置（只改这一处；上报点由配置的 decision 段决定）：
m7_dir = algo_home_path / 'algo/nuoya_algo/forecast_backbones/m7'
M7_CONFIG = m7_dir / 'config/z1_zeroshot_pstar.yaml'     # 零样本 + p*
# M7_CONFIG = m7_dir / 'config/configs.yaml'             # 原始零样本（模型 0.5 档直出）
# [M7 适配] 输出表：m7_0 = 原始零样本；m7_1 = 零样本 + p*。换配置时一起改；表名按线上约定改
M7_OUT_TAG = 'm7_1'
out_table_name = f'ds_out_{M7_OUT_TAG}_{PREDICT_OBJECT}_{predict_type}'  # 输出表的具体表名

if predict_type == 'ultra_short':
    PRED_LEN = 192
    TIME_INTERVAL = '15min'
elif predict_type == 'short':
    PRED_LEN = 480
    TIME_INTERVAL = '1d'
##########################

########## 模型初始化 ##########
# [M7 适配] 零样本：没有 checkpoint 目录，配置直接取线上部署的仓库文件（M7_CONFIG）
# [M7 适配] 线上站点信息表（plantid / plant_pointname / GCCAPACITY / LONGITUDE / LATITUDE，表头自动识别）。
#   station 列按 plant_pointname 对数据里的 plant_guangfu1358；容量只取 GCCAPACITY（每站固定容量）。换成线上实时那份的路径
m7_station_meta = str(m7_dir / 'info.csv')
# [M7 适配] ⚠️ Timestral-209M 权重目录（含 config.json + model.safetensors），换成线上实际路径
timestral_ckpt = str(algo_home_path / 'algo/checkpoints/Timestral-209M')

config_m7 = yaml.safe_load(open(M7_CONFIG, 'r', encoding='utf-8'))
config_m7['timestral']['checkpoint'] = timestral_ckpt          # [M7 适配]
config_m7['dataset']['station_meta'] = m7_station_meta         # [M7 适配]
config_m7['dataset']['station_meta_cols'] = None               # [M7 适配] info.csv 表头自动识别；自定义表头时在这里写映射
m7_point = (config_m7.get('decision') or {}).get('point')      # [M7 适配] 配置里写的上报点（null = 0.5 档直出）
model_m7 = TransformersAdaptedM7Model(config_m7).eval()
if model_m7.decision_point != m7_point:                        # [M7 适配] 决策层没按配置接上就停，避免安静地退回 0.5 档直出
    raise ValueError(f'M7 决策层 {model_m7.decision_point!r} 与配置 {M7_CONFIG} 的 decision.point={m7_point!r} 不一致')
M7_COL = type(model_m7).__name__                               # unified_inference / rolling_inference 的预测列名
print(f'[M7] 配置 {M7_CONFIG}；底座 {timestral_ckpt}；站表 {m7_station_meta}（{len(model_m7.station_meta.ids())} 个站）；'
      f'context={model_m7.context_length} pred={model_m7.prediction_length}；'
      f'点预测 {model_m7.decision_point or "模型 0.5 档直出"}'
      + (f'（{model_m7.decision_levels} 档 → {model_m7.decision_grid} 点）' if model_m7.decision_point else '')
      + f'；输出表 {out_table_name}')

######### 按 report_time 时间循环推理 #########
# report_time = "2026-10-01 10:00:00"

time_list = pd.date_range(
    start="2026-10-01 10:00:00",
    end="2026-10-03 10:00:00",
    freq=TIME_INTERVAL,
).strftime("%Y-%m-%d %H:%M:%S").tolist()

_m7_input_reported = False                                     # [M7 适配] 输入列核对只在第一次起报打一次

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
        # [M7 适配] 站名 → 站号、剔除站表里没有容量的站（p* 要站表容量，没容量的站推不了）
        meta = model_m7.station_meta
        ds_input = ds_input.copy()
        ds_input['station_id'] = ds_input['station'].map(meta.id_of)
        no_meta = sorted(ds_input.loc[ds_input['station_id'].isna(), 'station'].unique().tolist())
        ds_input = ds_input[ds_input['station_id'].notna()].astype({'station_id': 'int64'})
        no_cap = sorted(ds_input.loc[~ds_input['station_id'].map(lambda s: ((meta.get(int(s)) or {}).get('cap_mw') or 0) > 0), 'station'].unique().tolist())
        ds_input = ds_input[ds_input['station_id'].map(lambda s: ((meta.get(int(s)) or {}).get('cap_mw') or 0) > 0)]
        if no_meta or no_cap:
            print(f'[M7] {report_time} 剔除站表里找不到的站 {no_meta}、没有容量的站 {no_cap}；剩 {ds_input["station"].nunique()} 个站 {len(ds_input)} 行')
        if ds_input.empty:
            raise ValueError(f'{report_time} 没有可推理的站（站表 {m7_station_meta} 的 plant_pointname 要和数据 station 列一致）')
        feats = ['observe_power', 'general_GHI', 'general_GHI_predict', 'general_TEMP', 'general_TEMP_predict']
        if not _m7_input_reported:                             # [M7 适配] 核对模型要的列在数据里叫什么（✗ = 没对上，该协变量不喂）
            lines, _ = model_m7.input_report(['station', 'station_id'] + feats)
            print('[M7] ======== 输入列核对（配置列 → 推理数据列）========')
            for ln in lines:
                print(f'[M7]   {ln}')
            _m7_input_reported = True

        with warnings.catch_warnings(record=True) as w:        # [M7 适配] 两条路径都收 warning
            warnings.simplefilter('always')
            if predict_type == 'ultra_short':
                ds_out = unified_inference(
                    model_m7,
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
                    'future_target': [],                       # [M7 适配] M7 没有插值列（M8 这里是 tsfm_predict）
                    'future_len': PRED_LEN,
                }
                ds_out = rolling_inference(
                    model_m7,
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
            print(f'[M7] {report_time} warning: {m}')
        n_none = int(ds_out[M7_COL].isna().sum())
        if n_none:
            print(f'[M7] {report_time} 有 {n_none}/{len(ds_out)} 行没有预测（批量失败后逐样本兜底也失败），原因见上面的 warning')

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
