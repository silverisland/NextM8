import copy
import datetime
import logging
import os
import sys
import time
from typing import Optional
import argparse
import yaml
import warnings

import pandas as pd


sys.path.append("/home/nfdw/nuoya/algo/nuoya_algo/ensemble_model")
sys.path.append("/home/nfdw/nuoya/")

from weather_aggregation_rules_v2_1 import RULES
from feature_aggregation import aggregate_features
from etl_fix_forecast2 import fix_observe_power_predict

from detect_data_abnormal import check_ds_format_abnormal
from ods_transform.dataset_inferface import DataSetInterface
from inference import multi_station_inference, ds_to_ods, MultiStationInference
from ods_transform.trace_store import save_ds_table, load_ds_table

warnings.filterwarnings("ignore")
multi_logger = logging.getLogger('PredictTool.model1')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

INFO_PATH = r"/jtdata/products/data/info.csv"


class MultiInfer:
    def __init__(
            self,
            report_time: str,
            data_path: dict = None,
            predict_type: str = 'short',
            checkpoints_dir: str = "",
            config: dict | str = None,
            train_data_saved_dir: str = "",
            infer_data_save_dir: str = "",
            save_path: str = "",
            timeout: Optional[int] = None,
            log_path: str = "",
    ):
        # 日志
        log_path = "/home/nfdw/nuoya/logs/"
        log_filename = f"multi_infer_{predict_type}_{report_time.replace(' ', '_').replace(':', '')}.log"
        # checkpoint
        self.checkpoints_dir = "/home/nfdw/nuoya/algo/checkpoints/checkpoints_moev3_1_100m_4dr2"  # 站级的 ckpt
        self.province_checkpoints_dir = "/home/nfdw/nuoya/algo/checkpoints/ckpt_province"  # 省级的 ckpt
        # ods和ds保存路径
        self.train_data_saved_dir = "/home/nfdw/nuoya/data/train"
        self.infer_data_save_dir = "/home/nfdw/nuoya/data/infer"
        self.infer_data_save_dir_no_cache = "/home/nfdw/nuoya/data/infer_no_cache"
        # 结果保存
        save_path = "/jtdata/products/data/result/nuoya"
        # 输入气象源
        self.data_path = {
            "power": "",
            "observe_weather": "",
            "SOLARGIS": "",
            "solar_southern": "",
            "GDFS_JTHFGNED_MID_JTNWQISX": "",
            "EC_JTKRD7RO": "",
        }
        # 需要预测的区域
        self.areas = ["province_guangxi_solar"]
        # nuoya配置文件
        self.config = "/home/nfdw/nuoya/algo/nuoya_algo/ensemble_model/chunk_config.yaml"

        self.report_time = report_time
        self.logger = multi_logger
        os.makedirs(log_path, exist_ok=True)
        log_file = os.path.join(log_path, log_filename)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        self.logger.info(f"日志文件：{log_file}")

        self.save_path = os.path.join(save_path, predict_type)

        # 历史/预报长度
        self.history_len = 672
        self.predict_type = predict_type
        if self.predict_type == 'short':
            self.predict_seq_lens = 480
            self.logger.info(f"设置预测长度：{self.predict_seq_lens}")

        elif self.predict_type == 'ultra_short':
            self.predict_seq_lens = 192
            self.logger.info(f"设置预测长度：{self.predict_seq_lens}")
        else:
            self.predict_seq_lens = 480

        self.logger.info(f"预测类型：{self.predict_type}， 设置预测长度：{self.predict_seq_lens}，"
                         f"使用气象源：{self.data_path.keys().__str__()}")

        # fix: 生成 start_time, end_time 的代码封装为函数
        self._refresh_time_range()
        # # 起报时间
        # self.forecast_time = datetime.datetime.strptime(self.report_time, "%Y-%m-%d %H:%M:%S")
        # # 起报时间下一个时间点上开始预测的第一个点
        # self.forecast_s_time = self.forecast_time + datetime.timedelta(minutes=15)

        # # 构造预测结果保持点时间段
        # if predict_type == 'short':
        #     self.result_s_time = (self.forecast_time + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0)
        #     self.result_e_time = (self.forecast_time + datetime.timedelta(days=4)).replace(hour=23, minute=45, second=0)
        # else:
        #     self.result_s_time = self.forecast_s_time
        #     self.result_e_time = self.forecast_s_time + datetime.timedelta(hours=5) - datetime.timedelta(minutes=15)

        with open(self.config, 'r', encoding="utf-8") as f:
            self.config_dict = yaml.safe_load(f)

        self.only_power = False

        ## 09.23新增：初始化 MultiStationInference 对象来替代 multi_station_inference() 的旧的推理方式
        # 站级模型对象
        config_station = copy.deepcopy(self.config_dict)
        config_station["data_check"]["future_len"] = self.predict_seq_lens
        self.station_inference_obj = MultiStationInference(config=config_station, checkpoints_dir=self.checkpoints_dir)
        # 省级模型对象
        config_province = copy.deepcopy(self.config_dict)
        config_province["data_check"]["future_len"] = self.predict_seq_lens
        self.prov_inference_obj = MultiStationInference(config=config_province, checkpoints_dir=self.province_checkpoints_dir)
        # tsfm模型对象
        config_tsfm = copy.deepcopy(self.config_dict)
        config_tsfm["backstop_strategy"]["tsfm_only"] = True
        config_tsfm["pre_trained_model"]["future_len"] = self.predict_seq_lens
        config_tsfm["data_check"]["future_len"] = self.predict_seq_lens
        self.tsfm_inference_obj = MultiStationInference(config=config_tsfm, checkpoints_dir=self.checkpoints_dir)

    def _refresh_time_range(self):
        """刷新时间"""
        # 起报时间
        self.forecast_time = datetime.datetime.strptime(self.report_time, "%Y-%m-%d %H:%M:%S")
        # 起报时间下一个时间点上开始预测的第一个点
        self.forecast_s_time = self.forecast_time + datetime.timedelta(minutes=15)

        # 构造预测结果保持点时间段
        if self.predict_type == 'short':
            self.result_s_time = (self.forecast_time + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0)
            self.result_e_time = (self.forecast_time + datetime.timedelta(days=4)).replace(hour=23, minute=45, second=0)
        else:
            self.result_s_time = self.forecast_s_time
            self.result_e_time = self.forecast_s_time + datetime.timedelta(hours=5) - datetime.timedelta(minutes=15)
        

    def _clean_cache(self):
        """
        因为ods cache 缓存存在历史数据污染，每次调用前后删除缓存；
        如果未来有性能问题，则不用清理缓存
        """
        for a in self.areas:
            for suffix in ("short", "ultra_short"):
                path = os.path.join(self.infer_data_save_dir_no_cache, "ods", f"{a}_{suffix}_ods_infer_cache.parquet")  # 清理指定 no_cache 缓存路径
                try:
                    if os.path.exists(path):
                        os.remove(path)
                        self.logger.info(f"[{suffix}]缓存已删除：{path}")
                    else:
                        self.logger.info(f"[{suffix}]缓存路径不存在：{path}")
                except Exception as e:
                    self.logger.warning(f"[{suffix}]删除缓存失败：{path}, err_msg: {e}")

    def get_feature_store(self, cur_report_time=""):
        if cur_report_time == "":
            cur_report_time = self.report_time
        start_t = time.time()
        self.logger.info(
            f"multi_infer: start to get feature..., params: "
            f"cur_report_time: {cur_report_time}, "
            f"data_path: {self.data_path}, "
            f"save_path: {self.save_path}, "
            f"predict_type: {self.predict_type}, "
            f"predict_seq_lens: {self.predict_seq_lens}"
            f"checkpoints_dir: {self.checkpoints_dir}, "
            f"config: {self.config},"
            f"only_power: {self.only_power}"
        )

        stations_df = pd.read_csv(INFO_PATH)
        stations_df = stations_df[["plant_pointname", "LONGITUDE", "LATITUDE", "GCCAPACITY", "province_pointname"]]
        stations_df.rename(
            columns={'plant_pointname': 'plantId', 'LONGITUDE': 'lng', 'LATITUDE': 'lat', 'GCCAPACITY': 'cap_power_on'},
            inplace=True)

        error_area = []  # 统计预测异常的区域
        for area in self.areas:
            # 区分solar和wind
            area_type = "_solar" if area.endswith("_solar") else "_wind"
            province = area.replace(area_type, "")
            station_name_suffix = "plant_guangfu" if area.endswith("_solar") else "plant_fengdian"
            station_df = stations_df[stations_df["province_pointname"] == province]
            station_df = station_df[station_df["plantId"].str.startswith(station_name_suffix)]
            area_df = station_df[["plantId", "lng", "lat", "cap_power_on"]].copy()

            self.logger.info(f"=======start predict area: {area}, area_df shape: {area_df.shape}=======")
            df, success_flag = DataSetInterface(
                save_history_data_dir=self.train_data_saved_dir,
                save_infer_data_dir=self.infer_data_save_dir,
                logger=self.logger,
            ).prepare_infer_data(
                forecast_time=cur_report_time,
                predict_type=self.predict_type,
                data_path=self.data_path,
                area_name=area,
                plants_df=area_df,
                predict_seq_lens=self.predict_seq_lens,
                history_len=self.history_len,
                only_power=self.only_power,
            )
            # 插入 ETL 数据保存，如果后续有新表再这下面平铺添加，比如TSFM推理会产生的新列形成的表
            save_ds_table(
                df,
                base_path=self.train_data_saved_dir,
                table_name='ds_trajectory_input',  # 表名
                area=area,
                predict_type=self.predict_type,
                report_time=cur_report_time,
                dtype="float16",
                logger=self.logger
            )

    def handle(self):
        start_t = time.time()
        self._refresh_time_range()  # 09.23 fix 时间范围导致的裁剪bug
        self.logger.info(
            f"multi_infer: start to handle..., params: "
            f"report_time: {self.report_time}, "
            f"data_path: {self.data_path}, "
            f"save_path: {self.save_path}, "
            f"predict_type: {self.predict_type}, "
            f"predict_seq_lens: {self.predict_seq_lens}"
            f"checkpoints_dir: {self.checkpoints_dir}, "
            f"config: {self.config},"
            f"only_power: {self.only_power}"
        )

        stations_df = pd.read_csv(INFO_PATH)
        stations_df = stations_df[["plant_pointname", "LONGITUDE", "LATITUDE", "GCCAPACITY", "province_pointname"]]
        stations_df.rename(
            columns={'plant_pointname': 'plantId', 'LONGITUDE': 'lng', 'LATITUDE': 'lat', 'GCCAPACITY': 'cap_power_on'},
            inplace=True)

        error_area = []  # 统计预测异常的区域
        for area in self.areas:
            # 区分solar和wind
            area_type = "_solar" if area.endswith("_solar") else "_wind"
            province = area.replace(area_type, "")
            station_name_suffix = "plant_guangfu" if area.endswith("_solar") else "plant_fengdian"
            station_df = stations_df[stations_df["province_pointname"] == province]
            station_df = station_df[station_df["plantId"].str.startswith(station_name_suffix)]
            area_df = station_df[["plantId", "lng", "lat", "cap_power_on"]].copy()

            self.logger.info(f"=======start predict area: {area}, area_df shape: {area_df.shape}=======")

            ### 09.23 临时调试的数据读取代码 ###
            # df = pd.read_parquet('/home/nfdw/nuoya/tmp/short/date=2026-09-22/202609221000_20260922100000.parquet').drop(columns=['update_time'])  # 09.23 改造了 multi_station_inference 为一个类
            # df = pd.read_parquet('/home/nfdw/nuoya/tmp/ultra_short/date=2026-09-23/202609230215_20260923020000.parquet').drop(columns=['update_time'])  # 09.23 改造了 multi_station_inference 为一个类
            import glob
            rtime = datetime.datetime.strptime(self.report_time, "%Y-%m-%d %H:%M:%S").strftime("%Y%m%d%H%M")
            tmp_file_pattern = f'/home/nfdw/nuoya/data/train/ds_trajectory_input/area={area}/predict_type={self.predict_type}/date={self.report_time.split()[0]}/{rtime}_*.parquet'
            tmp_parquet_file = sorted(glob.glob(tmp_file_pattern))[0]
            df = pd.read_parquet(tmp_parquet_file).drop(columns=['update_time'])
            success_flag = True 
            ### end 09.23 临时调试的数据读取代码 ### 
            # df, success_flag = DataSetInterface(
            #     save_history_data_dir=self.train_data_saved_dir,
            #     save_infer_data_dir=self.infer_data_save_dir,
            #     logger=self.logger,
            # ).prepare_infer_data(
            #     forecast_time=self.report_time,
            #     predict_type=self.predict_type,
            #     data_path=self.data_path,
            #     area_name=area,
            #     plants_df=area_df,
            #     predict_seq_lens=self.predict_seq_lens,
            #     history_len=self.history_len,
            #     only_power=self.only_power,
            # )
            # # 插入 ETL 数据保存，如果后续有新表再这下面平铺添加，比如TSFM推理会产生的新列形成的表
            # save_ds_table(
            #     df,
            #     base_path=self.train_data_saved_dir,
            #     table_name='ds_trajectory_input',  # 表名
            #     area=area,
            #     predict_type=self.predict_type,
            #     report_time=self.report_time,
            #     dtype="float16",
            #     logger=self.logger
            # )
            # # =========== 无 ods cache 保存 ds_trajectory_input_no_ods_cache  =============
            # self.logger.info(f"开始清理ods cache，并保存 no cache ds_trajectory, ods cache路径：{self.infer_data_save_dir_no_cache}")
            # self._clean_cache()  # 先清理ods cache
            # df_no_ods_cache, _ = DataSetInterface(
            #     save_history_data_dir=self.train_data_saved_dir,
            #     save_infer_data_dir=self.infer_data_save_dir_no_cache,
            #     logger=self.logger,
            # ).prepare_infer_data(
            #     forecast_time=self.report_time,
            #     predict_type=self.predict_type,
            #     data_path=self.data_path,
            #     area_name=area,
            #     plants_df=area_df,
            #     predict_seq_lens=self.predict_seq_lens,
            #     history_len=self.history_len,
            #     only_power=self.only_power,
            # )
            # # 插入 ETL 数据保存，如果后续有新表再这下面平铺添加，比如TSFM推理会产生的新列形成的表
            # save_ds_table(
            #     df_no_ods_cache,
            #     base_path=self.train_data_saved_dir,
            #     table_name='ds_trajectory_input_no_ods_cache',  # 表名
            #     area=area,
            #     predict_type=self.predict_type,
            #     report_time=self.report_time,
            #     dtype="float16",
            #     logger=self.logger
            # )
            # ===============================================
            if not success_flag:
                self.logger.info(
                    f"multi_infer: finish generate infer area failed, area name: {area}, cost time: {time.time() - start_t}")
                error_area.append(area)
                continue

            self.logger.info(
                f"multi_infer: finish generate infer data..., shape: {df.shape}, cost time: {time.time() - start_t}")

            self.logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>RULES')
            self.logger.info(RULES)
            df = aggregate_features(df, RULES)
            for i in range(9):
                df[f'ssrd_pos_{i + 1}'] = df['solar_southern_GHI']
                df[f't2m_pos_{i + 1}'] = df['solar_southern_TEMP']
                df[f'ssrd_pos_{i + 1}_predict'] = df['solar_southern_GHI_predict']
                df[f't2m_pos_{i + 1}_predict'] = df['solar_southern_TEMP_predict']

            df['GHI_SOLARGIS'] = df['general_GHI']
            df['TEMP_SOLARGIS'] = df['general_TEMP']
            df['GHI_SOLARGIS_predict'] = df['general_GHI_predict']
            df['TEMP_SOLARGIS_predict'] = df['general_TEMP_predict']
            self.logger.info(df.iloc[0])

            # 先检查数据，然后把站点分2类，异常类走tsfm，正常类走MoE V3.1
            if self.only_power:
                df_abnormal_part = df.copy()
                df_normal_part = pd.DataFrame()
                self.logger.info(f"multi_infer: only_power模式，跳过数据检查，全部数据送入tsfm: {df_abnormal_part.shape}")
            else:
                self.logger.info(f"multi_infer: 数据检查开始： {time.time() - start_t} s")
                self.logger.info(f"multi_infer: 数据检查开始 shape: {df.shape}")

                # TODO: 缩小 col_hist, col_pred 范围
                col_his, col_pred = [], []
                for col in df.columns:
                    if col in ['timestamp_win', 'station', 'observe_power_future']:
                        continue
                    if col.endswith('_predict'):
                        col_pred.append(col)
                    else:
                        col_his.append(col)

                # TODO: 传入检查长度, 只检查 RULES 配置里我们输出的列，降低敏感性
                df_result = check_ds_format_abnormal(df[['timestamp_win', 'station'] + col_his + col_pred], # TODO:缩小 col_his, col_pred 范围
                                                     pred_col_len=self.predict_seq_lens)
                self.logger.info(f"multi_infer: 数据检查结束： {time.time() - start_t} s")

                # 异常数据
                df_abnormal = df_result[df_result['is_abnormal'] == True]
                df_abnormal_part = df.merge(df_abnormal, on=['timestamp_win', 'station'], how='inner').drop(
                    columns=['is_abnormal'])
                self.logger.info(
                    f"发现异常数据 shape: {df_abnormal_part.shape} ， 站点：{df_abnormal_part['station'].tolist()}")

                # 正常数据
                df_normal = df_result[df_result['is_abnormal'] == False]
                df_normal_part = df.merge(df_normal, on=['timestamp_win', 'station'], how='inner').drop(
                    columns=['is_abnormal'])
                self.logger.info(f"正常数据 shape: {df_normal_part.shape} ")

            # 提取 province 行（station == area），只存在于 normal 中
            df_province = pd.DataFrame()
            if not df_normal_part.empty and area in df_normal_part['station'].values:
                df_province = df_normal_part[df_normal_part['station'] == area].copy()
                df_normal_part.drop(df_normal_part[df_normal_part['station'] == area].index, inplace=True)
                self.logger.info(f"提取 province 行: {area}, shape={df_province.shape}")
            else:
                self.logger.warning(f"province 行未找到: {area}")

            df_res_list = []
            # 站级正常数据-MoE v3.1 入口
            if df_normal_part.shape[0] > 0 and self.only_power is False:
                self.logger.info(f"站级推理开始，checkpoint: {self.checkpoints_dir}")
                # config_copy = copy.deepcopy(self.config_dict)
                # config_copy["data_check"]["future_len"] = self.predict_seq_lens
                # result_df_normal_part = multi_station_inference(
                #     ds_dataframe=df_normal_part,
                #     df_plants_info=None,
                #     checkpoints_dir=self.checkpoints_dir,
                #     forecasting_type=self.predict_type,
                #     config=config_copy,
                # )
                result_df_normal_part = self.station_inference_obj.inference(ds_dataframe=df_normal_part)  # 09.23 改造附加
                df_res_list.append(result_df_normal_part)
                self.logger.info(f"======站级推理完成======")

            # 省级正常数据-MoE v3.1 入口
            if df_province.shape[0] == 1:
                df_province = pd.concat([df_province] * 2, ignore_index=True)
                df_province.loc[1, "station"] = "unknow_province"
            if df_province.shape[0] > 0:
                self.logger.info(f"省级推理开始, checkpoint: {self.province_checkpoints_dir}")
                # config_province = copy.deepcopy(self.config_dict)
                # config_province["data_check"]["future_len"] = self.predict_seq_lens
                # result_df_province = multi_station_inference(
                #     ds_dataframe=df_province,
                #     df_plants_info=None,
                #     checkpoints_dir=self.province_checkpoints_dir,
                #     forecasting_type=self.predict_type,
                #     config=config_province,
                # )
                result_df_province = self.prov_inference_obj.inference(ds_dataframe=df_province)  # 09.23 改造更新
                df_res_list.append(result_df_province)
                self.logger.info(f"======省级推理完成======")

            # 异常数据-tsfm入口
            if df_abnormal_part.shape[0] == 1:  # 只有一行数据会报错，复制一行，但station不能相同 
                df_abnormal_part = pd.concat([df_abnormal_part, df_abnormal_part], ignore_index=True)
                df_abnormal_part.loc[1, "station"] = "unknow"
            if df_abnormal_part.shape[0] > 0:
                # self.config_dict["backstop_strategy"]["tsfm_only"] = True
                # self.config_dict["pre_trained_model"]["future_len"] = self.predict_seq_lens
                # self.config_dict["data_check"]["future_len"] = self.predict_seq_lens
                # self.logger.info(f"动态修改config：{self.config_dict}")
                # result_df_abnormal_part = multi_station_inference(
                #     ds_dataframe=df_abnormal_part,
                #     df_plants_info=None,
                #     checkpoints_dir=self.checkpoints_dir,
                #     forecasting_type=self.predict_type,
                #     config=self.config_dict,
                # )
                result_df_abnormal_part = self.tsfm_inference_obj.inference(ds_dataframe=df_abnormal_part)
                df_res_list.append(result_df_abnormal_part)

            # 合并结果（支持多个结果依次 merge）
            if len(df_res_list) > 1:
                ds_result = pd.concat(df_res_list, ignore_index=True)
            elif len(df_res_list) == 1:
                ds_result = df_res_list[0]
            else:
                ds_result = pd.DataFrame()
                self.logger.warning("MoE V3.1和tsf均未推理出结果！")

            ds_result.rename(columns={"ensemble": "observe_power_predict"}, inplace=True)
            ds_result = ds_result[~ds_result['station'].isin(['unknow', 'unknow_province'])]

            try:  # 保存DS数据
                save_cols = [
                    'timestamp_win',
                    'station',
                    'observe_power',
                    'observe_power_future',
                    'ssrd_pos_1',
                    'ssrd_pos_2',
                    'ssrd_pos_3',
                    'ssrd_pos_4',
                    'ssrd_pos_5',
                    'ssrd_pos_6',
                    'ssrd_pos_7',
                    'ssrd_pos_8',
                    'ssrd_pos_9',
                    't2m_pos_1',
                    't2m_pos_2',
                    't2m_pos_3',
                    't2m_pos_4',
                    't2m_pos_5',
                    't2m_pos_6',
                    't2m_pos_7',
                    't2m_pos_8',
                    't2m_pos_9',
                    'GHI_SOLARGIS',
                    'TEMP_SOLARGIS',
                    'ssrd_pos_1_predict',
                    'ssrd_pos_2_predict',
                    'ssrd_pos_3_predict',
                    'ssrd_pos_4_predict',
                    'ssrd_pos_5_predict',
                    'ssrd_pos_6_predict',
                    'ssrd_pos_7_predict',
                    'ssrd_pos_8_predict',
                    'ssrd_pos_9_predict',
                    't2m_pos_1_predict',
                    't2m_pos_2_predict',
                    't2m_pos_3_predict',
                    't2m_pos_4_predict',
                    't2m_pos_5_predict',
                    't2m_pos_6_predict',
                    't2m_pos_7_predict',
                    't2m_pos_8_predict',
                    't2m_pos_9_predict',
                    'GHI_SOLARGIS_predict',
                    'TEMP_SOLARGIS_predict',
                ]
                # 合入predict_power
                df["timestamp_win"] = pd.to_datetime(df["timestamp_win"])
                ds_result["timestamp_win"] = pd.to_datetime(ds_result["timestamp_win"])
                df = pd.merge(df, ds_result, how='inner', on=['timestamp_win', 'station'])
                new_col = ['observe_power_predict']
                if self.predict_type == "ultra_short":
                    ef_config = {
                        'initial_threshold_max': 0.27,
                        'initial_threshold_min': -0.27,
                        'rate': 0.1,
                        'keep_first': 10
                    }
                    ef_province_config = {
                        'initial_threshold_max': 0.27,
                        'initial_threshold_min': -0.27,
                        'rate': 0.1,
                        'keep_first': 1
                    }
                    self.logger.info("start into EF module.")
                    try:
                        df_province_input = df[df['station'] == area]
                        df_station_input = df[~(df['station'] == area)]
                        self.logger.info(f"ef input province time: {list(set(df_province_input['timestamp_win'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()))}")
                        self.logger.info(f"ef input station time: {list(set(df_station_input['timestamp_win'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()))}")
                        ef_input_path = os.path.join(self.train_data_saved_dir, self.predict_type)
                        df_province_output = fix_observe_power_predict(df_province_input, ef_input_path,
                                                                       config=ef_province_config)

                        df_station_output = fix_observe_power_predict(df_station_input, ef_input_path,
                                                                      config=ef_config)
                        new_col += ['observe_power_predict_error_fix_16p']
                        # 上下合并 station 与 province
                        df = pd.concat([df_station_output, df_province_output], ignore_index=True)
                    except Exception as e:
                        self.logger.warning(f"EF修正失败，跳过EF修正，使用原始预测值：{e}")
                save_ds_data_cache(
                    df=df[save_cols + new_col], report_time=self.report_time, area=area, predict_type=self.predict_type,
                    base_save_path=self.train_data_saved_dir
                )
            except Exception as e:
                ds_result["ensemble"] = ds_result["observe_power_predict"]
                self.logger.error(f"multi_infer: save ds data failed, error msg: {e}")

            # save_ds_data_cache 之后处理 ensemble 列：observe_power_predict 保持不动，
            # 仅当 error_fix 列非空时用其覆盖 ensemble；最后同步回 ds_result 供 ODS 转换
            if 'observe_power_predict' in df.columns:
                df["ensemble"] = df["observe_power_predict"]
                if 'observe_power_predict_error_fix_16p' in df.columns:
                    fix_mask = df['observe_power_predict_error_fix_16p'].notna()
                    if fix_mask.any():
                        df.loc[fix_mask, "ensemble"] = df.loc[fix_mask, "observe_power_predict_error_fix_16p"]
                        self.logger.info(f"EF修正应用到 ensemble: {int(fix_mask.sum())} 行")
                df["timestamp_win"] = pd.to_datetime(df["timestamp_win"])
                ds_result = ds_result.drop(columns=['ensemble'], errors='ignore').merge(
                    df[['timestamp_win', 'station', 'ensemble']], on=['timestamp_win', 'station'], how='left')
            else:
                ds_result["ensemble"] = ds_result["observe_power_predict"]
            result_df = ds_to_ods(ds_result)
            value_cols = result_df.columns.difference(['dtime'])
            result_df[value_cols] = result_df[value_cols].clip(lower=0)  # 负值置零

            result_df = result_df[
                (result_df['dtime'] >= self.result_s_time) & (result_df['dtime'] <= self.result_e_time)]
            # 夜间结果置零
            result_df = zero_night_hours(result_df)

            if self.predict_type == "short":
                date_str = datetime.datetime.strptime(self.report_time, '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d')
            else:
                date_str = datetime.datetime.strptime(self.report_time, '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M')

            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            
            from pathlib import Path
            save_full_path = os.path.join(self.save_path,  f"hw_nuoya_{date_str}_{self.predict_type}_{area}_exp0923.parquet")   # TODO: 临时的文件后缀 exp0923

            result_df.to_parquet(save_full_path)
            # result_df.to_parquet(Path('/home/nfdw/nuoya/tmp') / f"hw_nuoya_{date_str}_{self.predict_type}_{area}.parquet" )   # TODO: 更改了回测的目录
            self.logger.info(f"结果保存成功：{save_full_path}, shape: {result_df.shape}")

            self.logger.info(f"multi_infer: finish handle... cost time: {time.time() - start_t}")

    def inference(self):
        pass


def zero_night_hours(
        df: pd.DataFrame,
        start_time: str = "20:00",
        end_time: str = "06:00",
        dtime_col: str = "dtime",
) -> pd.DataFrame:
    """
    将 DataFrame 中指定夜间时段的预测值置零。
    时间精度到分钟，支持 15 分钟频率的精确边界控制。

    Args:
        df: 输入 DataFrame，需包含时间列
        start_time: 夜间开始时刻，格式 HH:MM（含），默认 "20:00"
        end_time: 夜间结束时刻，格式 HH:MM（含），默认 "06:00"
        dtime_col: 时间列名，默认 'dtime'

    Returns:
        处理后的 DataFrame（已置零夜间预测值）
    """
    if dtime_col not in df.columns:
        raise ValueError(f"时间列 '{dtime_col}' 不存在，现有列: {list(df.columns)}")

    df = df.copy()
    time_str = df[dtime_col].dt.strftime('%H:%M')
    if start_time <= end_time:
        mask = (time_str >= start_time) & (time_str <= end_time)
    else:
        mask = (time_str >= start_time) | (time_str <= end_time)

    pred_cols = [c for c in df.columns if c != dtime_col]
    zeros = int(mask.sum())
    if zeros > 0:
        df.loc[mask, pred_cols] = 0.0
        multi_logger.info(f"夜间置零: {zeros}/{len(df)} ({zeros / len(df) * 100:.1f}%)")
    else:
        multi_logger.info(f"夜间置零: 0/{len(df)} (0.0%) - 该时间范围内无夜间数据")

    multi_logger.info(f"夜间时段: {start_time} ~ {end_time}")
    multi_logger.info(f"影响列数: {len(pred_cols)} 个预测列")
    return df


def save_ds_data_cache(df, report_time, area, predict_type, base_save_path):
    report_dt = datetime.datetime.strptime(report_time, '%Y-%m-%d %H:%M:%S')
    date_str = report_dt.strftime('%Y-%m-%d')
    time_str = report_dt.strftime('%H:%M')
    dir_path = os.path.join(
        base_save_path, predict_type,
        f"area={area}", f"date={date_str}", f"time={time_str}",
    )
    os.makedirs(dir_path, exist_ok=True)
    date_full_str = datetime.datetime.strptime(report_time, '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M')
    save_path = os.path.join(dir_path, f"hw_nuoya_ds_{date_full_str}_{predict_type}_{area}.parquet")
    df.to_parquet(save_path, index=False)
    multi_logger.info(f"[{report_time}] ✓ DS数据保存成功: {save_path}, shape={df.shape}")


if __name__ == '__main__':
    # ===============short===================
    # start = datetime.datetime(2026, 8, 25, 10,00,00)
    # end   = datetime.datetime(2026, 9, 16, 10,00,00)
    # report_time_str_list = [  # 前后包含
    #     (start + datetime.timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S")
    #     for i in range((end - start).days + 1)
    # ]

    # infer_object = MultiInfer(
    #     report_time=report_time_str_list[0],
    #     predict_type="short",
    # )

    # infer_object = MultiInfer(
    #     report_time='2026-08-29 10:00:00',
    #     predict_type="short",
    # )
    # infer_object.handle()

    # for t in report_time_str_list:
    #     infer_object.get_feature_store(t)

    # ===============ultra short===================
    start = datetime.datetime(2026, 8, 26, 2, 0, 00)
    end   = datetime.datetime(2026, 8, 31, 23, 45, 00)
    delta = datetime.timedelta(minutes=15)
    report_time_str_list = [  # 前后包含
        (start + i * delta).strftime("%Y-%m-%d %H:%M:%S")
        for i in range(int((end - start) / delta) + 1)
    ]
    infer_object = MultiInfer(
        report_time=report_time_str_list[0],
        predict_type="ultra_short",
    )

    # for t in report_time_str_list:
    #     infer_object.get_feature_store(t)

    # start = datetime.datetime(2026, 8, 25, 10,00,00)
    # end   = datetime.datetime(2026, 9, 15, 10,00,00)
    # report_time_str_list = [  # 前后包含
    #     (start + datetime.timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S")
    #     for i in range((end - start).days + 1)
    # ]

    # infer_object = MultiInfer(
    #     report_time=report_time_str_list[0],  # 测试需要给的 dummy 时间，无实际意义
    #     predict_type="ultra_short",
    # )
    
    for t in report_time_str_list:
        infer_object.report_time = t  # 回测态动态更改该时间
        try:
            infer_object.handle()
        except Exception as e:
            print(f'## SIKIP BUG ##: report_time={t} ', str(e))
            continue