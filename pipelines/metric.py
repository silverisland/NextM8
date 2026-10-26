"""
    DS层 => 内部分析 / Overlap
    按月展平之后计算 RMSE
    月份划分按之前的方法

    ODS层 => 业务方 / Non Overlap
    按天展平之后计算RMSE -> 取一个月中每天的指标平均
"""


import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Union, List
from collections import OrderedDict

import numpy as np
import pandas as pd


# 月度均值口径：剔除样本天数 <= 2 天的月份后，各月等权求均值。
# 与 eval_shell/make_models_summary.py 汇总口径一致（原在汇总脚本里做二次重算，
# 现下沉到产表环节，让各模型逐站表/ summary.json 直接就是剔除小月后的值）。
# 注意：判定用"该月有数据的日指标天数"（ODS 表 day_count 行），而非日历天数。
MIN_DAYS = 3
_ODS_DAY_ROW = "day_count"          # ODS 月度表（.T 后）里"该月有数据天数"所在行
_ODS_METRIC_ROWS = ["rmse_acc", "mae_acc", "ACC_acc"]


def _monthly_mean_exclude_short_months(table, day_row, metric_rows):
    """对 ODS 月度明细表（_calc_ods_metric 转置后）剔除样本天数 < MIN_DAYS 的月份列，
    再对 metric_rows 各指标行按各月等权求均值。

    table       : 转置后的月度表（index=行名，列=月份位置编号）
    day_row     : 天数所在行行名（ODS 为 'day_count'，即该月有数据的日指标天数）
    metric_rows : 需要求均值的指标行行名（如 ['rmse_acc','mae_acc','ACC_acc']）
    -> Series(index=metric_rows)，值为剔除小月后的月均值；无有效月份时 NaN。
    """
    days = pd.to_numeric(table.loc[day_row], errors="coerce")
    month_cols = [c for c in table.columns
                  if pd.notna(days.get(c)) and float(days[c]) >= MIN_DAYS]
    if not month_cols:
        return pd.Series(float("nan"), index=metric_rows)
    return table.loc[metric_rows, month_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)


class ODSMetric:
    def __init__(self):
        self.name2method = {
            "rmse_acc": self.cal_rmse_acc,
            "mae_acc": self.cal_mae_acc,
            "ACC_acc": self.cal_ACC_acc,
        }

    def cal_mae_acc(self, pred, real, cap):
        """入参数据支持数组样式，np.array   pd.Seres
            cap:是数值，非零
        """
        # MAE精度
        return 1 - np.mean(np.abs(real - pred)) / cap

    def cal_rmse_acc(self, pred, real, cap):
        # 功率RMSE精度
        return 1 - (np.sqrt(np.nansum((real - pred) ** 2) / len(pred)) / cap)

    def cal_ACC_acc(self, pred, real, cap):
        sum = 0
        for pred_one, real_one in zip(pred, real):
            PM = real_one
            if real_one < 0.2 * cap:
                PM = 0.2 * cap
            sum += ((real_one - pred_one) / PM) ** 2
        return 1 - np.sqrt(sum / len(pred))


class DSMetric:
    def __init__(self):
        self.name2method = OrderedDict()
        self.name2method["RMSE_SCORE"] = self.calc_score
        self.name2method["RMSE"] = self.calc_rmse
        self.name2method["MAE"] = self.calc_mae

    def calc_mse(self, pred, gt, cap=None):
        return np.nanmean((gt - pred) ** 2)

    def calc_mae(self, pred, gt, cap=None):
        return np.nanmean(np.abs(gt - pred))

    def calc_rmse(self, pred, gt, cap=None):
        return np.sqrt(self.calc_mse(gt, pred))

    # def calc_rmse(self, gt, pred, cap=None):
    #     return np.sqrt(np.nansum((gt - pred) ** 2) / len(pred))

    def calc_score(self, pred, gt, cap):
        return 1 - self.calc_rmse(gt, pred) / cap


@dataclass
class Dayconfig:
    num_points = 96
    ind_midnight_from_9am = 59
    ind_4th_hour = 15
    step_size = 15
    step = timedelta(minutes=15)


class UltraShortConverter:
    ods_time_col: str = 'dtime'

    def __init__(self):
        self.day_config = Dayconfig()

    def select_data(self, df, data_cols):
        for col in data_cols:
            df[col] = list(map(lambda x: x[self.day_config.ind_4th_hour], df[col]))
        return df.copy()

    def infer_time(self, df, time_col):
        df[self.ods_time_col] = list(map(
            lambda x: x + self.day_config.step * (self.day_config.ind_4th_hour + 1),
            df[time_col]
        ))
        df['interval'] = [240] * len(df)
        df['interval_day'] = [0] * len(df)
        df['report_date'] = list(map(lambda x: x.strftime("%Y-%m-%d"), df[time_col]))
        df['report_time'] = list(map(lambda x: x.strftime("%H:%M"), df[time_col]))
        df.drop(columns=[time_col], inplace=True)
        return df.copy()

    def convert(self, df, time_col, data_cols):
        df = self.select_data(df, data_cols)
        df = self.infer_time(df, time_col)
        return df.copy()


class DayAheadConverter:
    ods_time_col: str = 'dtime'

    def __init__(self):
        self.day_config = Dayconfig()
        self.start_ind = self.day_config.ind_midnight_from_9am
        self.end_ind = self.day_config.ind_midnight_from_9am + self.day_config.num_points

    def select_data(self, df, data_cols):
        for col in data_cols:
            df[col] = list(map(lambda x: x[self.start_ind:self.end_ind], df[col]))
        return df.copy()

    def infer_time(self, df, time_col):
        df[self.ods_time_col] = list(map(
            lambda x: [x + self.day_config.step * (i + 1) for i in range(self.start_ind, self.end_ind)],
            df[time_col]
        ))
        df['interval'] = [[self.day_config.step_size * (i + 1) for i in range(self.start_ind, self.end_ind)]] * len(df)
        df['interval_day'] = [[self.day_config.step_size * (i + 1) // (24 * 60) for i in range(self.start_ind, self.end_ind)]] * len(df)
        df['report_date'] = list(map(
            lambda x: [x.strftime("%Y-%m-%d")] * self.day_config.num_points,
            df[time_col]
        ))
        df['report_time'] = list(map(
            lambda x: [x.strftime("%H:%M")] * self.day_config.num_points,
            df[time_col]
        ))
        return df.copy()

    def convert(self, df, time_col, data_cols, keep_cols=()):
        df = self.select_data(df, data_cols)
        df = self.infer_time(df, time_col)
        out_dict = {
            col: np.array(df[col].tolist()).flatten()
            for col in [self.ods_time_col, 'report_date', 'report_time', 'interval', 'interval_day'] + data_cols
        }
        # 保留实体列（如 station）：每条源行会展开成 num_points 行，
        # 实体值需按每行展开长度 repeat，才能让下游"逐站分组"还能对上。
        lens = np.array([len(x) for x in df[self.ods_time_col]])
        for col in keep_cols:
            out_dict[col] = np.repeat(df[col].to_numpy(), lens)
        out_df = pd.DataFrame.from_dict(out_dict)
        return out_df


class SolarMetricCalculator:
    """
        1. Only deal with 1 target for evaluation
        2. Assume groundtruth is available
        3. Calculate metric by day
    """
    ods_time_col: str = 'dtime'
    ods_pred_col: str = 'predict_power'
    ods_out_cols: tuple = ('dtime', 'report_date', 'report_time', 'interval', 'interval_day', 'predict_power', 'ModelVersion')

    def __init__(
            self,
            model_version: str,
            cap_power: float,
            save_dir: str = './',
            path_to_groundtruth: str = '',
            time_col: str = 'timestamp_win',
            groundtruth_col: str = 'observe_power_future',
            predict_col: str = 'observe_power_predict',
            station_col: str = '',
            cap_by_station: Union[dict, pd.Series] = None,
    ):
        self.model_version = model_version
        self.cap_power = cap_power
        self.save_dir = save_dir
        self.time_col = time_col
        self.gt_col = groundtruth_col
        self.pred_col = predict_col
        self.station_col = station_col
        # 多站数据:同一 time_col 会被多个站共享, 行的唯一键是 (time_col, station_col)。
        # 外置真值文件必须按全键读取并合并, 否则会把预测配到"同时刻其它站"的真值上。
        self.gt_keys = ([self.time_col] if not self.station_col
                        else [self.time_col, self.station_col])
        self.gt_df = self.load_ds(path_to_groundtruth, self.gt_keys + [self.gt_col])
        if self.gt_df is None:
            # 未提供外部真值文件：data_cols 仍带上 gt_col，
            # 以支持"预测表自带真值列"的用法（generate_report 会保留传入的真值）。
            self.data_cols = [self.pred_col, self.gt_col]
        else:
            self.gt_df[self.time_col] = pd.to_datetime(self.gt_df[self.time_col])
            self.data_cols = [self.pred_col, self.gt_col]
        # 逐站装机容量（MW）：{"1002": 500.0, ...} 或 Series（index=站名）。
        # 提供后按站分组回测（口径与参考版"逐站 cap_power"一致）：
        #   每站用自己的 cap 算日指标，fleet 值 = 各站同日期标值的均值，
        #   月度/均值再自然聚合。未提供的站回退全局 cap_power 并打 warn。
        if cap_by_station is None:
            cap_by_station = {}
        self.cap_by_station = dict(cap_by_station)
        self.ultrashort_converter = UltraShortConverter()
        self.dayahead_converter = DayAheadConverter()

    @staticmethod
    def read_data(path: str, columns=None):
        if columns is None:
            return pd.read_parquet(path, engine='pyarrow')
        else:
            return pd.read_parquet(path, columns=columns, engine='pyarrow')

    def load_ds(self, data: Union[pd.DataFrame, str], columns=None):
        if isinstance(data, pd.DataFrame):
            return data.copy()
        elif os.path.isfile(data):
            df = self.read_data(data, columns)
            return df
        elif os.path.isdir(data):
            df = pd.concat([self.read_data(os.path.join(data, _), columns) for _ in os.listdir(data)])
            return df
        else:
            return None

    def merge_groundtruth(self, df: pd.DataFrame):
        # 真值来源：优先用传入 df 自带 gt_col；没有时才从外部 gt_df merge。
        if (self.gt_col not in df) and (self.gt_df is not None):
            df = df.merge(self.gt_df, on=self.gt_keys, how='left')
        return df

    def convert2ultrashort_ods(self, data: Union[pd.DataFrame, str], convert_only=True):
        """
            If groundtruth data is not provided, return data in strict ODS format;
            else return data for metric calculation
        """
        df_local = self.load_ds(data)
        df_local[self.time_col] = pd.to_datetime(df_local[self.time_col])
        df_local.dropna(inplace=True)

        df_local = self.ultrashort_converter.convert(df_local, self.time_col, self.data_cols)
        df_local['ModelVersion'] = [self.model_version] * len(df_local)
        if convert_only:
            df_local.rename(columns={self.pred_col: self.ods_pred_col}, inplace=True)
            return df_local[list(self.ods_out_cols)]
        else:
            return df_local

    def convert2dayahead_ods(self, data: Union[pd.DataFrame, str], convert_only=True):
        """
            If groundtruth data is not provided, return data in strict ODS format;
            else return data for metric calculation
        """
        df = self.load_ds(data)
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        df_local = df[(df[self.time_col].dt.hour == 9) & (df[self.time_col].dt.minute == 0)].copy()
        df_local.dropna(inplace=True)

        out_df = self.dayahead_converter.convert(df_local, self.time_col, self.data_cols)
        out_df['ModelVersion'] = [self.model_version] * len(out_df)
        if convert_only:
            out_df.rename(columns={self.pred_col: self.ods_pred_col}, inplace=True)
            return out_df[list(self.ods_out_cols)]
        else:
            return out_df

    def _station_cap(self, df):
        """返回（per-station cap: Series, index=station 名, 缺失站按全局 cap 兜底）。

        cap 来源优先级：传入的 cap_by_station（由调用方从仓库根 info.csv 构建，
        每站装机容量的唯一权威）优先于 df 自带 cap_power_on 列（unified_inference
        保留源表特征列时），避免历史/漂移的 parquet cap 覆盖权威值。全缺的站回退
        self.cap_power 并 warn。
        """
        if self.station_col and self.cap_by_station:
            map_ = pd.Series(self.cap_by_station)
        elif self.station_col and "cap_power_on" in df.columns:
            map_ = df.drop_duplicates(self.station_col).set_index(self.station_col)["cap_power_on"]
        else:
            return None
        missing = map_.index[~map_.astype(float).notna()]
        if len(missing):
            map_.loc[missing] = self.cap_power
            print(f"  [warn] {len(missing)} 个站缺少有效 cap，回退全局 {self.cap_power}")
        return map_

    def _calc_metric(self, df, groupby, metric_handler, cap_by_station: pd.Series = None):
        """分组计算指标。指定 cap_by_station 时按"组内主力站"取该站 cap：DS 分组恰好
        单站单行（同源行各月一条），用组内众数站即可；ODS 展开后同组全为同一站。"""
        out_cols = [groupby] + list(metric_handler.name2method.keys())
        out_dict = {k: [] for k in out_cols}
        for key, group in df.groupby(by=groupby):
            out_dict[groupby].append(key)
            if cap_by_station is not None and self.station_col in group.columns:
                st = group[self.station_col].mode().iloc[0]
                cap = float(cap_by_station.get(st, self.cap_power))
            else:
                cap = self.cap_power
            for k, method in metric_handler.name2method.items():
                out_dict[k].append(method(
                    np.array(group[self.pred_col].tolist()).flatten(),
                    np.array(group[self.gt_col].tolist()).flatten(),
                    cap
                ))
        return pd.DataFrame.from_dict(out_dict)

    def _calc_ods_metric(self, df, time_col, cap_by_station: pd.Series = None):
        """
            Calculate metric by day, then average to monthly.
            Output:
                - day count
                - sample count
                - metrics
        """
        groupby = 'Date'
        df[groupby] = list(map(lambda x: x.date(), df[time_col]))
        metric_df = self._calc_metric(df, groupby, ODSMetric(), cap_by_station=cap_by_station)
        # resample to get monthly metrics
        metric_df[groupby] = pd.to_datetime(metric_df[groupby])
        metric_df.set_index(groupby, inplace=True)
        # 使用显式月末偏移，兼容已移除 'M' 重采样别名的 pandas 3。
        mean_df = metric_df.resample(pd.offsets.MonthEnd()).mean()
        mean_df['day_count'] = metric_df.iloc[:, 0].resample(pd.offsets.MonthEnd()).count()
        mean_df = mean_df.round(4)
        mean_df.index = mean_df.index.to_period('M')
        mean_df.index = mean_df.index.astype(str)
        mean_df.reset_index(names=['YearMonth'], inplace=True)
        mean_df = mean_df[['YearMonth', 'day_count'] + list(ODSMetric().name2method.keys())]
        return mean_df.T

    def _calc_ds_metric(self, df, time_col, cap_by_station: pd.Series = None):
        """
            Calculate by month directly.
            Output:
                - day count
                - sample count
                - metrics
        """
        groupby = 'YearMonth'
        df[groupby] = list(map(lambda x: x.strftime('%Y-%m'), df[time_col]))
        df['Date'] = list(map(lambda x: x.date(), df[time_col]))
        metric_df = self._calc_metric(df, groupby, DSMetric(), cap_by_station=cap_by_station)
        metric_df = metric_df.merge(df.value_counts([groupby]).reset_index(name='sample_count'), on=groupby)
        metric_df = metric_df.merge(df.groupby(groupby)['Date'].nunique().reset_index(name='day_count'), on=groupby)
        metric_df = metric_df.round(4)
        # Get columns in fixed order
        metric_df = metric_df[[groupby, 'day_count', 'sample_count'] + list(DSMetric().name2method.keys())]
        return metric_df.T

    def calc_ultrashort_ods(self, df, cap_by_station=None):
        # Scenario1: forecast timepoint at 4 hours later, non-overlap
        df_local = self.convert2ultrashort_ods(df, convert_only=False)
        return self._calc_ods_metric(df_local, self.ods_time_col, cap_by_station=cap_by_station)

    def calc_short_ods(self, df, cap_by_station=None):
        # Scenario2: forecast next day's pv power at 9am, non-overlap
        df_local = self.convert2dayahead_ods(df, convert_only=False)
        return self._calc_ods_metric(df_local, self.ods_time_col, cap_by_station=cap_by_station)

    def calc_ultrashort(self, df, cap_by_station=None):
        # Scenario3:  forecast timepoint at 4 hours later; DS
        df_local = df.copy()
        df_local = self.ultrashort_converter.select_data(df_local, self.data_cols)
        return self._calc_ds_metric(df_local, self.time_col, cap_by_station=cap_by_station)

    def calc_short(self, df, cap_by_station=None):
        # Scenario4: forecast next day's pv power at 9am; DS
        df_local = df[(df[self.time_col].dt.hour == 9) & (df[self.time_col].dt.minute == 0)].copy()
        df_local = self.dayahead_converter.select_data(df_local, self.data_cols)
        return self._calc_ds_metric(df_local, self.time_col, cap_by_station=cap_by_station)

    def calc_48hours(self, df, cap_by_station=None):
        # Scenario5: forecast next 48 hours; DS
        df_local = df.copy()
        return self._calc_ds_metric(df_local, self.time_col, cap_by_station=cap_by_station)

    def generate_report(self, data, writer=None, sheet_name_fn=None):
        df = self.load_ds(data)
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        # 真值来源：
        #   - 有外部 GT（gt_df 非空）: 丢弃传入的真值列，统一从 gt_df merge（旧行为）
        #   - 无外部 GT: 保留传入 df 自带的真值列（如预测表由 unified_inference + test parquet 组装而来）
        if self.gt_df is not None and self.gt_col in df.columns:
            df.drop(columns=[self.gt_col], inplace=True)
        df = self.merge_groundtruth(df)
        df.dropna(inplace=True)

        # 逐站 cap（方案A核心）：优先 df 自带 cap_power_on，其次 __init__ 传入映射；
        # 有站级 cap 时按站分组回测，fleet 值 = 各站同日期指标均值。
        cap_by_station = self._station_cap(df)

        daily_metric_dfs = {
            'ods_ultra_short': self.calc_ultrashort_ods(df, cap_by_station=cap_by_station),
            'ods_short': self.calc_short_ods(df, cap_by_station=cap_by_station),
            'ultra_short': self.calc_ultrashort(df, cap_by_station=cap_by_station),
            'short': self.calc_short(df, cap_by_station=cap_by_station),
            '48hours': self.calc_48hours(df, cap_by_station=cap_by_station)
        }
        # ods_* 月表的月份列是位置编号 0..k-1（k=覆盖到的月数）；日指标 resample 到月后
        # 不会自带 'mean' 列。此前用 .drop(columns=12) 是针对某内部表（最后一个月列恰为 12）
        # 的硬编码，对 12 个月全齐的测试数据是静默 no-op、对部分月份数据会 KeyError，
        # 这里改为"若存在 'mean' 列则去掉，再对现有各月列取 mean"，两条路径都与旧行为一致。
        for key in ('ods_ultra_short', 'ods_short'):
            ods = daily_metric_dfs[key]
            if 'mean' in ods.columns:
                ods.drop(columns='mean', inplace=True)
            # 月均值：剔除 sample/样本天数 <=2 天的月份后，各月等权（与汇总口径一致）。
            # 天数用 day_count 行（该月有日指标的数据天数），不算日历天数。
            ods['mean'] = _monthly_mean_exclude_short_months(
                ods, _ODS_DAY_ROW, _ODS_METRIC_ROWS)

        # 不传 writer 时自建（旧行为）；传共享 writer 时仅追加 sheet（多套指标合并进同一 xlsx）
        own_file = writer is None
        if own_file:
            writer = pd.ExcelWriter(os.path.join(self.save_dir, f'{self.model_version}-metrics.xlsx'),
                                    engine='xlsxwriter')
        # note: 传共享 writer 时调用方负责 with/close 与 flush
        # sheet_name_fn（可选）：把 metric key 映射成最终 sheet 名。多站评测时调用方可
        # 用它把每个站写成独立 sheet（如 {站}_{指标}），默认 None 保持旧行为（sheet 名=metric key）。
        name_fn = sheet_name_fn or (lambda k: k)
        for k, df in daily_metric_dfs.items():
            df.to_excel(writer, sheet_name=name_fn(k))
        if own_file:
            writer.close()

        return daily_metric_dfs

# ============================================================================
# mF 超短期 16 Horizon ODS 评估补充
#
# 说明：
#   1. 保留上方 SolarMetricCalculator 及其既有行为不变；
#   2. 仅新增一个子类，用于分别评估未来第 1~16 个 15 分钟预测点；
#   3. 每个 horizon 继续复用现有 ODSMetric / _calc_ods_metric 业务口径；
#   4. generate_mf_report() 返回 dict，并保存独立 Excel 文件，避免覆盖原报告。
# ============================================================================


class MFUltraShortMetricCalculator(SolarMetricCalculator):
    """mF 模型的 16-horizon 超短期 ODS 评估器。

    输入 DataFrame 与原 ``SolarMetricCalculator`` 保持一致：
        - ``timestamp_win``：预测起报时间；
        - ``observe_power_predict``：长度至少为 16 的预测序列；
        - ground truth 由 ``path_to_groundtruth`` 提供，目标列同样为长度至少 16 的序列。

    新增输出：
        ``ods_ultra_short_1`` ... ``ods_ultra_short_16``

    其中 ``ods_ultra_short_16`` 与原 ``ods_ultra_short`` 使用相同的第 16 个点、
    相同的 +240min 时间偏移以及相同的 ODS 指标计算口径。
    """

    mf_pred_len: int = 16

    def _validate_horizon(self, horizon: int) -> int:
        """校验 1-based horizon，并返回对应的 0-based 序列索引。"""
        if not isinstance(horizon, (int, np.integer)):
            raise TypeError(f"horizon 必须为整数，当前为 {type(horizon).__name__}。")
        if horizon < 1 or horizon > self.mf_pred_len:
            raise ValueError(
                f"horizon 必须位于 [1, {self.mf_pred_len}]，当前为 {horizon}。"
            )
        return int(horizon) - 1

    def _select_horizon_data(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """从预测/标签序列中取指定 horizon 的单点数据。"""
        horizon_index = self._validate_horizon(horizon)
        df_local = df.copy()

        for col in self.data_cols:
            def _select_one(value, col_name=col):
                try:
                    return value[horizon_index]
                except (TypeError, IndexError, KeyError) as exc:
                    raise ValueError(
                        f"列 {col_name!r} 的样本无法读取第 {horizon} 个预测点；"
                        f"mF 评估要求该列每个样本至少包含 {self.mf_pred_len} 个点。"
                    ) from exc

            df_local[col] = list(map(_select_one, df_local[col]))

        return df_local

    def _infer_horizon_time(
        self,
        df: pd.DataFrame,
        horizon: int,
    ) -> pd.DataFrame:
        """按照 horizon 推导 ODS 时间字段。

        horizon=1  -> +15min, interval=15
        horizon=16 -> +240min, interval=240
        """
        self._validate_horizon(horizon)
        df_local = df.copy()
        df_local[self.time_col] = pd.to_datetime(df_local[self.time_col])

        step = self.ultrashort_converter.day_config.step
        step_size = self.ultrashort_converter.day_config.step_size
        interval_minutes = step_size * horizon

        df_local[self.ods_time_col] = list(map(
            lambda x: x + step * horizon,
            df_local[self.time_col],
        ))
        df_local['interval'] = [interval_minutes] * len(df_local)
        df_local['interval_day'] = [interval_minutes // (24 * 60)] * len(df_local)
        df_local['report_date'] = list(map(
            lambda x: x.strftime("%Y-%m-%d"),
            df_local[self.time_col],
        ))
        df_local['report_time'] = list(map(
            lambda x: x.strftime("%H:%M"),
            df_local[self.time_col],
        ))
        df_local.drop(columns=[self.time_col], inplace=True)
        return df_local

    def convert2ultrashort_ods_at_horizon(
        self,
        data: Union[pd.DataFrame, str],
        horizon: int,
        convert_only: bool = True,
    ) -> pd.DataFrame:
        """将指定 horizon 转换为超短期 ODS 数据。"""
        df_local = self.load_ds(data)
        if df_local is None:
            raise ValueError("未读取到可用于 mF 超短期评估的数据。")

        df_local[self.time_col] = pd.to_datetime(df_local[self.time_col])
        df_local.dropna(inplace=True)

        df_local = self._select_horizon_data(df_local, horizon=horizon)
        # 单点选择后不再需要站级分组（mF 每 horizon 只算单点 ODS），为保
        # 持与父类一致仍在数据传递中带上 station 列（下派生类无用，单站时为空）。
        df_local = self._infer_horizon_time(df_local, horizon=horizon)
        df_local['ModelVersion'] = [self.model_version] * len(df_local)

        if convert_only:
            df_local.rename(columns={self.pred_col: self.ods_pred_col}, inplace=True)
            return df_local[list(self.ods_out_cols)]
        return df_local

    def calc_ultrashort_ods_at_horizon(
        self,
        df: pd.DataFrame,
        horizon: int,
    ) -> pd.DataFrame:
        """计算指定第 1~16 个预测点的 ODS 指标。"""
        df_local = self.convert2ultrashort_ods_at_horizon(
            df,
            horizon=horizon,
            convert_only=False,
        )
        # mF 每 horizon 是逐点单站/单点评估，不引入逐站 cap 分组（与父类多站
        # 按站回测不同），保持历史单站口径不变。
        return self._calc_ods_metric(df_local, self.ods_time_col)

    @staticmethod
    def _postprocess_mf_ods_metric(metric_df: pd.DataFrame) -> pd.DataFrame:
        """沿用原 ods_ultra_short 的月度报告后处理方式。

        月均值：剔除样本天数 <=2 天的月份后各月等权（与 SolarMetricCalculator
        及汇总口径一致），不再硬编码前 12 列。
        """
        out = metric_df.copy()
        if 'mean' in out.columns:
            out = out.drop(columns='mean')
        if 12 in out.columns:
            out = out.drop(columns=12)
        out['mean'] = _monthly_mean_exclude_short_months(
            out, _ODS_DAY_ROW, _ODS_METRIC_ROWS)
        return out

    def generate_mf_report(self, data: Union[pd.DataFrame, str]):
        """生成 mF 第 1~16 horizon 的 ODS 评估结果并保存 Excel。

        Returns
        -------
        OrderedDict[str, pd.DataFrame]
            key 固定为 ``ods_ultra_short_1`` ... ``ods_ultra_short_16``。
        """
        df = self.load_ds(data)
        if df is None:
            raise ValueError("未读取到可用于 mF 超短期评估的数据。")

        df[self.time_col] = pd.to_datetime(df[self.time_col])

        # 与原 generate_report 保持一致：若输入结果中已有标签，仍以 gt_df 为准。
        if self.gt_col in df.columns:
            df.drop(columns=[self.gt_col], inplace=True)

        df = self.merge_groundtruth(df)
        if self.gt_col not in df.columns:
            raise ValueError(
                f"mF 指标计算缺少 ground truth 列 {self.gt_col!r}；"
                "请通过 path_to_groundtruth 提供真实值。"
            )
        df.dropna(inplace=True)

        metric_dfs = OrderedDict()
        for horizon in range(1, self.mf_pred_len + 1):
            key = f'ods_ultra_short_{horizon}'
            metric_df = self.calc_ultrashort_ods_at_horizon(df, horizon=horizon)
            metric_dfs[key] = self._postprocess_mf_ods_metric(metric_df)

        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(
            self.save_dir,
            f'{self.model_version}-mf-ultrashort-metrics.xlsx',
        )
        with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
            for key, metric_df in metric_dfs.items():
                metric_df.to_excel(writer, sheet_name=key)

        return metric_dfs

    def generate_report(self, data: Union[pd.DataFrame, str]):
        """保持与原评估器一致的调用入口，mF 子类返回 16 个 horizon 结果。"""
        return self.generate_mf_report(data)
