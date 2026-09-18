# -*- coding: utf-8 -*-

"""
数据集相关的共有Dataset类文件，所有模型必须兼容此输入。

适配：
    transformers==4.40.1
    datasets>=2.14 (可选依赖; 不装则仅 DataFrame 路径可用)

设计：
    UnifiedDataset 将 读取到的ds层数据转化为对应的batch数据。

    v2 变更: 支持两种后端, 由入参类型自动选择
        - "numpy" 后端: 入参为 pd.DataFrame。__init__ 时全量物化为 numpy,
          后续索引最快, 但内存 = 数据总量。行为与 v1 完全一致。
        - "arrow"  后端: 入参为 datasets.Dataset / parquet 路径 / glob / 路径列表。
          数据保持在 Arrow 内存映射中, __init__ 只做一次性的类型规整(可缓存),
          按行取数时才落到内存。内存占用与数据量无关。

    batch 数据格式不变:
    {
        "column_name_1": value_1,         # torch.Tensor
        ...
        "index" : value_index,
    }
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, default_collate

try:                                    # datasets 为可选依赖
    import datasets as hfds
    from datasets import Dataset as HFDataset
except ImportError:                     # pragma: no cover
    hfds = None
    HFDataset = ()                      # 使 isinstance 判断恒为 False


class UnifiedDataset(Dataset):
    """
    将 pandas.DataFrame 或 datasets.Dataset / parquet 文件 转化为 Dataset 对象。

    列约定:
        - timestamp_cols: 时间列。单元格可以是标量或等长的时间序列(list/np.ndarray)。
          支持 pd.Timestamp / np.datetime64 / 时间字符串 / epoch 整数
          (单位由 epoch_unit 显式指定; 缺省时按量级自动识别并 warning),
          统一归一化为 np.datetime64[ns]。单列内时区约定需一致。
          注意: 纯数字字符串(如 "1700000000")会优先按 epoch 整数解析, 而非日期字符串;
          含分隔符的日期字符串(如 "2024-01-01")不受影响。
        - entity_cols: 实体标识列(一般为字符串), 要求逐行唯一, 仅用于行筛选/对齐,
          不参与网络计算; 出现重复键时 warning 且 entity_to_index 保留首次出现。
        - 其余列为 feature: 数值或等长 List[数值]/np.ndarray(暂不支持变长),
          统一映射为 feature_dtype(默认 np.float32)。
        - label_cols: 指定为 label 的列。仅在 for_hf_trainer=True 时出现在
          __getitem__ 输出中(位于 "labels" 键下); 默认模式下 label 列不会出现在
          __getitem__ 结果里(但仍会从 feature 自动推断中排除, 并参与类型规整与列保留)。

    参数:
        data:
            pd.DataFrame                  -> numpy 后端(全量物化, 行为同 v1)
            datasets.Dataset              -> arrow 后端(惰性, 内存映射)
            str / Path / list[str]        -> 按 parquet 读取为 datasets.Dataset

        timestamp_output:
            "epoch_s"(默认): __getitem__ 直接返回 int64 epoch 秒, 可过 default_collate。
            "epoch_ns": __getitem__ 直接返回 int64 epoch 纳秒, 可过 default_collate。
                注意: NaT 在 epoch 输出中为 int64 最小值, 下游可用
                ts > torch.iinfo(torch.int64).min 做掩码。
            "datetime64": __getitem__ 返回 np.datetime64。语义直观, 但不能直接过
                DataLoader 的 default_collate(torch 无 datetime dtype), 请配合
                UnifiedDataset.collate_fn 使用(时间列在 batch 中变为 int64 epoch-ns 张量);

        epoch_unit:
            为了解析输入时间戳的数值型(epoch)时间列的单位: "s"/"ms"/"us"/"ns"。仅对 epoch 数值列生效,
            datetime64/Timestamp/时间字符串列不受影响。多时间列共用同一单位,
            单位不同的列请在传入前自行转为 datetime64。
            None(默认): 按该列最大值的量级自动识别, 并发出 warning 提示显式指定。
            注意: arrow 后端为避免全列扫描, 仅按前若干行的量级推断。

        num_proc:
            arrow 后端做类型规整(map)时的并行进程数。

    示例:
        >>> # ---- DataFrame 路径, 与 v1 完全一致 ----
        >>> ds = UnifiedDataset(df, entity_cols=["station_id"])

        >>> # ---- 大数据: parquet 直接进, 不经过 DataFrame ----
        >>> ds = UnifiedDataset(
        ...     "clean/**/*.parquet",
        ...     timestamp_cols=["timestamp_win"],
        ...     entity_cols=["station_id"],
        ...     label_cols=["observe_power_future"],
        ...     timestamp_output="epoch_s",
        ...     epoch_unit="s",
        ...     for_hf_trainer=True,
        ... )
        >>> len(ds), ds.backend
        (12345678, 'arrow')

        >>> # ---- 已有 datasets 对象 ----
        >>> ds = UnifiedDataset(hf_dataset, entity_cols=["station_id"])

    限制:
        - 不支持变长序列: list/ndarray 单元格必须等长;
        - arrow 后端下 data_dict 不再全量存在, 请改用 .column(name) 按需物化单列;
        - .timestamp / .entity_to_index / .entity_keys 在 v2 中改为惰性属性,
          arrow 后端下首次访问会物化整列, 大数据集慎用;
        - NaT 在 epoch 输出中为 int64 最小值, 掩码: ts > torch.iinfo(torch.int64).min。
    """

    # epoch 单位识别表: (上限, 单位); 按量级 10位→s, 13位→ms, 16位→us, 19位→ns
    _EPOCH_UNITS = ((1e11, "s"), (1e14, "ms"), (1e17, "us"), (float("inf"), "ns"))
    # epoch 单位的典型峰值量级, 用于显式指定时的 sanity check
    _UNIT_TYPICAL_PEAK = {"s": 1e10, "ms": 1e13, "us": 1e16, "ns": 1e19}
    # 各单位相对秒的倍数, 用于 arrow 后端的整数重标定
    _UNIT_FACTOR = {"s": 1, "ms": 10**3, "us": 10**6, "ns": 10**9}

    def __init__(
        self,
        data,                                          # [改动] df -> data, 支持多来源
        timestamp_cols: list[str] | None = None,
        entity_cols: list[str] | None = None,
        feature_cols: list[str] | None = None,
        label_cols: list[str] | None = None,
        feature_dtype: np.dtype = np.float32,
        timestamp_output: str = "epoch_s",
        epoch_unit: str | None = None,
        for_hf_trainer: bool = False,
        num_proc: int = 1,                             # [新增] arrow 后端 map 并行度
        keep_in_memory: bool = False,                  # [新增] 读 parquet 时是否直接进内存
        features=None,                                 # [新增] 显式 schema, 防多文件类型漂移
        precast: bool = False,                         # [新增] 一次性把数值列压到 feature_dtype
        _ts_plan: dict | None = None,                  # 内部使用: 克隆时透传时间列解析方案
    ):
        if timestamp_output not in ("datetime64", "epoch_ns", "epoch_s"):
            raise ValueError(f"timestamp_output 仅支持 'datetime64'/'epoch_ns'/'epoch_s', 收到: {timestamp_output!r}")
        self.timestamp_output = timestamp_output
        if epoch_unit not in (None, "s", "ms", "us", "ns"):
            raise ValueError(f"epoch_unit 仅支持 None/'s'/'ms'/'us'/'ns', 收到: {epoch_unit!r}")
        self.epoch_unit = epoch_unit
        self.feature_dtype = feature_dtype
        self.for_hf_trainer = for_hf_trainer
        self.num_proc = num_proc
        self.precast = precast

        # arrow 后端内部统一以该单位存 int64; datetime64 输出模式需要 ns 精度
        self._store_unit = "s" if timestamp_output == "epoch_s" else "ns"

        # 时间列解析方案: col -> (kind, src_unit); kind in {"datetime64", "epoch"}
        # arrow 后端据此在"取行时"换算, 不重写底表
        self._ts_plan: dict[str, tuple[str, str | None]] = dict(_ts_plan or {})

        # 惰性属性的缓存
        self._timestamp_cache: np.ndarray | None = None
        self._entity_keys: list | None = None
        self._entity_to_index: dict | None = None
        self._entity_index_built = False

        # ---------- 0. 识别来源, 归一为 DataFrame 或 HFDataset ----------  [新增]
        source, all_columns = self._resolve_source(data, keep_in_memory, features)

        # ---------- 1. 解析三类列 ----------  (逻辑同 v1, 只是列名来源抽象了)
        timestamp_cols = ["timestamp_win"] if timestamp_cols is None else list(timestamp_cols)
        entity_cols = [] if entity_cols is None else list(entity_cols)

        if missing := [c for c in timestamp_cols if c not in all_columns]:
            raise KeyError(f"timestamp_cols 不在数据中: {missing}")
        self.timestamp_cols = timestamp_cols

        self.entity_cols = [c for c in entity_cols if c in all_columns]
        if dropped := set(entity_cols) - set(self.entity_cols):
            warnings.warn(f"entity_cols 不在数据中, 已忽略: {sorted(dropped)}")

        label_cols = [] if label_cols is None else list(label_cols)
        if missing := [c for c in label_cols if c not in all_columns]:
            raise KeyError(f"label_cols 不在数据中: {missing}")
        self.label_cols = label_cols

        if feature_cols is None:
            reserved = set(self.timestamp_cols) | set(self.entity_cols) | set(self.label_cols)
            self.feature_cols = [c for c in all_columns if c not in reserved]
        else:
            if missing := [c for c in feature_cols if c not in all_columns]:
                raise KeyError(f"feature_cols 不在数据中: {missing}")
            if overlap := set(feature_cols) & (set(self.timestamp_cols) | set(self.entity_cols)):
                raise ValueError(f"feature_cols 与 timestamp/entity 列重叠: {sorted(overlap)}")
            self.feature_cols = list(feature_cols)

        # ---------- 2. 按后端分别准备数据 ----------
        if self.backend == "numpy":
            self._init_numpy_backend(source)
        else:
            self._init_arrow_backend(source)

        # ---------- 3. 兼容属性 ----------
        self.column_names = self.feature_cols

    # ==================================================================
    # 来源识别                                                    [新增]
    # ==================================================================
    def _resolve_source(self, data, keep_in_memory: bool, features=None):
        """返回 (source, all_columns), 并设置 self.backend / self.length。"""
        # 1) parquet 路径 / glob / 路径列表
        if isinstance(data, (str, Path)) or (
            isinstance(data, (list, tuple)) and data and isinstance(data[0], (str, Path))
        ):
            if hfds is None:
                raise ImportError("从 parquet 构建需要安装 datasets: pip install datasets")
            files = [str(p) for p in (data if isinstance(data, (list, tuple)) else [data])]
            ds = hfds.load_dataset(
                "parquet", data_files={"train": files},
                keep_in_memory=keep_in_memory,
                features=features,      # 多文件时某列全空会被推断成 null 类型, 显式给 schema 可避免
            )["train"]
            self.backend = "arrow"
            self.length = len(ds)

        # 2) 已经是 datasets 对象
        elif isinstance(data, HFDataset):
            ds = data
            self.backend = "arrow"
            self.length = len(ds)

        elif hfds is not None and isinstance(data, hfds.DatasetDict):
            raise TypeError("收到 DatasetDict, 请先取出具体 split, 如 data['train']")

        elif hfds is not None and isinstance(data, hfds.IterableDataset):
            raise TypeError(
                "不支持 IterableDataset(流式): 本类需要随机索引与 __len__。"
                "请改用 load_dataset(..., streaming=False)"
            )

        # 3) DataFrame
        elif isinstance(data, pd.DataFrame):
            ds = data
            self.backend = "numpy"
            self.length = len(data)

        else:
            raise TypeError(
                f"不支持的数据源类型: {type(data)}; "
                f"支持 pd.DataFrame / datasets.Dataset / parquet 路径"
            )

        if self.length == 0:
            raise ValueError("数据为空, 无法构建 Dataset")

        all_columns = list(ds.columns) if self.backend == "numpy" else list(ds.column_names)
        return ds, all_columns

    # ==================================================================
    # numpy 后端 (原 v1 逻辑, 未改动)
    # ==================================================================
    def _init_numpy_backend(self, df: pd.DataFrame) -> None:
        self._hf = None
        self._numeric_cols = []          # numpy 后端在物化时已完成 dtype 归一
        self.data_dict: dict[str, np.ndarray] = {}
        for col in self.label_cols:
            self.data_dict[col] = self._to_feature_array(df[col], col, self.feature_dtype)
        for col in self.feature_cols:
            self.data_dict[col] = self._to_feature_array(df[col], col, self.feature_dtype)
        for col in self.timestamp_cols:
            self.data_dict[col] = self._to_timestamp_array(df[col], col, self.epoch_unit)
        for col in self.entity_cols:
            if col in self.timestamp_cols:
                continue
            self.data_dict[col] = df[col].to_numpy()

        if self.timestamp_output == "epoch_ns":
            self._epoch_dict = {c: self.data_dict[c].astype(np.int64) for c in self.timestamp_cols}
        elif self.timestamp_output == "epoch_s":
            self._epoch_dict = {
                c: self.data_dict[c].astype("datetime64[s]").astype(np.int64)
                for c in self.timestamp_cols
            }
        else:
            self._epoch_dict = {}

    # ==================================================================
    # arrow 后端                                                  [新增]
    # ==================================================================
    def _init_arrow_backend(self, ds) -> None:
        """只做"规划", 不重写底表。

        v3 变更(内存/磁盘): 原实现对每个时间列和每个数值列各调一次
        cast_column/map, 而 cast_column 是一次全表复制(整张表所有列都会被
        重新写进新的 arrow cache)。列一多就是 N 倍磁盘放大, 千万行量级会写掉
        几十上百 GB 中间文件。现在:
            - 时间列: 记录 (kind, src_unit), 在 _format_ts 里按行换算 -> 零重写
            - 数值列: 取行时 astype 到 feature_dtype -> 零重写
            - 仅字符串时间列不得不 map 一次(建议在上游就转成 epoch 整数)
        理想情况下整个 __init__ 只是打开内存映射, 与数据量无关。
        """
        self.data_dict = {}          # arrow 后端下为空, 请用 .column()
        self._epoch_dict = {}

        for col in self.timestamp_cols:
            ds = self._plan_timestamp(ds, col)

        keep = list(dict.fromkeys(
            self.timestamp_cols + self.entity_cols + self.feature_cols + self.label_cols
        ))
        if drop := [c for c in ds.column_names if c not in keep]:
            ds = ds.remove_columns(drop)      # 只改 schema, 不复制数据

        if self.precast:
            ds = self._precast_numeric(ds)

        self._hf = ds.with_format("numpy", columns=keep)
        self._check_fixed_length()

    def _precast_numeric(self, ds):
        """可选: 一次性把 float64 数值列压到 feature_dtype。

        代价是一次全表重写(磁盘写一份新 cache), 收益是之后每个 epoch 少读一半字节。
        训练轮数多、上游又没做 float32 规整时值得开; 只跑一两轮就别开。
        关键是所有列合并成 **一次** cast, 而不是每列一次。
        """
        target = hfds.Value(np.dtype(self.feature_dtype).name)
        new_features, changed = dict(ds.features), []
        for col in dict.fromkeys(self.feature_cols + self.label_cols):
            feat = ds.features[col]
            try:
                if isinstance(feat, hfds.Value):
                    if feat.dtype != target.dtype and feat.dtype.startswith("float"):
                        new_features[col] = target
                        changed.append(col)
                elif hasattr(feat, "feature"):
                    inner = getattr(feat.feature, "dtype", None)
                    if inner and inner != target.dtype and inner.startswith("float"):
                        new_features[col] = type(feat)(target)
                        changed.append(col)
            except Exception as e:                       # noqa: BLE001
                warnings.warn(f"列 '{col}' 规划 cast 失败, 跳过: {e}")
        if not changed:
            return ds
        warnings.warn(
            f"precast=True: 将对 {changed} 执行一次全表重写(整表所有列都会复制到新 cache), "
            f"请确认磁盘余量充足"
        )
        return ds.cast(hfds.Features(new_features), num_proc=self.num_proc)

    def _plan_timestamp(self, ds, col: str):
        """决定该时间列如何在取行时换算; 只有字符串列会真正改写数据。"""
        dtype, is_array = self._feature_dtype_str(ds.features[col])

        # a) Arrow 原生 timestamp: numpy format 直接给 datetime64, 无需任何转换
        if isinstance(dtype, str) and dtype.startswith("timestamp"):
            self._ts_plan[col] = ("datetime64", None)
            return ds

        # b) 数值 epoch: 只记源单位, 换算推迟到取行时
        if isinstance(dtype, str) and (dtype.startswith("int") or dtype.startswith("float")):
            if col not in self._ts_plan:
                self._ts_plan[col] = ("epoch", self._infer_epoch_unit_arrow(ds, col))
            return ds

        # c) 字符串 / 其他: 只能 map 一次
        if is_array:
            raise TypeError(
                f"时间列 '{col}' 是非数值的数组列(inner dtype={dtype}), "
                f"arrow 后端不支持; 请在上游转成 epoch 整数"
            )
        warnings.warn(
            f"时间列 '{col}' 为非数值类型({dtype}), 将用 pd.to_datetime 逐批解析, "
            f"并触发一次全表重写。大数据集请在上游(如 DuckDB)预先转成 epoch 整数"
        )
        store_unit = self._store_unit

        def _parse(batch):
            dt = pd.to_datetime(pd.Series(batch[col]), utc=True, errors="coerce").dt.tz_localize(None)
            vals = np.asarray(dt, dtype=f"datetime64[{store_unit}]").astype(np.int64)
            return {col: vals}

        self._ts_plan[col] = ("epoch", store_unit)
        return ds.map(_parse, batched=True, num_proc=self.num_proc,
                      desc=f"parse timestamp {col}")

    @staticmethod
    def _feature_dtype_str(feat) -> tuple[str | None, bool]:
        """取出 feature 的标量 dtype 字符串, 并标明它是不是数组列。
        [修复] 原实现对 Sequence/List 列直接 str(feat), 会漏判成"非数值"
        而走到 pandas 解析分支, 向量时间戳列因此崩溃。"""
        if hasattr(feat, "feature"):                 # Sequence / List
            inner = feat.feature
            return getattr(inner, "dtype", None), True
        return getattr(feat, "dtype", None), False

    def _infer_epoch_unit_arrow(self, ds, col: str) -> str:
        """arrow 后端不做全列扫描, 只按前若干行的量级推断。"""
        if self.epoch_unit is not None:
            return self.epoch_unit
        n = min(len(ds), 256)
        raw = ds[:n][col]
        try:
            sample = np.asarray([v for v in raw if v is not None], dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"时间列 '{col}' 前 {n} 行无法解析为数值(可能含 null 或混合类型): {e}"
            ) from e
        peak = float(np.nanmax(np.abs(sample))) if sample.size else 0.0
        unit = self._epoch_unit(peak)
        warnings.warn(
            f"时间列 '{col}' 未显式指定 epoch_unit, 已按前 {n} 行的量级识别为 '{unit}' "
            f"(峰值 {peak:.3g}); arrow 后端不做全列扫描, 强烈建议显式传入 epoch_unit"
        )
        return unit

    @staticmethod
    def _scale_epoch(v, src_unit: str, dst_unit: str):
        """epoch 整数在单位间换算, 保持标量/数组的形态不变。

        [修复] NaT 哨兵(int64 最小值)与 NaN 先用掩码摘出来, 换算后再写回,
        原实现直接把哨兵值参与乘法, 整数溢出。
        """
        NAT = np.iinfo(np.int64).min
        a = np.asarray(v)
        scalar = a.ndim == 0

        if a.dtype.kind == "f":
            nat = np.isnan(a)
        elif a.dtype.kind in "iu":
            nat = a == NAT
        else:
            nat = np.zeros(a.shape, dtype=bool)

        src = UnifiedDataset._UNIT_FACTOR[src_unit]
        dst = UnifiedDataset._UNIT_FACTOR[dst_unit]
        safe = np.where(nat, 0, a).astype(np.int64)
        out = safe * (dst // src) if dst >= src else safe // (src // dst)
        out = np.where(nat, NAT, out).astype(np.int64)
        return out[()] if scalar else out

    def _check_fixed_length(self, n_probe: int = 64) -> None:
        """抽查前若干行, 确认数组列定长, 同时缓存需要 dtype 对齐的数值列。
        [修复] 原实现只记录长度、从不比较, 变长数据会一路跑到 collate 才炸。"""
        n = min(self.length, n_probe)
        probe = self._hf[:n]
        self._array_lengths: dict[str, int] = {}
        self._numeric_cols: list[str] = []
        for col in list(dict.fromkeys(self.feature_cols + self.label_cols)):
            vals = probe[col]
            first = np.asarray(vals[0])
            if first.dtype.kind in "fiub":
                self._numeric_cols.append(col)
            elif first.dtype.kind == "O":
                # 上游未清洗时最常见的两种成因: 列里有 null, 或数组列变长。
                # 不拦住的话会一路漏到 collate, 报一个完全指不到这里的错。
                raise TypeError(
                    f"列 '{col}' 在 numpy format 下是 object 类型, 通常意味着"
                    f"列内存在 null 或变长数组。请在上游(DuckDB)填充/过滤后再进; "
                    f"若为标识列请移入 entity_cols"
                )
            lens = {int(np.asarray(v).shape[0]) for v in vals if np.asarray(v).ndim >= 1}
            if not lens:
                continue
            if len(lens) > 1:
                raise ValueError(
                    f"列 '{col}' 在前 {n} 行内序列长度不一致(暂不支持变长序列): {sorted(lens)}"
                )
            self._array_lengths[col] = lens.pop()

    # ==================================================================
    # 惰性属性 (v1 中为 __init__ 里的实例属性)                     [改动]
    # ==================================================================
    @property
    def timestamp(self) -> np.ndarray:
        """datetime64[ns] 全列, 便于按时间筛选行。arrow 后端首次访问会物化整列。"""
        if self._timestamp_cache is None:
            col = self.timestamp_cols[0]
            if self.backend == "numpy":
                self._timestamp_cache = self.data_dict[col]
            else:
                warnings.warn(
                    f"访问 .timestamp 会把 '{col}' 整列物化到内存 "
                    f"({self.length} 行 × 8B ≈ {self.length * 8 / 1e9:.2f} GB); "
                    f"仅做时间切分时建议改用 .filter_by_time()"
                )
                self._timestamp_cache = self._ts_as_datetime64(col)
        return self._timestamp_cache

    def _ts_as_datetime64(self, col: str) -> np.ndarray:
        """把某个时间列整列物化成 datetime64[ns](arrow 后端专用)。"""
        kind, src = self._ts_plan.get(col, ("epoch", self._store_unit))
        raw = self._hf[col]
        if kind == "datetime64":
            return np.asarray(raw).astype("datetime64[ns]")
        return np.asarray(self._scale_epoch(raw, src, "ns")).astype("datetime64[ns]")

    # 每行大约要 2 个 python 对象 + 1 个 dict 槽位, 经验值约 2.5KB/千行
    _ENTITY_INDEX_MAX_ROWS = 2_000_000

    def _build_entity_index(self) -> None:
        if self._entity_index_built:
            return
        if self.entity_cols and self.length > self._ENTITY_INDEX_MAX_ROWS:
            # .tolist() 会把整列变成 N 个 python str 对象, 再加一个等大的 dict,
            # 千万行量级是 20~30GB。惰性只是把 OOM 推迟到某次不经意的访问。
            raise RuntimeError(
                f"{self.length} 行的数据集不适合构建 entity 索引"
                f"(预计占用 {self.length * 2.5e-6:.1f} GB 以上)。"
                f"请先 select()/filter_by_time() 取子集再访问 entity_to_index/entity_keys"
            )
        self._entity_index_built = True
        if not self.entity_cols:
            return
        cols = [self.column(c).tolist() for c in self.entity_cols]
        keys = list(zip(*cols))
        if len(self.entity_cols) == 1:
            keys = [k[0] for k in keys]
        self._entity_keys = keys
        self._entity_to_index = {}
        for i, k in enumerate(keys):
            if k in self._entity_to_index:
                warnings.warn(
                    f"entity 键 {k!r} 在第 {self._entity_to_index[k]} 行与第 {i} 行重复, "
                    f"entity_to_index 仅保留首次出现"
                )
            else:
                self._entity_to_index[k] = i

    @property
    def entity_keys(self) -> list | None:
        self._build_entity_index()
        return self._entity_keys

    @property
    def entity_to_index(self) -> dict | None:
        self._build_entity_index()
        return self._entity_to_index

    _COLUMN_WARN_BYTES = 512 * 1024**2          # 单列物化超过 512MB 就提醒

    def column(self, name: str) -> np.ndarray:
        """按需物化单列。arrow 后端下替代 data_dict[name]。"""       # [新增]
        if self.backend == "numpy":
            return self.data_dict[name]
        width = self._array_lengths.get(name, 1)
        est = self.length * width * np.dtype(self.feature_dtype).itemsize
        if est > self._COLUMN_WARN_BYTES:
            warnings.warn(
                f"物化列 '{name}' 预计占用约 {est / 1e9:.2f} GB "
                f"({self.length} 行 × {width} × {np.dtype(self.feature_dtype).itemsize}B); "
                f"如只需部分行请先 select()"
            )
        # return np.asarray(self._hf[name])
        arr = np.asarray(self._hf[name])
        # v3 把数值列的 cast 推迟到了取数时, column() 是绕过 _get_row 的旁路,
        # 这里必须同样归一, 否则 ds[i][col] 是 float32 而 ds.column(col) 是 float64。
        if name in self._numeric_cols and arr.dtype != self.feature_dtype:
            arr = arr.astype(self.feature_dtype, copy=False)
        return arr

    # ==================================================================
    # 行筛选 / 切分                                                [新增]
    # ==================================================================
    def _clone_kwargs(self) -> dict:
        # [修复] 克隆时必须把时间列的解析方案原样带过去。
        # v2 曾在这里回传 epoch_unit, 新实例会对已换算过的列再换算一次;
        # v3 改为不改写底表, 因此直接透传 _ts_plan, 语义更明确也更安全。
        return dict(
            timestamp_cols=self.timestamp_cols,
            entity_cols=self.entity_cols,
            feature_cols=self.feature_cols,
            label_cols=self.label_cols,
            feature_dtype=self.feature_dtype,
            timestamp_output=self.timestamp_output,
            epoch_unit=self.epoch_unit,
            for_hf_trainer=self.for_hf_trainer,
            num_proc=self.num_proc,
            _ts_plan=self._ts_plan,
        )

    def select(self, indices) -> "UnifiedDataset":
        """按行号取子集, arrow 后端零拷贝(只改 indices 映射)。"""
        if self.backend == "arrow":
            return UnifiedDataset(self._hf.select(indices), **self._clone_kwargs())
        raise NotImplementedError("numpy 后端请直接切分原 DataFrame")

    def filter_by_time(self, start=None, end=None) -> "UnifiedDataset":
        """按时间半开区间 [start, end) 切分, 返回新的 UnifiedDataset, 不物化整列。

        [修复] 原实现在 numpy 后端返回行号数组、arrow 后端返回 Dataset,
        返回类型不一致。numpy 后端本就没保留原 df, 这里直接明确不支持。
        """
        if self.backend != "arrow":
            raise NotImplementedError(
                "numpy 后端请直接切分原 DataFrame 后重建, 例如 "
                "UnifiedDataset(df[df['ts'] < cutoff], ...)"
            )
        col = self.timestamp_cols[0]
        unit = self._store_unit

        def _to_epoch(x) -> int:
            # [修复] 数值边界会被 pd.Timestamp 按纳秒解释(1970 年附近),
            # 静默给出全量结果, 必须显式拒绝
            if isinstance(x, (int, float, np.integer, np.floating)):
                raise TypeError(
                    f"filter_by_time 的边界不支持数值类型(收到 {x!r}): "
                    f"pd.Timestamp 会把数值当纳秒解释, 极易静默出错。"
                    f"请传入日期字符串 / pd.Timestamp / np.datetime64"
                )
            return int(np.datetime64(pd.Timestamp(x), unit).astype(np.int64))

        lo = _to_epoch(start) if start is not None else None
        hi = _to_epoch(end) if end is not None else None

        # 只物化这一列的 int64(N × 8B), 再用 select 建立行号映射。
        # 比 Dataset.filter 省一次全表重写, 也避开了 filter 对当前 format 的歧义。
        ts = self._ts_as_datetime64(col).astype(f"datetime64[{unit}]").astype(np.int64)
        mask = np.ones(len(ts), dtype=bool)
        if lo is not None or hi is not None:
            # [修复] NaT 的 int64 最小值哨兵不参与区间比较:
            # 否则只给 end 时 NaT 行会因 v < hi 恒真而漏进结果
            mask &= ts != np.iinfo(np.int64).min
        if lo is not None:
            mask &= ts >= lo
        if hi is not None:
            mask &= ts < hi
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            raise ValueError(f"时间区间 [{start}, {end}) 内没有样本")
        return UnifiedDataset(self._hf.select(idx), **self._clone_kwargs())

    # ==================================================================
    @staticmethod
    def _epoch_unit(peak: float) -> str:
        """按量级识别 epoch 单位: 10位→s, 13位→ms, 16位→us, 19位→ns。"""
        for limit, unit in UnifiedDataset._EPOCH_UNITS:
            if peak < limit:
                return unit
        return "ns"

    # ------------------------------------------------------------------
    # 列转换 (numpy 后端专用, 与 v1 完全一致)
    # ------------------------------------------------------------------
    @staticmethod
    def _to_stacked_array(s: pd.Series, col: str) -> np.ndarray:
        arr = s.to_numpy()
        first = UnifiedDataset._first_valid(arr)
        if isinstance(first, (list, tuple, np.ndarray)):
            try:
                arr = np.stack([np.asarray(v) for v in arr])
            except ValueError as e:
                raise ValueError(f"列 '{col}' 内序列长度不一致(暂不支持变长序列): {e}") from e
        return arr

    @staticmethod
    def _to_feature_array(s: pd.Series, col: str, dtype) -> np.ndarray:
        arr = UnifiedDataset._to_stacked_array(s, col)
        if np.issubdtype(arr.dtype, np.number):
            return arr.astype(dtype, copy=False)
        try:
            return s.to_numpy(dtype=dtype, na_value=np.nan)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"列 '{col}' 无法转换为 {np.dtype(dtype)}; "
                f"若为标识列请移入 entity_cols, 时间列请移入 timestamp_cols。原始错误: {e}"
            ) from e

    @staticmethod
    def _to_timestamp_array(s: pd.Series, col: str, epoch_unit: str | None = None) -> np.ndarray:
        arr = UnifiedDataset._to_stacked_array(s, col)
        try:
            return UnifiedDataset._to_datetime64(arr, col, epoch_unit)
        except Exception as e:
            raise TypeError(f"时间列 '{col}' 解析失败: {e}") from e

    @staticmethod
    def _to_datetime64(arr: np.ndarray, col: str, epoch_unit: str | None = None) -> np.ndarray:
        arr = np.asarray(arr)
        shape = arr.shape
        if np.issubdtype(arr.dtype, np.datetime64):
            return arr.astype("datetime64[ns]")
        flat = arr.ravel()

        if np.issubdtype(arr.dtype, np.number):
            return UnifiedDataset._epoch_to_dt64(flat, col, epoch_unit).reshape(shape)

        try:
            epochs = flat.astype(np.int64)
        except (TypeError, ValueError, OverflowError):
            epochs = None
        if epochs is not None:
            return UnifiedDataset._epoch_to_dt64(epochs, col, epoch_unit).reshape(shape)

        dt = pd.to_datetime(flat, utc=True)
        dt = dt.tz_localize(None)
        return np.asarray(dt, dtype="datetime64[ns]").reshape(shape)

    @staticmethod
    def _epoch_to_dt64(values: np.ndarray, col: str, epoch_unit: str | None = None) -> np.ndarray:
        if np.issubdtype(values.dtype, np.floating):
            valid = ~np.isnan(values)
            peak = float(np.abs(values[valid]).max(initial=0)) if valid.any() else 0.0
            target = values
        else:
            target = values.astype(np.int64)
            peak = float(np.abs(target).max(initial=0)) if target.size else 0.0

        if epoch_unit is None:
            unit = UnifiedDataset._epoch_unit(peak)
            warnings.warn(
                f"时间列 '{col}' 未显式指定 epoch_unit, 已按量级自动识别为 '{unit}' "
                f"(峰值 {peak:.3g}); 建议在正式流程中显式传入 epoch_unit 以避免误判"
            )
        else:
            unit = epoch_unit
            typical = UnifiedDataset._UNIT_TYPICAL_PEAK[unit]
            if peak > 0 and not (typical / 1e4 < peak < typical * 1e2):
                warnings.warn(
                    f"时间列 '{col}' 峰值 {peak:.3g} 与 epoch_unit='{unit}' 的典型量级不符, "
                    f"请确认单位是否正确"
                )

        dt = pd.to_datetime(target, unit=unit)
        return np.asarray(dt, dtype="datetime64[ns]")

    @staticmethod
    def _first_valid(arr: np.ndarray, max_scan: int = 64):
        for v in arr[:max_scan]:
            try:
                if pd.isna(v):
                    continue
            except (TypeError, ValueError):
                pass
            return v
        return None

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.length

    def _format_ts(self, col: str, v):
        """arrow 后端: 按该列的解析方案换算到 timestamp_output 表示。"""
        kind, src = self._ts_plan.get(col, ("epoch", self._store_unit))

        if kind == "datetime64":                      # 底层就是 Arrow timestamp
            if self.timestamp_output == "datetime64":
                return np.asarray(v).astype("datetime64[ns]")[()] \
                    if np.asarray(v).ndim == 0 else np.asarray(v).astype("datetime64[ns]")
            a = np.asarray(v).astype(f"datetime64[{self._store_unit}]").astype(np.int64)
            return a[()] if a.ndim == 0 else a

        # kind == "epoch": 底层是 src 单位的整数
        if self.timestamp_output == "datetime64":
            a = self._scale_epoch(v, src, "ns")
            a = np.asarray(a).astype("datetime64[ns]")
            return a[()] if a.ndim == 0 else a
        return self._scale_epoch(v, src, self._store_unit)

    def _coerce_dtype(self, row: dict) -> dict:
        """取行时把数值列对齐到 feature_dtype。
        v3: 以此替代 __init__ 里对每个数值列的 cast_column(每次一遍全表复制)。"""
        for c in self._numeric_cols:
            v = row[c]
            if isinstance(v, np.ndarray) and v.dtype != self.feature_dtype:
                row[c] = v.astype(self.feature_dtype, copy=False)
            elif isinstance(v, np.generic) and v.dtype != self.feature_dtype:
                row[c] = v.astype(self.feature_dtype)
        return row

    def _get_row(self, index: int) -> dict:
        """返回该行所有需要的列, 时间列已是输出表示。"""
        if self.backend == "numpy":
            ts_src = self._epoch_dict if self.timestamp_output in ("epoch_ns", "epoch_s") else self.data_dict
            row = {c: self.data_dict[c][index] for c in self.feature_cols}
            row.update({c: self.data_dict[c][index] for c in self.label_cols})
            for c in self.timestamp_cols:
                row[c] = ts_src[c][index]
            for c in self.entity_cols:
                if c not in self.timestamp_cols:
                    row[c] = self.data_dict[c][index]
            return row

        row = dict(self._hf[int(index)])
        for c in self.timestamp_cols:
            row[c] = self._format_ts(c, row[c])
        return self._coerce_dtype(row)

    def _assemble(self, row: dict, index: int) -> dict:
        sample = {col: row[col] for col in self.feature_cols}
        for col in self.timestamp_cols:
            sample[col] = row[col]
        for col in self.entity_cols:
            if col in self.timestamp_cols:
                continue
            sample[col] = row[col]
        sample["index"] = index

        if self.for_hf_trainer:
            labels = {col: row[col] for col in self.label_cols}
            return {"batch_dict": sample, "labels": labels}
        return sample

    def __getitem__(self, index: int) -> dict:
        return self._assemble(self._get_row(index), index)

    def __getitems__(self, indices: list[int]) -> list[dict]:
        """[新增] 批量取数钩子。arrow 后端上一次取一批远快于逐条,
        PyTorch 2.x 的 DataLoader 会自动优先调用它。"""
        if self.backend == "numpy":
            return [self[i] for i in indices]

        batch = self._hf[[int(i) for i in indices]]
        out = []
        for pos, idx in enumerate(indices):
            row = {k: v[pos] for k, v in batch.items()}
            for c in self.timestamp_cols:
                row[c] = self._format_ts(c, row[c])
            out.append(self._assemble(self._coerce_dtype(row), int(idx)))
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def collate_fn(
        batch: list[dict],
        timestamp_output: str = 'epoch_s',
        drop_str_type: bool = True,
        for_hf_trainer: bool = False,
    ) -> dict:
        """
        兼容 datetime64/str 的 DataLoader collate:
        时间列 → int64 epoch(ns/s) 张量(NaT → int64 最小值); str(实体列) → list[str];
        其余走 default_collate。时间特征提取仍在下游 nn.Module 内完成, 职责不变。
        """
        if timestamp_output == 'epoch_ns':
            target_dtype = "datetime64[ns]"
        elif timestamp_output == 'epoch_s':
            target_dtype = "datetime64[s]"
        else:
            raise ValueError(f"collate_fn 不支持的 timestamp_output: {timestamp_output!r}")

        def _is_str_type(v) -> bool:
            if isinstance(v, (str, np.str_)):
                return True
            if isinstance(v, np.ndarray):
                if v.dtype.kind in ('U', 'S'):
                    return True
                if v.dtype.kind == 'O' and v.size > 0 and isinstance(v.flat[0], (str, np.str_)):
                    return True
            if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], (str, np.str_)):
                return True
            return False

        def _process_sample(sample: dict) -> dict:
            processed = {}
            for k, v in sample.items():
                if drop_str_type and _is_str_type(v):
                    continue
                is_scalar = isinstance(v, np.datetime64)
                is_array = isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.datetime64)
                if is_scalar or is_array:
                    v = v.astype(target_dtype).astype(np.int64)
                processed[k] = v
            return processed

        if for_hf_trainer:
            return {
                "batch_dict": default_collate([_process_sample(s["batch_dict"]) for s in batch]),
                "labels":     default_collate([_process_sample(s["labels"])     for s in batch]),
            }

        return default_collate([_process_sample(sample) for sample in batch])