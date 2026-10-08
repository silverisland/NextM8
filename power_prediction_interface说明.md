# 功率序列预测填补接口说明

本文对应 `power_prediction_interface.py` 在 2026-09-24 的实际内容，说明当前的 v0、v1、v2 三个接口。

> 使用建议：实际使用时优先测试 V1（`build_observe_power_new_v1`）；如果推理超时，再切换到 V2（`build_observe_power_new_v2`），按截断位置批量推理。这里是人工选择接口的使用顺序，当前代码不会在超时后自动切换。

> 上线备注：实际上线时，需要将代码中的模型推理调用（当前为 `model.predict(...)`）替换为 **M5 模型的 `unified_inference` 接口**。具体入参、预测长度配置和返回结果格式应按该接口适配，确保预测结果能继续用于后续对齐和填补。当前代码尚未完成此替换。

## 1. 功能概述

每行数据包含观测序列 `observe_power` 和基础序列 `final_power`。接口在观测序列的尾部范围内寻找首个缺失位置，以对应位置截断 `final_power`，调用模型预测，再填补观测截断段中的缺失值。

最终结果写入返回表的 `observe_power` 列，不新增 `observe_power_new` 列。

- 参与处理的行：结果是长度为 672 的一维数组。
- 跳过处理的行：直接使用原始 `final_power`，不强制调整长度。
- 返回表保留原来的行顺序、索引和其他列。
- `timestamp_win` 仅作为记录标识，不因截断而调整。

## 2. 当前文件的使用前提

当前三个函数都没有 `model` 参数，而是在函数内部访问外部变量 `model`。因此，模型必须存在于这些函数所属的全局命名空间中。

文件底部还保留了直接使用 `df` 的业务调用代码。这意味着：

- 当前文件适用于已经准备好 `df` 和 `model` 的业务执行环境。
- 普通 `import power_prediction_interface` 会执行文件底部代码；当前没有预先定义 `df`，会报 `NameError`。
- 在另一个脚本中定义同名的 `df` 或 `model`，不会自动将它们传入被导入模块的全局命名空间。

如需将该文件作为可导入模块使用，需要另行将底部业务调用封装为函数或放到主程序入口，并明确模型的传入或初始化方式。本文只说明当前代码，没有修改这一结构。

## 3. 输入数据

输入是 `pandas.DataFrame`，需要包含以下列：

| 列名 | 含义 | 参与处理时的要求 |
|---|---|---|
| `timestamp_win` | 时间标识 | 传给模型并用于 v2 对齐 |
| `station` | 站点名称 | 传给模型并用于 v2 对齐 |
| `observe_power` | 原始观测功率序列 | 长度 672 的一维数值数组，可有缺失值 |
| `final_power` | 基础功率序列，也是模型输入的来源 | 长度 672 的一维数值数组，不含缺失值 |

有效数组会转换为浮点数。`0` 是有效功率值，不是缺失值。

注意：模型输入中的列名仍然是 `observe_power`，但该列装入的是经过截断、或截断后补齐的 `final_power`。

## 4. 参数与三个接口的区别

当前函数签名为：

```python
build_observe_power_new_v0(
    df,
    tail_length,
    prediction_length=192,
    prediction_name="TransformersAdaptedTimesFM3Model",
)

build_observe_power_new_v1(
    df,
    tail_length=8,
    prediction_length=192,
    prediction_name="TransformersAdaptedTimesFM3Model",
)

build_observe_power_new_v2(
    df,
    tail_length=8,
    prediction_length=192,
    prediction_name="TransformersAdaptedTimesFM3Model",
)
```

参数含义：

| 参数 | 说明 |
|---|---|
| `tail_length` | 从观测序列末尾向前取多少个点，在该范围内从左向右寻找首个缺失位置 |
| `prediction_length` | v0 传给模型的预测长度；v1、v2 当前仅用它检查参数关系，不传给模型 |
| `prediction_name` | 模型返回表中存储预测数组的列名 |

调用时应使用整数，满足 `0 < tail_length <= 672` 且 `prediction_length >= tail_length`。

| 接口 | 模型输入 | 模型调用方式 | 预测长度如何设置 |
|---|---|---|---|
| v0 | 截断后在左侧重复填入 `final_power[0]`，补齐至 672 | 每个需要预测的行单独调用 | `model.predict(model_input, prediction_length=prediction_length)` |
| v1 | 直接使用截断后的序列，不补齐 | 每个需要预测的行单独调用 | `model.predict(model_input)`，由模型自身配置决定 |
| v2 | 直接使用截断后的序列，不补齐 | 截断位置相同的行合成一批，每批调用一次 | `model.predict(model_input)`，由模型自身配置决定 |

例如三行分别截掉 3、5、3 个点：

- v0：输入长度分别为 672、672、672，调用模型 3 次。
- v1：输入长度分别为 669、667、669，调用模型 3 次。
- v2：两条长度 669 的序列合成一批，长度 667 的序列另一批，调用模型 2 次。

v0 人工补入了历史首值，模型看到的输入与 v1/v2 不同，因此三个版本的预测数值不保证相同。v1 与 v2 是否预测一致，也取决于真实模型的批处理行为。

## 5. 逐行处理规则

### 5.1 跳过处理的情况

以下任一情况出现，该行不调用模型，也不进行观测尾部覆盖，最终 `observe_power` 直接取原始 `final_power`：

| 条件 | 最终输出 |
|---|---|
| `final_power` 为 `None` 或单个 `NaN` | 原值 |
| `final_power` 为空数组、长度不为 672，或不是一维数组 | 原值，不补齐也不截取 |
| `final_power` 中含 `None` / `NaN` | 原数组，保留缺失值 |
| `observe_power` 为 `None`、单个标量、长度不为 672，或不是一维数组 | 原始 `final_power` |

因此，返回表的每一行不一定都是长度 672 的数组。只有实际参与覆盖或填补的行保证长度为 672。

### 5.2 观测尾部没有缺失

如果最后 `tail_length` 个观测点全部有效：

```python
final[-tail_length:] = observed[-tail_length:]
```

不调用模型，将这段观测覆盖到 `final_power` 的副本中，再作为最终 `observe_power`。

### 5.3 观测尾部存在缺失

1. 取 `observe_power[-tail_length:]`，从左向右查找第一个缺失点。
2. 将该缺失点在原始 672 点数组中的下标记为 `cut_position`。
3. 取 `final_power[:cut_position]` 作为模型历史输入；v0 还会在左侧补齐。
4. 截断长度为 `cut_length = 672 - cut_position`。
5. 从模型返回的未来预测中取前 `cut_length` 个点。
6. 只用这些预测填补 `observe_power[cut_position:]` 中的缺失位置，保留该段已有观测值。
7. 将补好的尾部写回原始 `final_power` 副本的对应位置。

等价结果为：

```python
filled_tail = np.where(
    np.isnan(observed[cut_position:]),
    predicted[:cut_length],
    observed[cut_position:],
)

new_observe_power = np.concatenate([
    final_power[:cut_position],
    filled_tail,
])
```

这里截断前使用的是 `final_power`，不是原始观测。即使搜索范围内、首个缺失点之前存在有效观测，也仍保留对应的 `final_power` 值。

## 6. 一个具体例子

数组长度为 672，设置 `tail_length=8`，搜索范围是下标 664 到 671。

假设观测最后 8 个点为：

```text
下标：664  665  666  667  668  669  670  671
观测：10   11   NaN  13   NaN  15   16   17
```

首个缺失点为 666，所以 `cut_position=666`，`cut_length=6`。

- v0 模型输入：6 个 `final_power[0]`，后接 `final_power[:666]`，总长 672。
- v1/v2 模型输入：`final_power[:666]`，长度 666。

若预测前 6 个点为 `[100, 101, 102, 103, 104, 105]`，填补后的尾部是：

```text
下标：666  667  668  669  670  671
结果：100  13   102  15   16   17
```

最终数组为 `final_power[:666]` 拼接上述 6 个点，长度恰好为 672。

## 7. 模型返回要求与顺序

模型返回一个 DataFrame，包含标识列以及 `prediction_name` 指定的预测数组列：

```text
timestamp_win | station | TransformersAdaptedTimesFM3Model
```

预测数组的第 0 个点必须对应输入历史序列结束后的第一个时刻，也就是本次截断位置。代码不会自行判断预测的时间起点。

- v0：每次输入一行，检查返回恰好一行，再读取预测数组。
- v1：每次输入一行，直接读取返回表第一行；依赖模型保证记录对应正确。
- v2：每次输入一个等长批次，通过 `timestamp_win + station` 对齐返回结果。批内该组合必须唯一，模型返回顺序可以变化。
- v2 使用 `np.stack` 堆叠同批预测，同批返回的预测数组也需要等长。
- 预测数组必须至少覆盖本次截断长度，否则抛出异常。

v2 按截断位置分组的执行顺序，不会改变最终 DataFrame 的行顺序。代码通过原始行位置回写结果，原表索引重复也不会导致回写错位。

## 8. 调用示例

下面示例适用于已经在同一业务执行环境中定义好三个函数、`df` 和 `model` 的情况，不是当前文件的独立导入示例。

```python
# 模型固定需要 672 点历史输入时
result_v0 = build_observe_power_new_v0(
    df,
    tail_length=8,
    prediction_length=192,
)

# 每行独立推理，历史输入不补齐
result_v1 = build_observe_power_new_v1(
    df,
    tail_length=8,
    prediction_name="TransformersAdaptedTimesFM3Model",
)

# 按截断位置分批推理，历史输入不补齐
result_v2 = build_observe_power_new_v2(
    df,
    tail_length=8,
    prediction_name="TransformersAdaptedTimesFM3Model",
)
```

v1/v2 的预测长度应在模型自身配置中设置。仅改变函数的 `prediction_length` 参数，不会改变这两个版本实际向模型请求的预测长度。

## 9. 文件底部的业务调用与回退

当前文件底部业务代码执行以下逻辑：

1. 将原始 `observe_power` 保存为 `observe_power_original`。
2. `FILL_OBSERVE_POWER=True` 时调用 v1，搜索长度为 8。
3. 若填补过程中抛出任何异常，打印异常，将整张表的 `observe_power` 改为 `final_power`。
4. 关闭填补开关时，也将整张表的 `observe_power` 改为 `final_power`。

整表回退是当前明确保留的业务行为，不是逐行失败隔离：一行异常可能让此前其他行的填补结果也不被采用。

## 10. 填补下标日志

当前 v2 的说明字符串仍提到打印下标，但实际函数体已没有对应的 `print`，因此当前版本不会自动打印每行填补下标。

如后续需要恢复，可在 v2 的 `observed_tail = observed[rows, cut_position:]` 后加入：

```python
for row, row_tail in zip(rows, observed_tail):
    row_position = valid_rows[row]
    fill_indices = cut_position + np.flatnonzero(np.isnan(row_tail))
    print(
        f"行位置={row_position}, "
        f"df索引={df.index[row_position]}, "
        f"station={df.iloc[row_position]['station']}, "
        f"填补下标={fill_indices.tolist()}"
    )
```

下标从 0 开始，表示准备使用预测填补的位置。日志会按批次处理顺序打印，不一定按原表顺序；返回表仍保持原顺序。该片段仅在文档中展示，没有重新加入源码。

## 11. 边界与限制

- `tail_length=672` 且第一个观测点缺失时，v1/v2 的模型输入为空数组；模型必须支持该输入，或调用方限制搜索范围。
- v0 在相同情况下会生成 672 个首值作为模型输入。这仅保证输入长度，不保证其预测效果。
- 当前没有过滤输入中的 `Inf`，也没有检查预测值是否为有限数值。预测缺失位置上的 `NaN` / `Inf` 会被直接写入。
- 有效形状的数组仍需要能转换成数值；非数值字符串等内容会引发异常，而不是按缺失值处理。
- v2 一次堆叠所有有效行，并将同一截断位置的行全部送入模型，未设置批量大小上限。大数据量时需要关注内存和模型容量。
- `df.copy()` 不会递归复制单元格内所有数组；函数执行中会复制参与填补的数组，但跳过行可能继续引用原始 `final_power` 数组。返回后若要原地修改这些数组，需要自行复制。

## 12. pytest 与当前验证状态

依赖为 `numpy`、`pandas`、`pytest`。测试文件是 `test_power_prediction_interface.py`，包含假数据构造、模拟模型以及命令行入口。

测试设计覆盖：

- 逐行截断、左侧补齐、有效观测保留。
- `final_power` 为空、含缺失或长度异常时原样输出。
- `observe_power` 为 `None` 或长度异常时跳过。
- 参与处理的行结果为 672 点。
- v2 相同截断位置合批、预测返回乱序时按键对齐。
- 原表顺序和索引保留、异常回退、日志下标。

**2026-09-24 实际运行状态：测试在收集阶段失败，尚未执行测试用例。**

首先遇到的问题是接口文件底部执行了未定义的 `df`：

```text
NameError: name 'df' is not defined
```

测试文件还保留了旧函数名 `build_observe_power_new`、`build_observe_power_new_no_padding`、`fill_recent_observe_power`，并按旧签名传入 `model`。当前接口已改为 v0/v1/v2 和全局模型变量，测试还需要同步这些变化；日志断言也需要与实际打印行为保持一致。

此前的 84 项通过记录对应旧版代码，不能作为当前文件已经通过测试的结论。

解决导入问题并同步测试后，可在两个 Python 文件所在目录运行：

```bash
python -m pytest -q test_power_prediction_interface.py
```

测试文件已有的直接运行和演示入口为：

```bash
python test_power_prediction_interface.py
python test_power_prediction_interface.py --demo
```

这两个入口当前同样受上述导入和接口不一致问题影响。
