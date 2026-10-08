"""填补观测尾部：逐行补齐版、逐行不补齐版，以及按截断位置分组的 v2。"""

import numpy as np
import pandas as pd


def build_observe_power_new_v0(df, tail_length, prediction_length = 192, prediction_name = 'TransformersAdaptedTimesFM3Model'):
    """逐行预测填补，模型输入左补首值至 672 点。

    final_power 为空、含空值或不是长度 672 的一维数组时原样输出，不处理。
    observe_power 为 None 或不是长度 672 的一维数组时，也直接输出 final_power。
    只有参与覆盖或填补的行保证输出长度为 672，跳过的行不调整长度。
    其余行从最后 tail_length 个点中寻找首个缺失位置，再截断并推理。
    每次给模型传入一行 DataFrame，要求返回一行及 prediction_name 列。
    返回 df 的副本，覆盖 observe_power；timestamp_win 保持不变。
    """
    if not 0 < tail_length <= 672 or prediction_length < tail_length:
        raise ValueError(
            "需要满足 0 < tail_length <= 672 且 prediction_length >= tail_length"
        )

    result = df.copy()
    output = df["final_power"].tolist()
    keys = ["timestamp_win", "station"]

    for row_number, (final_value, observed_value) in enumerate(
        zip(df["final_power"], df["observe_power"])
    ):
        # 无效 final_power 原样输出，不读取观测值，也不调用模型。
        final = np.asarray(final_value)
        if final.shape != (672,) or np.asarray(pd.isna(final)).any():
            continue

        observed = np.asarray(observed_value)
        if observed.shape != (672,):
            continue
        final = final.astype(float, copy=True)
        observed = observed.astype(float, copy=False)

        missing_positions = np.flatnonzero(np.isnan(observed[-tail_length:]))
        if missing_positions.size == 0:
            final[-tail_length:] = observed[-tail_length:]
            output[row_number] = final
            continue

        # 首个缺失点作为截断位置，该位置对应预测结果的第 0 点。
        cut_position = 672 - tail_length + missing_positions[0]
        cut_length = 672 - cut_position
        model_values = np.concatenate([
            np.full(cut_length, final[0]), final[:cut_position]
        ])

        # 每行独立调用一次模型。
        model_input = df.iloc[[row_number]][keys].copy()
        model_input["observe_power"] = [model_values]
        predictions = model.predict(model_input, prediction_length=prediction_length)
        if len(predictions) != 1:
            raise ValueError("单行推理必须返回一行预测结果")
        predicted = np.asarray(predictions[prediction_name].iloc[0], dtype=float)
        if predicted.ndim != 1 or predicted.size < cut_length:
            raise ValueError(f"第 {row_number} 行模型预测长度不足以覆盖截断部分")

        # 回填到长度 672 的 final 副本中，仅替换尾部，不改变数组长度。
        tail = observed[cut_position:]
        final[cut_position:] = np.where(np.isnan(tail), predicted[:cut_length], tail)
        output[row_number] = final

    result["observe_power"] = output
    return result


def build_observe_power_new_v1(df, tail_length = 8, prediction_length = 192, prediction_name = 'TransformersAdaptedTimesFM3Model'):
    """逐行预测填补，模型输入直接使用截断后的序列，不补齐。

    final_power 为空、含空值或不是长度 672 的一维数组时原样输出，不处理。
    observe_power 为 None 或不是长度 672 的一维数组时，也直接输出 final_power。
    不补齐仅针对模型输入；参与覆盖或填补的行输出长度为 672。
    跳过的行不调整长度。
    其余行从最后 tail_length 个点中寻找首个缺失位置，再截断并推理。
    每次给模型传入一行 DataFrame，要求返回一行及 prediction_name 列。
    返回 df 的副本，覆盖 observe_power；timestamp_win 保持不变。
    若整段截断，模型须支持长度为 0 的输入。
    """
    if not 0 < tail_length <= 672 or prediction_length < tail_length:
        raise ValueError(
            "需要满足 0 < tail_length <= 672 且 prediction_length >= tail_length"
        )

    result = df.copy()
    output = df["final_power"].tolist()
    keys = ["timestamp_win", "station"]

    for row_number, (final_value, observed_value) in enumerate(
        zip(df["final_power"], df["observe_power"])
    ):
        # 无效 final_power 原样输出，不读取观测值，也不调用模型。
        final = np.asarray(final_value)
        if final.shape != (672,) or np.asarray(pd.isna(final)).any():
            continue

        observed = np.asarray(observed_value)
        if observed.shape != (672,):
            continue
        final = final.astype(float, copy=True)
        observed = observed.astype(float, copy=False)

        missing_positions = np.flatnonzero(np.isnan(observed[-tail_length:]))
        if missing_positions.size == 0:
            final[-tail_length:] = observed[-tail_length:]
            output[row_number] = final
            continue

        # 首个缺失点作为截断位置，该位置对应预测结果的第 0 点。
        cut_position = 672 - tail_length + missing_positions[0]
        cut_length = 672 - cut_position
        model_values = final[:cut_position].copy()

        # 每行独立调用一次模型。
        model_input = df.iloc[[row_number]][keys].copy()
        model_input["observe_power"] = [model_values]
        predictions = model.predict(model_input)

        predicted = np.asarray(predictions[prediction_name].iloc[0], dtype=float)
        if predicted.ndim != 1 or predicted.size < cut_length:
            raise ValueError(f"第 {row_number} 行模型预测长度不足以覆盖截断部分")

        # 回填到长度 672 的 final 副本中，仅替换尾部，不改变数组长度。
        tail = observed[cut_position:]
        final[cut_position:] = np.where(np.isnan(tail), predicted[:cut_length], tail)
        output[row_number] = final

    result["observe_power"] = output
    return result

def build_observe_power_new_v2(
    df, tail_length=8, prediction_length=192,
    prediction_name='TransformersAdaptedTimesFM3Model',
):
    """按首个缺失位置分组推理，模型输入不补齐到 672。

    同一截断位置的行只调用一次模型，每批输入数组等长。
    模型调用方式与逐行不补齐版一致：model.predict(model_input)。
    预测长度由模型自身配置，返回数组须覆盖本批截断长度。
    每批 timestamp_win + station 组合须唯一，用于对齐可能乱序的模型结果。
    有效行填补后长度为 672；无效 final_power 原样作为 observe_power。
    observe_power 为 None 或不是长度 672 的一维数组时，也直接输出 final_power。
    timestamp_win 保持不变，返回 df 的副本，不修改原始数据。
    打印参与预测填补的原始行位置、df 索引、站点及数组下标（从 0 开始）。
    """
    if not 0 < tail_length <= 672 or prediction_length < tail_length:
        raise ValueError(
            "需要满足 0 < tail_length <= 672 且 prediction_length >= tail_length"
        )

    result = df.copy()
    output = df["final_power"].tolist()
    valid_rows = [
        i for i, (value, observed_value) in enumerate(zip(output, df["observe_power"]))
        if np.asarray(value).shape == (672,) and not np.asarray(pd.isna(value)).any()
        and np.asarray(observed_value).shape == (672,)
    ]
    if not valid_rows:
        result["observe_power"] = output
        return result

    valid_df = df.iloc[valid_rows]
    final = np.stack(valid_df["final_power"]).astype(float)
    observed = np.stack(valid_df["observe_power"]).astype(float)
    if observed.shape != final.shape:
        raise ValueError("有效行的 observe_power 必须为长度 672 的一维数组")

    tail = observed[:, -tail_length:]
    missing = np.isnan(tail)
    needs_prediction = missing.any(axis=1)
    final[~needs_prediction, -tail_length:] = tail[~needs_prediction]

    keys = ["timestamp_win", "station"]
    cut_positions = 672 - tail_length + missing.argmax(axis=1)
    for cut_position in np.unique(cut_positions[needs_prediction]):
        # 截断位置相同的行放在同一批，例如位置 666 的所有行一起推理。
        rows = np.flatnonzero(needs_prediction & (cut_positions == cut_position))
        cut_length = 672 - cut_position
        model_input = valid_df.iloc[rows][keys].copy()
        if model_input.duplicated(keys).any():
            raise ValueError("同批推理的 timestamp_win + station 组合必须唯一")
        model_input["observe_power"] = list(final[rows, :cut_position])
        predictions = model.predict(model_input)
        # 按键恢复输入顺序，避免模型返回乱序导致错配。
        predictions = predictions.set_index(keys, verify_integrity=True)
        predicted = np.stack(predictions.loc[
            pd.MultiIndex.from_frame(model_input[keys]), prediction_name
        ]).astype(float)
        if predicted.ndim != 2 or predicted.shape[1] < cut_length:
            raise ValueError("模型预测长度不足以覆盖截断部分")

        observed_tail = observed[rows, cut_position:]
        final[rows, cut_position:] = np.where(
            np.isnan(observed_tail), predicted[:, :cut_length], observed_tail
        )

    for row, values in zip(valid_rows, final):
        output[row] = values
    result["observe_power"] = output
    return result



###### observe_power近期数据填补， 这个放在multi_infer.py中 ds数据构造后面
df['observe_power_original'] = df['observe_power']   #保存未治理的observe_power，后面observe_power会被覆盖
FILL_OBSERVE_POWER = True   #治理开关
if FILL_OBSERVE_POWER:
    try: 
        t_lenth = 8                          
        df = build_observe_power_new_v1(df, t_lenth)
    except Exception as e:
        print(e)
        df['observe_power'] = df['final_power']
else: 
    df['observe_power'] = df['final_power']