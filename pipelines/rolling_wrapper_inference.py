import math
import numpy as np
import pandas as pd
from pathlib import Path
from operator import itemgetter
from pipelines.pipeline_wrapper_inference import unified_inference


def check_and_report(df, cols, target_len, label):
    if not cols:
        raise ValueError(f"{label}列在数据中不存在：{cols}")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label}列在数据中不存在：{missing}")
    if not cols:
        raise ValueError(f"{label}列为空，请检查配置")

    ok_mask = pd.Series(True, index=df.index)
    for col in cols:
        valid = df[col].map(lambda x: hasattr(x, '__len__') and len(x) >= target_len)
        ok_mask &= valid
    return ok_mask


def data_check(df, config, past_len, pred_length):

    past_cols   = [col for col in config['past_covariate'] + config['past_target'] if col in df.columns]
    future_cols = [col for col in config['future_covariate'] if col in df.columns]

    past_mask   = check_and_report(df, past_cols,   past_len, "历史")
    future_mask = check_and_report(df, future_cols, pred_length, "预报")

    final_mask = past_mask & future_mask
    total_bad = int((~final_mask).sum())
    if total_bad:
        print(f"合计数据长度不合格行数 = {total_bad} / {len(df)}")

    df_checked = df[final_mask].reset_index(drop=True)
    if df_checked.empty:
        raise ValueError(
            f"过滤后为空：历史列需长度 {past_len}，未来列需 {pred_length}，请检查数据。"
        )
    return df_checked


def build_sliding_windows(df, config, full_covariate,
                          context_length, prediction_length, total_remain, remaining,
                          entity_cols):
    past_target      = config['past_target'][0]
    past_target_full = past_target + '_full'
    future_targets   = config['future_target']
    is_last = (remaining == total_remain - 1)

    # ---- 统一切片定义 ----
    if is_last:
        past_end_index = -prediction_length
        past_slice   = slice(past_end_index - context_length, past_end_index)
        future_slice = slice(past_end_index, None)
        target_slice = slice(past_end_index, None)
    else:
        past_end_index = context_length + remaining * prediction_length
        past_slice   = slice(past_end_index - context_length, past_end_index)
        future_slice = slice(past_end_index, past_end_index + prediction_length)
        target_start = remaining * prediction_length
        target_slice = slice(target_start, target_start + prediction_length)

    past_target_slice = slice(-context_length, None)

    get_past   = itemgetter(past_slice)
    get_future = itemgetter(future_slice)
    get_ptgt   = itemgetter(past_target_slice)
    get_tgt    = itemgetter(target_slice)

    current_df = df[list(entity_cols)].copy()

    # ---- 历史气象 ----
    for past_col, full_col in zip(config['past_covariate'], full_covariate):
        current_df[past_col] = df[full_col].map(get_past)

    # ---- 预测气象 ----
    for fut_col, full_col in zip(config['future_covariate'], full_covariate):
        current_df[fut_col] = df[full_col].map(get_future)

    # ---- 历史光功率 ----
    current_df[past_target] = df[past_target_full].map(get_ptgt)

    # ---- 未来光功率 ----
    for col in future_targets:
        current_df[col] = df[col].map(get_tgt)

    return current_df


def concat_series(a, b):
    """按行拼接两个序列列"""
    return [list(x) + list(y) for x, y in zip(a.to_numpy(), b.to_numpy())]

CONFIG = {'past_covariate': ['general_ghi', 'general_temp'],
          'future_covariate': ['general_ghi_predict', 'general_temp_predict'],
          'past_target':  ['observe_power'],
          'future_target': ['chronos'],
          'future_len': 192,
}

def rolling_inference(pipeline_backbone, ds_input, config=CONFIG,
                      timestamp_cols=('timestamp_win',),
                      entity_cols=('timestamp_win', 'station'),
                      feature_cols=None,
                      batch_size: int = 128,
                      num_workers: int = 0,
                      timestamp_output: str = 'epoch_s',
                      prediction_to_numpy: bool = True):
    """
    
    参数:
        model: 预测模型
        df: 输入DataFrame
        config: rolling配置
        timestamp_cols: 时间戳列名
        entity_cols: 实体列名
        batch_size: 批处理大小
        num_workers: 并行工作进程数
        timestamp_output: 时间戳输出格式
        prediction_to_numpy: 是否将预测转换为numpy数组
    
    返回:
        包含所有预测结果的DataFrame
    """
    if ds_input.empty:
        return pd.DataFrame()

    # 模型数据长度
    context_length = pipeline_backbone.context_length 
    prediction_length = pipeline_backbone.prediction_length
    # 预测长度
    future_len = config['future_len']
    total_remain = math.ceil(future_len/prediction_length)   # 剩余需要预测的次数

    # ---- 1. 校验 ----
    try:
        ds_input = data_check(ds_input, config, context_length, future_len)
    except Exception as e:
        raise ValueError(e)

    # ---- 2. 构造 full 列（局部，不改 config）----
    full_covariate = [c + '_full' for c in config['past_covariate']]
    # 气象数据特征拼接为全量序列
    for past_col, fut_col, full_col in zip(config['past_covariate'], config['future_covariate'], full_covariate) :
        ds_input[full_col] = concat_series(ds_input[past_col], ds_input[fut_col])

    past_target = config['past_target'][0]
    past_target_full = past_target + '_full'
    # 光功率数据特征拼接为全量序列
    ds_input[past_target_full] = ds_input[past_target]

    # ---- 3. 固定行索引，循环里按它对齐 ----
    df_key = ds_input.set_index(list(entity_cols)).index   # 循环外算一次
    prediction_col = type(pipeline_backbone).__name__
    for remaining in range(total_remain):
        current_df = build_sliding_windows(
            ds_input, 
            config,
            full_covariate,
            context_length, 
            prediction_length, 
            total_remain, 
            remaining,
            entity_cols)

        # 调用统一推理接口进行预测
        ds_output = unified_inference(
            pipeline_backbone=pipeline_backbone,
            ds_input=current_df,
            timestamp_cols=timestamp_cols,
            entity_cols=entity_cols,
            feature_cols=feature_cols,
            batch_size=batch_size,
            num_workers=num_workers,
            timestamp_output=timestamp_output,
            prediction_to_numpy=prediction_to_numpy,
        )
        take_len = min(prediction_length, future_len - remaining * prediction_length)
        # ---- 关键：不 merge，直接按 index 对齐取预测列 ----
        pred = ds_output.set_index(list(entity_cols))[prediction_col]
        pred = pred.reindex(df_key).to_numpy()          # 对齐到 df 的行

        # ---- 拼接到 observe_power_full ----
        ds_input[past_target_full] = [
            np.concatenate([np.asarray(a), np.asarray(b[-take_len:])])
            for a, b in zip(ds_input[past_target_full].to_numpy(), pred)
        ]
        
    ds_input[prediction_col] = ds_input[past_target_full].apply(lambda x: x[-future_len:])
    ds_output = ds_input[list(entity_cols) + [prediction_col]]    
    return ds_output
 


# 项目根目录路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 数据集目录路径（包含训练和测试数据）
_E2E_DATA_DIR = Path(r'D:\codeset\pv\pv_predition1\dataset\YLJ_0812_90_train_10_test')
    
    
def _first_parquet_path() -> Path:
    """获取数据集目录中的第一个parquet文件路径"""
    first = sorted(_E2E_DATA_DIR.glob("*.parquet"))[0]
    return first

def load_model_from_checkpoint(model_name, checkpoint_dir, device):
    import yaml
    from pipelines.model_registry import build_model
    from safetensors.torch import load_file

    checkpoint_dir = Path(checkpoint_dir)
    config_path = checkpoint_dir / "model_config.yaml"
    safetensors_path = checkpoint_dir / "model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"缺少模型配置文件: {config_path}")
    if not safetensors_path.is_file():
        raise FileNotFoundError(f"缺少模型权重文件: {safetensors_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)
    model = build_model(model_name=model_name, model_config=model_config)
    state_dict = load_file(str(safetensors_path))
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint 与模型 {model_name} 结构不匹配: "
            f"missing={missing}, unexpected={unexpected}"
        )
    model.to(device)
    model.eval()
    return model, model_config

if __name__ == '__main__':
    """主程序入口：加载模型配置并进行预测测试"""
    model_name = 'm3'
    # checkpoint_dir = r'D:\codeset\pv\master_new2\master_scroll\NextGenPVForecast\runs\m1\model_result'
    checkpoint_dir = r'D:\codeset\pv\master_new2\master_scroll\NextGenPVForecast\runs\model_result_m3_nwloss\model_result'
    # checkpoint_dir = '/home/ai4eng_devuser2/NextGenPVForecast/runs/m5'
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_config = load_model_from_checkpoint(model_name, checkpoint_dir, device)

    # 加载数据
    ds_input_str = _first_parquet_path()
    ds_input = pd.read_parquet(ds_input_str)
    for col in ['observe_power_future', 'observe_power_chronos_pred'] + CONFIG['future_covariate']: # 为了测试480预测长度，临时添加
        ds_input[col] = ds_input[col].apply(lambda x: np.concatenate([x, x, x[:96]])) 
    # 执行预测
    from pipelines.rolling_wrapper_inference import rolling_inference
    result = rolling_inference(model, ds_input, config=CONFIG)
    prediction_col = type(model).__name__
    # 验证预测结果长度是否正确
    assert len(result[prediction_col].iloc[0]) == CONFIG['future_len']