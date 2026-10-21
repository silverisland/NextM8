"""瘦身守护（2026-10-14）：j1_final2 配方（小尺寸模型）在固定合成 batch 上的前向输出与瘦身前逐位一致。
参考权重 / 输出由瘦身前的代码生成（M8_WRITE_REFERENCE=1 pytest ... 重新生成），之后任何改动都要能严格加载同一份
state_dict 并复现同样的 logits / 分位 / 损失——保证删掉的只是没用到的分支。"""
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from forecast_backbones.m8.config import load_config
from forecast_backbones.m8.transformer_models import TransformersAdaptedM8Model
from forecast_backbones.m8.tests.conftest import ROOT, write_station_meta

FIX = Path(__file__).parent / "fixtures"
REF_W, REF_OUT = FIX / "j1_small_reference.safetensors", FIX / "j1_small_reference.npz"


def _j1_small(tmp_path):
    cfg = load_config(str(ROOT / "forecast_backbones/m8/config/j1_final2.yaml"))
    cfg["dataset"]["station_meta"] = str(write_station_meta(tmp_path, drop_coords_for=(1002,)))   # 1002 缺坐标 → 走包络
    cfg["dataset"]["station_meta_cols"] = None                                                    # 变体写的是 info.csv 表头，合成站表是标准列名
    cfg["model"].update({"d_model": 32, "d_ff": 64, "head_hidden": 32, "n_heads": 2, "blocks": ["conv", "deltanet"], "pos_in_patch_dim": 8})
    cfg["features"]["hint_col"] = "timesfm_predict"                                               # 合成 batch 里的插值列名
    # 参考文件按 2026-10-20 之前的默认四对 NWP 生成（之后默认删了 Solargis 两对）：钉住四对，守的是代码不变、旧 checkpoint 照常复现
    cfg["features"]["past_future"] = [["general_ghi", "general_ghi_predict"], ["general_temp", "general_temp_predict"],
                                      ["GHI_solargis", "ghi_solargis_predict"], ["temp_solargis", "temp_solargis_predict"]]
    cfg["features"]["static"]["n_station_slots"] = 64                                             # 站点嵌入缩小，参考文件才小
    cfg["run"]["out_dir"] = str(tmp_path / "run")
    return cfg


def test_j1_forward_matches_reference(tmp_path, batch):
    torch.manual_seed(0)
    m = TransformersAdaptedM8Model(_j1_small(tmp_path)); m.eval()
    if os.environ.get("M8_WRITE_REFERENCE") == "1":
        FIX.mkdir(exist_ok=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in m.state_dict().items()}, str(REF_W))
        with torch.no_grad():
            out = m(batch["batch_dict"], labels=batch["labels"])
        np.savez(REF_OUT, logits=out.logits.numpy(), quantiles=out.aux["quantiles"].numpy(), loss=float(out.loss),
                 p_clear=out.aux["p_clear"], night=out.aux["night_loss"].astype(np.int8), channels=np.array(m.fe.channel_names))
        pytest.skip("写入参考文件")
    assert REF_W.exists() and REF_OUT.exists(), "缺参考文件：M8_WRITE_REFERENCE=1 用瘦身前的代码生成"
    ref = np.load(REF_OUT)
    assert list(ref["channels"]) == list(m.fe.channel_names)                                        # 通道布局不变
    m.load_state_dict(load_file(str(REF_W)), strict=True)                                            # 参数名集合不变
    with torch.no_grad():
        out = m(batch["batch_dict"], labels=batch["labels"])
    np.testing.assert_allclose(out.aux["quantiles"].numpy(), ref["quantiles"], rtol=0, atol=1e-5)
    np.testing.assert_allclose(out.logits.numpy(), ref["logits"], rtol=0, atol=1e-5)
    np.testing.assert_allclose(out.aux["p_clear"], ref["p_clear"], rtol=0, atol=1e-6)
    assert np.array_equal(out.aux["night_loss"].astype(np.int8), ref["night"])
    assert abs(float(out.loss) - float(ref["loss"])) < 1e-6
