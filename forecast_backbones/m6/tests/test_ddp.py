"""两进程 DDP（CPU、gloo）真跑 train_m6：各进程训完的 LoRA 权重逐位相同（梯度确实在同步）、同一步停下、
产物只由主进程写；文本站号走“主进程转换、其余进程复用”那条路。"""
import json
import os
import socket
import subprocess
import sys
import textwrap

import pytest
import torch
import yaml

pytest.importorskip("timesfm3")
from .conftest import ROOT, make_synthetic_frame, write_station_meta


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


DRIVER = textwrap.dedent('''
    import sys, torch
    sys.path.insert(0, sys.argv[4])
    import forecast_backbones.m6.train as T
    from forecast_backbones.m6 import _dist
    from forecast_backbones.m6.lora import lora_state_dict
    out = sys.argv[3]
    real_train = T._M6Trainer.train
    def train(self, *a, **k):
        r = real_train(self, *a, **k)
        torch.save({"sd": lora_state_dict(self.model.backbone.model), "step": self.state.global_step,
                    "world": _dist.world_size()}, f"{out}.rank{_dist.rank()}.pt")
        return r
    T._M6Trainer.train = train
    T.train_m6(sys.argv[1], sys.argv[2], out_dir=out)
''')


def test_two_process_ddp_training_keeps_ranks_in_sync(tmp_path, tiny_ckpt_dir):
    df = make_synthetic_frame(n_days=12, issue_every_min=60, seed=0, start="2025-06-08")
    df["station_id"] = df["station_id"].astype(str)
    df.to_parquet(tmp_path / "data.parquet", index=False)
    cfg = {"timesfm3": {"checkpoint": str(tiny_ckpt_dir), "per_core_batch_size": 4},
           "dataset": {"station_meta": str(write_station_meta(tmp_path))}, "eval": {"batch_size": 64},
           "calibration": {"enabled": False}, "decision": {"point": {"default": "median", "by_step": []}},
           "finetune": {"enabled": True,
                        "schedule": {"batch_size": 4, "lr": 1e-3, "precision": "fp32", "logging_steps": 1, "max_steps": 6,
                                     "eval_every_steps": 2, "progress_bar": False, "sampling": "station_day"},
                        "split": {"train_start": None, "val_blocks": [["2025-06-12", "2025-06-14"]], "test_start": "2025-06-19",
                                  "test_end": None, "cal_days": 0, "purge_days": 2, "embargo_days": 7},
                        "early_stop": {"patience": 10}}}
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    (tmp_path / "driver.py").write_text(DRIVER, encoding="utf-8")
    out = tmp_path / "run"
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", f"--master_port={_free_port()}",
           str(tmp_path / "driver.py"), str(tmp_path / "cfg.yaml"), str(tmp_path / "data.parquet"), str(out), str(ROOT)]
    env = {**os.environ, "GLOO_SOCKET_IFNAME": "lo0"} if sys.platform == "darwin" else None   # 本机两进程走回环；mac 上 gloo 挑到别的网卡会 uv_accept 失败
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=600, env=env)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    r0, r1 = torch.load(f"{out}.rank0.pt"), torch.load(f"{out}.rank1.pt")
    assert r0["world"] == r1["world"] == 2 and r0["step"] == r1["step"] == 6
    assert r0["sd"].keys() == r1["sd"].keys() and all(torch.equal(r0["sd"][k], r1["sd"][k]) for k in r0["sd"])
    assert any(k.endswith("lora_B") and float(v.abs().sum()) > 0 for k, v in r0["sd"].items())
    log = json.loads((out / "train_log.json").read_text(encoding="utf-8"))
    assert [h["step"] for h in log["history"]] == [0, 2, 4, 6]
    assert (out / "model.safetensors").exists() and not (out / "station_routing.json").exists()
    assert (out / ".data" / "data_with_station_id.parquet").exists()
    assert p.stdout.count("[m6-lora] step 0 ") == 1                          # 只有主进程打分、打印
