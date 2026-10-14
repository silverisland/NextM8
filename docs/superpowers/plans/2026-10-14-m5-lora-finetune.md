# M5 TimesFM-3 + LoRA 微调（自包含，M8 训练框架）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 M6 的 LoRA 微调搬进 M5（删掉 conformal），复用 M8 的训练 / 评估框架，在 `m8_v1` 上拿到"零样本 vs LoRA"基线；线上零样本类行为逐位不变。

**Architecture:** 一个模型类 `TransformersAdaptedM5Model`，按 model_config 里有没有 `dataset.station_meta` 分两种模式：legacy（线上零样本，代码逐字保留）与 full（站表 → 容量 / 站号；TimesFM-3 torch 可微路径 + LoRA；M8 决策层与 aux schema）。训练、评估、逐站报表从 M8 复制进 `forecast_backbones/m5/` 并改成 TimesFM 版；LoRA / 可微底座从 M6 复制。站点几何默认全关（D8），基线只差 LoRA 权重。

**Tech Stack:** Python 3.10+（本机 3.14）、torch、HF transformers Trainer、timesfm 3.0.2（模块名 `timesfm3`）、pvlib、safetensors、pytest。

**Spec:** `docs/superpowers/specs/2026-10-14-m5-lora-finetune-design.md`（含 D8：站点几何默认关）

## Global Constraints

- 仓库根 `R=/Users/tqa946816/Documents/华为/NextGenPVForecast`，所有命令在 `R` 下执行。
- 本机测试解释器：`PY=/private/tmp/claude-501/-Users-tqa946816-Documents----NextGenPVForecast/56e02fe2-d7e0-4557-984b-fd80d7f01e47/scratchpad/venv314/bin/python`（已装 timesfm 3.0.2、torch 2.14.1、transformers 4.57.6）。服务器解释器 `/home/ma-user/ts3/bin/python`。
- 另一个 Claude 会话也在往这个仓库提交：**只 `git add` 本计划列出的路径**，禁止 `git add -A` / `git add .` / `git stash`；`train_shell/`、`eval_shell/eval_m8.sh`、`experiments/`、`nanwang_online/` 里用户未提交的改动一律不碰（只新增 `train_shell/train_m5*.sh`、`eval_shell/eval_m5.sh` 这三个文件并只 add 它们）。
- 每个 commit 信息以这两行结尾（空一行后）：`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`。
- M6、M8 的任何文件不改。`forecast_backbones/m5` 下的 .py / .yaml 不得引用 `forecast_backbones.m6` / `forecast_backbones.m8`（Task 1 的测试守住）。
- legacy 模式（没有 `dataset.station_meta`）的 forward 输出、调用底座的参数、实例属性、`hasattr(model, "config") == False` 与重构前逐位相同。
- M6 来的文件保留 `_m6_eps_patched` / `_m6_original` 两个标记名不改（M5 与 M6 同进程时共用同一个垫片标记，防重复打补丁）。
- 训练精度默认 fp32（V100 不支持 bf16）。TimesFM-3 分位固定 9 档 `(0.1, …, 0.9)`。
- PyYAML 里浮点写 `1.0e-4`（`1e-4` 会被读成字符串）。

## 与 spec 的出入（实现时按这里）

1. **不改 `train_shell/train_script.py`**：它在用户未提交的 `train_shell/` 里，而且 M5 与 M8 一样通过 `python -m forecast_backbones.m5.train`（`train_m5.sh`）训练，`train_script.py` 原本也没有 m5 分支。
2. **legacy 模式不暴露 `config` / `compute_metrics`**：pipeline 可能按 `hasattr` 判断，重构前的类没有它们；full 模式 `config` 返回合并后的 cfg，`compute_metrics` 没有调用方，不加。
3. **`config.load_config` 只服务 full 模式**（必须有 `dataset.station_meta`）；legacy 模式仍由类内 `_merge(DEFAULT_CFG, …)` 合并，与今天相同。模式只看**原始** model_config 有没有 `dataset.station_meta`；train.py / evaluate.py 传给模型的都是合并后的 cfg（DEFAULT_CFG 自带站表路径），所以变体 yaml 不用写站表。
4. **`REMOVED_KEYS` 加上 legacy 的 timesfm3 列键**（`context_len / pred_len / past_future_cols / past_only_cols / return_quantiles / predict_quantile`）：full 配置里写了它们直接报错并指出新位置，防止把 configs.yaml 加上站表后被静默忽略。

## Review Focus

1. **线上 release 配置**（`model_config.yaml`，带 `run` / `model` 段、没有 `dataset.station_meta`）必须仍走 legacy、逐位不变——Task 5 的 `online_yaml` 用例与属性用例。
2. **数据里的站不在站表里** → 明确报错（指出站号与 `station_meta.csv`），不能静默用错容量——Task 4 `test_unknown_station_raises`。
3. **历史功率 / 协变量含 NaN、整行 NaN** → full 模式 torch 路径插值后有限（整行置 0），loss / logits 有限——Task 6 `test_nan_history_and_missing_labels`。
4. **标签缺测（NaN）** → 只从 loss 里掩掉，不产生 NaN 梯度——同上。
5. **混合成员集成与评估时换决策**：零样本 run 与 LoRA run 一起集成、`--decision-yaml g2_cs_ceiling.yaml` 在不重训的情况下恢复 M8 晴空上限——Task 9 `test_ensemble_and_decision_override`。

---

## 文件结构

```
forecast_backbones/m5/
  transformer_models.py   改：一个类两种模式（legacy 逐字保留 + full 新增）；load_station_meta / save_weights / load_weights
  transformer_trainer.py  改：evaluate_model 按模式分派（legacy 原样 / full = M8 schema）；build_training_args（自 M8）
  config.py               新：DEFAULT_CFG、load_config、REMOVED_KEYS、deep_merge / interpolate / dump_config / raw_override
  rowctx.py               新：SolarCache（自 M8）、RowCtx（M8 同字段）、RowContextBuilder
  decision.py             自 M8；physics_clip 支持 clear_sky_cap_ratio=None（无晴空上限）
  solar.py _parquet.py data.py rows.py _dist.py _arrays.py losses.py station_meta.csv   自 M8（改名）
  backbone.py lora.py _torch_compat.py                                         自 M6（改名）
  train.py evaluate.py station_report.py collect_runs.py                       自 M8 并改成 TimesFM / LoRA 版
  README.md               新
  config/configs.yaml     不动（线上零样本）
  config/finetune.yaml    新：DEFAULT_CFG 落盘
  config/variants/        新：z0_zeroshot / l1_lora_r8 / d2_pstar_point / g1_solar_cov / g2_cs_ceiling
  tests/conftest.py       改：合成数据 + 小 TimesFM-3 权重 fixture
  tests/…                 见各任务
删除：m5/conformal.py、m5/conformal_replay.py、m5/tests/test_conformal.py、eval_shell/conformal_m5.sh
新增：train_shell/train_m5.sh、train_shell/train_m5_variants.sh、eval_shell/eval_m5.sh
```

---

### Task 1: 自包含支撑模块 + 删 conformal + 测试夹具

**Files:**
- Create（复制）：`forecast_backbones/m5/{solar,_parquet,data,rows,_dist,_arrays,losses,decision}.py`、`forecast_backbones/m5/station_meta.csv`（自 M8）；`forecast_backbones/m5/{backbone,lora,_torch_compat}.py`（自 M6）
- Create（复制）：`forecast_backbones/m5/tests/{test_lora,test_backbone_torch,test_torch_compat}.py`（自 M6 tests）
- Create：`forecast_backbones/m5/tests/test_standalone.py`
- Modify：`forecast_backbones/m5/tests/conftest.py`（整体替换）
- Delete：`forecast_backbones/m5/conformal.py`、`forecast_backbones/m5/conformal_replay.py`、`forecast_backbones/m5/tests/test_conformal.py`、`eval_shell/conformal_m5.sh`

**Interfaces:**
- Produces：`m5.solar.{StationMeta, station_cap_mw, attach_station_id, window_timestamps, solar_geometry, clock_pseudo_geometry, norm_station_name}`；`m5.decision.{MODEL_LEVELS, physics_clip, enforce_monotone, select_point, level_value, p_star, quantile_cols}`；`m5.losses.pinball_loss(q, y, levels, mask)`；`m5.backbone.TorchBackbone`（`.from_evaluator(ev)`、`.tiny_for_tests(seed)`、`.predict_quantiles(contexts[B,L], past_future[B,C,L+H]|None, past_only[B,C,L]|None, *, horizon, use_znorm, use_symmetric_averaging, make_positive) -> Tensor[B,H,9]`）；`m5.lora.{LoRALinear, inject_lora, freeze_all_but_lora, lora_state_dict, save_lora_weights, load_lora_weights, load_lora_state_dict, trainable_summary}`；`m5.data._dataset_with_station_id`、`m5.data._check_presplit_dir`；`m5.rows.{scored_row_index, issue_grid_row_index, StationDaySampler, row_stations, SHORT_ISSUE_TIME}`；`m5._dist.{is_main, rank, world_size, barrier, broadcast_from_main}`。
- Produces（测试夹具，`tests/conftest.py`）：`CAP`、`L`、`H`、`write_station_meta(tmp_path, drop_coords_for=())`、`make_frame(...)`、`write_splits(...)`、`full_cfg(tmp_path, ckpt, **over) -> dict`；fixtures `tiny_ckpt_dir`（session）、`tiny_evaluator`、`collate`、`dataset`、`batch`。

- [ ] **Step 1: 写 `tests/test_standalone.py`（会失败：m5 里还没有站表，且后续文件还没复制）**

```python
"""M5 自包含（2026-10-14）：forecast_backbones/m5 不引用 forecast_backbones/m6、m8 的任何代码或数据文件；
pipelines / interfaces 等所有模型共享的部分照常引用。"""
import re
import subprocess
import sys
from pathlib import Path

M5 = Path(__file__).resolve().parents[1]
ROOT = M5.parents[1]
_REF = re.compile(r"(from\s+\.\.m[68]\b|forecast_backbones\.m[68]\b|forecast_backbones/m[68]\b|from\s+m[68]\b|import\s+m[68]\b)")


def test_no_m6_m8_reference_in_m5_sources():
    hits = []
    for p in sorted(M5.rglob("*")):
        if p.suffix not in (".py", ".yaml", ".sh") or "__pycache__" in p.parts or p.samefile(Path(__file__)):
            continue
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if _REF.search(line):
                hits.append(f"{p.relative_to(ROOT)}:{n}: {line.strip()[:100]}")
    assert not hits, "m5 里还引用 m6 / m8：\n" + "\n".join(hits)


def test_importing_m5_does_not_load_m6_or_m8():
    mods = sorted(f"forecast_backbones.m5.{p.stem}" for p in M5.glob("*.py"))
    code = ("import sys, importlib; sys.path.insert(0, %r)\n"
            "for m in %r:\n"
            "    try:\n"
            "        importlib.import_module(m)\n"
            "    except ModuleNotFoundError as e:\n"
            "        if not str(e.name).startswith('timesfm3'):\n"
            "            raise\n"
            "bad = sorted(m for m in sys.modules if m.startswith(('forecast_backbones.m6', 'forecast_backbones.m8')))\n"
            "print('LOADED=' + ','.join(bad))" % (str(ROOT), mods))
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().splitlines()[-1] == "LOADED=", r.stdout


def test_station_meta_lives_in_m5():
    assert (M5 / "station_meta.csv").exists()


def test_conformal_is_gone():
    assert not (M5 / "conformal.py").exists() and not (M5 / "conformal_replay.py").exists()
    assert not (ROOT / "eval_shell" / "conformal_m5.sh").exists()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `$PY -m pytest -q forecast_backbones/m5/tests/test_standalone.py`
Expected: FAIL（`test_station_meta_lives_in_m5`、`test_conformal_is_gone` 失败）

- [ ] **Step 3: 复制文件并改名（M8 来源用全局 m8→m5；M6 来源只改路径与两处文案，保留 `_m6_eps_patched`）**

```bash
M5=forecast_backbones/m5; M6=forecast_backbones/m6; M8=forecast_backbones/m8
for f in solar.py _parquet.py data.py rows.py _dist.py _arrays.py losses.py decision.py; do
  sed -e 's/m8/m5/g; s/M8/M5/g' "$M8/$f" > "$M5/$f"
done
cp "$M8/station_meta.csv" "$M5/station_meta.csv"
for f in backbone.py lora.py _torch_compat.py; do
  sed -e 's#forecast_backbones/m6#forecast_backbones/m5#g; s/forecast_backbones\.m6/forecast_backbones.m5/g; s/m6 lora + output_head/m5 lora + output_head/; s/与 m6 假定/与 m5 假定/' "$M6/$f" > "$M5/$f"
done
for f in test_lora.py test_backbone_torch.py test_torch_compat.py; do
  sed -e 's/forecast_backbones\.m6/forecast_backbones.m5/g' "$M6/tests/$f" > "$M5/tests/$f"
done
git rm -q forecast_backbones/m5/conformal.py forecast_backbones/m5/conformal_replay.py \
          forecast_backbones/m5/tests/test_conformal.py eval_shell/conformal_m5.sh
```

- [ ] **Step 4: 整体替换 `forecast_backbones/m5/tests/conftest.py`**

```python
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]          # NextGenPVForecast/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import functools
import json
import numpy as np
import pandas as pd
import pytest

CAP = {1001: 500.0, 1002: 300.0}
COORDS = {1001: (30.0, 101.0, 4600.0), 1002: (30.0, 101.0, 4600.0)}
L, H = 672, 192


def write_station_meta(tmp_path, drop_coords_for=()):
    p = Path(tmp_path) / "station_meta.csv"
    lines = ["station_id,station,lat,lon,alt_m,cap_mw"]
    for s, (la, lo, al) in COORDS.items():
        lines.append(f"{s},S{s},,,,{CAP[s]}" if s in drop_coords_for else f"{s},S{s},{la},{lo},{al},{CAP[s]}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _bell(ts: pd.DatetimeIndex) -> np.ndarray:
    h = ts.hour + ts.minute / 60.0
    return np.where((h >= 7) & (h <= 19), np.sin(np.pi * (h - 7) / 12.0), 0.0).clip(min=0)


def make_frame(n_days=3, stations=(1001, 1002), issue_every_min=60, seed=0, start="2025-06-08", sigma=0.15, nan_frac=0.0):
    """物理尺度的合成行表（与 M8 测试同一构造）：GHI ≈ 1000·bell·k_day W/m²，功率 = cap·0.85·bell·k_day·(1+ε)，温度 20 + 8·bell。"""
    rng = np.random.default_rng(seed)
    rows = []
    grid = pd.date_range(pd.Timestamp(start) - pd.Timedelta(days=7), periods=(7 + n_days + 2) * 96, freq="15min")
    bell = _bell(grid)
    for s in stations:
        cap = CAP[s]
        k_day = np.repeat(rng.uniform(0.4, 1.0, len(grid) // 96 + 1), 96)[: len(grid)]
        ghi_true = 1000.0 * bell * k_day
        power = cap * 0.85 * bell * k_day * (1 + np.clip(rng.normal(0, sigma, len(grid)), -0.8, 0.8))
        power = np.where(bell > 0, power, 0.0)
        bias = np.repeat(rng.uniform(-0.1, 0.1, len(grid) // 96 + 1), 96)[: len(grid)]
        ghi_nwp = np.clip(ghi_true * (1 + bias), 0, None)
        temp = 20.0 + 8.0 * bell
        if nan_frac > 0:
            power = power.copy(); power[rng.random(len(grid)) < nan_frac] = np.nan
        issues = pd.date_range(pd.Timestamp(start), pd.Timestamp(start) + pd.Timedelta(days=n_days) - pd.Timedelta(minutes=15),
                               freq=f"{issue_every_min}min")
        for S in issues:
            i = grid.get_loc(S)
            if i - L + 1 < 0 or i + H >= len(grid):
                continue
            hist = slice(i - L + 1, i + 1); fut = slice(i + 1, i + 1 + H)
            rows.append({
                "timestamp_win": S, "station_id": s, "cap_power_on": cap,
                "observe_power": power[hist].astype(np.float32), "observe_power_future": power[fut].astype(np.float32),
                "general_ghi": ghi_nwp[hist].astype(np.float32), "general_ghi_predict": ghi_nwp[fut].astype(np.float32),
                "general_temp": temp[hist].astype(np.float32), "general_temp_predict": temp[fut].astype(np.float32),
            })
    return pd.DataFrame(rows)


def write_splits(tmp_path, n_train_days=4, n_val_days=2, seed=0, train_stations=(1001, 1002), val_stations=(1001, 1002)):
    """train/valid.parquet + split_manifest.json（与 pipelines.time_splits 产物同构的最小集）。"""
    d = Path(tmp_path) / "splits"; d.mkdir(parents=True, exist_ok=True)
    tr = make_frame(n_days=n_train_days, stations=train_stations, issue_every_min=60, seed=seed, start="2025-06-08")
    va = make_frame(n_days=n_val_days, stations=val_stations, issue_every_min=60, seed=seed + 1, start="2025-06-20")
    va["split_block"] = 0
    tr.to_parquet(d / "train.parquet", index=False); va.to_parquet(d / "valid.parquet", index=False)
    (d / "split_manifest.json").write_text(json.dumps({"generated_by": "pipelines.time_splits",
        "plan": {"val_blocks": [["2025-06-20", "2025-06-22"]], "test_start": "2025-07-01"}}), encoding="utf-8")
    return d


def full_cfg(tmp_path, ckpt, **over):
    """full 模式最小配置：小 TimesFM-3 权重 + 两站站表 + 很短的训练预算；over 按段深合并。"""
    from forecast_backbones.m5.config import deep_merge
    base = {"dataset": {"station_meta": str(write_station_meta(tmp_path))},
            "timesfm3": {"checkpoint": str(ckpt), "device": "cpu", "per_core_batch_size": 8},
            "run": {"out_dir": str(Path(tmp_path) / "run")},
            "train": {"schedule": {"batch_size": 4, "max_steps": 4, "eval_every_steps": 2, "warmup_steps": 1,
                                   "logging_steps": 1, "progress_bar": False},
                      "early_stop": {"patience": 3}},
            "eval": {"batch_size": 8}}
    return deep_merge(base, over)


@pytest.fixture(scope="session")
def tiny_ckpt_dir(tmp_path_factory):
    """随机小 TimesFM-3（8/16 patch、32 维、2 层）存成 HF 目录一次，全部测试复用同一份权重。"""
    pytest.importorskip("timesfm3")
    from forecast_backbones.m5.backbone import TorchBackbone
    d = tmp_path_factory.mktemp("tiny_tfm3")
    TorchBackbone.tiny_for_tests(seed=0).model.save_pretrained(str(d))
    return d


@pytest.fixture
def tiny_evaluator(tiny_ckpt_dir):
    """每个测试一个全新的 evaluator（后续测试会往 evaluator.model 注入 LoRA，不能跨测试共享实例）。"""
    from timesfm3 import ModelConfig, TimesFM3Evaluator
    return TimesFM3Evaluator(ModelConfig(checkpoint_path=str(tiny_ckpt_dir), per_core_batch_size=4, device="cpu"))


@pytest.fixture
def collate():
    from pipelines.transformer_dataset import UnifiedDataset
    return functools.partial(UnifiedDataset.collate_fn, for_hf_trainer=True)


@pytest.fixture
def dataset():
    from pipelines.transformer_dataset import UnifiedDataset
    df = make_frame(n_days=2, issue_every_min=120)
    return UnifiedDataset(df, timestamp_cols=["timestamp_win"], entity_cols=[], label_cols=["observe_power_future"], for_hf_trainer=True)


@pytest.fixture
def batch(dataset, collate):
    import torch
    loader = torch.utils.data.DataLoader(dataset, batch_size=6, collate_fn=collate, shuffle=False)
    return next(iter(loader))
```

- [ ] **Step 5: 跑 M5 测试确认通过**

Run: `$PY -m pytest -q forecast_backbones/m5/tests`
Expected: PASS（test_standalone 4 个、复制来的 lora / backbone / torch_compat、原有 test_transformer_trainer 全过）

- [ ] **Step 6: Commit**

```bash
git add forecast_backbones/m5/{solar,_parquet,data,rows,_dist,_arrays,losses,decision,backbone,lora,_torch_compat}.py \
        forecast_backbones/m5/station_meta.csv \
        forecast_backbones/m5/tests/{conftest,test_standalone,test_lora,test_backbone_torch,test_torch_compat}.py
git commit -m "refactor(m5): 删 conformal；自 M8 / M6 复制站表、决策、数据、LoRA、可微底座，M5 自包含（测试守住不引用 m6 / m8）

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
（`git rm` 已把删除放进暂存区，随本次提交。）

---

### Task 2: 配置模块 `config.py` + `finetune.yaml` + 变体

**Files:**
- Create：`forecast_backbones/m5/config.py`、`forecast_backbones/m5/config/finetune.yaml`、`forecast_backbones/m5/config/variants/{z0_zeroshot,l1_lora_r8,d2_pstar_point,g1_solar_cov,g2_cs_ceiling}.yaml`
- Test：`forecast_backbones/m5/tests/test_config.py`

**Interfaces:**
- Produces：`DEFAULT_CFG: dict`、`REMOVED_KEYS: dict[tuple[str,...], str]`、`load_config(src: str|dict|None) -> dict`、`deep_merge(base, override) -> dict`、`interpolate(cfg) -> dict`、`dump_config(cfg, path) -> None`、`raw_override(src) -> dict`、`LORA_TARGETS: list[str]`。

- [ ] **Step 1: 写 `tests/test_config.py`**

```python
from pathlib import Path

import pytest
import yaml

M5 = Path(__file__).resolve().parents[1]
ROOT = M5.parents[1]


def test_finetune_yaml_is_default_cfg_on_disk():
    from forecast_backbones.m5.config import load_config
    assert load_config(str(M5 / "config" / "finetune.yaml")) == load_config(None)


def test_defaults_match_spec():
    from forecast_backbones.m5.config import load_config
    c = load_config(None)
    assert c["features"]["derived"] == {"solar_elevation": False, "clear_sky_power": False}
    assert c["features"]["past_future"] == [["general_ghi", "general_ghi_predict"], ["general_temp", "general_temp_predict"]]
    assert c["decision"]["physics"] == {"night_zero": False, "clear_sky_cap_ratio": None, "capacity_cap": True, "clip_negative": True}
    assert c["decision"]["point"]["default"] == "median"
    assert c["train"]["loss"] == {"night_weight": 1.0, "normalize_by_cap": True}
    assert c["train"]["schedule"]["precision"] == "fp32" and c["train"]["schedule"]["lr"] == 1.0e-4
    assert c["finetune"]["enabled"] is True and c["finetune"]["method"] == "lora"
    assert c["finetune"]["lora"] == {"rank": 8, "alpha": 16, "dropout": 0.05,
                                     "target_modules": ["query_proj", "key_proj", "value_proj", "out_proj"]}
    assert c["train"]["weights_path"] == "runs/m5_baseline/model.safetensors"
    sm = Path(c["dataset"]["station_meta"])
    assert sm.parts[:2] == ("forecast_backbones", "m5") and (ROOT / sm).exists()


@pytest.mark.parametrize("path,hint", [
    (("finetune", "schedule"), "train.schedule"),
    (("finetune", "lora_path"), "train.weights_path"),
    (("timesfm3", "context_len"), "dataset.context_len"),
    (("timesfm3", "past_future_cols"), "features.past_future"),
    (("calibration",), "conformal"),
])
def test_removed_keys_raise_with_new_location(path, hint):
    from forecast_backbones.m5.config import load_config
    over = {}
    d = over
    for k in path[:-1]:
        d = d.setdefault(k, {})
    d[path[-1]] = {"enabled": True} if path == ("calibration",) else 1
    with pytest.raises(ValueError, match=hint):
        load_config(over)


def test_validation_errors():
    from forecast_backbones.m5.config import load_config
    with pytest.raises(ValueError, match="method"):
        load_config({"finetune": {"method": "dora"}})
    with pytest.raises(ValueError, match="inference_path"):
        load_config({"finetune": {"inference_path": "jax"}})
    with pytest.raises(ValueError, match="rank"):
        load_config({"finetune": {"lora": {"rank": 0}}})
    with pytest.raises(ValueError, match="clear_sky_cap_ratio"):
        load_config({"decision": {"physics": {"clear_sky_cap_ratio": 0}}})
    with pytest.raises(ValueError, match="station_meta"):
        load_config({"dataset": {"station_meta": None}})


@pytest.mark.parametrize("name,check", [
    ("z0_zeroshot", lambda c: c["finetune"]["enabled"] is False and c["finetune"]["inference_path"] == "torch"),
    ("l1_lora_r8", lambda c: c["finetune"]["enabled"] is True and c["finetune"]["lora"]["rank"] == 8),
    ("d2_pstar_point", lambda c: c["decision"]["point"]["default"] == "p_star"),
    ("g1_solar_cov", lambda c: c["features"]["derived"] == {"solar_elevation": True, "clear_sky_power": True}),
    ("g2_cs_ceiling", lambda c: c["decision"]["physics"]["clear_sky_cap_ratio"] == 1.05),
])
def test_variants_load(name, check):
    from forecast_backbones.m5.config import load_config
    assert check(load_config(str(M5 / "config" / "variants" / f"{name}.yaml")))


def test_configs_yaml_stays_zero_shot():
    raw = yaml.safe_load((M5 / "config" / "configs.yaml").read_text(encoding="utf-8"))
    assert "station_meta" not in (raw.get("dataset") or {}) and "finetune" not in raw
```

- [ ] **Step 2: 跑测试确认失败**

Run: `$PY -m pytest -q forecast_backbones/m5/tests/test_config.py`
Expected: FAIL（`ModuleNotFoundError: forecast_backbones.m5.config`）

- [ ] **Step 3: 写 `forecast_backbones/m5/config.py`**

```python
# -*- coding: utf-8 -*-
# forecast_backbones/m5/config.py
"""M5（TimesFM-3 + LoRA 微调）的训练 / 评估配置：yaml → dict，与 DEFAULT_CFG 深合并，解析 ${run.out_dir}。
只服务 full 模式（必须有 dataset.station_meta）；线上零样本（legacy）配置由 transformer_models 里的 DEFAULT_CFG 合并，不经过这里。
config/finetune.yaml 是 DEFAULT_CFG 的落盘（tests/test_config.py 守住两边一致）。"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

__all__ = ["DEFAULT_CFG", "REMOVED_KEYS", "LORA_TARGETS", "load_config", "deep_merge", "interpolate", "dump_config", "raw_override"]


def deep_merge(base: dict, override: dict) -> dict:
    """dict 递归合并；非 dict（含 list）整体替换。"""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def interpolate(cfg: dict) -> dict:
    """把字符串里的 ${run.out_dir} 替换成 cfg['run']['out_dir']。只支持这一个占位符。"""
    out_dir = str(cfg.get("run", {}).get("out_dir", ""))

    def _walk(x):
        if isinstance(x, dict):
            return {k: _walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_walk(v) for v in x]
        if isinstance(x, str) and "${run.out_dir}" in x:
            return x.replace("${run.out_dir}", out_dir)
        return x

    return _walk(cfg)


def dump_config(cfg: dict, path) -> None:
    """配置写成 yaml（元组转列表，键序不变）。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plain = json.loads(json.dumps(cfg, ensure_ascii=False, default=str))
    Path(path).write_text(yaml.safe_dump(plain, allow_unicode=True, sort_keys=False), encoding="utf-8")


def raw_override(config_path) -> dict:
    """yaml 路径 / dict / None → 尚未合并、未插值的 override dict。"""
    if config_path is None:
        return {}
    if isinstance(config_path, dict):
        return config_path
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}


LORA_TARGETS = ["query_proj", "key_proj", "value_proj", "out_proj"]

DEFAULT_CFG: dict = {
    "run": {"model_name": "m5", "seeds": [0], "out_dir": "runs/m5_baseline"},
    "dataset": {
        "pv_col": "observe_power",
        "label_col": "observe_power_future",
        "ts_col": "timestamp_win",
        "station_id_col": "station_id",
        # 只取每站固定容量 cap_mw（损失归一与两套指标的分母）与站号（seen / unseen 报表）；经纬度只在几何变体（g1 / g2）里用
        "station_meta": "forecast_backbones/m5/station_meta.csv",
        "station_meta_cols": None,          # 站表列名映射 {内部名: 你表里的列名}，同 M8
        "tz": "Asia/Shanghai",
        "step_min": 15,
        "known_stations": None,             # 训练入口自动写入训练集站号（只用于报表）
        "context_len": 672,
        "pred_len": 192,
    },
    "timesfm3": {
        "checkpoint": "/data/tangqingang/timesfm-3.0-pytorch",
        "device": "auto",
        "per_core_batch_size": 64,          # 只影响 evaluator 路径
        "nan_policy": "interp",
        "make_positive": True,
        "use_znorm": False,
        "use_symmetric_averaging": True,
        "eager_init": True,
    },
    "features": {
        "past_future": [["general_ghi", "general_ghi_predict"], ["general_temp", "general_temp_predict"]],
        "past_only": [],
        # D8：几何协变量默认关，输入与线上 M5 相同；g1 变体打开（太阳高度角 / 晴空出力各一个 past_future 通道）
        "derived": {"solar_elevation": False, "clear_sky_power": False},
    },
    "finetune": {
        "enabled": True,
        "inference_path": "torch",          # enabled=false 时零样本走哪条：torch（与 LoRA 同一可微路径）| evaluator（predict_batch）
        "method": "lora",                   # 目前只有 lora；后续 PEFT 变体的开关位
        "lora": {"rank": 8, "alpha": 16, "dropout": 0.05, "target_modules": list(LORA_TARGETS)},
        "unfreeze_output_head": True,
    },
    "train": {
        "schedule": {"lr": 1.0e-4, "weight_decay": 0.01, "batch_size": 16, "max_steps": 4000, "epochs_max": 10,
                     "warmup_steps": 0, "grad_clip": 1.0, "precision": "fp32", "logging_steps": 20,
                     "eval_every_steps": 250, "sampling": "random", "progress_bar": True, "num_workers": 0},
        # night_weight：夜间步在损失里的权重（1 = 同权）。normalize_by_cap：q、y 都除以该行 cap 再算 pinball（false = 真实单位）
        "loss": {"night_weight": 1.0, "normalize_by_cap": True},
        "early_stop": {"metric": "nanwang_after_pstar", "patience": 5, "nrmse_issue_every_min": 60, "smooth_k": 1},
        "ema_decay": None,                  # 只影子可训参数（LoRA + output_head）
        "weights_path": "${run.out_dir}/model.safetensors",   # 存在即在构造时加载（只含 LoRA + output_head）
    },
    "decision": {
        # D8：clear_sky_cap_ratio=null → 无晴空上限，只钳到容量；g2 变体 = 1.05（与 M8 同口径）
        "physics": {"night_zero": False, "clear_sky_cap_ratio": None, "capacity_cap": True, "clip_negative": True},
        "point": {"default": "median", "by_step": [], "floor_ratio": 0.2, "tau_grid_step": 0.005,
                  "tail": "flat", "shrink_factor": None},
    },
    "eval": {"batch_size": 64, "report_dir": "${run.out_dir}/reports"},
}

# 旧（M6 / legacy M5）键 → 新位置或删除原因；出现即报错，不让 deep_merge 静默收下
REMOVED_KEYS: dict = {
    ("calibration",): "M5 不做 conformal 校准（2026-10-14 删除）",
    ("stacking",): "M5 不做堆叠",
    ("variability",): "M5 不分多变度档",
    ("data",): "切分由 train.py 的 --splits-dir / --train-parquet 给",
    ("finetune", "lora_path"): "train.weights_path",
    ("finetune", "loss"): "train.loss",
    ("finetune", "masks"): "train.loss.night_weight",
    ("finetune", "schedule"): "train.schedule",
    ("finetune", "split"): "train.py 的 --splits-dir",
    ("finetune", "early_stop"): "train.early_stop",
    ("features", "nwp_members"): "M5 不用 NWP 集合成员",
    ("features", "target_transform"): "train.loss.normalize_by_cap",
    ("timesfm3", "context_len"): "dataset.context_len",
    ("timesfm3", "pred_len"): "dataset.pred_len",
    ("timesfm3", "past_future_cols"): "features.past_future",
    ("timesfm3", "past_only_cols"): "features.past_only",
    ("timesfm3", "return_quantiles"): "full 模式恒返回 9 档分位",
    ("timesfm3", "predict_quantile"): "decision.point.default（如 quantile:0.9）",
}


def _check_removed(override: dict) -> None:
    hit = []
    for path, hint in REMOVED_KEYS.items():
        d = override
        for k in path:
            if not isinstance(d, dict) or k not in d:
                break
            d = d[k]
        else:
            hit.append(f"{'.'.join(path)} → {hint}")
    if hit:
        raise ValueError("M5 配置里有已删除 / 已搬家的键：" + "；".join(hit))


def load_config(src) -> dict:
    override = raw_override(src)
    _check_removed(override)
    cfg = interpolate(deep_merge(DEFAULT_CFG, override))
    if not cfg["dataset"].get("station_meta"):
        raise ValueError("M5 训练 / 评估（full 模式）必须给 dataset.station_meta：站表提供每站容量 cap_mw 与站号")
    ft = cfg["finetune"]
    if ft["method"] != "lora":
        raise ValueError(f"finetune.method 目前只支持 lora，得到 {ft['method']!r}")
    if ft["inference_path"] not in ("torch", "evaluator"):
        raise ValueError(f"finetune.inference_path 须是 torch 或 evaluator，得到 {ft['inference_path']!r}")
    if int(ft["lora"]["rank"]) <= 0:
        raise ValueError(f"finetune.lora.rank 须 > 0，得到 {ft['lora']['rank']!r}")
    ratio = cfg["decision"]["physics"].get("clear_sky_cap_ratio")
    if ratio is not None and not float(ratio) > 0:
        raise ValueError(f"decision.physics.clear_sky_cap_ratio 须为 null（无晴空上限）或正数，得到 {ratio!r}")
    return cfg
```

- [ ] **Step 4: 写 `config/finetune.yaml`**

```yaml
# M5 训练 / 评估配置（full 模式）= config.py 的 DEFAULT_CFG 落盘；改默认值两边一起改（tests/test_config.py 守住一致）。
# 线上零样本配置是 configs.yaml（没有 dataset.station_meta → legacy 模式），不要拿它训练。
run:
  model_name: m5
  seeds: [0]
  out_dir: runs/m5_baseline
dataset:
  pv_col: observe_power
  label_col: observe_power_future
  ts_col: timestamp_win
  station_id_col: station_id
  station_meta: forecast_backbones/m5/station_meta.csv   # 只取每站固定容量 cap_mw 与站号；经纬度只在几何变体（g1 / g2）里用
  station_meta_cols: null
  tz: Asia/Shanghai
  step_min: 15
  known_stations: null
  context_len: 672
  pred_len: 192
timesfm3:
  checkpoint: /data/tangqingang/timesfm-3.0-pytorch
  device: auto
  per_core_batch_size: 64
  nan_policy: interp
  make_positive: true
  use_znorm: false
  use_symmetric_averaging: true
  eager_init: true
features:
  past_future:
    - [general_ghi, general_ghi_predict]
    - [general_temp, general_temp_predict]
  past_only: []
  derived: {solar_elevation: false, clear_sky_power: false}   # D8：几何默认关（g1 变体打开）
finetune:
  enabled: true
  inference_path: torch          # enabled=false 时零样本走哪条：torch | evaluator
  method: lora
  lora:
    rank: 8
    alpha: 16
    dropout: 0.05
    target_modules: [query_proj, key_proj, value_proj, out_proj]
  unfreeze_output_head: true
train:
  schedule:
    lr: 1.0e-4
    weight_decay: 0.01
    batch_size: 16
    max_steps: 4000
    epochs_max: 10
    warmup_steps: 0
    grad_clip: 1.0
    precision: fp32              # V100 不支持 bf16
    logging_steps: 20
    eval_every_steps: 250
    sampling: random
    progress_bar: true
    num_workers: 0
  loss:
    night_weight: 1.0
    normalize_by_cap: true       # q、y 都除以该行 cap 再算 pinball；false = 真实单位（M6 旧口径）
  early_stop:
    metric: nanwang_after_pstar
    patience: 5
    nrmse_issue_every_min: 60
    smooth_k: 1
  ema_decay: null
  weights_path: ${run.out_dir}/model.safetensors
decision:
  physics:
    night_zero: false
    clear_sky_cap_ratio: null    # D8：无晴空上限，只钳到容量（g2 变体 = 1.05，与 M8 同口径）
    capacity_cap: true
    clip_negative: true
  point:
    default: median
    by_step: []
    floor_ratio: 0.2
    tau_grid_step: 0.005
    tail: flat
    shrink_factor: null
eval:
  batch_size: 64
  report_dir: ${run.out_dir}/reports
```

- [ ] **Step 5: 写五个变体**

`config/variants/z0_zeroshot.yaml`:
```yaml
# 零样本基线：不挂 LoRA、不训练，与 LoRA run 走同一条 torch 推理 + 决策 + 打分链；第 0 步打分即终评，不写权重
finetune:
  enabled: false
  inference_path: torch
```
`config/variants/l1_lora_r8.yaml`:
```yaml
# LoRA 基线（= finetune.yaml 默认）：rank 8 / alpha 16 / dropout 0.05，attention 四个投影 + 解冻 output_head；输入只有 ghi、温度
finetune:
  enabled: true
  lora: {rank: 8, alpha: 16}
```
`config/variants/d2_pstar_point.yaml`:
```yaml
# 决策层：提交曲线取南网短期最优点 p*（不重训，evaluate.py --decision-yaml 覆盖）
decision:
  point: {default: p_star}
```
`config/variants/g1_solar_cov.yaml`:
```yaml
# 几何协变量（不进本轮基线）：太阳高度角、晴空出力作为两个 past_future 通道喂 TimesFM；需要站表经纬度，需重训
features:
  derived: {solar_elevation: true, clear_sky_power: true}
```
`config/variants/g2_cs_ceiling.yaml`:
```yaml
# 晴空上限（不进本轮基线）：决策层截到 min(1.05·p_clear, cap)，与 M8 同口径；只动决策层，可 --decision-yaml 覆盖已训好的 run
decision:
  physics: {clear_sky_cap_ratio: 1.05}
```

- [ ] **Step 6: 跑测试确认通过**

Run: `$PY -m pytest -q forecast_backbones/m5/tests/test_config.py forecast_backbones/m5/tests/test_standalone.py`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add forecast_backbones/m5/config.py forecast_backbones/m5/config/finetune.yaml forecast_backbones/m5/config/variants \
        forecast_backbones/m5/tests/test_config.py
git commit -m "feat(m5): 训练配置 config.py + finetune.yaml（几何默认关、无晴空上限、fp32）+ 变体 z0 / l1 / d2 / g1 / g2；旧键报错并指出新位置

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: 决策层支持"无晴空上限"

**Files:**
- Modify：`forecast_backbones/m5/decision.py`（`physics_clip`）
- Test：`forecast_backbones/m5/tests/test_decision.py`

**Interfaces:**
- Produces：`physics_clip(q, night, p_clear, cap, clear_sky_cap_ratio: float | None = 1.05, capacity_cap=True, night_ceiling=True, clip_negative=True) -> np.ndarray`；`None` = 上界只看容量（`capacity_cap=False` 时无上界），不看 p_clear。数值时与 M8 完全相同。

- [ ] **Step 1: 写 `tests/test_decision.py`**

```python
import numpy as np

from forecast_backbones.m5.decision import physics_clip


def _q(v, T=4, K=3):
    return np.full((1, T, K), float(v))


def test_no_clear_sky_ceiling_clips_only_at_capacity():
    q = _q(120.0); q[0, 1] = 30.0
    pc = np.array([[0.0, 0.0, 50.0, np.nan]])                       # 夜间 / 夜间 / 白天 / 缺测
    out = physics_clip(q, np.zeros((1, 4), bool), pc, np.array([[100.0]]), clear_sky_cap_ratio=None)
    assert np.allclose(out[0, :, 0], [100.0, 30.0, 100.0, 100.0])   # 只钳到容量；p_clear=0 的夜间步不被钳成 0


def test_no_ceiling_and_no_capacity_cap_leaves_values():
    out = physics_clip(_q(120.0), np.zeros((1, 4), bool), np.zeros((1, 4)), np.array([[100.0]]),
                       clear_sky_cap_ratio=None, capacity_cap=False)
    assert np.allclose(out, 120.0)


def test_none_ratio_still_clips_negative_and_applies_night_mask():
    night = np.array([[False, True, False, False]])
    out = physics_clip(_q(-5.0), night, np.zeros((1, 4)), np.array([[100.0]]), clear_sky_cap_ratio=None)
    assert np.allclose(out, 0.0)
    out2 = physics_clip(_q(40.0), night, np.zeros((1, 4)), np.array([[100.0]]), clear_sky_cap_ratio=None)
    assert np.allclose(out2[0, :, 0], [40.0, 0.0, 40.0, 40.0])


def test_numeric_ratio_keeps_m8_behaviour():
    pc = np.array([[0.0, 40.0, 200.0, np.nan]])
    out = physics_clip(_q(120.0), np.zeros((1, 4), bool), pc, np.array([[100.0]]), clear_sky_cap_ratio=1.05)
    assert np.allclose(out[0, :, 0], [0.0, 42.0, 100.0, 100.0])      # night_ceiling 默认 True：p_clear=0 → 0
    out_nc = physics_clip(_q(120.0), np.zeros((1, 4), bool), pc, np.array([[100.0]]), clear_sky_cap_ratio=1.05,
                          night_ceiling=False)
    assert np.allclose(out_nc[0, :, 0], [100.0, 42.0, 100.0, 100.0])
```

- [ ] **Step 2: 跑测试确认失败**

Run: `$PY -m pytest -q forecast_backbones/m5/tests/test_decision.py`
Expected: FAIL（`TypeError: unsupported operand type(s) for *: 'NoneType' and ...`）

- [ ] **Step 3: 改 `physics_clip`**

把 `forecast_backbones/m5/decision.py` 里的整个 `physics_clip` 函数替换为：

```python
def physics_clip(q, night, p_clear, cap, clear_sky_cap_ratio: float | None = 1.05, capacity_cap: bool = True,
                 night_ceiling: bool = True, clip_negative: bool = True) -> np.ndarray:
    """白天 clip 到 min(clear_sky_cap_ratio·p_clear, cap)，night 掩码为 True 的步归零。
    clear_sky_cap_ratio=None（M5 默认，D8）：没有晴空上限，上界只看容量（capacity_cap=False 时无上界），p_clear 不参与，
      夜间也不会被隐式钳成 0。
    p_clear 非有限值（缺测/NaN）时该步退化为只用装机容量 cap 钳位（不受 capacity_cap 开关影响），
    不设晴空上界，避免 NaN 传播到预报。
    night_ceiling=True（默认，m6 口径）：p_clear<=0 的步 upper=0 → 预报被钳成 0。注意这等于隐式的夜间归零：
      夜间晴空 GHI 恒为 0，所以就算 night 掩码全 False、clear_sky_cap_ratio 再大，夜间也会被钳成 0。
    night_ceiling=False（night_zero=false 的口径）：p_clear<=0 的步与非有限值同样处理，只用 cap 钳位，
      夜间 / 退化站窗外的输出由模型自己给，决策层不做任何形式的夜间归零。
    clip_negative=False：负值也不截到 0（严格裸对比用；负功率物理上不可能，默认仍截）。"""
    q = np.asarray(q, float).copy()
    pc = np.asarray(p_clear, float)
    capv = np.broadcast_to(np.asarray(cap, float), pc.shape)
    if clear_sky_cap_ratio is None:
        upper = capv if capacity_cap else np.full(pc.shape, np.inf)
    else:
        upper = clear_sky_cap_ratio * pc
        if capacity_cap:
            upper = np.minimum(upper, capv)
        free = ~np.isfinite(pc) if night_ceiling else (~np.isfinite(pc) | (pc <= 0.0))
        upper = np.where(free, capv, upper)
    q = np.minimum(q, upper[..., None])
    if clip_negative:
        q = np.maximum(q, 0.0)
    q[np.asarray(night, bool)] = 0.0
    return q
```

- [ ] **Step 4: 跑测试确认通过**

Run: `$PY -m pytest -q forecast_backbones/m5/tests/test_decision.py`
Expected: PASS（4 个）

- [ ] **Step 5: Commit**

```bash
git add forecast_backbones/m5/decision.py forecast_backbones/m5/tests/test_decision.py
git commit -m "feat(m5): physics_clip 支持 clear_sky_cap_ratio=None（无晴空上限，只钳容量、夜间不被隐式归零）；数值时与 M8 相同

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: 站点上下文 `rowctx.py`

**Files:**
- Create：`forecast_backbones/m5/rowctx.py`
- Test：`forecast_backbones/m5/tests/test_rowctx.py`

**Interfaces:**
- Consumes：`m5.solar.{window_timestamps, solar_geometry, clock_pseudo_geometry, StationMeta, station_cap_mw}`。
- Produces：`RowCtx`（dataclass，字段与 M8 同名：`station_id[B] int, cap[B], elev_deg[B,T], ghi_clear[B,T], p_clear[B,T], night[B,T] bool, k_steps[B,H], var_signal[B], degraded[B] bool, cs_ratio[B], envelope_used[B] bool`，T=L+H）；`RowContextBuilder(cfg, station_meta)`，方法 `build(batch_dict) -> RowCtx`、`station_caps(sid[B]) -> ndarray[B]`。M5 恒定：`k_steps=0`、`var_signal=0`、`cs_ratio=1`、`envelope_used=False`。

- [ ] **Step 1: 写 `tests/test_rowctx.py`**

```python
import numpy as np
import pandas as pd
import pytest

from forecast_backbones.m5.config import load_config
from forecast_backbones.m5.rowctx import RowContextBuilder
from forecast_backbones.m5.solar import StationMeta
from .conftest import write_station_meta, L, H


def _bd(sids, ts):
    return {"station_id": np.asarray(sids), "timestamp_win": np.asarray([pd.Timestamp(t).value // 10**9 for t in ts])}


def _builder(tmp_path, drop=()):
    meta = StationMeta.from_csv(write_station_meta(tmp_path, drop_coords_for=drop))
    return RowContextBuilder(load_config({"dataset": {"station_meta": "unused.csv"}}), meta)


def test_caps_shapes_and_night(tmp_path):
    ctx = _builder(tmp_path).build(_bd([1001, 1002], ["2025-06-20 10:00", "2025-06-20 10:00"]))
    assert ctx.cap.tolist() == [500.0, 300.0] and ctx.station_id.tolist() == [1001, 1002]
    assert ctx.elev_deg.shape == (2, L + H) and ctx.p_clear.shape == (2, L + H) and ctx.night.shape == (2, L + H)
    assert not ctx.night[0, L - 1]                       # 起报时刻 10:00 是白天
    assert ctx.night[0, L + 63]                          # 未来第 64 步 = 次日 02:00 是夜间
    assert (ctx.p_clear <= ctx.cap[:, None] + 1e-9).all() and not ctx.degraded.any()
    assert ctx.k_steps.shape == (2, H) and (ctx.k_steps == 0).all()
    assert (ctx.var_signal == 0).all() and (ctx.cs_ratio == 1).all() and not ctx.envelope_used.any()


def test_missing_coords_degrade_with_warning(tmp_path):
    with pytest.warns(UserWarning, match="1002"):
        ctx = _builder(tmp_path, drop=(1002,)).build(_bd([1001, 1002], ["2025-06-20 12:00", "2025-06-20 12:00"]))
    assert ctx.degraded.tolist() == [False, True]
    assert ctx.p_clear[1, L - 1] == pytest.approx(300.0)  # 钟表几何：白天 p_clear = 容量


def test_unknown_station_raises(tmp_path):
    with pytest.raises(ValueError, match="station_meta"):
        _builder(tmp_path).build(_bd([9999], ["2025-06-20 10:00"]))


def test_missing_station_id_column_raises(tmp_path):
    with pytest.raises(ValueError, match="station_id"):
        _builder(tmp_path).build({"timestamp_win": np.asarray([0])})
```

- [ ] **Step 2: 跑测试确认失败**

Run: `$PY -m pytest -q forecast_backbones/m5/tests/test_rowctx.py`
Expected: FAIL（`ModuleNotFoundError: forecast_backbones.m5.rowctx`）

- [ ] **Step 3: 写 `forecast_backbones/m5/rowctx.py`**

```python
# -*- coding: utf-8 -*-
# forecast_backbones/m5/rowctx.py
"""每批的站点上下文：站号 → 固定容量（站表 cap_mw）；窗口时间戳 → 太阳高度角 / 晴空 GHI / 晴空功率 / 夜间。
M5 默认（D8）只用 cap 与 station_id；几何量只进 aux 供报表，features.derived 或 decision.physics.clear_sky_cap_ratio 打开时
才影响预测。SolarCache 自 M8 复制（pvlib 按 (站, 日) 缓存 96 步）；RowCtx 字段与 M8 同名，evaluate.run_inference 原样可用。"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from .solar import window_timestamps, solar_geometry, clock_pseudo_geometry, StationMeta, station_cap_mw
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from forecast_backbones.m5.solar import window_timestamps, solar_geometry, clock_pseudo_geometry, StationMeta, station_cap_mw

_DEGRADED_WARNED: set = set()


def _to_np(v) -> np.ndarray:
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    return np.asarray(v)


class SolarCache:
    """(station_id, 日期) → 该日 15 分钟网格上的 (elev_deg[96], ghi_clear[96])。时间戳不在网格上的行直接算、不缓存。"""

    def __init__(self, station_meta: StationMeta, tz: str, step_min: int = 15, max_entries: int = 400_000):
        self.meta, self.tz, self.step_s = station_meta, tz, int(step_min) * 60
        self.n_day = 86400 // self.step_s
        self.max_entries = int(max_entries)
        self._elev: dict = {}
        self._ghi: dict = {}

    def geometry(self, sid: int, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        """ts [n, T] datetime64[s]（naive 本地墙钟）→ (elev[n,T], ghi_clear[n,T], degraded)。"""
        if not self.meta.has_coords(sid):
            e, g = clock_pseudo_geometry(ts)
            return e, g, True
        m = self.meta.get(sid)
        epoch = ts.astype("datetime64[s]").astype(np.int64)
        if (epoch % self.step_s).any():
            e, g = solar_geometry(ts, m["lat"], m["lon"], m["alt_m"], self.tz)
            return e, g, False
        days = (epoch // 86400).astype(np.int64)
        slots = ((epoch - days * 86400) // self.step_s).astype(np.int64)
        uniq = np.unique(days)
        missing = [int(d) for d in uniq if (sid, int(d)) not in self._elev]
        if missing:
            if len(self._elev) > self.max_entries:
                self._elev.clear(); self._ghi.clear()
            grid = (np.asarray(missing, np.int64)[:, None] * 86400 + np.arange(self.n_day)[None, :] * self.step_s)
            e, g = solar_geometry(grid.astype("datetime64[s]"), m["lat"], m["lon"], m["alt_m"], self.tz)
            for i, d in enumerate(missing):
                self._elev[(sid, d)] = e[i].astype(np.float32); self._ghi[(sid, d)] = g[i].astype(np.float32)
        etab = np.stack([self._elev[(sid, int(d))] for d in uniq]); gtab = np.stack([self._ghi[(sid, int(d))] for d in uniq])
        idx = np.searchsorted(uniq, days)
        return etab[idx, slots].astype(float), gtab[idx, slots].astype(float), False


@dataclass
class RowCtx:
    station_id: np.ndarray      # [B] int
    cap: np.ndarray             # [B]
    elev_deg: np.ndarray        # [B, T]
    ghi_clear: np.ndarray       # [B, T]
    p_clear: np.ndarray         # [B, T]
    night: np.ndarray           # [B, T] bool（elev <= 0）
    k_steps: np.ndarray         # [B, H]，M5 恒 0（M8 的 NWP 晴空指数，这里不算）
    var_signal: np.ndarray      # [B]，M5 恒 0
    degraded: np.ndarray        # [B] bool：站表缺坐标，走钟表几何
    cs_ratio: np.ndarray = None # [B]，M5 恒 1
    envelope_used: np.ndarray = None   # [B]，M5 恒 False


class RowContextBuilder:
    def __init__(self, cfg: dict, station_meta: StationMeta):
        ds = cfg["dataset"]
        self.L, self.H = int(ds["context_len"]), int(ds["pred_len"])
        self.ts_col, self.sid_col, self.step_min = ds["ts_col"], ds["station_id_col"], int(ds["step_min"])
        self.meta = station_meta
        self.solar = SolarCache(station_meta, ds["tz"], self.step_min)
        self._cap_cache: dict[int, float] = {}

    def station_caps(self, sid: np.ndarray) -> np.ndarray:
        """[B] 每行所属站的固定容量（站表 cap_mw）；站表没有的站直接报错。"""
        for s in np.unique(sid):
            if int(s) not in self._cap_cache:
                self._cap_cache[int(s)] = station_cap_mw(self.meta, int(s))
        return np.array([self._cap_cache[int(s)] for s in sid], float)

    def build(self, batch_dict: dict) -> RowCtx:
        if self.sid_col not in batch_dict:
            raise ValueError(f"batch_dict 缺 {self.sid_col!r} 列：M5 full 模式按站查 station_meta 取容量，必须带数值 station_id")
        ts_epoch = _to_np(batch_dict[self.ts_col]).astype(np.int64).reshape(-1)
        sid = _to_np(batch_dict[self.sid_col]).astype(int).reshape(-1)
        B, T = len(sid), self.L + self.H
        cap = self.station_caps(sid)
        ts = window_timestamps(ts_epoch, self.L, self.H, self.step_min)
        elev = np.empty((B, T)); ghi = np.empty((B, T)); degraded = np.zeros(B, bool)
        for s in np.unique(sid):
            rows = np.flatnonzero(sid == s)
            e, g, d = self.solar.geometry(int(s), ts[rows])
            elev[rows] = e; ghi[rows] = g; degraded[rows] = d
        if degraded.any():
            new = [int(s) for s in np.unique(sid[degraded]) if int(s) not in _DEGRADED_WARNED]
            if new:
                warnings.warn(f"station_meta 缺坐标，几何退化为钟表几何（p_clear = 装机容量, elev = ±10）：station_id={new}"
                              f"（M5 默认不用几何，只影响 aux 与 g1 / g2 变体）")
                _DEGRADED_WARNED.update(new)
        p_clear = np.where(degraded[:, None], np.where(ghi > 0, cap[:, None], 0.0), ghi / 1000.0 * cap[:, None])
        return RowCtx(station_id=sid, cap=cap, elev_deg=elev, ghi_clear=ghi, p_clear=p_clear, night=elev <= 0.0,
                      k_steps=np.zeros((B, self.H)), var_signal=np.zeros(B), degraded=degraded,
                      cs_ratio=np.ones(B), envelope_used=np.zeros(B, bool))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `$PY -m pytest -q forecast_backbones/m5/tests/test_rowctx.py forecast_backbones/m5/tests/test_standalone.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add forecast_backbones/m5/rowctx.py forecast_backbones/m5/tests/test_rowctx.py
git commit -m "feat(m5): rowctx——站表固定容量 + 太阳几何（SolarCache 自 M8），RowCtx 与 M8 同字段

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: 模型类——冻结 legacy、模式切换、full 模式构造与权重

**Files:**
- Create：`forecast_backbones/m5/tests/legacy_m5_forward.py`（冻结副本）、`forecast_backbones/m5/tests/test_legacy_contract.py`
- Modify：`forecast_backbones/m5/transformer_models.py`

**Interfaces:**
- Consumes：`m5.config.load_config`、`m5.rowctx.RowContextBuilder`、`m5.decision.MODEL_LEVELS`、`m5.solar.StationMeta`、`m5.lora.*`、`m5.backbone.TorchBackbone`、`m5._torch_compat.ensure_rmsnorm`。
- Produces（模块级）：`load_station_meta(path, cols=None) -> StationMeta`、`save_weights(model, path)`、`load_weights(model, path)`；类属性 `mode: "legacy"|"full"`；full 模式：`cfg`、`config`（property）、`station_meta`、`ctx_builder`、`levels`、`torch_path`、`backbone`（TorchBackbone 或 None）、`weights_loaded`、`set_known_stations(ids) -> list[int]`、`reseed(seed)`、`n_params()`（可训）、`n_total_params()`、`_attach_torch_path()`。Task 6 加 forward / decide / predict_raw / loss。

- [ ] **Step 1: 冻结旧代码作对照**

```bash
{ echo '# 冻结副本：2026-10-14 重构前的 forecast_backbones/m5/transformer_models.py（逐字），只供 test_legacy_contract 对照，不要改。'; \
  git show HEAD:forecast_backbones/m5/transformer_models.py; } > forecast_backbones/m5/tests/legacy_m5_forward.py
```

- [ ] **Step 2: 写 `tests/test_legacy_contract.py`**

```python
"""legacy 模式（线上 nanwang_online / eval_pretrain / experiments 用的零样本配置，没有 dataset.station_meta）
与重构前的 M5 逐位相同：同输入 → 同 logits / aux、同样的底座调用参数、同样的实例属性。"""
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from forecast_backbones.m5.transformer_models import TransformersAdaptedM5Model, TransformersAdaptedTimesFM3Model
from . import legacy_m5_forward as old
from .conftest import write_station_meta

L, H = 672, 192
M5 = Path(__file__).resolve().parents[1]


class _RecordingBackbone:
    """predict_batch 替身：记录全部输入与关键字参数；forecast = 0.5×context 末 H 段 + 0.01×第一协变量末 H 段；
    quantiles = forecast × [0.6..1.4]；return_quantiles=False 时 quantiles=None。"""

    def __init__(self):
        self.calls = []

    def predict_batch(self, inputs, horizon, **kw):
        rec = {k: ([np.array(a, copy=True) for a in v] if isinstance(v, list) else v) for k, v in kw.items()}
        self.calls.append(([np.array(x, copy=True) for x in inputs], rec))
        pf = kw.get("past_future_covariates")
        outs = []
        for i, x in enumerate(inputs):
            fc = 0.5 * x[0, -horizon:] + (0.01 * pf[i][0, -horizon:] if pf else 0.0)
            q = fc[:, None] * np.linspace(0.6, 1.4, 9)[None, :] if kw.get("return_quantiles", True) else None
            outs.append(types.SimpleNamespace(forecast=fc[None, :], quantiles=None if q is None else q[None]))
        return iter(outs)


def _batch(B=3, nan=False, drop=()):
    rng = np.random.default_rng(0)
    t = lambda n: torch.as_tensor(rng.uniform(0, 100, (B, n)), dtype=torch.float32)
    bd = {"timestamp_win": torch.arange(B) * 3600, "observe_power": t(L),
          "general_ghi": t(L), "general_ghi_predict": t(H), "general_temp": t(L), "general_temp_predict": t(H)}
    if nan:
        bd["observe_power"][0, 10:20] = float("nan"); bd["general_ghi"][1, :] = float("nan")
    for c in drop:
        bd.pop(c)
    return bd


def _pair(cfg):
    cfg = {**cfg, "timesfm3": {**(cfg.get("timesfm3") or {}), "eager_init": False}}
    new, ref = TransformersAdaptedM5Model(cfg), old.TransformersAdaptedM5Model(cfg)
    for m in (new, ref):
        m._backbone = _RecordingBackbone(); m._backbone_device = "cpu"
    return new, ref


CASES = {
    "default": ({}, {}),
    "q0.9": ({"timesfm3": {"predict_quantile": 0.9}}, {}),
    "no_quantiles": ({"timesfm3": {"return_quantiles": False}}, {}),
    "nan_interp": ({}, {"nan": True}),
    "missing_pair": ({}, {"drop": ("general_temp_predict",)}),
    "past_only": ({"timesfm3": {"past_only_cols": ["general_temp"]}}, {}),
    "online_yaml": (yaml.safe_load((M5 / "config" / "configs.yaml").read_text(encoding="utf-8")), {}),
    "release_like": ({"run": {"model_name": "m5"}, "model": {}, "dataset": {"pv_col": "observe_power"}}, {}),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_legacy_forward_bit_identical(case):
    cfg, bkw = CASES[case]
    new, ref = _pair(cfg)
    assert new.mode == "legacy"
    a, b = new(_batch(**bkw), labels={"observe_power_future": torch.zeros(3, H)}), ref(_batch(**bkw))
    assert torch.equal(a.logits, b.logits) and torch.equal(a.predictions, b.predictions)
    assert a.loss is None and b.loss is None
    assert (a.aux is None) == (b.aux is None)
    if a.aux is not None:
        assert set(a.aux) == set(b.aux) == {"quantiles"} and torch.equal(a.aux["quantiles"], b.aux["quantiles"])
    (ia, ka), (ib, kb) = new._backbone.calls[0], ref._backbone.calls[0]
    assert len(ia) == len(ib) and all(np.array_equal(x, y) for x, y in zip(ia, ib))
    assert sorted(ka) == sorted(kb)
    for k in ka:
        va, vb = ka[k], kb[k]
        if isinstance(va, list):
            assert len(va) == len(vb) and all(np.array_equal(x, y) for x, y in zip(va, vb)), k
        else:
            assert va == vb, k


def test_legacy_attributes_used_by_callers():
    new, ref = _pair({})
    for k in ("target_col", "context_len", "pred_len", "checkpoint", "device_pref", "past_future_cols", "past_only_cols",
              "per_core_batch_size", "return_quantiles", "predict_quantile", "nan_policy", "make_positive", "use_znorm",
              "use_symmetric_averaging", "_warned_pairs"):
        assert getattr(new, k) == getattr(ref, k), k
    assert not hasattr(new, "config") and not hasattr(ref, "config")          # pipeline 按 hasattr 判断的契约不变
    new.past_future_cols = [("general_ghi", "general_ghi_predict")]            # experiments/nwp_covariate/backends.py 的用法
    new._warned_pairs = set()
    new(_batch())
    assert len(new._backbone.calls[-1][1]["past_future_covariates"][0]) == 1


def test_registry_and_alias():
    from pipelines.model_registry import import_model_class
    assert import_model_class("m5") is TransformersAdaptedM5Model
    assert TransformersAdaptedTimesFM3Model is TransformersAdaptedM5Model


def test_finetune_without_station_meta_raises():
    with pytest.raises(ValueError, match="station_meta"):
        TransformersAdaptedM5Model({"timesfm3": {"eager_init": False}, "finetune": {"enabled": True}})


def test_station_meta_selects_full_mode(tmp_path):
    m = TransformersAdaptedM5Model({"dataset": {"station_meta": str(write_station_meta(tmp_path))},
                                    "timesfm3": {"eager_init": False}})
    assert m.mode == "full" and m.backbone is None and m.cfg["finetune"]["enabled"] is True
    assert m.config is m.cfg and m.levels == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    assert m.set_known_stations([1002, 1001, 1001]) == [1001, 1002] and m.cfg["dataset"]["known_stations"] == [1001, 1002]


def test_full_mode_missing_station_meta_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="station_meta"):
        TransformersAdaptedM5Model({"dataset": {"station_meta": str(tmp_path / "nope.csv")}, "timesfm3": {"eager_init": False}})
```

- [ ] **Step 3: 跑测试确认失败**

Run: `$PY -m pytest -q forecast_backbones/m5/tests/test_legacy_contract.py`
Expected: FAIL——`test_legacy_forward_bit_identical` 在 `assert new.mode == "legacy"` 处 `AttributeError: mode`；三个 full / 报错用例 FAIL；`test_registry_and_alias`、`test_legacy_attributes_used_by_callers` PASS（说明属性契约在改动前就成立）

- [ ] **Step 4: 改 `transformer_models.py`（5 处 Edit，legacy 代码一字不动）**

4a. 文件第 1–2 行：把
```python
# -*- coding: utf-8 -*-
"""TimesFM-3 零样本封装（2026-09-07，流程已对官方 API 核对）。
```
替换为
```python
# -*- coding: utf-8 -*-
"""M5：TimesFM-3。两种模式，按 model_config 里有没有 dataset.station_meta 切换（2026-10-14）：

  legacy（没有 dataset.station_meta：线上 nanwang_online、eval_pretrain、experiments/nwp_covariate 的零样本配置）
      行为与重构前逐位相同，下面"零样本封装"的契约原样有效；tests/legacy_m5_forward.py 冻结了旧代码作对照。
  full（给了 dataset.station_meta：forecast_backbones/m5/train.py / evaluate.py / station_report.py 用）
      配置走 config.load_config（默认值见 config.py / config/finetune.yaml）；站表只取每站固定容量 cap_mw 与站号；
      协变量 = features.past_future（默认与线上相同：ghi、温度）+ 可选几何通道（features.derived，默认关）；
      底座：finetune.enabled → torch 可微路径挂 LoRA（并解冻 output_head）；否则零样本（finetune.inference_path：torch | evaluator）；
      决策层与 M8 相同（单调化 → 钳位 → median / p*），默认无晴空上限；labels 给定时 .loss = 9 档 pinball（默认容量归一）；
      aux 与 M8 同 schema。权重文件（train.weights_path）只存 LoRA + output_head。

────────────────────────────── 以下为 legacy 模式的原文档 ──────────────────────────────
TimesFM-3 零样本封装（2026-09-07，流程已对官方 API 核对）。
```

4b. 在 `from interfaces.modeloutput import TransformersAdaptedModelOutput`（except 分支那一行）之后、`DEFAULT_CFG = {` 之前插入：
```python


def _dep(name: str):
    """full 模式的依赖按需导入：legacy 模式（线上）不 import pvlib / timesfm3.torch。"""
    import importlib
    if __package__:
        try:
            return importlib.import_module(f".{name}", __package__)
        except ImportError:
            pass
    return importlib.import_module(f"forecast_backbones.m5.{name}")


def _full_mode(model_config) -> bool:
    """给了 dataset.station_meta → full 模式（训练 / 评估）；否则 legacy（线上零样本，行为与重构前相同）。"""
    ds = model_config.get("dataset") if isinstance(model_config, dict) else None
    return isinstance(ds, dict) and bool(ds.get("station_meta"))


def load_station_meta(path, cols: dict | None = None):
    """dataset.station_meta（任意路径）+ dataset.station_meta_cols（列名映射）。路径不存在直接报错；
    只有仓库内默认相对路径（forecast_backbones/...）在别的工作目录下解析失败时才回退到 m5 自带那份。"""
    StationMeta = _dep("solar").StationMeta
    p = Path(path)
    if not p.exists():
        fallback = Path(__file__).parent / "station_meta.csv"
        if not p.is_absolute() and p.parts[:1] == ("forecast_backbones",) and fallback.exists():
            warnings.warn(f"station_meta {p} 不存在（相对当前工作目录解析失败）：回退到 {fallback}")
            return StationMeta.from_csv(fallback, cols)
        raise FileNotFoundError(f"station_meta {p} 不存在（dataset.station_meta；相对路径按当前工作目录解析）")
    return StationMeta.from_csv(p, cols)


def save_weights(model, path) -> None:
    """只存 LoRA + output_head（键名相对 backbone.model，safetensors）；底座不存。"""
    lora = _dep("lora")
    if getattr(model, "backbone", None) is None:
        raise RuntimeError("save_weights 需要 torch 路径（full 模式且 finetune.enabled 或 inference_path=torch）")
    lora.save_lora_weights(Path(path), lora.lora_state_dict(model.backbone.model))


def load_weights(model, path) -> None:
    lora = _dep("lora")
    lora.load_lora_state_dict(model.backbone.model, lora.load_lora_weights(Path(path)))
```
并在文件顶部 `import sys` 下面加一行 `import warnings`。

4c. `__init__` 开头：把
```python
        super().__init__()

        # config 文件驱动：config 里的 dataset/timesfm3 段优先，缺省回落到 DEFAULT_CFG。
```
替换为
```python
        super().__init__()
        if _full_mode(model_config):
            self.mode = "full"
            self._init_full(model_config)
            return
        self.mode = "legacy"
        if ((model_config or {}).get("finetune") or {}).get("enabled"):
            raise ValueError("finetune.enabled 需要 full 模式：请在配置里给 dataset.station_meta（站表提供每站容量 cap_mw 与站号）")

        # config 文件驱动：config 里的 dataset/timesfm3 段优先，缺省回落到 DEFAULT_CFG。
```

4d. `_ensure_backbone`：把
```python
        if self._backbone is not None and self._backbone_device == want:
            return
        from timesfm3 import ModelConfig, TimesFM3Evaluator   # 懒 import
```
替换为
```python
        if self._backbone is not None and self._backbone_device == want:
            return
        if self.mode == "full":
            _dep("_torch_compat").ensure_rmsnorm()            # torch < 2.4：须在 import timesfm3.torch 之前
        from timesfm3 import ModelConfig, TimesFM3Evaluator   # 懒 import
```
并把同函数末行
```python
        self._backbone_device = want
```
替换为
```python
        self._backbone_device = want
        if self.mode == "full" and self.torch_path and self.backbone is None:
            self._attach_torch_path()
```

4e. 在 `    # ------------------------------------------------------------------ forward` 这一行之前插入 full 模式构造段：
```python
    # ================================================================== full 模式：构造 / 框架契约 / 权重
    def _init_full(self, model_config: dict) -> None:
        self.cfg = _dep("config").load_config(model_config)
        ds, t3, fe, ft = (self.cfg[k] for k in ("dataset", "timesfm3", "features", "finetune"))
        self.target_col, self.label_col = ds["pv_col"], ds["label_col"]
        self.context_len, self.pred_len = int(ds["context_len"]), int(ds["pred_len"])
        # 与 legacy 同名的属性：_resolve_device / _ensure_backbone / _col 两种模式共用
        self.checkpoint, self.device_pref = t3["checkpoint"], t3.get("device", "auto")
        self.per_core_batch_size, self.nan_policy = int(t3["per_core_batch_size"]), t3["nan_policy"]
        self.make_positive, self.use_znorm = bool(t3["make_positive"]), bool(t3["use_znorm"])
        self.use_symmetric_averaging = bool(t3["use_symmetric_averaging"])
        self.return_quantiles, self.predict_quantile = True, None
        self.past_future_cols = [tuple(p) for p in fe["past_future"]]
        self.past_only_cols = list(fe["past_only"])
        self._warned_pairs = set()
        self.station_meta = load_station_meta(ds["station_meta"], ds.get("station_meta_cols"))
        self.ctx_builder = _dep("rowctx").RowContextBuilder(self.cfg, self.station_meta)
        self.levels = tuple(float(q) for q in _dep("decision").MODEL_LEVELS)
        self.torch_path = bool(ft["enabled"]) or str(ft["inference_path"]) == "torch"
        self.backbone = None                      # torch 路径的 TorchBackbone 子模块（_attach_torch_path 赋值）
        self._backbone, self._backbone_device = None, None
        self.weights_loaded = False
        self._progress_desc = None
        if bool(t3.get("eager_init", True)):
            self._ensure_backbone(self._resolve_device())

    @property
    def config(self) -> dict:
        """full 模式：train.py 把合并后的 cfg 落成 model_config.yaml。legacy 模式没有这个属性（与重构前一致）。"""
        if self.mode != "full":
            raise AttributeError("config")
        return self.cfg

    def set_known_stations(self, station_ids) -> list[int]:
        """训练入口调用：记下训练集里的站号（只用于报表 seen / unseen，TimesFM 本身不用），同步进 cfg 随 model_config.yaml 落盘。"""
        ids = sorted({int(s) for s in station_ids})
        self.cfg["dataset"]["known_stations"] = ids
        return ids

    def reseed(self, seed: int) -> None:
        """train.py 契约；M5 没有随机增强，空操作。"""

    def n_params(self) -> int:
        """可训参数数（LoRA + output_head；零样本为 0）。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def n_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _attach_torch_path(self) -> None:
        """从已加载的 evaluator 取底层 nn.Module 建 TorchBackbone；finetune.enabled 时挂 LoRA、冻结底座、加载 train.weights_path。"""
        lora = _dep("lora")
        self.backbone = _dep("backbone").TorchBackbone.from_evaluator(self._backbone)
        ft = self.cfg["finetune"]
        if ft["enabled"]:
            lo = ft["lora"]
            lora.inject_lora(self.backbone.model, list(lo["target_modules"]), int(lo["rank"]), float(lo["alpha"]),
                             float(lo["dropout"]))
            lora.freeze_all_but_lora(self.backbone.model, bool(ft["unfreeze_output_head"]))
            wp = Path(str(self.cfg["train"]["weights_path"]))
            if wp.exists():
                load_weights(self, wp)
                self.weights_loaded = True
        else:
            self.backbone.requires_grad_(False)
        self.backbone.eval()               # LoRA 的 dropout 默认 train 态：落回推理态，训练脚本显式 .train()

```

- [ ] **Step 5: 跑测试确认通过（含旧的 legacy 测试）**

Run: `$PY -m pytest -q forecast_backbones/m5/tests/test_legacy_contract.py forecast_backbones/m5/tests/test_transformer_trainer.py forecast_backbones/m5/tests/test_standalone.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add forecast_backbones/m5/transformer_models.py forecast_backbones/m5/tests/legacy_m5_forward.py \
        forecast_backbones/m5/tests/test_legacy_contract.py
git commit -m "refactor(m5): 一个类两种模式——没有 dataset.station_meta 走 legacy（逐位等于重构前，冻结副本对照），给了站表走 full（站表容量 / 站号、LoRA 挂载与权重只存 LoRA + output_head）

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: full 模式 forward / 决策 / 损失 / predict_raw

**Files:**
- Modify：`forecast_backbones/m5/transformer_models.py`
- Test：`forecast_backbones/m5/tests/test_model_full.py`

**Interfaces:**
- Consumes：Task 5 的构造段；`TorchBackbone.predict_quantiles`；`m5.decision.{physics_clip, enforce_monotone, select_point, level_value, p_star}`；`m5.losses.pinball_loss`。
- Produces：`decide(q_raw[B,H,K] ndarray, night_fut[B,H] bool, p_clear_fut[B,H], cap[B], decision: dict|None=None) -> (point[B,H], q_cal[B,H,K], median[B,H], pstar[B,H], night_physics[B,H])`；`predict_raw(batch_dict) -> (q[B,H,9] ndarray 真实单位, RowCtx)`；forward（full）返回 `.logits[B,H]`、`.loss`（给了 labels）、`.aux` 键：`quantiles(Tensor, 带梯度) quantiles_cal p_star median levels_cal night night_loss elev_deg p_clear var_bin var_signal station_id cap degraded`；`compute_loss(outputs, labels, num_items_in_batch=None)`。

- [ ] **Step 1: 写 `tests/test_model_full.py`**

```python
import numpy as np
import pytest
import torch

pytest.importorskip("timesfm3")
from forecast_backbones.m5.transformer_models import TransformersAdaptedM5Model, save_weights
from forecast_backbones.m5.lora import LoRALinear
from .conftest import full_cfg, write_station_meta

L, H = 672, 192


def _model(tmp_path, ckpt, **over):
    return TransformersAdaptedM5Model(full_cfg(tmp_path, ckpt, **over))


def test_lora_at_init_equals_zero_shot(tmp_path, tiny_ckpt_dir, batch):
    bd = batch["batch_dict"]
    lo = _model(tmp_path, tiny_ckpt_dir)
    zs = _model(tmp_path, tiny_ckpt_dir, finetune={"enabled": False})
    assert any(isinstance(x, LoRALinear) for x in lo.backbone.modules())
    assert not any(isinstance(x, LoRALinear) for x in zs.backbone.modules())
    with torch.no_grad():
        a, b = lo(bd), zs(bd)
    assert torch.allclose(a.aux["quantiles"], b.aux["quantiles"], atol=1e-6)
    assert torch.allclose(a.logits, b.logits, atol=1e-5)
    assert zs.n_params() == 0 and 0 < lo.n_params() < lo.n_total_params()


def test_torch_and_evaluator_zero_shot_agree(tmp_path, tiny_ckpt_dir, batch):
    bd = batch["batch_dict"]
    a = _model(tmp_path, tiny_ckpt_dir, finetune={"enabled": False, "inference_path": "evaluator"})
    b = _model(tmp_path, tiny_ckpt_dir, finetune={"enabled": False, "inference_path": "torch"})
    assert a.backbone is None and b.backbone is not None
    with torch.no_grad():
        assert torch.allclose(a(bd).aux["quantiles"], b(bd).aux["quantiles"], atol=1e-4)


def test_default_inputs_are_two_nwp_channels_and_g1_adds_geometry(tmp_path, tiny_ckpt_dir, batch):
    bd = batch["batch_dict"]
    seen = {}
    for name, over in (("base", {}), ("g1", {"features": {"derived": {"solar_elevation": True, "clear_sky_power": True}}})):
        m = _model(tmp_path, tiny_ckpt_dir, **over)
        orig = m.backbone.predict_quantiles

        def spy(ctx, pf, po, _n=name, _o=orig, **kw):
            seen[_n] = pf
            return _o(ctx, pf, po, **kw)

        m.backbone.predict_quantiles = spy
        with torch.no_grad():
            m(bd)
    B = len(bd["timestamp_win"])
    assert seen["base"].shape == (B, 2, L + H)
    assert seen["g1"].shape == (B, 4, L + H)
    assert np.array_equal(seen["g1"][:, :2], seen["base"]) and (seen["g1"][:, 2] < 0).any()   # 第 3 通道 = 太阳高度角（夜间为负）


def test_loss_grad_only_on_lora_and_head_and_cap_normalisation(tmp_path, tiny_ckpt_dir, batch):
    bd, labels = batch["batch_dict"], batch["labels"]
    sid = np.asarray(bd["station_id"]).reshape(-1)
    keep = torch.as_tensor(np.flatnonzero(sid == sid[0]))                      # 单站：cap 是一个数
    one = {k: v[keep] for k, v in bd.items()}
    lab = {k: v[keep] for k, v in labels.items()}
    m = _model(tmp_path, tiny_ckpt_dir, finetune={"lora": {"dropout": 0.0}})
    m.train()
    out = m(one, labels=lab)
    assert out.loss.requires_grad and torch.isfinite(out.loss)
    out.loss.backward()
    grads = {n for n, p in m.named_parameters() if p.grad is not None}
    assert grads and all(".lora_" in n or ".output_head." in n for n in grads)
    raw = _model(tmp_path, tiny_ckpt_dir, finetune={"lora": {"dropout": 0.0}}, train={"loss": {"normalize_by_cap": False}})
    raw.train()
    cap = float(out.aux["cap"][0])
    assert float(raw(one, labels=lab).loss) == pytest.approx(float(out.loss) * cap, rel=1e-4)


def test_nan_history_and_missing_labels(tmp_path, tiny_ckpt_dir, batch):
    bd = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch["batch_dict"].items()}
    labels = {k: v.clone() for k, v in batch["labels"].items()}
    bd["observe_power"][0, 100:300] = float("nan")
    bd["general_ghi"][1, :] = float("nan")                                     # 整行缺测 → 置 0
    labels["observe_power_future"][0, :50] = float("nan")
    m = _model(tmp_path, tiny_ckpt_dir)
    m.train()
    out = m(bd, labels=labels)
    assert torch.isfinite(out.loss) and torch.isfinite(out.aux["quantiles"]).all() and torch.isfinite(out.logits).all()
    out.loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_decide_default_has_no_clear_sky_ceiling_and_g2_restores_it(tmp_path):
    m = TransformersAdaptedM5Model({"dataset": {"station_meta": str(write_station_meta(tmp_path))},
                                    "timesfm3": {"eager_init": False}})
    cap = np.array([100.0]); q = np.full((1, 4, 9), 30.0)
    night = np.array([[True, False, False, False]]); pc = np.array([[0.0, 20.0, 80.0, 80.0]])
    point, *_ = m.decide(q, night, pc, cap)
    assert np.allclose(point, 30.0)                                            # 默认：夜间不归零、无晴空上限
    dec = {**m.cfg["decision"], "physics": {**m.cfg["decision"]["physics"], "clear_sky_cap_ratio": 1.05}}
    point2, *_ = m.decide(q, night, pc, cap, dec)
    assert np.allclose(point2, [[30.0, 21.0, 30.0, 30.0]])                     # g2：min(1.05·p_clear, cap)


def test_predict_raw_matches_forward(tmp_path, tiny_ckpt_dir, batch):
    m = _model(tmp_path, tiny_ckpt_dir)
    bd = batch["batch_dict"]
    q, ctx = m.predict_raw(bd)
    with torch.no_grad():
        out = m(bd)
    np.testing.assert_allclose(q, out.aux["quantiles"].numpy(), atol=1e-6)
    assert np.array_equal(ctx.station_id, out.aux["station_id"])
    point, *_ = m.decide(q, ctx.night[:, L:], ctx.p_clear[:, L:], ctx.cap)
    np.testing.assert_allclose(point, out.logits.numpy(), atol=1e-5)
    assert set(out.aux) == {"quantiles", "quantiles_cal", "levels_cal", "p_star", "median", "night", "night_loss",
                            "elev_deg", "p_clear", "var_bin", "var_signal", "station_id", "cap", "degraded"}


def test_weights_round_trip_only_lora_and_head(tmp_path, tiny_ckpt_dir, batch):
    from safetensors.torch import load_file
    m = _model(tmp_path, tiny_ckpt_dir)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("lora_B"):
                p.normal_(0, 0.1)
    w = tmp_path / "w" / "model.safetensors"
    save_weights(m, w)
    keys = set(load_file(str(w)))
    assert keys and all(k.endswith((".lora_A", ".lora_B")) or k.startswith("output_head.") for k in keys)
    m2 = _model(tmp_path, tiny_ckpt_dir, train={"weights_path": str(w)})
    assert m2.weights_loaded and not m.weights_loaded
    bd = batch["batch_dict"]
    m0 = _model(tmp_path, tiny_ckpt_dir)
    with torch.no_grad():
        q1, q2, q0 = m(bd).aux["quantiles"], m2(bd).aux["quantiles"], m0(bd).aux["quantiles"]
    assert torch.allclose(q1, q2, atol=1e-6) and not torch.allclose(q0, q2, atol=1e-6)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `$PY -m pytest -q forecast_backbones/m5/tests/test_model_full.py`
Expected: FAIL（forward 走 legacy 分支 / `AttributeError: decide`）

- [ ] **Step 3: 在 `transformer_models.py` 加 full 推理段与 forward 分派**

3a. 在 Task 5 插入的构造段之后（仍在 `# ---- forward` 之前）追加：
```python
    # ================================================================== full 模式：推理 / 决策 / 损失
    def _covariates(self, batch_dict: dict, ctx):
        """→ (past_future [B,C,L+H] | None, past_only [B,C,L] | None)。缺列的协变量对只提示一次并跳过（与 legacy 同）。"""
        L, H = self.context_len, self.pred_len
        pf = []
        for hist_c, fut_c in self.past_future_cols:
            if hist_c not in batch_dict or fut_c not in batch_dict:
                if (hist_c, fut_c) not in self._warned_pairs:
                    print(f"[timesfm3] 协变量对 {(hist_c, fut_c)} 在 batch 里缺列，本次推理跳过（只提示一次）")
                    self._warned_pairs.add((hist_c, fut_c))
                continue
            pf.append(np.concatenate([self._col(batch_dict, hist_c)[:, -L:], self._col(batch_dict, fut_c)[:, :H]], axis=1))
        d = self.cfg["features"]["derived"]
        if d.get("solar_elevation"):
            pf.append(ctx.elev_deg.astype(np.float32))
        if d.get("clear_sky_power"):
            pf.append(ctx.p_clear.astype(np.float32))
        po = [self._col(batch_dict, c)[:, -L:] for c in self.past_only_cols]
        return (np.stack(pf, axis=1).astype(np.float32) if pf else None,
                np.stack(po, axis=1).astype(np.float32) if po else None)

    def _quantiles(self, tgt: np.ndarray, pf, po) -> torch.Tensor:
        """→ [B,H,9] float32 真实单位。torch 路径可微（训练时带梯度）；evaluator 路径走 predict_batch（无梯度）。"""
        if self.torch_path:
            q = self.backbone.predict_quantiles(tgt, pf, po, horizon=self.pred_len, use_znorm=self.use_znorm,
                                                use_symmetric_averaging=self.use_symmetric_averaging,
                                                make_positive=self.make_positive)
            return q if q.dtype == torch.float32 else q.float()
        kwargs = dict(horizon=self.pred_len, return_quantiles=True, make_positive=self.make_positive,
                      use_znorm=self.use_znorm, use_symmetric_averaging=self.use_symmetric_averaging)
        if pf is not None:
            kwargs["past_future_covariates"] = list(pf)
        if po is not None:
            kwargs["past_only_covariates"] = list(po)
        qs = []
        for o in self._backbone.predict_batch([t[None, :] for t in tgt], **kwargs):
            if getattr(o, "quantiles", None) is None:
                raise ValueError("TimesFM-3 evaluator 没有返回分位数")
            q = np.asarray(o.quantiles)
            qs.append(q[0] if q.ndim == 3 else q)
        return torch.as_tensor(np.stack(qs), dtype=torch.float32, device=torch.device(self._backbone_device or "cpu"))

    def _raw(self, batch_dict: dict):
        if self._backbone is None:
            raise RuntimeError("backbone 未加载：timesfm3.eager_init=False 时需先 _ensure_backbone(device)")
        tgt = self._col(batch_dict, self.target_col)[:, -self.context_len:]
        ctx = self.ctx_builder.build(batch_dict)
        pf, po = self._covariates(batch_dict, ctx)
        return self._quantiles(tgt, pf, po), ctx

    @torch.no_grad()
    def predict_raw(self, batch_dict: dict):
        """推理态：→ (q [B,H,9] 真实单位 numpy, ctx)。不过决策层，供 evaluate.py 集成先平均分位再统一决策。"""
        if self.mode != "full":
            raise RuntimeError("predict_raw 只在 full 模式可用（配置里给 dataset.station_meta）")
        was = self.training
        self.eval()
        try:
            q, ctx = self._raw(batch_dict)
            return q.detach().float().cpu().numpy().astype(float), ctx
        finally:
            self.train(was)

    def decide(self, q_raw: np.ndarray, night_fut: np.ndarray, p_clear_fut: np.ndarray, cap: np.ndarray, decision: dict | None = None):
        """决策层（numpy，没有可训参数，评估时可换规则不用重训）。decision 缺省用 cfg['decision']。
        → (point [B,H], q_cal, median, pstar, night_physics)。clear_sky_cap_ratio=None（默认）时无晴空上限。"""
        dm = _dep("decision")
        dec = decision if decision is not None else self.cfg["decision"]
        ph, pt = dec["physics"], dec["point"]
        if pt.get("tail", "flat") != "flat":
            raise ValueError(f"decision.point.tail: only 'flat' is implemented (got {pt.get('tail')!r})")
        nz, cn = bool(ph["night_zero"]), bool(ph.get("clip_negative", True))
        night_physics = night_fut if nz else np.zeros_like(night_fut)
        ratio = ph.get("clear_sky_cap_ratio")
        q_cal = dm.physics_clip(dm.enforce_monotone(q_raw), night_physics, p_clear_fut, cap[:, None],
                                None if ratio is None else float(ratio), bool(ph["capacity_cap"]),
                                night_ceiling=nz, clip_negative=cn)
        shrink = float(pt.get("shrink_factor") or 1.0)
        point = dm.select_point(q_cal, self.levels, cap, pt["default"], list(pt.get("by_step") or ()), shrink,
                                float(pt["floor_ratio"]), float(pt["tau_grid_step"]))
        median = dm.level_value(q_cal, self.levels, 0.5)
        pstar = dm.p_star(q_cal, self.levels, cap[:, None], float(pt["floor_ratio"]), float(pt["tau_grid_step"]))
        return point, q_cal, median, pstar, night_physics

    def _forward_full(self, batch_dict: dict, labels=None) -> TransformersAdaptedModelOutput:
        L = self.context_len
        q_t, ctx = self._raw(batch_dict)                                          # [B,H,9]；torch 路径训练时带梯度
        dev = q_t.device
        night_fut, p_clear_fut = ctx.night[:, L:], ctx.p_clear[:, L:]
        point, q_cal, median, pstar, night_physics = self.decide(
            q_t.detach().float().cpu().numpy().astype(float), night_fut, p_clear_fut, ctx.cap)
        logits = torch.as_tensor(point, dtype=torch.float32, device=dev)
        aux = {
            "quantiles": q_t,
            "quantiles_cal": torch.as_tensor(q_cal, dtype=torch.float32, device=dev),
            "levels_cal": list(self.levels),
            "p_star": torch.as_tensor(pstar, dtype=torch.float32, device=dev),
            "median": torch.as_tensor(median, dtype=torch.float32, device=dev),
            "night": night_physics, "night_loss": night_fut,
            "elev_deg": ctx.elev_deg[:, L:], "p_clear": p_clear_fut,
            "var_bin": np.zeros(len(ctx.cap), int), "var_signal": ctx.var_signal, "station_id": ctx.station_id,
            "cap": ctx.cap, "degraded": ctx.degraded,
        }
        loss = None if labels is None else self._loss(q_t, labels, night_fut, ctx.cap)
        return TransformersAdaptedModelOutput(loss=loss, logits=logits, predictions=logits, aux=aux)

    def _loss(self, q_t: torch.Tensor, labels, night_fut: np.ndarray, cap: np.ndarray) -> torch.Tensor:
        """9 档 pinball，监督未来 pred_len 步；标签非有限的步掩掉；夜间步按 train.loss.night_weight 加权；
        normalize_by_cap（默认）时 q、y 都除以该行 cap。"""
        y = labels[self.label_col] if isinstance(labels, dict) else labels
        y = torch.as_tensor(np.asarray(y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y, np.float32),
                            device=q_t.device)[:, : self.pred_len]
        lc = self.cfg["train"]["loss"]
        mask = torch.isfinite(y).to(q_t.dtype)
        nw = float(lc.get("night_weight", 1.0))
        if nw != 1.0:
            night = torch.as_tensor(np.asarray(night_fut, bool), device=q_t.device)
            mask = mask * torch.where(night, torch.full_like(mask, nw), torch.ones_like(mask))
        y = torch.nan_to_num(y)
        if bool(lc.get("normalize_by_cap", True)):
            c = torch.as_tensor(np.asarray(cap, np.float32), device=q_t.device)
            q_t, y = q_t / c[:, None, None], y / c[:, None]
        return _dep("losses").pinball_loss(q_t, y, self.levels, mask)

    def compute_loss(self, outputs, labels, num_items_in_batch=None) -> torch.Tensor:
        if outputs.loss is not None:
            return outputs.loss
        aux = outputs.aux
        return self._loss(aux["quantiles"], labels, aux["night_loss"], aux["cap"])

```

3b. forward 开头：把
```python
    def forward(self, batch_dict: dict, labels=None) -> TransformersAdaptedModelOutput:
        # labels 仅为接口兼容，忽略。
```
替换为
```python
    def forward(self, batch_dict: dict, labels=None) -> TransformersAdaptedModelOutput:
        if self.mode == "full":
            return self._forward_full(batch_dict, labels)
        # labels 仅为接口兼容，忽略。（以下为 legacy 模式，逐字沿用重构前代码）
```

- [ ] **Step 4: 跑测试确认通过**

Run: `$PY -m pytest -q forecast_backbones/m5/tests`
Expected: PASS（含 test_legacy_contract 仍全过）

- [ ] **Step 5: Commit**

```bash
git add forecast_backbones/m5/transformer_models.py forecast_backbones/m5/tests/test_model_full.py
git commit -m "feat(m5): full 模式 forward——两对 NWP 协变量（几何通道可选）→ LoRA 可微底座 → M8 决策层（默认无晴空上限）；pinball 容量归一；predict_raw / decide 供集成评估

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: `transformer_trainer.py` 按模式分派 + TrainingArguments

**Files:**
- Modify：`forecast_backbones/m5/transformer_trainer.py`（整体替换）
- Test：`forecast_backbones/m5/tests/test_transformer_trainer.py`（追加一个用例）

**Interfaces:**
- Produces：`evaluate_model(model, test_dataset, HF_collate_fn, config) -> DataFrame`（legacy 原样；full = M8 schema：`timestamp_win, station, station_id, cap, observe_power_predict, observe_power_future, p_star, median, night, elev_deg, p_clear, var_bin, var_signal, degraded, q_0.1..q_0.9`）；`build_training_args(cfg, out_dir, max_steps=None) -> TrainingArguments`；`MODEL_LEVELS`、`QUANTILE_COLS`。

- [ ] **Step 1: 在 `tests/test_transformer_trainer.py` 末尾追加**

```python
def test_evaluate_model_full_mode_schema(tmp_path, tiny_ckpt_dir):
    pytest.importorskip("timesfm3")
    from .conftest import full_cfg, make_frame
    df = make_frame(n_days=1, issue_every_min=360)
    ds = UnifiedDataset(df, timestamp_cols=["timestamp_win"], entity_cols=[], label_cols=["observe_power_future"],
                        for_hf_trainer=True)
    collate = functools.partial(UnifiedDataset.collate_fn, for_hf_trainer=True)
    m = TransformersAdaptedM5Model(full_cfg(tmp_path, tiny_ckpt_dir))
    res = evaluate_model(m, ds, collate, m.cfg)
    assert len(res) == len(df)
    assert {"timestamp_win", "station", "station_id", "cap", "observe_power_predict", "observe_power_future", "p_star",
            "median", "night", "elev_deg", "p_clear", "var_bin", "var_signal", "degraded", *QUANTILE_COLS} <= set(res.columns)
    assert set(res["cap"]) == {500.0, 300.0} and set(res["station"]) == {"S1001", "S1002"}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `$PY -m pytest -q forecast_backbones/m5/tests/test_transformer_trainer.py`
Expected: 新用例 FAIL（结果表缺 `station` / `cap` 等列）

- [ ] **Step 3: 整体替换 `forecast_backbones/m5/transformer_trainer.py`**

```python
# -*- coding: utf-8 -*-
# forecast_backbones/m5/transformer_trainer.py
"""M5 的框架契约入口：evaluate_model(model, test_dataset, HF_collate_fn, config) -> DataFrame（registry.build_evaluator("m5")）。
  legacy 模式（线上零样本配置）：结果表与重构前相同（timestamp_win / station_id / observe_power_predict / observe_power_future / q_*）。
  full 模式（给了站表）：结果表与 M8 同 schema（另带 station / cap / p_star / median / night / elev_deg / p_clear / var_bin /
  var_signal / degraded），train.py 的打分（score_nanwang / nrmse）直接吃。
build_training_args：train.py 用的 HF TrainingArguments（与 M8 相同）。"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

MODEL_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
QUANTILE_COLS = [f"q_{float(l):g}" for l in MODEL_LEVELS]


def _np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def evaluate_model(model, test_dataset, HF_collate_fn, config) -> pd.DataFrame:
    if getattr(model, "mode", "legacy") == "full":
        return _evaluate_full(model, test_dataset, HF_collate_fn)
    return _evaluate_legacy(model, test_dataset, HF_collate_fn, config)


def _evaluate_legacy(model, test_dataset, HF_collate_fn, config) -> pd.DataFrame:
    """逐 batch 零样本推理。列名从 config.dataset 取（ts_col / label_col / station_id_col），缺省与 M6 一致；
    绝不把 label_col 喂给模型。站标识列若被 collate 丢掉（字符串站名）则结果表里没有 station_id 列。"""
    cfg = config or {}
    ds_cfg = cfg.get("dataset") or {}
    ts_col = ds_cfg.get("ts_col", "timestamp_win")
    label_col = ds_cfg.get("label_col", "observe_power_future")
    sid_col = ds_cfg.get("station_id_col", "station_id")
    bs = int((cfg.get("eval") or {}).get("batch_size", 64))
    H = int(model.pred_len)
    loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, collate_fn=HF_collate_fn)
    rows = []
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, dict) and "batch_dict" in batch:
                    bd, labels = batch["batch_dict"], batch.get("labels")
                else:
                    bd, labels = batch, None
                bd = {k: v for k, v in bd.items() if k != label_col}
                truth = _np(labels[label_col]) if labels is not None and label_col in labels else None
                out = model(bd)
                logits = _np(out.logits)
                q = _np(out.aux["quantiles"]) if out.aux and out.aux.get("quantiles") is not None else None
                ts = _np(bd[ts_col]).astype(np.int64).reshape(-1)
                sid = _np(bd[sid_col]).reshape(-1) if sid_col in bd else None
                for i in range(logits.shape[0]):
                    rec = {"timestamp_win": pd.Timestamp(int(ts[i]), unit="s")}
                    if sid is not None:
                        rec["station_id"] = int(sid[i])
                    rec["observe_power_predict"] = logits[i].astype(np.float32)
                    rec["observe_power_future"] = (truth[i][:H].astype(np.float32) if truth is not None
                                                  else np.full(H, np.nan, np.float32))
                    if q is not None:
                        for j, c in enumerate(QUANTILE_COLS):
                            rec[c] = q[i, :, j].astype(np.float32)
                    rows.append(rec)
    finally:
        model.train(was_training)
    return pd.DataFrame(rows)


def _evaluate_full(model, test_dataset, HF_collate_fn) -> pd.DataFrame:
    cfg = model.cfg
    bs = int(cfg["eval"].get("batch_size", 64))
    ts_col, label_col, H = cfg["dataset"]["ts_col"], cfg["dataset"]["label_col"], int(cfg["dataset"]["pred_len"])
    loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, collate_fn=HF_collate_fn)
    show = bool(cfg["train"]["schedule"].get("progress_bar", True))
    desc = str(getattr(model, "_progress_desc", None) or "[m5] 推理")
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, unit="batch", disable=not show, mininterval=2.0, dynamic_ncols=True):
            if isinstance(batch, dict) and "batch_dict" in batch:
                bd, labels = batch["batch_dict"], batch.get("labels")
            else:
                bd, labels = batch, None
            bd = {k: v for k, v in bd.items() if k != label_col}          # 绝不喂未来真值
            truth = _np(labels[label_col]) if labels is not None and label_col in labels else None
            out = model(bd)
            q = _np(out.aux["quantiles"]); logits = _np(out.logits)
            pstar = _np(out.aux["p_star"]); med = _np(out.aux["median"])
            ts = _np(bd[ts_col]).astype(np.int64).reshape(-1)
            for i in range(logits.shape[0]):
                sid = int(out.aux["station_id"][i])
                rec = {
                    "timestamp_win": pd.Timestamp(int(ts[i]), unit="s"),
                    "station": model.station_meta.name(sid), "station_id": sid, "cap": float(out.aux["cap"][i]),
                    "observe_power_predict": logits[i].astype(np.float32),
                    "observe_power_future": (truth[i][:H].astype(np.float32) if truth is not None else np.full(H, np.nan, np.float32)),
                    "p_star": pstar[i].astype(np.float32), "median": med[i].astype(np.float32),
                    "night": out.aux["night"][i].astype(bool), "elev_deg": out.aux["elev_deg"][i].astype(np.float32),
                    "p_clear": out.aux["p_clear"][i].astype(np.float32),
                    "var_bin": int(out.aux["var_bin"][i]), "var_signal": float(out.aux["var_signal"][i]),
                    "degraded": bool(out.aux["degraded"][i]),
                }
                for j, c in enumerate(QUANTILE_COLS):
                    rec[c] = q[i, :, j].astype(np.float32)
                rows.append(rec)
    return pd.DataFrame(rows)


def build_training_args(cfg: dict, out_dir, max_steps=None):
    """max_steps：None = 用 train.schedule.max_steps；0（或配置里 null）= 不限步数，按 epochs_max 训。"""
    from transformers import TrainingArguments
    s = cfg["train"]["schedule"]
    steps = s.get("max_steps") if max_steps is None else max_steps
    cuda = torch.cuda.is_available()
    bf16 = str(s.get("precision", "fp32")) == "bf16" and cuda and torch.cuda.is_bf16_supported()
    fp16 = str(s.get("precision", "fp32")) == "fp16" and cuda
    eval_kw = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters else "evaluation_strategy"
    return TrainingArguments(
        output_dir=str(Path(out_dir) / "hf"), per_device_train_batch_size=int(s["batch_size"]),
        num_train_epochs=float(s["epochs_max"]), learning_rate=float(s["lr"]), weight_decay=float(s["weight_decay"]),
        max_grad_norm=float(s["grad_clip"]), bf16=bf16, fp16=fp16, use_cpu=not cuda,
        lr_scheduler_type="cosine", warmup_steps=int(s.get("warmup_steps", 0)),
        save_strategy="no", logging_steps=int(s.get("logging_steps", 50)), report_to=[],
        remove_unused_columns=False, disable_tqdm=not bool(s.get("progress_bar", True)), dataloader_pin_memory=cuda,
        dataloader_num_workers=int(s.get("num_workers", 0)),
        max_steps=(int(steps) if steps else -1), seed=int((cfg["run"].get("seeds") or [0])[0]),
        ddp_timeout=7200, **{eval_kw: "no"})
```

- [ ] **Step 4: 跑测试确认通过**

Run: `$PY -m pytest -q forecast_backbones/m5/tests`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add forecast_backbones/m5/transformer_trainer.py forecast_backbones/m5/tests/test_transformer_trainer.py
git commit -m "feat(m5): evaluate_model 按模式分派（legacy 原样 / full = M8 结果表 schema）；build_training_args 自 M8

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: 训练入口 `train.py` + `collect_runs.py`

**Files:**
- Create（复制后改）：`forecast_backbones/m5/train.py`、`forecast_backbones/m5/collect_runs.py`
- Test：`forecast_backbones/m5/tests/test_train.py`

**Interfaces:**
- Consumes：`TransformersAdaptedM5Model`、`save_weights`、`load_weights`、`evaluate_model`、`build_training_args`、`m5.config.{DEFAULT_CFG, load_config, deep_merge, dump_config, raw_override}`、`m5.solar.station_cap_mw`、`m5.data.*`、`m5.rows.*`、`m5._dist`。
- Produces：`train_m5(config_path, splits_dir=None, out_dir=None, max_steps=None, tag=None, refit=False, train_parquet=None) -> dict`（键 `summary` DataFrame、`station_table`、`history`、`best`）；`WeightEma(model, decay)`（`.shadow`、`.update(model)`、`.applied(model)`、`.load_into(model)`）；`score_nanwang(model, ds_val, collate, cfg, res=None) -> (float, table, res)`、`seen_unseen_scores(table)`、`nrmse_pct(res)`、`ultra_proxy_pct(res)`、`WEIGHTS_FILE="model.safetensors"`、`CONFIG_FILE="model_config.yaml"`。产物：`model.safetensors`（零样本 run 没有）、`model_config.yaml`、`train_log.json`（零样本有 `"zero_shot": true`）、`result_val.parquet`、`reports/summary.csv`（列见 `_summary_row`）、`reports/station_table.csv`、`split_manifest.json`。

- [ ] **Step 1: 写 `tests/test_train.py`**

```python
import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

pytest.importorskip("timesfm3")
from .conftest import full_cfg, write_splits


def test_lora_run_end_to_end_with_step0_zero_shot(tmp_path, tiny_ckpt_dir):
    from forecast_backbones.m5.train import train_m5
    splits = write_splits(tmp_path, n_train_days=2, n_val_days=2, train_stations=(1001,), val_stations=(1001, 1002))
    out = tmp_path / "lora"
    train_m5(full_cfg(tmp_path, tiny_ckpt_dir, train={"schedule": {"lr": 1.0e-2}}), splits_dir=splits, out_dir=out, tag="l1")
    log = json.loads((out / "train_log.json").read_text(encoding="utf-8"))
    assert [h["step"] for h in log["history"]][:3] == [0, 2, 4]                 # 第 0 步（零样本）+ 每 2 步
    s = pd.read_csv(out / "reports" / "summary.csv").iloc[0]
    assert s["nanwang_zeroshot"] == pytest.approx(log["history"][0]["nanwang_after_pstar"])
    assert s["gain"] == pytest.approx(s["nanwang_short_D+1"] - s["nanwang_zeroshot"])
    assert s["method"] == "lora" and s["rank"] == 8 and s["n_trainable"] > 0 and s["n_unseen_stations"] == 1
    assert (out / "model.safetensors").exists() and not (out / ".fresh" / "model.safetensors").exists()
    assert (out / "result_val.parquet").exists() and (out / "reports" / "station_table.csv").exists()
    saved = yaml.safe_load((out / "model_config.yaml").read_text(encoding="utf-8"))
    assert saved["dataset"]["known_stations"] == [1001] and saved["finetune"]["enabled"] is True


def test_zero_shot_run_scores_without_training_and_matches_lora_step0(tmp_path, tiny_ckpt_dir):
    from forecast_backbones.m5.train import train_m5
    splits = write_splits(tmp_path, n_train_days=2, n_val_days=2)
    z = train_m5(full_cfg(tmp_path, tiny_ckpt_dir, finetune={"enabled": False}), splits_dir=splits, out_dir=tmp_path / "z0", tag="z0")
    zs = z["summary"].iloc[0]
    assert not (tmp_path / "z0" / "model.safetensors").exists() and (tmp_path / "z0" / "model_config.yaml").exists()
    assert zs["best_step"] == 0 and zs["n_evals"] == 1 and zs["method"] == "zero_shot" and zs["n_trainable"] == 0
    assert zs["gain"] == 0.0 and np.isfinite(zs["nanwang_short_D+1"])
    log = json.loads((tmp_path / "z0" / "train_log.json").read_text(encoding="utf-8"))
    assert log["zero_shot"] is True and log["best_step"] == 0
    l = train_m5(full_cfg(tmp_path, tiny_ckpt_dir), splits_dir=splits, out_dir=tmp_path / "l1", tag="l1")
    assert l["history"][0]["nanwang_after_pstar"] == pytest.approx(zs["nanwang_short_D+1"], abs=1e-6)   # B=0 → 与零样本同分
    with pytest.raises(ValueError, match="零样本"):
        train_m5(full_cfg(tmp_path, tiny_ckpt_dir, finetune={"enabled": False}), splits_dir=splits,
                 out_dir=tmp_path / "zr", refit=True)


def test_refit_with_ema_saves_final_weights(tmp_path, tiny_ckpt_dir):
    from forecast_backbones.m5.train import train_m5
    splits = write_splits(tmp_path, n_train_days=2, n_val_days=2)
    out = tmp_path / "refit"
    train_m5(full_cfg(tmp_path, tiny_ckpt_dir, train={"ema_decay": 0.5}), splits_dir=splits, out_dir=out, tag="rf", refit=True)
    s = pd.read_csv(out / "reports" / "summary.csv").iloc[0]
    assert (out / "model.safetensors").exists() and s["best_step"] == 4 and np.isnan(s["nanwang_short_D+1"])
    assert s["ema_decay"] == 0.5


def test_weight_ema_shadows_only_trainable_params():
    from forecast_backbones.m5.train import WeightEma
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 2))
    m[0].requires_grad_(False)
    frozen = m[0].weight.detach().clone()
    ema = WeightEma(m, decay=0.9)
    assert set(ema.shadow) == {"1.weight", "1.bias"}
    w0 = m[1].weight.detach().clone()
    with torch.no_grad():
        m[1].weight.add_(1.0)
    ema.update(m)
    d1 = min(0.9, 2 / 11)
    assert torch.allclose(ema.shadow["1.weight"], d1 * w0 + (1 - d1) * (w0 + 1.0))
    live = m[1].weight.detach().clone()
    with ema.applied(m):
        assert torch.allclose(m[1].weight, ema.shadow["1.weight"]) and torch.equal(m[0].weight, frozen)
    assert torch.equal(m[1].weight, live)
    ema.load_into(m)
    assert torch.allclose(m[1].weight, ema.shadow["1.weight"])
    with pytest.raises(ValueError, match="可训"):
        WeightEma(torch.nn.Linear(2, 2).requires_grad_(False), 0.9)


def test_collect_runs_lists_zero_shot_and_lora(tmp_path, tiny_ckpt_dir):
    from forecast_backbones.m5.train import train_m5
    from forecast_backbones.m5.collect_runs import collect
    splits = write_splits(tmp_path, n_train_days=2, n_val_days=2)
    train_m5(full_cfg(tmp_path, tiny_ckpt_dir, finetune={"enabled": False}), splits_dir=splits, out_dir=tmp_path / "b" / "z0", tag="z0")
    train_m5(full_cfg(tmp_path, tiny_ckpt_dir), splits_dir=splits, out_dir=tmp_path / "b" / "l1", tag="l1")
    df = collect([tmp_path / "b"])
    assert set(df["tag"]) == {"z0", "l1"} and {"nanwang_zeroshot", "gain", "method"} <= set(df.columns)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `$PY -m pytest -q forecast_backbones/m5/tests/test_train.py`
Expected: FAIL（`ModuleNotFoundError: forecast_backbones.m5.train`）

- [ ] **Step 3: 复制并全局改名**

```bash
sed -e 's/m8/m5/g; s/M8/M5/g' forecast_backbones/m8/train.py > forecast_backbones/m5/train.py
sed -e 's/m8/m5/g; s/M8/M5/g' forecast_backbones/m8/collect_runs.py > forecast_backbones/m5/collect_runs.py
```
之后在 `forecast_backbones/m5/train.py` 里：
- 两处 import（相对与绝对兜底）`from .features import station_cap_mw` / `from forecast_backbones.m5.features import station_cap_mw` 改成 `from .solar import station_cap_mw` / `from forecast_backbones.m5.solar import station_cap_mw`。
- 按 3a–3f 替换下列整段；其余函数（`ultra_proxy_pct`、`_subset`、`score_nanwang`、`seen_unseen_scores`、`nrmse_pct`、`score_nrmse`、`EmaCallback`、`_M5Trainer`、`load_presplit`、`expand_parquets`、`load_full`、`_refit_rows`）保持复制来的内容。

3a. 模块文档字符串（第 3–13 行的 `"""M5 训练入口：…"""`）整体替换为：
```python
"""M5 训练入口（TimesFM-3 + LoRA；框架自 M8）：
  1. 读 pipelines.time_splits 落好的切分目录（train.parquet / valid.parquet / split_manifest.json）
  2. HF Trainer：损失在模型 forward 里（9 档 pinball，未来 192 步，默认容量归一）；只训 LoRA + output_head
  3. EvalCallback：第 0 步（LoRA 的 B=0 = 零样本）先打一次分计入 history 并作为初始最优，之后每 eval_every_steps 步打分；
     早停 / 最优按 train.early_stop；最优权重存 model.safetensors（只含 LoRA + output_head）
  4. 训后：加载最优权重 → 验证集推理 → result_val.parquet → reports/summary.csv（含 nanwang_zeroshot / gain）、reports/station_table.csv
  5. finetune.enabled=false（零样本 z0）：不训练，第 0 步打分即终评，不写权重
  6. --refit / --train-parquet：与 M8 相同（无验证、固定步数后存最终权重）；零样本不支持
训练模型的构造从零样本起步：train.weights_path 改指向 out_dir 下从不写入的 .fresh/ 哨兵路径。"""
```

3b. `class WeightEma` 整体替换为：
```python
class WeightEma:
    """权重指数滑动平均（train.ema_decay）：只影子 requires_grad 的参数（LoRA + output_head），冻结的底座不复制。
    shadow ← d_n·shadow + (1−d_n)·θ，d_n = min(decay, (1+n)/(10+n))；applied(model) 期间这些参数换成 shadow，退出复原。"""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay, self.n, self.last_decay = float(decay), 0, float("nan")
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError(f"train.ema_decay 须在 [0, 1]，得到 {decay!r}")
        self.shadow = {k: p.detach().clone() for k, p in model.named_parameters() if p.requires_grad}
        if not self.shadow:
            raise ValueError("train.ema_decay 开了，但模型没有可训参数")

    def _params(self, model: torch.nn.Module) -> dict:
        return {k: p for k, p in model.named_parameters() if k in self.shadow}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.n += 1
        d = min(self.decay, (1.0 + self.n) / (10.0 + self.n))
        self.last_decay = d
        for k, p in self._params(model).items():
            s = self.shadow[k]
            s.mul_(d).add_(p.detach().to(s.device), alpha=1.0 - d)

    @contextlib.contextmanager
    def applied(self, model: torch.nn.Module):
        params = self._params(model)
        backup = {k: p.detach().clone() for k, p in params.items()}
        with torch.no_grad():
            for k, p in params.items():
                p.copy_(self.shadow[k])
        try:
            yield
        finally:
            with torch.no_grad():
                for k, p in params.items():
                    p.copy_(backup[k])

    @torch.no_grad()
    def load_into(self, model: torch.nn.Module) -> None:
        """把 shadow 永久写进模型（refit 的最终权重）。"""
        for k, p in self._params(model).items():
            p.copy_(self.shadow[k])
```

3c. `class EvalCallback`：在 `__init__` 末尾（`self.last_extra: dict = {}` 之后）加一行
```python
        self.zeroshot = None                   # 第 0 步（还没走过优化步 = 零样本）的南网 D+1
```
把 `_evaluate` 整个方法替换为：
```python
    def _evaluate(self, state, control):
        loss = next((h["loss"] for h in reversed(state.log_history) if "loss" in h), None)
        if _dist.is_main():
            with (self.ema.applied(self.model) if self.ema is not None else contextlib.nullcontext()):   # 验证与存最优都用 EMA 权重
                score = self._score()
                if self.zeroshot is None:
                    self.zeroshot = (float(score) if self.metric == "nanwang_after_pstar"
                                     else float(score_nanwang(self.model, self.ds_val, self.collate, self.cfg)[0]))
                stop = self._record(int(state.global_step or 0), float(state.epoch or 0.0), score, loss)
        else:
            stop = False
        if _dist.broadcast_from_main(stop) and control is not None:
            control.should_training_stop = True

    def on_train_begin(self, args, state, control, **kwargs):
        """第 0 步（LoRA 的 B=0，输出与零样本逐位相同）先打一次分，计入 history 并作为初始最优；patience 从它起算。"""
        self._evaluate(state, control)
        return control
```

3d. `_summary_row` 整体替换为：
```python
def _summary_row(cfg, tag, out, model, best_step, n_evals, minutes, n_val, n_train,
                 nw=float("nan"), nr=float("nan"), seen=float("nan"), unseen=float("nan"), n_unseen=0, ultra=float("nan"),
                 zs=float("nan")) -> dict:
    """reports/summary.csv 的一行（列序固定；collect_runs 按列名取）。refit 时分数列全是 NaN。
    nanwang_zeroshot = 第 0 步（零样本）的南网 D+1；gain = 最终 D+1 − 零样本。"""
    ft, lo = cfg["finetune"], cfg["finetune"]["lora"]
    on = bool(ft["enabled"])
    ratio = cfg["decision"]["physics"].get("clear_sky_cap_ratio")
    return {"tag": tag or cfg["run"].get("tag") or out.name, "nanwang_short_D+1": nw, "nanwang_zeroshot": zs, "gain": nw - zs,
            "nrmse_pct": nr, "nanwang_ultra_proxy": ultra,
            "nanwang_seen": seen, "nanwang_unseen": unseen, "n_unseen_stations": n_unseen,
            "best_step": best_step, "n_evals": n_evals, "train_minutes": minutes,
            "method": ft["method"] if on else "zero_shot", "rank": int(lo["rank"]) if on else 0,
            "alpha": float(lo["alpha"]) if on else float("nan"),
            "target_modules": "+".join(lo["target_modules"]) if on else "",
            "unfreeze_head": bool(ft["unfreeze_output_head"]) and on,
            "n_trainable": model.n_params(), "lr": float(cfg["train"]["schedule"]["lr"]),
            "derived": "+".join(k for k, v in cfg["features"]["derived"].items() if v),
            "cs_ceiling": float("nan") if ratio is None else float(ratio),
            "ema_decay": float("nan") if cfg["train"].get("ema_decay") is None else float(cfg["train"]["ema_decay"]),
            "smooth_k": int(cfg["train"]["early_stop"].get("smooth_k", 1) or 1),
            "n_val_rows": n_val, "n_train_rows": n_train}


def _zero_shot_report(model, ds_val, collate, train_cfg, cfg, tag, out, report_dir, n_train) -> dict:
    """finetune.enabled=false：不训练，第 0 步（零样本）打分即终评；不写权重文件。"""
    t0 = time.time()
    model.eval()
    nw, station_tab, _ = score_nanwang(model, ds_val, collate, train_cfg)
    nr, res_hourly = score_nrmse(model, ds_val, collate, train_cfg, int(cfg["train"]["early_stop"].get("nrmse_issue_every_min", 60)))
    res_hourly.to_parquet(out / "result_val.parquet", index=False)
    seen, unseen, n_unseen = seen_unseen_scores(station_tab)
    up = ultra_proxy_pct(res_hourly)
    summary = pd.DataFrame([_summary_row(cfg, tag, out, model, best_step=0, n_evals=1, minutes=(time.time() - t0) / 60.0,
                                         n_val=len(ds_val), n_train=n_train, nw=nw, nr=nr, seen=seen, unseen=unseen,
                                         n_unseen=n_unseen, ultra=up, zs=nw)])
    summary.to_csv(report_dir / "summary.csv", index=False)
    station_tab.to_csv(report_dir / "station_table.csv", index=False)
    metric = str(cfg["train"]["early_stop"].get("metric", "nanwang_after_pstar"))
    score = {"nanwang_after_pstar": nw, "nrmse": nr, "nanwang_ultra_proxy": up}[metric]
    hist = [{"step": 0, "epoch": 0.0, "nanwang_after_pstar": nw, metric: score, "train_loss": None,
             "best": score, "bad_evals": 0, "time": time.time()}]
    (out / "train_log.json").write_text(json.dumps({"history": hist, "best": score, "best_step": 0, "metric": metric,
                                                    "zero_shot": True, "final": summary.to_dict("records")[0]},
                                                   indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(summary.to_string(index=False)); print(station_tab.to_string(index=False))
    print(f"[m5] 零样本：没有训练、没有权重文件；产物 {out / CONFIG_FILE} / {out / 'result_val.parquet'} / {report_dir}")
    return {"summary": summary, "station_table": station_tab, "history": hist, "best": score}
```

3e. `def train_m5(...)` 整个函数替换为：
```python
def train_m5(config_path, splits_dir=None, out_dir=None, max_steps=None, tag: str | None = None, refit: bool = False,
             train_parquet=None) -> dict:
    """splits_dir：time_splits 落好的切分目录（有验证集，可早停）；train_parquet：现成的训练 parquet，全量训练、无验证（等于 refit）。二选一。
    finetune.enabled=false（零样本 z0）：不训练，第 0 步打分即终评，不写权重。"""
    if (splits_dir is None) == (train_parquet is None):
        raise ValueError("--splits-dir 与 --train-parquet 二选一：切分目录（带验证集）或现成的训练 parquet（全量、无验证）")
    if train_parquet is not None:
        refit = True
    if out_dir is not None:
        config_path = deep_merge(raw_override(config_path), {"run": {"out_dir": str(out_dir)}})
    cfg = load_config(config_path)
    finetune = bool(cfg["finetune"]["enabled"])
    if refit and not finetune:
        raise ValueError("零样本（finetune.enabled: false）没有可训练参数：不需要 --refit / --train-parquet，测试集直接用 evaluate.py")
    out = Path(cfg["run"]["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    if out_dir is not None:
        cfg["eval"]["report_dir"] = str(out / "reports")
    report_dir = Path(cfg["eval"]["report_dir"]); report_dir.mkdir(parents=True, exist_ok=True)
    sch = cfg["train"]["schedule"]
    sampling = str(sch.get("sampling") or "random")
    if sampling not in ("random", "station_day"):
        raise ValueError(f"train.schedule.sampling={sampling!r} 不在 ('random', 'station_day')")
    if splits_dir is not None:
        _check_presplit_dir(splits_dir, need_cal=False)
    train_cfg = {**cfg, "train": {**cfg["train"], "weights_path": str(out / ".fresh" / WEIGHTS_FILE)}}
    args = build_training_args(train_cfg, out, max_steps)
    if refit and int(args.max_steps) <= 0:
        raise ValueError("全量重训（refit）没有验证集、不能早停：train.schedule.max_steps（或 --max-steps）必须 > 0，"
                         "按该配方在验证切分上各种子 best_step 的上限给")
    _ = args.device                        # 多进程时在这里建进程组、绑卡；须先于模型构造
    if args.n_gpu > 1:
        raise ValueError(f"单进程看到 {args.n_gpu} 张卡：HF 会退到 DataParallel，dict batch 不支持。多卡请用 accelerate launch / torchrun")
    main = _dist.is_main()
    model = TransformersAdaptedM5Model(train_cfg)
    model.to(args.device)
    scratch = out / ".data"
    station_col, ts_col = str(cfg["dataset"]["station_id_col"]), str(cfg["dataset"]["ts_col"])
    def _load(reuse: bool):
        if train_parquet is not None:
            return load_full(train_parquet, cfg, model.station_meta, scratch, reuse=reuse)
        return load_presplit(splits_dir, cfg, model.station_meta, scratch, reuse=reuse, refit=refit)
    if main:
        loaded = _load(reuse=False)
    _dist.barrier()
    if not main:
        loaded = _load(reuse=True)
    ds_train, ds_val, man = loaded
    known = model.set_known_stations(row_stations(ds_train, station_col).tolist())
    train_cfg["dataset"]["known_stations"] = known
    collate = functools.partial(UnifiedDataset.collate_fn, for_hf_trainer=True)
    eval_every = sch.get("eval_every_steps")
    if main:
        dump_config(deep_merge(deep_merge(DEFAULT_CFG, raw_override(config_path)), {"dataset": {"known_stations": known}}), out / CONFIG_FILE)
        (out / "split_manifest.json").write_text(json.dumps({"generated_by": "forecast_backbones.m5.train",
            "splits_dir": None if splits_dir is None else str(Path(splits_dir).resolve()),
            "train_parquet": man.get("train_parquet"), "plan": man.get("plan"), "refit": bool(refit),
            "counts": {"train": len(ds_train), "val": 0 if ds_val is None else len(ds_val)}}, indent=2, ensure_ascii=False), encoding="utf-8")
        ft, lo = cfg["finetune"], cfg["finetune"]["lora"]
        what = (f"LoRA rank={lo['rank']} alpha={lo['alpha']} dropout={lo['dropout']} targets={'+'.join(lo['target_modules'])} "
                f"解冻 output_head={ft['unfreeze_output_head']}" if finetune else f"零样本（{ft['inference_path']} 路径，不训练）")
        print(f"[m5] tag={tag or cfg['run'].get('tag')} {what}；可训参数 {model.n_params():,} / 总参数 {model.n_total_params():,}", flush=True)
        how = ("无验证、不早停（refit）" if refit else
               f"验证{f'每 {eval_every} 步' if eval_every else '每个 epoch 末'}（含第 0 步零样本），早停指标 "
               f"{cfg['train']['early_stop']['metric']}，patience={cfg['train']['early_stop']['patience']}")
        print(f"[m5] 训练集站 {len(known)} 个；train={len(ds_train)} val={0 if ds_val is None else len(ds_val)} rows；"
              f"每卡 batch {int(args.per_device_train_batch_size)}；预算 {args.max_steps} 步；{how}；采样 {sampling}；device={args.device}", flush=True)
    if not finetune:
        return _zero_shot_report(model, ds_val, collate, train_cfg, cfg, tag, out, report_dir, len(ds_train)) if main else {}
    best_path = out / WEIGHTS_FILE
    ema_decay = cfg["train"].get("ema_decay")
    ema = WeightEma(model, float(ema_decay)) if ema_decay is not None else None
    if main and ema is not None:
        print(f"[m5] 权重 EMA 开：decay={ema.decay}（只影子 LoRA + output_head）；验证与最优权重都用 EMA 权重")
    cb = None if refit else EvalCallback(model, ds_val, collate, train_cfg, best_path=best_path,
                                         patience=int(cfg["train"]["early_stop"]["patience"]),
                                         log_path=out / "train_log.json", eval_every_steps=eval_every, ema=ema)
    callbacks = ([EmaCallback(ema, model)] if ema is not None else []) + ([cb] if cb else [])
    model.train()
    trainer = _M5Trainer(model=model, args=args, data_collator=collate, train_dataset=ds_train, callbacks=callbacks,
                         sampling=sampling, station_col=station_col, ts_col=ts_col)
    t0 = time.time()
    trainer.train()
    if not main:
        return {}
    row = functools.partial(_summary_row, cfg, tag, out, model, n_train=len(ds_train))
    if refit:
        if ema is not None:
            ema.load_into(model)
        save_weights(model, best_path)
        hist = [{"step": int(h["step"]), "train_loss": float(h["loss"])} for h in trainer.state.log_history if "loss" in h]
        summary = pd.DataFrame([row(best_step=int(args.max_steps), n_evals=0, minutes=(time.time() - t0) / 60.0, n_val=0)])
        summary.to_csv(report_dir / "summary.csv", index=False)
        (out / "train_log.json").write_text(json.dumps({"history": hist, "best": None, "best_step": int(args.max_steps),
                                                        "patience": None, "metric": None, "refit": True, "ema_decay": ema_decay,
                                                        "final": summary.to_dict("records")[0]},
                                                       indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(summary.to_string(index=False))
        print(f"[m5] 产物：{best_path} / {out / CONFIG_FILE} / {report_dir}（全量重训没有验证分数；测试集分数用 evaluate.py / eval_m5.sh）")
        return {"summary": summary, "station_table": None, "history": hist, "best": None}
    if cb.best_step is None or not best_path.exists():
        raise ValueError(f"训练全程没有产出可用的最优权重：早停指标 {cb.metric} 在验证集上自始至终非有限，{best_path} 不存在。"
                         f"请检查验证集是否覆盖完整天、station_meta 是否填好。")
    load_weights(model, best_path)
    model.eval()
    nw_score, station_tab, _ = score_nanwang(model, ds_val, collate, train_cfg)
    nr, res_hourly = score_nrmse(model, ds_val, collate, train_cfg, int(cfg["train"]["early_stop"].get("nrmse_issue_every_min", 60)))
    res_hourly.to_parquet(out / "result_val.parquet", index=False)
    nw_seen, nw_unseen, n_unseen = seen_unseen_scores(station_tab)
    summary = pd.DataFrame([row(best_step=cb.best_step, n_evals=len(cb.history), minutes=(time.time() - t0) / 60.0, n_val=len(ds_val),
                                nw=nw_score, nr=nr, seen=nw_seen, unseen=nw_unseen, n_unseen=n_unseen,
                                ultra=ultra_proxy_pct(res_hourly), zs=float(cb.zeroshot))])
    summary.to_csv(report_dir / "summary.csv", index=False)
    station_tab.to_csv(report_dir / "station_table.csv", index=False)
    log = json.loads((out / "train_log.json").read_text(encoding="utf-8"))
    log["final"] = summary.to_dict("records")[0]
    (out / "train_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(summary.to_string(index=False)); print(station_tab.to_string(index=False))
    print(f"[m5] 产物：{best_path} / {out / CONFIG_FILE} / {out / 'result_val.parquet'} / {report_dir}")
    return {"summary": summary, "station_table": station_tab, "history": cb.history, "best": cb.best}
```

3f. `if __name__ == "__main__":` 段里：`ap = argparse.ArgumentParser(description="M5 PVRex 训练")` 改成 `ap = argparse.ArgumentParser(description="M5 TimesFM-3 + LoRA 微调（finetune.enabled=false 时只做零样本打分）")`；`--config` 的默认值改成 `"forecast_backbones/m5/config/finetune.yaml"`；最后一行调用保持 `train_m5(a.config, a.splits_dir, out_dir=a.out, max_steps=a.max_steps, tag=a.tag, refit=a.refit, train_parquet=a.train_parquet)`。

3g. `forecast_backbones/m5/collect_runs.py`：把 `cols = [c for c in [...] if c in df.columns]` 里的列表替换为
```python
["tag", "nanwang_short_D+1", "nanwang_zeroshot", "gain", "nanwang_seen", "nanwang_unseen", "nrmse_pct",
 "nanwang_ultra_proxy", "best_step", "method", "rank", "n_trainable", "lr", "derived", "cs_ceiling", "ema_decay",
 "train_minutes", "run_dir"]
```

- [ ] **Step 4: 检查复制后的残留**

Run: `grep -n "fe\.\|blocks\|patch_len\|d_model\|cpm\|cov_dropout\|PVRex\|from .features" forecast_backbones/m5/train.py`
Expected: 无输出（若有，按 3a–3f 的替换补齐）

- [ ] **Step 5: 跑测试确认通过**

Run: `$PY -m pytest -q forecast_backbones/m5/tests`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add forecast_backbones/m5/train.py forecast_backbones/m5/collect_runs.py forecast_backbones/m5/tests/test_train.py
git commit -m "feat(m5): 训练入口（自 M8）——第 0 步零样本打分进 history 与 summary（nanwang_zeroshot / gain），零样本 run 不训练不写权重，EMA 只影子 LoRA + output_head

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: 评估 `evaluate.py` + 逐站报表 `station_report.py`

**Files:**
- Create（复制后改）：`forecast_backbones/m5/evaluate.py`、`forecast_backbones/m5/station_report.py`
- Test：`forecast_backbones/m5/tests/test_evaluate.py`

**Interfaces:**
- Consumes：`train_m5` 产物目录；`TransformersAdaptedM5Model.{predict_raw, decide, station_meta, levels, cfg}`；`m5.train.{score_nanwang, seen_unseen_scores, nrmse_pct, ultra_proxy_pct, WEIGHTS_FILE, CONFIG_FILE}`。
- Produces：`evaluate(runs, test_paths, out_dir=None, point=None, decision_yaml=None, issue_every_min=60, batch_size=None, device=None, tag=None, progress=True) -> dict`（键 `summary` dict、`station_table`、`members`、`result`）；`load_member(run_dir, device)`；`station_report.evaluate_full(runs, test_paths, out_dir=None, engine="unified", point=None, decision_yaml=None, batch_size=None, device=None, tag=None, progress=True) -> dict`。

- [ ] **Step 1: 写 `tests/test_evaluate.py`**

```python
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("timesfm3")
from .conftest import full_cfg, make_frame, write_splits

M5 = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runs(tmp_path_factory, tiny_ckpt_dir):
    """一个 LoRA run、一个零样本 run（验证集里 1002 是训练没见过的站）。"""
    from forecast_backbones.m5.train import train_m5
    tmp = tmp_path_factory.mktemp("ev")
    splits = write_splits(tmp, n_train_days=2, n_val_days=2, train_stations=(1001,), val_stations=(1001, 1002))
    train_m5(full_cfg(tmp, tiny_ckpt_dir, train={"schedule": {"lr": 1.0e-2}}), splits_dir=splits, out_dir=tmp / "l1", tag="l1")
    train_m5(full_cfg(tmp, tiny_ckpt_dir, finetune={"enabled": False}), splits_dir=splits, out_dir=tmp / "z0", tag="z0")
    return tmp, splits


def test_evaluate_matches_training_eval_for_lora_and_zero_shot(runs, tmp_path):
    from forecast_backbones.m5.evaluate import evaluate
    tmp, splits = runs
    for name in ("l1", "z0"):
        r = evaluate([tmp / name], [str(splits / "valid.parquet")], out_dir=tmp_path / name, progress=False)
        tr = pd.read_csv(tmp / name / "reports" / "summary.csv").iloc[0]
        assert abs(r["summary"]["nanwang_short_D+1"] - tr["nanwang_short_D+1"]) < 1e-3, name
        assert abs(r["summary"]["nrmse_pct"] - tr["nrmse_pct"]) < 1e-3, name
        assert r["summary"]["n_unseen_stations"] == 1 and r["summary"]["point_rule"] == "median"


def test_ensemble_and_decision_override(runs, tmp_path):
    from forecast_backbones.m5.evaluate import evaluate
    tmp, splits = runs
    e = evaluate([tmp / "l1", tmp / "z0"], [str(splits / "valid.parquet")], out_dir=tmp_path / "ens",
                 decision_yaml=str(M5 / "config" / "variants" / "g2_cs_ceiling.yaml"), progress=False)
    assert e["summary"]["n_members"] == 2 and len(e["members"]) == 3 and np.isfinite(e["summary"]["nanwang_short_D+1"])
    dec = json.loads((tmp_path / "ens" / "reports" / "decision.json").read_text(encoding="utf-8"))
    assert dec["physics"]["clear_sky_cap_ratio"] == 1.05
    p = evaluate([tmp / "l1"], [str(splits / "valid.parquet")], out_dir=tmp_path / "p",
                 decision_yaml=str(M5 / "config" / "variants" / "d2_pstar_point.yaml"), progress=False)
    assert p["summary"]["point_rule"] == "p_star"


def test_lora_run_without_weights_is_rejected(runs, tmp_path):
    from forecast_backbones.m5.evaluate import evaluate
    tmp, splits = runs
    bad = tmp_path / "bad"
    shutil.copytree(tmp / "l1", bad)
    (bad / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="LoRA"):
        evaluate([bad], [str(splits / "valid.parquet")], out_dir=tmp_path / "x", progress=False)


def test_station_report_native(runs, tmp_path):
    from forecast_backbones.m5.station_report import evaluate_full
    tmp, _ = runs
    tdir = tmp_path / "test"; tdir.mkdir()
    for s in (1001, 1002):
        df = make_frame(n_days=3, stations=(s,), issue_every_min=15, seed=5 + s, start="2025-06-24")
        df.insert(1, "station", f"S{s}")
        df.to_parquet(tdir / f"station={s}.parquet", index=False)
    r = evaluate_full([tmp / "l1"], [str(tdir)], out_dir=tmp_path / "f", engine="native", progress=False)
    tab = r["station_table"]
    assert list(tab["station"]) == ["S1001", "S1002"] and tab["seen"].tolist() == [True, False]
    assert np.isfinite(tab[["nanwang_short_D+1", "nanwang_ultra_short"]].astype(float).to_numpy()).all()
    assert (tmp_path / "f" / "metrics_ens" / "S1001-metrics.xlsx").exists()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `$PY -m pytest -q forecast_backbones/m5/tests/test_evaluate.py`
Expected: FAIL（`ModuleNotFoundError: forecast_backbones.m5.evaluate`）

- [ ] **Step 3: 复制、全局改名、改 import**

```bash
sed -e 's/m8/m5/g; s/M8/M5/g' forecast_backbones/m8/evaluate.py > forecast_backbones/m5/evaluate.py
sed -e 's/m8/m5/g; s/M8/M5/g' forecast_backbones/m8/station_report.py > forecast_backbones/m5/station_report.py
```
在 `forecast_backbones/m5/station_report.py` 里把两处 `from .features import station_cap_mw` / `from forecast_backbones.m5.features import station_cap_mw` 改成 `from .solar import station_cap_mw` / `from forecast_backbones.m5.solar import station_cap_mw`。

- [ ] **Step 4: 替换 `evaluate.load_member`**

```python
def load_member(run_dir, device) -> TransformersAdaptedM5Model:
    """train_m5 的产物目录 → 推理态模型。LoRA run 必须有 model.safetensors；零样本 run（finetune.enabled=false）没有权重文件。"""
    run = Path(run_dir)
    cfg_path, w = run / CONFIG_FILE, run / WEIGHTS_FILE
    if not cfg_path.exists():
        raise FileNotFoundError(f"{run} 缺 {CONFIG_FILE}（要 train_m5 的产物目录）")
    raw = raw_override(str(cfg_path))
    finetune = bool((raw.get("finetune") or {}).get("enabled", True))
    if finetune and not w.exists():
        raise FileNotFoundError(f"{run} 缺 {WEIGHTS_FILE}：finetune.enabled=true 的 run 必须有 LoRA 权重")
    cfg = deep_merge(raw, {"train": {"weights_path": str(w)}, "timesfm3": {"device": str(device)}})
    m = TransformersAdaptedM5Model(cfg)
    if finetune and not m.weights_loaded:
        raise RuntimeError(f"{w} 没有加载上")
    if not m.cfg["dataset"].get("known_stations"):
        warnings.warn(f"{cfg_path} 没有 known_stations：所有站都按已见站处理")
    m.to(device); m.eval()
    return m
```

- [ ] **Step 5: 检查残留并跑测试**

Run: `grep -n "from .features\|model\"\]\[\"quantiles\|PVRex" forecast_backbones/m5/evaluate.py forecast_backbones/m5/station_report.py`
Expected: 无输出

Run: `$PY -m pytest -q forecast_backbones/m5/tests`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add forecast_backbones/m5/evaluate.py forecast_backbones/m5/station_report.py forecast_backbones/m5/tests/test_evaluate.py
git commit -m "feat(m5): evaluate / station_report（自 M8）——零样本成员无权重可加载、LoRA 成员缺权重报错；集成分位平均与 --decision-yaml 换决策不重训

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Shell 脚本 + README

**Files:**
- Create：`train_shell/train_m5.sh`、`train_shell/train_m5_variants.sh`、`eval_shell/eval_m5.sh`、`forecast_backbones/m5/README.md`
- Test：`forecast_backbones/m5/tests/test_shells.py`

**Interfaces:**
- Consumes：`python -m forecast_backbones.m5.train`、`…m5.evaluate`、`…m5.station_report`、`…m5.collect_runs`。
- Produces：`bash train_shell/train_m5.sh [--splits_dir|--train_parquet] [--config] [--gpus] [--seed] [--max_steps] [--eval_every] [--batch_size] [--precision] [--tag] [--run_id] [--runs_base] [--refit] [--overwrite] [--venv-python] [--allow-cpu]` → `runs/m5/<run_id>/`；`train_m5_variants.sh` → `runs/m5/<batch_tag>/<变体>[_s<seed>]/` 与 `variants_summary.csv`；`eval_m5.sh --runs … --test … [--full] [--decision_yaml]`。

- [ ] **Step 1: 写 `tests/test_shells.py`**

```python
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SH = [ROOT / "train_shell" / "train_m5.sh", ROOT / "train_shell" / "train_m5_variants.sh", ROOT / "eval_shell" / "eval_m5.sh"]


@pytest.mark.parametrize("p", SH, ids=lambda p: p.name)
def test_shell_syntax_and_no_m8_leftovers(p):
    assert p.exists(), p
    assert subprocess.run(["bash", "-n", str(p)], capture_output=True).returncode == 0
    bad = [l for l in p.read_text(encoding="utf-8").splitlines()
           if "forecast_backbones/m8" in l or "forecast_backbones.m8" in l or "runs/m8" in l or "PVRex" in l]
    assert not bad, bad


def test_train_m5_defaults_and_zero_shot_check():
    txt = SH[0].read_text(encoding="utf-8")
    assert 'CONFIG="${BASE_DIR}/forecast_backbones/m5/config/finetune.yaml"' in txt
    assert 'SPLITS_DIR="${BASE_DIR}/data/splits/m8_v1"' in txt
    assert "zero_shot" in txt                                  # 零样本 run 没有 model.safetensors 也算成功


def test_readme_exists():
    assert (ROOT / "forecast_backbones" / "m5" / "README.md").exists()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `$PY -m pytest -q forecast_backbones/m5/tests/test_shells.py`
Expected: FAIL（文件不存在）

- [ ] **Step 3: 复制并全局改名**

```bash
sed -e 's/m8/m5/g; s/M8/M5/g' train_shell/train_m8.sh > train_shell/train_m5.sh
sed -e 's/m8/m5/g; s/M8/M5/g' train_shell/train_m8_variants.sh > train_shell/train_m5_variants.sh
sed -e 's/m8/m5/g; s/M8/M5/g' eval_shell/eval_m8.sh > eval_shell/eval_m5.sh
```

- [ ] **Step 4: 改三个脚本的头注释与默认值（用这个脚本做精确替换，找不到就报错）**

```bash
$PY - <<'PYEOF'
from pathlib import Path
import re

def edit(path, pairs):
    p = Path(path); s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert s.count(old) >= 1, (path, old[:60])
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8")

def header(path, new_header):
    """替换第 2 行起、到第一个非注释行之前的注释头（第 1 行 #!/bin/bash 保留）。"""
    p = Path(path); lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
    i = 1
    while i < len(lines) and lines[i].startswith("#"):
        i += 1
    p.write_text(lines[0] + new_header + "".join(lines[i:]), encoding="utf-8")

header("train_shell/train_m5.sh", """#
# train_m5.sh —— M5（TimesFM-3 + LoRA 微调；finetune.enabled=false 时只做零样本打分）单个配置的训练脚本。
# 与 train_m8.sh 同一套约定：同一个 python 与 LD_LIBRARY_PATH、同一份切分目录、accelerate launch、配置只改运行时副本。
# 底座 ~200M 参数只训 LoRA + output_head；V100 用 fp32，显存不够把 --batch_size 降到 8。
#
# 用法：
#   bash train_m5.sh --splits_dir data/splits/m8_v1 --gpus 0                                              # 默认 = l1 LoRA 基线
#   bash train_m5.sh --splits_dir data/splits/m8_v1 --config forecast_backbones/m5/config/variants/z0_zeroshot.yaml --gpus 0
#   bash train_m5.sh --splits_dir data/splits/m8_v1 --gpus 0 --max_steps 20 --eval_every 10 --run_id smoke   # 冒烟
#   批量跑变体 / 多种子见 train_m5_variants.sh
""")
edit("train_shell/train_m5.sh", [
    ('SPLITS_DIR="${BASE_DIR}/data/splits/m1_m4_v1"', 'SPLITS_DIR="${BASE_DIR}/data/splits/m8_v1"'),
    ('CONFIG="${BASE_DIR}/forecast_backbones/m5/config/configs.yaml"', 'CONFIG="${BASE_DIR}/forecast_backbones/m5/config/finetune.yaml"'),
    ("M5 配置或变体（默认 forecast_backbones/m5/config/configs.yaml；变体在 config/variants/）",
     "M5 训练配置或变体（默认 forecast_backbones/m5/config/finetune.yaml；变体在 config/variants/；config/configs.yaml 是线上零样本配置，不能拿来训练）"),
    ("（配方选定后给最终模型；切分用 data/splits/m5_refit.yaml 落的目录，valid.parquet 为 0 行）",
     "（配方选定后给最终模型；切分用 data/splits/m8_refit.yaml 落的目录，valid.parquet 为 0 行；零样本 z0 不需要 refit）"),
    ("请给 --max_steps（j1_final2 用 6000）", "请给 --max_steps（按验证切分上各种子 best_step 的上限给）"),
    ('''for f in model.safetensors model_config.yaml train_log.json reports/summary.csv; do
  [ -f "$RUN_DIR/$f" ] || { echo "❌ 训练结束但缺少 $RUN_DIR/$f" >&2; exit 1; }
done''',
     '''for f in model_config.yaml train_log.json reports/summary.csv; do
  [ -f "$RUN_DIR/$f" ] || { echo "❌ 训练结束但缺少 $RUN_DIR/$f" >&2; exit 1; }
done
# 零样本 run（train_log.json 里 "zero_shot": true）不写权重；LoRA run 必须有 model.safetensors
if ! grep -q '"zero_shot": true' "$RUN_DIR/train_log.json" && [ ! -f "$RUN_DIR/model.safetensors" ]; then
  echo "❌ 训练结束但缺少 $RUN_DIR/model.safetensors" >&2; exit 1
fi'''),
])

header("train_shell/train_m5_variants.sh", """#
# train_m5_variants.sh —— 批量训练 M5 的多个变体：每个变体一张卡，卡数不够时排队；跑完汇总成一张表（collect_runs）。
#
# 用法（基线）：
#   bash train_m5_variants.sh --splits_dir data/splits/m8_v1 --variants z0_zeroshot --gpus 0 --batch_tag m5_base
#   bash train_m5_variants.sh --splits_dir data/splits/m8_v1 --variants l1_lora_r8 --seeds 0,1,2 --gpus 1,2,3 --batch_tag m5_base
#   冒烟：... --variants z0_zeroshot,l1_lora_r8 --max_steps 20 --eval_every 10 --gpus 0 --batch_tag smoke
# d2_pstar_point / g2_cs_ceiling 只动决策层，不用训：对已训好的 run 用 eval_m5.sh --decision_yaml
# 产物：runs/m5/<batch_tag>/<变体>[_s<seed>]/…，汇总 runs/m5/<batch_tag>/variants_summary.csv
""")
edit("train_shell/train_m5_variants.sh", [
    ('SPLITS_DIR="${BASE_DIR}/data/splits/m1_m4_v1"', 'SPLITS_DIR="${BASE_DIR}/data/splits/m8_v1"'),
])

header("eval_shell/eval_m5.sh", """#
# eval_m5.sh —— M5 测试集评估：一个或多个训练产物目录（多个 = 集成，分位平均后再决策）→ 测试 parquet → 指标。
# 零样本 run（z0）没有权重文件，可以单独评，也可以和 LoRA run 一起集成。
#   默认（舰队快评）：10:00 行 + 整点行 → 南网 D+1（seen/unseen）/ nrmse → result_test.parquet + reports/
#   --full（逐站完整评测）：全部起报行 → ods_short / ods_ultra_short / nanwang_short_D+1 / nanwang_ultra_short
#          → metrics_ens/{站}-metrics.xlsx + {站}-summary.json + reports/station_metrics.csv
#
# 用法：
#   bash eval_m5.sh --runs runs/m5/m5_base/l1_lora_r8_s0 --test data/splits/m8_v1/valid.parquet
#   bash eval_m5.sh --runs runs/m5/m5_base/l1_lora_r8_s0 --test data/splits/m8_v1/valid.parquet \\
#        --decision_yaml forecast_backbones/m5/config/variants/d2_pstar_point.yaml          # p*，不重训
#   bash eval_m5.sh --runs runs/m5/m5_base/l1_lora_r8_s0,runs/m5/m5_base/l1_lora_r8_s1,runs/m5/m5_base/l1_lora_r8_s2 \\
#        --test "datasets/test/*.parquet" --full
""")
PYEOF
```
注：`header()` 写进去的 `\\` 在文件里是单个 `\`（续行注释，仅示意）。

- [ ] **Step 5: 人工过一遍残留**

Run: `grep -n "m8\|M8\|PVRex\|1.5M\|slstm\|e1_final\|j1_final\|a1_hybrid" train_shell/train_m5.sh train_shell/train_m5_variants.sh eval_shell/eval_m5.sh`
Expected: 只剩 `data/splits/m8_v1`、`m8_refit.yaml`、`train_m8.sh`（头注释里"与 train_m8.sh 同一套约定"）这类有意保留的引用；其余逐条改成 M5 语义

- [ ] **Step 6: 写 `forecast_backbones/m5/README.md`**

```markdown
# M5：TimesFM-3（零样本 + LoRA 微调）

一个模型类 `TransformersAdaptedM5Model`（旧名 `TransformersAdaptedTimesFM3Model`），两种模式：

| 模式 | 触发 | 谁在用 | 行为 |
|---|---|---|---|
| legacy | 配置里**没有** `dataset.station_meta` | 线上 `nanwang_online`、`eval_shell/eval_pretrain.py`、`experiments/nwp_covariate` | 与 2026-10-14 前逐位相同（`tests/legacy_m5_forward.py` 冻结对照） |
| full | 给了 `dataset.station_meta` | `train.py` / `evaluate.py` / `station_report.py` | 站表容量 + LoRA 可微路径 + M8 决策层 |

配置：`config/configs.yaml` = 线上零样本（不要拿来训练）；`config/finetune.yaml` = 训练默认（= `config.py` 的 `DEFAULT_CFG`）。

基线（D8：站点几何默认关）：输入只有 ghi、温度；决策层只钳容量、不加晴空上限；点预测取中位数。
零样本 `z0_zeroshot` 与 LoRA `l1_lora_r8` 只差 LoRA 权重，summary 里 `gain = nanwang_short_D+1 − nanwang_zeroshot`。
几何留作变体：`g1_solar_cov`（两个几何协变量，需重训）、`g2_cs_ceiling`（晴空上限 1.05，只动决策层）。

## 跑基线（服务器）

    bash train_shell/train_m5_variants.sh --splits_dir data/splits/m8_v1 --variants z0_zeroshot --gpus 0 --batch_tag m5_base
    bash train_shell/train_m5_variants.sh --splits_dir data/splits/m8_v1 --variants l1_lora_r8 --seeds 0,1,2 --gpus 1,2,3 --batch_tag m5_base
    python -m forecast_backbones.m5.evaluate --runs runs/m5/m5_base/l1_lora_r8_s0 --test data/splits/m8_v1/valid.parquet \
           --decision-yaml forecast_backbones/m5/config/variants/d2_pstar_point.yaml        # p*，不重训

## 站表

`station_meta.csv` 从 M8 复制，M5 / M6 / M8 各一份、各自维护；改站表要三处一起改。容量只取 `cap_mw`（每站固定值）。

## 权重

`model.safetensors` 只含 LoRA（`*.lora_A / *.lora_B`）与 `output_head.*`，键名相对 TimesFM 的 `model`；底座从 `timesfm3.checkpoint` 加载。
零样本 run 没有权重文件。
```

- [ ] **Step 7: 跑测试确认通过**

Run: `$PY -m pytest -q forecast_backbones/m5/tests`
Expected: PASS

- [ ] **Step 8: Commit（只 add 这四个文件 + 测试）**

```bash
git add train_shell/train_m5.sh train_shell/train_m5_variants.sh eval_shell/eval_m5.sh \
        forecast_backbones/m5/README.md forecast_backbones/m5/tests/test_shells.py
git commit -m "feat(m5): train_m5.sh / train_m5_variants.sh / eval_m5.sh（自 M8 脚本）+ README；零样本 run 无权重文件也算训练成功

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: 全量回归 + 服务器基线

**Files:** 无代码改动（发现问题回到对应任务修）。

- [ ] **Step 1: 本机全量回归**

Run: `$PY -m pytest -q forecast_backbones/m5/tests forecast_backbones/m6/tests forecast_backbones/m8/tests experiments/nwp_covariate/tests 2>&1 | tail -5`
Expected: M5 全过；M6 与重构前一致；M8 只有重构前就失败的 2 个（`test_reference.py::test_j1_forward_matches_reference`、`test_train_protocol.py::test_train_m8_ema_weights_are_what_gets_scored_and_saved`，本机浮点容差问题，与本次无关）；experiments 与重构前一致。对照基准：用 `git log` 找 Task 1 之前的提交，在一个临时 worktree（`git worktree add /tmp/... <sha>`）里跑同一条命令得到对照，跑完 `git worktree remove`（不用 `git stash`：另一个会话的未提交改动不能动）。

- [ ] **Step 2: 确认工作树里没有误提交 / 漏提交**

Run: `git status --short forecast_backbones/m5 train_shell/train_m5.sh train_shell/train_m5_variants.sh eval_shell/eval_m5.sh docs/superpowers`
Expected: 无输出（全部已提交）；`git log --stat -8` 里只有本计划列出的路径

- [ ] **Step 3: 服务器冒烟（用户在服务器执行；我给命令）**

```bash
cd <仓库> && git pull
bash train_shell/train_m5_variants.sh --splits_dir data/splits/m8_v1 --variants z0_zeroshot,l1_lora_r8 \
     --max_steps 20 --eval_every 10 --gpus 0 --batch_tag m5_smoke
```
看点：`runs/m5/m5_smoke/variants_summary.csv` 两行；`l1` 的 `train_log.json` 第 0 步分数 = `z0` 的 `nanwang_short_D+1`；日志里没有 `horizon=192 须为 output_patch_len 的整数倍` 报错；记下每步耗时与显存（`nvidia-smi`），显存不够加 `--batch_size 8`。

- [ ] **Step 4: 服务器基线（冒烟通过后）**

```bash
bash train_shell/train_m5_variants.sh --splits_dir data/splits/m8_v1 --variants z0_zeroshot --gpus 0 --batch_tag m5_base
bash train_shell/train_m5_variants.sh --splits_dir data/splits/m8_v1 --variants l1_lora_r8 --seeds 0,1,2 --gpus 1,2,3 --batch_tag m5_base
for s in 0 1 2; do
  bash eval_shell/eval_m5.sh --runs runs/m5/m5_base/l1_lora_r8_s$s --test data/splits/m8_v1/valid.parquet \
       --decision_yaml forecast_backbones/m5/config/variants/d2_pstar_point.yaml --out runs/m5/m5_base/pstar_s$s
done
bash eval_shell/eval_m5.sh --runs runs/m5/m5_base/z0_zeroshot --test data/splits/m8_v1/valid.parquet \
     --decision_yaml forecast_backbones/m5/config/variants/d2_pstar_point.yaml --out runs/m5/m5_base/pstar_z0
```
把 `runs/m5/m5_base/variants_summary.csv` 与四个 `pstar_*/reports/summary.csv` 贴回来，和 M8 e1_final（D+1 75.01 / seen 76.32 / unseen 70.91 / nrmse 11.13 / ultra_proxy 77.72）放进同一张表。
```
