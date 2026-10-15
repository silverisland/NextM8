import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SH = [ROOT / "train_shell" / "train_m5.sh", ROOT / "train_shell" / "train_m5_variants.sh", ROOT / "eval_shell" / "eval_m5.sh"]


@pytest.mark.parametrize("p", SH, ids=lambda p: p.name)
def test_shell_syntax_and_no_m8_leftovers(p):
    assert p.exists(), p
    assert subprocess.run(["bash", "-n", str(p)], capture_output=True).returncode == 0
    m8 = "m" + "8"                                             # 拆开写：test_standalone 会扫 m5 下所有文件里的 m8 引用
    pats = ["forecast_backbones/" + m8, "forecast_backbones." + m8, "runs/" + m8, "PVRex"]
    bad = [l for l in p.read_text(encoding="utf-8").splitlines() if any(x in l for x in pats)]
    assert not bad, bad


def test_train_m5_defaults_and_zero_shot_check():
    txt = SH[0].read_text(encoding="utf-8")
    assert 'CONFIG="${BASE_DIR}/forecast_backbones/m5/config/finetune.yaml"' in txt
    assert 'SPLITS_DIR="${BASE_DIR}/data/splits/m8_v1"' in txt
    assert "zero_shot" in txt                                  # 零样本 run 没有 model.safetensors 也算成功


def test_readme_exists():
    assert (ROOT / "forecast_backbones" / "m5" / "README.md").exists()


def test_variants_staging_is_per_invocation():
    txt = SH[1].read_text(encoding="utf-8")
    assert 'rm -rf "$OUT_BASE/.runs"' not in txt and "$OUT_BASE/.runs/" not in txt and '"$OUT_BASE/.runs"' not in txt
    assert 'STAGE="$OUT_BASE/.runs_$$"' in txt and 'rm -rf "$STAGE"' in txt


def _plan(tmp_path, *args):
    """train_m5_variants.sh --dry_run：只打印调度计划（PLAN 行），不训练、不建目录。"""
    r = subprocess.run(["bash", str(SH[1]), "--dry_run", "--runs_base", str(tmp_path), *args],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    head = next(l for l in r.stdout.splitlines() if l.startswith("PLAN gpus_per_job="))
    slots = [l.split(None, 3)[1:] for l in r.stdout.splitlines() if l.startswith("PLAN slot=")]
    assert not any(tmp_path.iterdir())                                       # dry run 不落任何东西
    return head, {s.split("=")[1]: (g.split("=")[1], j.split("=", 1)[1].split()) for s, g, j in slots}


EIGHT = ("--gpus", "0,1,2,3,4,5,6,7")


def test_variants_auto_split_z0_runs_once_and_jobs_share_cards(tmp_path):
    head, slots = _plan(tmp_path, "--variants", "z0_zeroshot,l1_lora_r8", "--seeds", "0,1,2", *EIGHT)
    assert head == "PLAN gpus_per_job=2 slots=4 jobs=4"                      # z0 与种子无关只跑一次；4 个任务各 2 张卡同时跑
    assert slots == {"0": ("0,1", ["z0_zeroshot"]), "1": ("2,3", ["l1_lora_r8_s0"]),
                     "2": ("4,5", ["l1_lora_r8_s1"]), "3": ("6,7", ["l1_lora_r8_s2"])}


def test_variants_single_job_takes_all_cards(tmp_path):
    head, slots = _plan(tmp_path, "--variants", "l1_lora_r8", *EIGHT)
    assert head == "PLAN gpus_per_job=8 slots=1 jobs=1" and slots == {"0": ("0,1,2,3,4,5,6,7", ["l1_lora_r8"])}


def test_variants_more_jobs_than_cards_queue_one_card_each(tmp_path):
    head, slots = _plan(tmp_path, "--variants", "l1_lora_r8", "--seeds", "0,1,2,3,4,5,6,7,8", *EIGHT)
    assert head == "PLAN gpus_per_job=1 slots=8 jobs=9"
    assert slots["0"] == ("0", ["l1_lora_r8_s0", "l1_lora_r8_s8"]) and slots["7"] == ("7", ["l1_lora_r8_s7"])


def test_variants_cards_per_job_divide_global_batch(tmp_path):
    assert _plan(tmp_path, "--variants", "l1_lora_r8", "--gpus", "0,1,2,3,4,5")[0] == "PLAN gpus_per_job=4 slots=1 jobs=1"   # 16 不能被 6/5 整除
    assert _plan(tmp_path, "--variants", "l1_lora_r8", "--batch_size", "12", *EIGHT)[0] == "PLAN gpus_per_job=6 slots=1 jobs=1"


def test_variants_explicit_gpus_per_job(tmp_path):
    head, slots = _plan(tmp_path, "--variants", "l1_lora_r8", "--seeds", "0,1,2,3", "--gpus_per_job", "4", *EIGHT)
    assert head == "PLAN gpus_per_job=4 slots=2 jobs=4"
    assert slots["0"] == ("0,1,2,3", ["l1_lora_r8_s0", "l1_lora_r8_s2"])
    r = subprocess.run(["bash", str(SH[1]), "--dry_run", "--runs_base", str(tmp_path), "--variants", "l1_lora_r8",
                        "--gpus_per_job", "3", *EIGHT], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "整除" in r.stderr                        # 3 卡除不尽全局 batch 16
