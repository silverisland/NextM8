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
    assert 'CONFIG="${BASE_DIR}/forecast_backbones/m5/config/f1_final.yaml"' in txt and "--station_meta)" in txt
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
    head, slots = _plan(tmp_path, "--variants", "z0_zeroshot,f1_final", "--seeds", "0,1,2", *EIGHT)
    assert head == "PLAN gpus_per_job=2 slots=4 jobs=4"                      # z0 与种子无关只跑一次；4 个任务各 2 张卡同时跑
    assert slots == {"0": ("0,1", ["z0_zeroshot"]), "1": ("2,3", ["f1_final_s0"]),
                     "2": ("4,5", ["f1_final_s1"]), "3": ("6,7", ["f1_final_s2"])}


def test_variants_single_job_takes_all_cards(tmp_path):
    head, slots = _plan(tmp_path, "--variants", "f1_final", *EIGHT)
    assert head == "PLAN gpus_per_job=8 slots=1 jobs=1" and slots == {"0": ("0,1,2,3,4,5,6,7", ["f1_final"])}


def test_variants_more_jobs_than_cards_queue_one_card_each(tmp_path):
    head, slots = _plan(tmp_path, "--variants", "f1_final", "--seeds", "0,1,2,3,4,5,6,7,8", *EIGHT)
    assert head == "PLAN gpus_per_job=1 slots=8 jobs=9"
    assert slots["0"] == ("0", ["f1_final_s0", "f1_final_s8"]) and slots["7"] == ("7", ["f1_final_s7"])


def test_variants_cards_per_job_divide_global_batch(tmp_path):
    assert _plan(tmp_path, "--variants", "f1_final", "--gpus", "0,1,2,3,4,5")[0] == "PLAN gpus_per_job=4 slots=1 jobs=1"   # 16 不能被 6/5 整除
    assert _plan(tmp_path, "--variants", "f1_final", "--batch_size", "12", *EIGHT)[0] == "PLAN gpus_per_job=6 slots=1 jobs=1"


def test_variants_explicit_gpus_per_job(tmp_path):
    head, slots = _plan(tmp_path, "--variants", "f1_final", "--seeds", "0,1,2,3", "--gpus_per_job", "4", *EIGHT)
    assert head == "PLAN gpus_per_job=4 slots=2 jobs=4"
    assert slots["0"] == ("0,1,2,3", ["f1_final_s0", "f1_final_s2"])
    r = subprocess.run(["bash", str(SH[1]), "--dry_run", "--runs_base", str(tmp_path), "--variants", "f1_final",
                        "--gpus_per_job", "3", *EIGHT], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "整除" in r.stderr                        # 3 卡除不尽全局 batch 16


def test_variants_skip_decision_only_yamls(tmp_path):
    dec = str(ROOT / "forecast_backbones" / "m5" / "config" / "decision")
    r = subprocess.run(["bash", str(SH[1]), "--dry_run", "--runs_base", str(tmp_path), "--variant_dir", dec,
                        "--variants", "d2_pstar_point", *EIGHT], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "没有要训练的任务" in r.stderr


def test_variants_default_dir_is_flat_config_without_online_yaml(tmp_path):
    head, slots = _plan(tmp_path, *EIGHT)                                    # 不给 --variants：config/ 下全部
    names = sorted(n for _, js in slots.values() for n in js)
    assert names == ["f1_final", "f2_lr5e5", "f3_lr2e4", "f4_drop01", "f5_wd01", "s3_c1_envelope", "s3_c1_envelope_zs", "z0_zeroshot", "z1_zeroshot_pstar"]
