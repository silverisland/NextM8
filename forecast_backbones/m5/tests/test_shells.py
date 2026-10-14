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
