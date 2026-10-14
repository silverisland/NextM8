"""M8 自包含（2026-10-13）：forecast_backbones/m8 不再引用 forecast_backbones/m6 的任何代码或数据文件；
所有模型共享的 pipelines / interfaces 仍然引用。"""
import re
import subprocess
import sys
from pathlib import Path

M8 = Path(__file__).resolve().parents[1]
ROOT = M8.parents[1]
_M6_REF = re.compile(r"(from\s+\.\.m6|forecast_backbones\.m6|forecast_backbones/m6|from\s+m6\b|import\s+m6\b)")


def test_no_m6_reference_in_m8_sources():
    hits = []
    for p in sorted(M8.rglob("*.py")) + sorted(M8.rglob("*.yaml")) + sorted(M8.rglob("*.sh")):
        if "__pycache__" in p.parts:
            continue
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if _M6_REF.search(line) and not p.samefile(Path(__file__)):
                hits.append(f"{p.relative_to(ROOT)}:{n}: {line.strip()[:100]}")
    assert not hits, "m8 里还引用 m6：\n" + "\n".join(hits)


def test_importing_m8_does_not_load_m6():
    code = ("import sys; sys.path.insert(0, %r); "
            "import forecast_backbones.m8.train, forecast_backbones.m8.evaluate, forecast_backbones.m8.station_report, "
            "forecast_backbones.m8.diagnose_test, forecast_backbones.m8.features, forecast_backbones.m8.transformer_models; "
            "bad = sorted(m for m in sys.modules if m.startswith('forecast_backbones.m6')); "
            "print('M6_LOADED=' + ','.join(bad))" % str(ROOT))
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "M6_LOADED=\n" in r.stdout or r.stdout.rstrip().endswith("M6_LOADED="), r.stdout


def test_default_station_meta_lives_in_m8():
    from forecast_backbones.m8.config import DEFAULT_CFG
    p = Path(DEFAULT_CFG["dataset"]["station_meta"])
    assert p.parts[:2] == ("forecast_backbones", "m8") and (ROOT / p).exists()
    sh = ROOT / "train_shell" / "train_m8.sh"
    if sh.exists():
        assert "forecast_backbones/m8/station_meta.csv" in sh.read_text(encoding="utf-8")


_TS_REF = re.compile(r"(from\s+\.\.\.pipelines\.time_splits|from\s+pipelines\.time_splits|import\s+pipelines\.time_splits)")


def test_no_time_splits_import_in_m8_sources():
    """全量训练（2026-10-14）：给一个训练 parquet 文件夹直接训练，m8 不再 import pipelines.time_splits（拼表 / 站号归一自带）。"""
    hits = []
    for p in sorted(M8.glob("*.py")):
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if _TS_REF.search(line):
                hits.append(f"{p.relative_to(ROOT)}:{n}: {line.strip()[:100]}")
    assert not hits, "m8 还在 import pipelines.time_splits：\n" + "\n".join(hits)


def test_training_modules_load_without_time_splits():
    code = ("import sys; sys.path.insert(0, %r); sys.modules['pipelines.time_splits'] = None; "
            "import forecast_backbones.m8.train, forecast_backbones.m8.data, forecast_backbones.m8.station_report, forecast_backbones.m8.evaluate; "
            "print('OK')" % str(ROOT))
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-2000:]
