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
