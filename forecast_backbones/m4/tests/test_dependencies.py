"""Deploy M4 alone and reject even optional sibling-model imports."""

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_no_sibling_imports():
    directory = ROOT / "forecast_backbones/m4"
    for path in directory.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            assert not any(
                n.startswith("forecast_backbones.")
                and not n.startswith("forecast_backbones.m4.")
                for n in names
            ), path


def test_standalone_train_reload_infer_and_full_report(tmp_path):
    deployment = tmp_path / "deployment"
    ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache", ".ruff_cache")
    for name in ("pipelines", "interfaces", "forecast_backbones/m4"):
        shutil.copytree(ROOT / name, deployment / name, ignore=ignore)
    code = """
import importlib.abc
import sys
class RejectSiblingModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('forecast_backbones.') and fullname.split('.')[1] != 'm4':
            raise AssertionError('Sibling model import attempted: ' + fullname)
sys.meta_path.insert(0, RejectSiblingModels())
from forecast_backbones.m4.transformer_models import TransformersAdaptedM4Model
assert TransformersAdaptedM4Model().n_params() > 0
import pytest
raise SystemExit(pytest.main(['forecast_backbones/m4/tests/test_workflow.py', '-q', '--disable-warnings']))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=deployment,
        env=dict(os.environ, PYTHONPATH=str(deployment)),
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-3000:]
