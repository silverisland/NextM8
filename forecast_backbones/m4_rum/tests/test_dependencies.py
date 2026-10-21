"""Rum must work without any other forecast model installed."""

import ast
import shutil
import subprocess
import sys
from pathlib import Path

from forecast_backbones.m4_rum.config import load_config

ROOT = Path(__file__).resolve().parents[3]
RUM = ROOT / "forecast_backbones" / "m4_rum"
PUBLIC = ROOT / "pipelines" / "forecasting"


def test_no_private_model_dependencies():
    for directory in (RUM, PUBLIC):
        for path in directory.rglob("*.py"):
            is_test = "tests" in path.relative_to(directory).parts
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                    # Relative imports cannot escape Rum into a sibling model.
                    assert node.level <= 1, f"{path}:{node.lineno}"
                else:
                    continue
                forbidden = [
                    name
                    for name in names
                    if name.startswith("forecast_backbones")
                    and not (is_test and name.startswith("forecast_backbones.m4_rum"))
                ]
                assert not forbidden, (
                    f"{path}:{node.lineno} imports a model implementation: {names}"
                )


def test_default_assets_do_not_require_another_model():
    for config in (None, RUM / "config" / "configs.yaml"):
        path = Path(load_config(config)["dataset"]["station_meta"])
        assert path.parts[:2] == ("pipelines", "forecasting")
        assert (ROOT / path).is_file()


def test_train_and_full_evaluation_without_m8_directory(tmp_path):
    """Deploy Rum alone and reject even optional sibling-model imports."""
    deployment = tmp_path / "deployment"
    ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache", ".ruff_cache")
    for name in ("pipelines", "interfaces"):
        shutil.copytree(ROOT / name, deployment / name, ignore=ignore)
    shutil.copytree(RUM, deployment / "forecast_backbones" / "m4_rum", ignore=ignore)
    assert not (deployment / "forecast_backbones" / "m8").exists()
    code = """
import importlib.abc
import sys

class RejectSiblingModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("forecast_backbones."):
            if fullname.split(".")[1] != "m4_rum":
                raise AssertionError("Private model import attempted: " + fullname)
        return None

sys.meta_path.insert(0, RejectSiblingModels())
from forecast_backbones.m4_rum.transformer_models import TransformersAdaptedM4RumModel
from forecast_backbones.m4_rum import train, evaluate
model = TransformersAdaptedM4RumModel()
assert model.n_params() == 310529
import pytest
result = pytest.main([
    "forecast_backbones/m4_rum/tests/test_workflow.py",
    "forecast_backbones/m4_rum/tests/test_pinball.py", "-q", "--disable-warnings",
])
assert not any(n.startswith("forecast_backbones.m8") for n in sys.modules)
raise SystemExit(result)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=deployment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-3000:]
