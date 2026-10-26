"""Training and evaluation must never silently switch Python environments."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(
    params=[
        "train_shell/train_m4_rum.sh",
        "eval_shell/eval_m4_rum.sh",
        "train_shell/compare_m4_rum.sh",
        "eval_shell/infer_m4_rum.sh",
    ]
)
def launcher(request, tmp_path):
    script = tmp_path / "checkout" / request.param
    script.parent.mkdir(parents=True)
    # Simulate a missing default even on machines where the server venv exists.
    script.write_text(
        (ROOT / request.param)
        .read_text()
        .replace(
            "/home/ma-user/ts3/bin/python", str(tmp_path / "missing-default-python")
        )
    )
    record = tmp_path / "executed.json"
    fake = tmp_path / "python"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if sys.argv[1] == '-c':\n"
        "    print(1)\n"
        "else:\n"
        "    with open(os.environ['PYTHON_SELECTION_RECORD'], 'w') as f:\n"
        "        json.dump(sys.argv[1:], f)\n"
        "    sys.exit(7)\n"
    )
    fake.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "PYTHON_SELECTION_RECORD": str(record),
        "OMP_NUM_THREADS": "1",
        "HF_DATASETS_CACHE": str(tmp_path / "hf_cache"),
    }
    env.pop("PYTHON", None)
    return script, fake, record, env


@pytest.mark.parametrize("selection", ["default", "environment", "cli"])
def test_missing_interpreter_never_runs_path_python(launcher, selection):
    script, _, record, env = launcher
    options = []
    if selection == "environment":
        env["PYTHON"] = "/missing/environment/python"
    elif selection == "cli":
        env["PYTHON"] = "python"  # Even a valid alternative must not be used.
        options = ["--venv-python", "/missing/cli/python"]
    result = subprocess.run(
        ["bash", str(script), *options],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Python 解释器不可执行" in result.stderr
    assert not record.exists()


@pytest.mark.parametrize("selection", ["environment", "cli"])
def test_explicit_interpreter_precedence_arguments_and_exit_status(launcher, selection):
    script, fake, record, env = launcher
    if selection == "environment":
        env["PYTHON"] = str(fake)
        options = []
    else:
        env["PYTHON"] = "/missing/environment/python"
        options = ["--venv-python", str(fake)]
    result = subprocess.run(
        ["bash", str(script), *options, "--out", "/run with spaces"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 7, result.stderr
    mode = {
        "train_m4_rum.sh": "train",
        "eval_m4_rum.sh": "evaluate",
        "compare_m4_rum.sh": "production",
        "infer_m4_rum.sh": "infer",
    }[script.name]
    assert json.loads(record.read_text()) == [
        "-m",
        "forecast_backbones.m4_rum." + mode,
        "--out",
        "/run with spaces",
    ]
