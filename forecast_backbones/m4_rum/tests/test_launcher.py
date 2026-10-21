"""Exercise the shell launcher without requiring CUDA or starting training."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "train_shell/train_m4_rum.sh"
CUDA_LIBS = "/usr/local/cuda-12.1/lib64:/usr/lib/x86_64-linux-gnu"


@pytest.fixture
def launch(tmp_path):
    python = tmp_path / "fake python"
    record = tmp_path / "launch.json"
    probe = tmp_path / "probe.json"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if sys.argv[1] == '-c':\n"
        "    with open(os.environ['PROBE_RECORD'], 'w') as f:\n"
        "        json.dump(os.environ.get('LD_LIBRARY_PATH'), f)\n"
        "    print(os.environ['FAKE_GPU_COUNT'])\n"
        "else:\n"
        "    with open(os.environ['LAUNCH_RECORD'], 'w') as f:\n"
        "        json.dump({'args': sys.argv[1:], "
        "'visible': os.environ.get('CUDA_VISIBLE_DEVICES'), "
        "'libraries': os.environ.get('LD_LIBRARY_PATH'), "
        "'probe_libraries': json.load(open(os.environ['PROBE_RECORD'])), "
        "'threads': os.environ['OMP_NUM_THREADS']}, f)\n"
        "    sys.exit(int(os.environ.get('TRAIN_EXIT', '0')))\n"
    )
    python.chmod(0o755)
    nproc = tmp_path / "nproc"
    nproc.write_text("#!/bin/sh\necho 16\n")
    nproc.chmod(0o755)

    def run(*options, count=2, exit_code=0, env_overrides=None, cli_python=True):
        env = dict(
            os.environ,
            PATH=f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            PYTHON=str(python),
            CUDA_VISIBLE_DEVICES="4,5",
            FAKE_GPU_COUNT=str(count),
            LAUNCH_RECORD=str(record),
            PROBE_RECORD=str(probe),
            TRAIN_EXIT=str(exit_code),
        )
        for key in ("PRE_CUDA_LD", "LD_LIBRARY_PATH", "OMP_NUM_THREADS"):
            env.pop(key, None)
        for key, value in (env_overrides or {}).items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        result = subprocess.run(
            [
                "bash",
                str(SCRIPT),
                *(["--venv-python", str(python)] if cli_python else []),
                *options,
                "--splits_dir",
                "/data/path with spaces",
                "--config",
                "recipe.yaml",
                "--max_steps",
                "20",
            ],
            env=env,
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        return result, json.loads(record.read_text()) if record.exists() else None

    return run


@pytest.mark.parametrize("gpus,visible", [("0,2", "0,2"), ("auto", "4,5")])
def test_multigpu_launch_and_argument_forwarding(launch, gpus, visible):
    result, recorded = launch("--gpus", gpus, "--port", "29709", "--cpu_threads", "3")
    assert result.returncode == 0, result.stderr
    assert recorded["visible"] == visible
    assert recorded["threads"] == "3"
    args = recorded["args"]
    assert args[:2] == ["-m", "torch.distributed.run"]
    assert "--nproc_per_node=2" in args
    assert "--master_port=29709" in args
    assert args[-8:] == [
        "-m",
        "forecast_backbones.m4_rum.train",
        "--splits-dir",
        "/data/path with spaces",
        "--config",
        "recipe.yaml",
        "--max-steps",
        "20",
    ]


@pytest.mark.parametrize("options,count", [(["--gpu", "0"], 1), (["--allow-cpu"], 0)])
def test_single_process_and_exit_status(launch, options, count):
    result, recorded = launch(*options, count=count, exit_code=7)
    assert result.returncode == 7
    assert recorded["args"][:2] == ["-m", "forecast_backbones.m4_rum.train"]


@pytest.mark.parametrize(
    "options,count,message",
    [
        (["--gpus", "0,0"], 2, "重复编号"),
        (["--gpus", "0,1"], 1, "仅检测到 1 张"),
        (["--gpus", "0,"], 2, "--gpus 必须"),
        ([], 0, "未检测到 GPU"),
        (["--port", "65536"], 2, "--port 必须"),
        (["--cpu_threads", "0"], 2, "--cpu_threads 必须"),
    ],
)
def test_invalid_launch_does_not_start_training(launch, options, count, message):
    result, recorded = launch(*options, count=count)
    assert result.returncode != 0
    assert message in result.stderr
    assert recorded is None


@pytest.mark.parametrize(
    "prefix,existing,expected",
    [
        (None, None, CUDA_LIBS),
        (None, "", CUDA_LIBS),
        (None, "/existing", f"{CUDA_LIBS}:/existing"),
        ("/custom/cuda", "/existing", "/custom/cuda:/existing"),
        ("", "/existing", "/existing"),
        ("", None, None),
    ],
)
def test_cuda_libraries_ready_for_probe_and_training(
    launch, prefix, existing, expected
):
    result, recorded = launch(
        env_overrides={"PRE_CUDA_LD": prefix, "LD_LIBRARY_PATH": existing}
    )
    assert result.returncode == 0, result.stderr
    assert recorded["libraries"] == recorded["probe_libraries"] == expected


@pytest.mark.parametrize(
    "options,omp,count,expected",
    [
        ([], None, 2, "8"),
        ([], None, 32, "1"),
        (["--allow-cpu"], None, 0, "16"),
        ([], "5", 2, "5"),
        (["--cpu_threads", "3"], "5", 2, "3"),
    ],
)
def test_thread_allocation_and_overrides(launch, options, omp, count, expected):
    result, recorded = launch(
        *options, count=count, env_overrides={"OMP_NUM_THREADS": omp}
    )
    assert result.returncode == 0, result.stderr
    assert recorded["threads"] == expected


@pytest.mark.parametrize("cli_python", [False, True])
def test_python_environment_and_cli_override(launch, cli_python):
    # Without a CLI override use $PYTHON; with one it must override even a bad $PYTHON.
    result, recorded = launch(
        cli_python=cli_python,
        env_overrides={"PYTHON": "/missing/python"} if cli_python else None,
    )
    assert result.returncode == 0, result.stderr
    assert recorded is not None
