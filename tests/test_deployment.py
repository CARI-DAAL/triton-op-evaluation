import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_OVERRIDE_KEYS = {
    "BASE_IMAGE",
    "CANN_SET_ENV_PATH",
    "PIP_EXTRA_INDEX_URL",
    "PYTHONUSERBASE",
    "PYTHON_BIN",
    "TORCH_NPU_SPEC",
    "TORCH_SPEC",
    "TRITON_ASCEND_SPEC",
    "TRITON_EVAL_API_PORT",
    "TRITON_EVAL_CARDS",
    "TRITON_EVAL_REDIS_URL",
    "TRITON_SPEC",
}


def _render_compose(filename: str) -> dict:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker Compose is required to render deployment configuration")

    env = os.environ.copy()
    for key in COMPOSE_OVERRIDE_KEYS:
        env.pop(key, None)
    result = subprocess.run(
        [
            docker,
            "compose",
            "--env-file",
            "/dev/null",
            "-f",
            filename,
            "config",
            "--format",
            "json",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _create_fake_cann(tmp_path: Path) -> Path:
    toolkit = tmp_path / "cann" / "ascend-toolkit"
    (toolkit / "aarch64-linux" / "lib64").mkdir(parents=True)
    (toolkit / "lib64").mkdir()
    set_env = toolkit / "set_env.sh"
    set_env.write_text(
        "\n".join(
            [
                'export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-host-toolkit}"',
                'export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-host-home}"',
                'export ASCEND_OPP_PATH="${ASCEND_OPP_PATH:-host-opp}"',
                'export LD_LIBRARY_PATH="cann-base:${LD_LIBRARY_PATH:-}"',
                "",
            ]
        )
    )
    return set_env


def test_production_compose_exposes_the_host_runtime_and_all_cards(monkeypatch):
    monkeypatch.setenv("TRITON_EVAL_API_PORT", "99999")
    config = _render_compose("docker-compose.yml")
    service = config["services"]["triton-eval"]

    assert service["environment"] == {
        "ASCEND_DRIVER_PATH": "/usr/local/Ascend/driver",
        "CANN_SET_ENV_PATH": (
            "/home/canada_group_folder/CANN/9.0.0.0430/ascend-toolkit/set_env.sh"
        ),
        "HOME": "/tmp",
        "PYTHONUSERBASE": "/home/n84449292/.local",
        "PYTHON_BIN": "/home/n84449292/miniconda3/envs/kernelgym/bin/python3",
        "TRITON_EVAL_API_HOST": "127.0.0.1",
        "TRITON_EVAL_API_PORT": "11907",
        "TRITON_EVAL_CARDS": "8,9,10,11,12,13,14,15",
        "TRITON_EVAL_REDIS_URL": "redis://127.0.0.1:6379/7",
    }
    assert service["privileged"] is True
    assert set(service["depends_on"]) == {"redis"}

    device_sources = {device["source"] for device in service["devices"]}
    assert device_sources == {
        "/dev/davinci_manager",
        "/dev/devmm_svm",
        "/dev/hisi_hdc",
        *(f"/dev/davinci{card}" for card in range(8, 16)),
    }

    volume_targets = {volume["target"] for volume in service["volumes"]}
    assert {
        "/home/n84449292/miniconda3",
        "/home/n84449292/.local",
        "/home/canada_group_folder/CANN/9.0.0.0430",
        "/usr/local/Ascend/driver",
        "/usr/local/bin/entrypoint.sh",
    } <= volume_targets


def test_production_compose_manages_an_ephemeral_redis_queue():
    config = _render_compose("docker-compose.yml")
    redis = config["services"]["redis"]

    assert redis["image"] == "redis:7-alpine"
    assert redis["network_mode"] == "host"
    assert redis["restart"] == "unless-stopped"
    assert redis["command"] == [
        "redis-server",
        "--bind",
        "127.0.0.1",
        "--port",
        "6379",
        "--save",
        "",
        "--appendonly",
        "no",
    ]


def test_isolated_compose_targets_npu0_without_production_resources():
    config = _render_compose("docker-compose.npu0-test.yml")
    service = config["services"]["triton-eval-npu0"]

    assert service["environment"]["TRITON_EVAL_API_PORT"] == "11908"
    assert service["environment"]["TRITON_EVAL_API_HOST"] == "127.0.0.1"
    assert service["environment"]["TRITON_EVAL_CARDS"] == "0"
    assert service["environment"]["TRITON_EVAL_REDIS_URL"] == "redis://127.0.0.1:6379/8"
    assert {device["source"] for device in service["devices"]} == {
        "/dev/davinci_manager",
        "/dev/devmm_svm",
        "/dev/hisi_hdc",
        "/dev/davinci0",
    }
    assert set(config["services"]) == {"triton-eval-npu0"}


@pytest.mark.parametrize(
    ("mode", "expected_args"),
    [
        ("api", ("args=-m uvicorn run_api:app --host 127.0.0.1 --port 12000",)),
        ("workers", ("args=run_worker.py --card 3",)),
        (
            "all",
            (
                "args=-m uvicorn run_api:app --host 127.0.0.1 --port 12000",
                "args=run_worker.py --card 3",
            ),
        ),
    ],
)
def test_container_launcher_initializes_host_runtime_before_selected_python(
    tmp_path, mode, expected_args
):
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        pytest.skip("Bubblewrap is required to execute the fixed-root container entrypoint")

    set_env = _create_fake_cann(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    custom_python = fake_bin / "custom-python"
    _write_executable(
        custom_python,
        """#!/usr/bin/env bash
printf 'runner=custom\n'
printf 'toolkit=%s\n' "${ASCEND_TOOLKIT_HOME:-}"
printf 'ld=%s\n' "${LD_LIBRARY_PATH:-}"
printf 'args=%s\n' "$*"
""",
    )
    _write_executable(
        fake_bin / "python3",
        """#!/usr/bin/env bash
printf 'runner=default\n'
printf 'toolkit=%s\n' "${ASCEND_TOOLKIT_HOME:-}"
printf 'ld=%s\n' "${LD_LIBRARY_PATH:-}"
printf 'args=%s\n' "$*"
""",
    )

    opt_root = tmp_path / "opt"
    (opt_root / "triton-eval").mkdir(parents=True)
    result = subprocess.run(
        [
            bwrap,
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--bind",
            str(opt_root),
            "/opt",
            "--ro-bind",
            str(PROJECT_ROOT),
            "/opt/triton-eval",
            "--chdir",
            "/opt/triton-eval",
            "/usr/bin/env",
            "-i",
            f"PATH={fake_bin}:/usr/bin:/bin",
            f"PYTHON_BIN={custom_python}",
            f"CANN_SET_ENV_PATH={set_env}",
            "ASCEND_TOOLKIT_HOME=baked-toolkit",
            "ASCEND_HOME_PATH=baked-home",
            "ASCEND_OPP_PATH=baked-opp",
            "LD_LIBRARY_PATH=baked-ld",
            "TRITON_EVAL_API_PORT=12000",
            "TRITON_EVAL_CARDS=3",
            "/bin/bash",
            "/opt/triton-eval/docker/entrypoint.sh",
            mode,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and "bwrap:" in result.stderr:
        pytest.skip(f"Bubblewrap is unavailable in this environment: {result.stderr.strip()}")

    assert result.returncode == 0, result.stderr
    assert "runner=custom" in result.stdout
    assert "toolkit=host-toolkit" in result.stdout
    assert f"{set_env.parent}/aarch64-linux/lib64" in result.stdout
    assert result.stdout.count("runner=custom") == len(expected_args)
    for expected in expected_args:
        assert expected in result.stdout


def test_host_launcher_adds_cann_arch_library_to_api_and_worker(tmp_path):
    set_env = _create_fake_cann(tmp_path)
    fake_python = tmp_path / "fake-python"
    _write_executable(
        fake_python,
        """#!/usr/bin/env bash
printf 'ld=%s\n' "${LD_LIBRARY_PATH:-}"
printf 'args=%s\n' "$*"
""",
    )

    log_dir = tmp_path / "logs"
    env_file = tmp_path / "service.env"
    env_file.write_text(
        "\n".join(
            [
                f"PYTHON_BIN={fake_python}",
                f"CANN_SET_ENV_PATH={set_env}",
                "TRITON_EVAL_API_HOST=127.0.0.1",
                "TRITON_EVAL_API_PORT=11907",
                "TRITON_EVAL_CARDS=8",
                f"LOG_DIR={log_dir}",
                "",
            ]
        )
    )
    env = os.environ.copy()
    env["TRITON_EVAL_ENV_FILE"] = str(env_file)
    env["LD_LIBRARY_PATH"] = "baseline-ld"

    result = subprocess.run(
        ["bash", "scripts/start.sh"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    arch_library = f"{set_env.parent}/aarch64-linux/lib64"
    assert arch_library in (log_dir / "api.log").read_text()
    assert arch_library in (log_dir / "worker_8.log").read_text()
