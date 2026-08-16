"""Durable deployment controls: pinned Redis, gateway bind, fail-closed wrapper."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REDIS_IMAGE = (
    "docker.io/library/redis:7.4.10-alpine3.21"
    "@sha256:e7723ff73d963f5cc6d9c4643ea3d989527a402a319239054e9472a7fb9219a2"
)
EVAL_IMAGE = "registry.example.com/triton-eval@sha256:" + "a" * 64
GATEWAY = "172.18.0.1"

COMPOSE_RENDER_ENV = {
    **{key: "" for key in (
        "BASE_IMAGE", "CANN_SET_ENV_PATH", "PIP_EXTRA_INDEX_URL", "PYTHONUSERBASE",
        "PYTHON_BIN", "TORCH_NPU_SPEC", "TORCH_SPEC", "TRITON_ASCEND_SPEC",
        "TRITON_EVAL_API_PORT", "TRITON_EVAL_CARDS", "TRITON_EVAL_REDIS_URL",
        "TRITON_SPEC",
    )},
    "TRITON_EVAL_API_HOST": GATEWAY,
    "TRITON_EVAL_API_PORT": "11907",
    "TRITON_EVAL_IMAGE": EVAL_IMAGE,
    "TRITON_EVAL_EVALUATOR_BUILD": EVAL_IMAGE,
}


def _render_compose(filename: str, extra_env=None) -> dict:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker Compose is required to render deployment configuration")
    env = os.environ.copy()
    env.pop("TRITON_EVAL_API_HOST", None)
    env.update(COMPOSE_RENDER_ENV)
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        [docker, "compose", "--env-file", "/dev/null", "-f", filename,
         "config", "--format", "json"],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True,
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
        "\n".join([
            'export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-host-toolkit}"',
            'export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-host-home}"',
            'export ASCEND_OPP_PATH="${ASCEND_OPP_PATH:-host-opp}"',
            'export LD_LIBRARY_PATH="cann-base:${LD_LIBRARY_PATH:-}"',
            "",
        ])
    )
    return set_env


# ---------------------------------------------------------------------------
# Compose rendering: durable Redis, gateway bind, immutable images
# ---------------------------------------------------------------------------


def test_production_compose_pins_the_reviewed_multiplatform_redis(tmp_path):
    config = _render_compose("docker-compose.yml")
    redis = config["services"]["redis"]
    assert redis["image"] == REDIS_IMAGE
    assert redis["network_mode"] == "host"
    assert redis["restart"] == "unless-stopped"
    assert redis["command"] == [
        "redis-server", "--bind", "127.0.0.1", "--port", "6379", "--save", "",
        "--appendonly", "yes", "--appendfsync", "always",
        "--aof-use-rdb-preamble", "yes",
    ]
    volumes = redis["volumes"]
    assert any(
        v.get("source") == "triton-eval-redis-v2" and v.get("target") == "/data"
        for v in volumes
    )
    assert "triton-eval-redis-v2" in config.get("volumes", {})


def test_production_compose_binds_the_api_to_the_gateway(tmp_path):
    config = _render_compose("docker-compose.yml")
    service = config["services"]["triton-eval"]
    assert service["environment"]["TRITON_EVAL_API_HOST"] == GATEWAY
    assert service["environment"]["TRITON_EVAL_API_PORT"] == "11907"
    assert service["network_mode"] == "host"
    assert service["image"] == EVAL_IMAGE
    assert service["environment"]["TRITON_EVAL_EVALUATOR_BUILD"] == EVAL_IMAGE


def test_production_compose_requires_an_image_input_without_default(tmp_path):
    """No mutable default: rendering without TRITON_EVAL_IMAGE fails closed."""
    import subprocess

    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker Compose is required to render deployment configuration")
    env = os.environ.copy()
    env.pop("TRITON_EVAL_API_HOST", None)
    env.update(COMPOSE_RENDER_ENV)
    env["TRITON_EVAL_IMAGE"] = ""
    result = subprocess.run(
        [docker, "compose", "--env-file", "/dev/null", "-f", "docker-compose.yml",
         "config", "--format", "json"],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "TRITON_EVAL_IMAGE" in result.stderr


def test_production_compose_mounts_secret_and_catalog_read_only(tmp_path):
    config = _render_compose("docker-compose.yml")
    service = config["services"]["triton-eval"]
    targets = {v["target"]: v for v in service["volumes"]}
    for target in ("/run/secrets/triton-eval", "/srv/triton-eval-release"):
        assert target in targets, target
        assert targets[target].get("read_only") is True
    for volume in service["volumes"]:
        assert "docker.sock" not in str(volume.get("source", ""))


def test_npu0_test_compose_keeps_gateway_and_durable_redis(tmp_path):
    config = _render_compose("docker-compose.npu0-test.yml")
    assert set(config["services"]) == {"triton-eval-npu0", "redis"}


# ---------------------------------------------------------------------------
# compose.sh wrapper semantics
# ---------------------------------------------------------------------------


def _fake_tools(tmp_path, preflight_exit=0):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    log = tmp_path / "calls.log"
    fake_python = bin_dir / "fake-python"
    _write_executable(
        fake_python,
        f"""#!/usr/bin/env bash
printf '%s\\n' "python $*" >> {log}
case "$*" in
  *deployment_preflight*)
    printf '{{"gateway": "%s"}}\\n' "${{TRITON_EVAL_PREFLIGHT_GATEWAY:-172.18.0.1}}"
    exit {preflight_exit}
    ;;
esac
exit 0
""",
    )
    fake_docker = bin_dir / "docker"
    _write_executable(
        fake_docker,
        f"""#!/usr/bin/env bash
printf '%s\\n' "docker $*" >> {log}
exit 0
""",
    )
    return bin_dir, fake_python, log


@pytest.mark.parametrize("action", ["up", "start", "restart"])
def test_startup_actions_run_preflight_exactly_once_first(tmp_path, action):
    bin_dir, fake_python, log = _fake_tools(tmp_path, preflight_exit=0)
    env = os.environ.copy() | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "TRITON_EVAL_PYTHON": str(fake_python),
        "TRITON_EVAL_PREFLIGHT_GATEWAY": GATEWAY,
    }
    result = subprocess.run(
        ["bash", "scripts/compose.sh", action, "-d"],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    preflight_calls = [c for c in calls if "deployment_preflight" in c]
    compose_calls = [c for c in calls if c.startswith("docker compose")]
    assert len(preflight_calls) == 1
    assert len(compose_calls) == 1
    assert calls[0] == preflight_calls[0]  # preflight strictly before compose
    # The validated gateway is exported for the composed service.
    assert result.stdout.strip().endswith(GATEWAY) or GATEWAY in result.stdout


@pytest.mark.parametrize("action", ["up", "start", "restart"])
def test_startup_actions_refuse_compose_when_preflight_fails(tmp_path, action):
    bin_dir, fake_python, log = _fake_tools(tmp_path, preflight_exit=1)
    env = os.environ.copy() | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "TRITON_EVAL_PYTHON": str(fake_python),
    }
    result = subprocess.run(
        ["bash", "scripts/compose.sh", action],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True,
    )
    assert result.returncode != 0
    calls = log.read_text().splitlines()
    assert not [c for c in calls if c.startswith("docker compose")]


@pytest.mark.parametrize("action", ["down", "logs", "ps", "config"])
def test_control_actions_skip_startup_validation(tmp_path, action):
    bin_dir, fake_python, log = _fake_tools(tmp_path, preflight_exit=1)
    env = os.environ.copy() | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "TRITON_EVAL_PYTHON": str(fake_python),
    }
    result = subprocess.run(
        ["bash", "scripts/compose.sh", action],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    assert not [c for c in calls if "deployment_preflight" in c]
    assert [c for c in calls if c.startswith("docker compose")]


# ---------------------------------------------------------------------------
# Launcher behavior (kept from the pre-durable baseline)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "expected_args"),
    [
        ("api", ("args=-m uvicorn run_api:app --host 127.0.0.1 --port 12000",)),
        ("workers", ("args=run_worker.py --card 3",)),
        ("all", (
            "args=-m uvicorn run_api:app --host 127.0.0.1 --port 12000",
            "args=run_worker.py --card 3",
        )),
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
            bwrap, "--ro-bind", "/", "/", "--dev", "/dev", "--bind",
            str(opt_root), "/opt", "--ro-bind", str(PROJECT_ROOT),
            "/opt/triton-eval", "--chdir", "/opt/triton-eval", "/usr/bin/env",
            "-i", f"PATH={fake_bin}:/usr/bin:/bin",
            f"PYTHON_BIN={custom_python}", f"CANN_SET_ENV_PATH={set_env}",
            "ASCEND_TOOLKIT_HOME=baked-toolkit", "ASCEND_HOME_PATH=baked-home",
            "ASCEND_OPP_PATH=baked-opp", "LD_LIBRARY_PATH=baked-ld",
            "TRITON_EVAL_API_PORT=12000", "TRITON_EVAL_CARDS=3",
            "/bin/bash", "/opt/triton-eval/docker/entrypoint.sh", mode,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0 and "bwrap:" in result.stderr:
        pytest.skip(f"Bubblewrap is unavailable: {result.stderr.strip()}")

    assert result.returncode == 0, result.stderr
    assert "runner=custom" in result.stdout
    assert "toolkit=host-toolkit" in result.stdout
    assert f"{set_env.parent}/aarch64-linux/lib64" in result.stdout
    # One extra interpreter call: the in-container readiness gate (heredoc `-`).
    assert result.stdout.count("runner=custom") == len(expected_args) + 1
    assert "args=-\n" in result.stdout
    for expected in expected_args:
        assert expected in result.stdout


def test_host_launcher_adds_cann_arch_library_to_api_and_worker(tmp_path):
    set_env = _create_fake_cann(tmp_path)
    fake_python = tmp_path / "fake-python"
    _write_executable(
        fake_python,
        """#!/usr/bin/env bash
printf 'ld=%s\\n' "${LD_LIBRARY_PATH:-}"
printf 'args=%s\\n' "$*"
""",
    )
    log_dir = tmp_path / "logs"
    env_file = tmp_path / "service.env"
    env_file.write_text("\n".join([
        f"PYTHON_BIN={fake_python}",
        f"CANN_SET_ENV_PATH={set_env}",
        "TRITON_EVAL_API_HOST=127.0.0.1",
        "TRITON_EVAL_API_PORT=11907",
        "TRITON_EVAL_CARDS=8",
        f"LOG_DIR={log_dir}",
        "",
    ]))
    env = os.environ.copy()
    env["TRITON_EVAL_ENV_FILE"] = str(env_file)
    env["LD_LIBRARY_PATH"] = "baseline-ld"

    result = subprocess.run(
        ["bash", "scripts/start.sh"], cwd=PROJECT_ROOT, env=env,
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    arch_library = f"{set_env.parent}/aarch64-linux/lib64"
    assert arch_library in (log_dir / "api.log").read_text()
    assert arch_library in (log_dir / "worker_8.log").read_text()
