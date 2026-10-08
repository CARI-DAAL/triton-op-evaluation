"""Durable deployment controls: pinned Redis, private instance bind, fail-closed wrapper."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REDIS_IMAGE = (
    "docker.io/library/redis:7.4.10-alpine3.21"
    "@sha256:e7723ff73d963f5cc6d9c4643ea3d989527a402a319239054e9472a7fb9219a2"
)
EVAL_IMAGE = "registry.example.com/triton-eval@sha256:" + "a" * 64
EVALUATOR_BUILD = EVAL_IMAGE.rsplit("@", 1)[1]
INSTANCE_HOST = "172.18.0.1"

COMPOSE_RENDER_ENV = {
    **{key: "" for key in (
        "BASE_IMAGE", "CANN_SET_ENV_PATH", "PIP_EXTRA_INDEX_URL", "PYTHONUSERBASE",
        "PYTHON_BIN", "TORCH_NPU_SPEC", "TORCH_SPEC", "TRITON_ASCEND_SPEC",
        "TRITON_EVAL_API_PORT", "TRITON_EVAL_CARDS", "TRITON_EVAL_REDIS_URL",
        "TRITON_SPEC",
    )},
    "TRITON_EVAL_API_HOST": INSTANCE_HOST,
    "TRITON_EVAL_API_PORT": "11907",
    "TRITON_EVAL_IMAGE": EVAL_IMAGE,
    "TRITON_EVAL_EVALUATOR_BUILD": EVALUATOR_BUILD,
}


def _render_compose(filename: str, extra_env=None) -> dict:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker Compose is required to render deployment configuration")
    env = {key: value for key, value in os.environ.items() if not key.startswith("TRITON_EVAL_")}
    env.update(COMPOSE_RENDER_ENV)
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        [docker, "compose", "--project-directory", str(PROJECT_ROOT),
         "--env-file", "/dev/null", "-f", filename,
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


def _complete_deployment_env(tmp_path):
    secret = tmp_path / "capabilities.json"
    secret.write_text(json.dumps({
        "capabilities": [{
            "token": "deployment-test-capability", "caller_id": "trial-1",
            "caller_class": "trial", "scopes": ["trial:submit", "trial:read"],
        }],
    }))
    secret.chmod(0o600)
    return COMPOSE_RENDER_ENV | {
        "TRITON_EVAL_REDIS_URL": "redis://127.0.0.1:6379/7",
        "TRITON_EVAL_CARDS": "0",
        "TRITON_EVAL_CAPABILITY_SECRET_PATH": str(secret),
    }


# ---------------------------------------------------------------------------
# Compose rendering: durable Redis, private instance bind, immutable images
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("project_name", [None, "evaluation-instance"])
def test_relocated_compose_preserves_paths_and_volume_identity(tmp_path, project_name):
    if shutil.which("docker") is None:
        pytest.skip("Docker Compose is required to render deployment configuration")
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("TRITON_EVAL_", "COMPOSE_"))
    }
    env.update(COMPOSE_RENDER_ENV)
    if project_name:
        env["COMPOSE_PROJECT_NAME"] = project_name
    result = subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts/compose.sh"), "config", "--format", "json"],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    expected_name = project_name or PROJECT_ROOT.name.lower()
    assert config["name"] == expected_name
    service = config["services"]["triton-eval"]
    assert service["build"]["context"] == str(PROJECT_ROOT)
    assert service["build"]["dockerfile"] == "docker/Dockerfile"
    logs = next(v for v in service["volumes"] if v["target"] == "/opt/triton-eval/logs")
    assert logs["source"] == str(PROJECT_ROOT / "logs")
    assert config["volumes"]["triton-eval-redis-v2"]["name"] == (
        f"{expected_name}_triton-eval-redis-v2"
    )


def test_production_compose_pins_the_reviewed_multiplatform_redis(tmp_path):
    config = _render_compose("docker/compose.yaml")
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


def test_production_compose_binds_the_api_to_the_private_host(tmp_path):
    config = _render_compose("docker/compose.yaml")
    service = config["services"]["triton-eval"]
    assert service["environment"]["TRITON_EVAL_API_HOST"] == INSTANCE_HOST
    assert service["environment"]["TRITON_EVAL_API_PORT"] == "11907"
    assert service["network_mode"] == "host"
    assert service["image"] == EVAL_IMAGE
    assert service["environment"]["TRITON_EVAL_EVALUATOR_BUILD"] == EVALUATOR_BUILD


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
        [docker, "compose", "--project-directory", str(PROJECT_ROOT),
         "--env-file", "/dev/null", "-f", "docker/compose.yaml",
         "config", "--format", "json"],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "TRITON_EVAL_IMAGE" in result.stderr


def test_production_compose_mounts_secret_read_only_without_catalog(tmp_path):
    config = _render_compose("docker/compose.yaml")
    service = config["services"]["triton-eval"]
    targets = {v["target"]: v for v in service["volumes"]}
    for target in ("/run/secrets/triton-eval",):
        assert target in targets, target
        assert targets[target].get("read_only") is True
    assert "/srv/triton-eval-release" not in targets
    assert not any("CATALOG" in name for name in service["environment"])
    for volume in service["volumes"]:
        assert "docker.sock" not in str(volume.get("source", ""))


@pytest.mark.parametrize("cards", ["0", "0,1,3"])
def test_same_compose_supports_single_or_multiple_cards(cards):
    config = _render_compose("docker/compose.yaml", {"TRITON_EVAL_CARDS": cards})
    assert config["services"]["triton-eval"]["environment"]["TRITON_EVAL_CARDS"] == cards


@pytest.mark.parametrize("field, value, expected", [
    ("reference_device_policy", "npu-strict", "npu-strict"),
    ("verify_budget_cap_seconds", "900", 900),
    ("benchmark_timeout_seconds", "120", 120),
    ("device_hold_mb", "0", 0),
    ("trial_rate_limit_per_minute", "30", 30),
    ("trusted_rate_limit_per_minute", "80", 80),
    ("trial_max_inflight", "3", 3),
    ("trusted_max_inflight", "5", 5),
    ("session_max_inflight", "6", 6),
    ("rollout_caller_ids", '["rl-a","rl-b"]', ["rl-a", "rl-b"]),
    ("rollout_rate_limit_per_minute", "750", 750),
    ("rollout_max_inflight", "128", 128),
    ("poison_death_threshold", "3", 3),
    ("card_affinity_avoidance", "false", False),
    ("canary_budget_seconds", "90", 90),
    ("canary_interval_seconds", "45.5", 45.5),
    ("card_reset_command", "site-reset --card {card}", "site-reset --card {card}"),
    ("card_recovery_max_step", "1", 1),
    ("card_recovery_retry_seconds", "120.5", 120.5),
    ("card_canary_on_startup", "false", False),
    ("verdict_cache_enabled", "false", False),
    ("verdict_cache_ttl_seconds", "7200", 7200),
    ("cpu_workers", "2", 2),
    ("precheck_budget_seconds", "20", 20),
    ("cann_fingerprint", "site-cann-2026", "site-cann-2026"),
    ("worker_hardware_model", "SiteAscendModel", "SiteAscendModel"),
    ("worker_driver_version", "site-driver-v3", "site-driver-v3"),
    ("worker_runtime_version", "site-runtime-v2", "site-runtime-v2"),
    ("worker_toolchain_version", "Site CANN build", "Site CANN build"),
])
def test_runtime_options_survive_compose_preflight_and_settings_parsing(
    tmp_path, monkeypatch, field, value, expected,
):
    from scripts.deployment_preflight import inspect_deployment
    from triton_evaluation.config import Settings

    env = _complete_deployment_env(tmp_path)
    env[f"TRITON_EVAL_{field.upper()}"] = value
    rendered = _render_compose("docker/compose.yaml", env)

    def docker_read(argv):
        if argv[:2] == ["docker", "compose"]:
            return 0, json.dumps(rendered), ""
        assert argv[:3] == ["docker", "image", "inspect"]
        return 0, json.dumps({"RepoDigests": [argv[-1]]}), ""

    config = inspect_deployment(env, docker_read)
    runtime_env = rendered["services"]["triton-eval"]["environment"]
    for name in os.environ:
        if name.startswith("TRITON_EVAL_"):
            monkeypatch.delenv(name)
    for name, runtime_value in runtime_env.items():
        monkeypatch.setenv(name, runtime_value)
    settings = Settings.from_env()
    assert settings.evaluator_build == config.evaluator_build == EVALUATOR_BUILD
    assert runtime_env["TRITON_EVAL_IMAGE"] == EVAL_IMAGE
    assert getattr(settings, field) == expected


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
printf 'compose-file=%s\\n' "${{TRITON_EVAL_COMPOSE_FILE:-}}" >> {log}
printf 'env-file=%s\\n' "${{TRITON_EVAL_ENV_FILE:-}}" >> {log}
case "$*" in
  *deployment_preflight*)
    printf '{{"api_host": "%s"}}\\n' "${{TRITON_EVAL_PREFLIGHT_INSTANCE_HOST:-172.18.0.1}}"
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
        "TRITON_EVAL_PREFLIGHT_INSTANCE_HOST": INSTANCE_HOST,
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
    # The validated instance address is exported for the composed service.
    assert result.stdout.strip().endswith(INSTANCE_HOST) or INSTANCE_HOST in result.stdout


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


@pytest.mark.parametrize("custom_compose", [False, True])
def test_preflight_and_startup_use_same_files_outside_repository(tmp_path, custom_compose):
    bin_dir, fake_python, log = _fake_tools(tmp_path)
    env_file = tmp_path / "deployment.env"
    env_file.write_text("IGNORED_TEST_VALUE=1\n")
    env = {key: value for key, value in os.environ.items() if not key.startswith("TRITON_EVAL_")}
    env.update({
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "TRITON_EVAL_PYTHON": str(fake_python),
        "TRITON_EVAL_ENV_FILE": str(env_file),
    })
    compose_file = PROJECT_ROOT / "docker/compose.yaml"
    if custom_compose:
        compose_file = tmp_path / "custom compose.yml"
        env["TRITON_EVAL_COMPOSE_FILE"] = str(compose_file)
    result = subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts/compose.sh"), "up"],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    assert f"compose-file={compose_file}" in calls
    assert f"env-file={env_file}" in calls
    assert (
        f"docker compose --project-directory {PROJECT_ROOT}"
        f" --env-file {env_file} -f {compose_file} up"
    ) in calls


def test_env_file_inputs_must_be_exported_before_startup(tmp_path):
    bin_dir, _, log = _fake_tools(tmp_path)
    env_file = tmp_path / "deployment.env"
    env_file.write_text("TRITON_EVAL_API_HOST=172.18.0.1\nTRITON_EVAL_API_PORT=11907\n")
    env = {key: value for key, value in os.environ.items() if not key.startswith("TRITON_EVAL_")}
    env.update({
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "TRITON_EVAL_PYTHON": sys.executable,
        "TRITON_EVAL_ENV_FILE": str(env_file),
    })
    result = subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts/compose.sh"), "up"],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "missing deployment inputs" in result.stderr
    assert "TRITON_EVAL_API_HOST" in result.stderr
    assert not log.exists()  # Failure occurs before any Docker invocation.


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


@pytest.mark.parametrize("build, expected_exit", [
    (EVALUATOR_BUILD, 0), ("sha256:" + "b" * 64, 1),
])
def test_container_readiness_checks_the_image_digest_build(
    tmp_path, build, expected_exit,
):
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        pytest.skip("Bubblewrap is required to run the container readiness gate")
    env = _complete_deployment_env(tmp_path)
    env["TRITON_EVAL_EVALUATOR_BUILD"] = build
    env["PYTHON_BIN"] = sys.executable
    env["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")
    env["PATH"] = os.environ["PATH"]
    env["CANN_SET_ENV_PATH"] = str(_create_fake_cann(tmp_path))
    opt_root = tmp_path / "opt"
    (opt_root / "triton-eval").mkdir(parents=True)
    result = subprocess.run(
        [
            bwrap, "--ro-bind", "/", "/", "--dev", "/dev", "--bind",
            str(opt_root), "/opt", "--ro-bind", str(PROJECT_ROOT),
            "/opt/triton-eval", "--chdir", "/opt/triton-eval", "/usr/bin/env",
            "-i", *(f"{name}={value}" for name, value in env.items()),
            "/bin/bash", "/opt/triton-eval/docker/entrypoint.sh", "/bin/echo", "runtime-ready",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0 and "bwrap:" in result.stderr:
        pytest.skip(f"Bubblewrap is unavailable: {result.stderr.strip()}")
    assert result.returncode == expected_exit, result.stderr
    assert ("runtime-ready" in result.stdout) is (expected_exit == 0)
    if expected_exit:
        assert "evaluator build must equal" in result.stderr


@pytest.mark.parametrize(
    ("mode", "expected_args"),
    [
        ("api", ("args=-m uvicorn scripts.run_api:app --host 127.0.0.1 --port 12000",)),
        ("workers", ("args=-m scripts.run_worker --cpu", "args=-m scripts.run_worker --card 3")),
        ("all", (
            "args=-m uvicorn scripts.run_api:app --host 127.0.0.1 --port 12000",
            "args=-m scripts.run_worker --cpu",
            "args=-m scripts.run_worker --card 3",
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


@pytest.mark.parametrize("cpu_workers", [0, 2])
def test_host_launcher_adds_cann_arch_library_to_api_and_worker(tmp_path, cpu_workers):
    set_env = _create_fake_cann(tmp_path)
    fake_python = tmp_path / "fake-python"
    _write_executable(
        fake_python,
        """#!/usr/bin/env bash
printf 'ld=%s\\n' "${LD_LIBRARY_PATH:-}"
printf 'args=%s\\n' "$*"
printf 'cwd=%s\\n' "$PWD"
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
        f"TRITON_EVAL_CPU_WORKERS={cpu_workers}",
        f"LOG_DIR={log_dir}",
        "",
    ]))
    env = os.environ.copy()
    env["TRITON_EVAL_ENV_FILE"] = str(env_file)
    env["LD_LIBRARY_PATH"] = "baseline-ld"

    result = subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts/start.sh")], cwd=tmp_path, env=env,
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    arch_library = f"{set_env.parent}/aarch64-linux/lib64"
    assert f"cwd={PROJECT_ROOT}" in (log_dir / "api.log").read_text()
    assert arch_library in (log_dir / "api.log").read_text()
    assert arch_library in (log_dir / "worker_8.log").read_text()
    assert "args=-m uvicorn scripts.run_api:app" in (log_dir / "api.log").read_text()
    assert "args=-m scripts.run_worker --card 8" in (log_dir / "worker_8.log").read_text()
    cpu_logs = sorted(log_dir.glob("worker_cpu_*.log"))
    assert len(cpu_logs) == cpu_workers
    for cpu_log in cpu_logs:
        assert arch_library in cpu_log.read_text()
        assert "args=-m scripts.run_worker --cpu" in cpu_log.read_text()


def test_stop_without_pid_file_matches_api_cpu_and_npu_workers(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "pkill",
        """#!/usr/bin/env bash
[[ "$1" == "-TERM" && "$2" == "-f" ]] || exit 2
printf '%s\\n' \\
  'python -m uvicorn scripts.run_api:app --host 127.0.0.1' \\
  'python -m scripts.run_worker --cpu' \\
  'python -m scripts.run_worker --card 3' \\
  'python -m scripts.run_worker --cardinality 3' | grep -E "$3"
""",
    )
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "LOG_DIR": str(tmp_path / "logs"),
        "TRITON_EVAL_ENV_FILE": str(tmp_path / "missing.env"),
    }
    result = subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts/stop.sh")], cwd=tmp_path, env=env,
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "python -m uvicorn scripts.run_api:app --host 127.0.0.1",
        "python -m scripts.run_worker --cpu",
        "python -m scripts.run_worker --card 3",
    ]


def test_api_entrypoint_loads_via_uvicorn_import_path(tmp_path):
    env = os.environ | _complete_deployment_env(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", (
            "from fastapi import FastAPI; "
            "from uvicorn.importer import import_from_string; "
            "from scripts.run_api import build_app; "
            "assert isinstance(import_from_string('scripts.run_api:app'), FastAPI); "
            "assert isinstance(build_app(), FastAPI)"
        )],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
