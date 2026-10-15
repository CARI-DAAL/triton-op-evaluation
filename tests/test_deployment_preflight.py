"""Deployment preflight: rendered runtime settings, fakeable Docker, fail closed."""
import ipaddress
import json

import pytest

from scripts.deployment_preflight import (
    CONTAINER_MANIFEST,
    PROJECT_ROOT,
    DeploymentConfig,
    PreflightError,
    inspect_deployment,
)
from triton_evaluation import versioning

INSTANCE_HOST = "172.18.0.1"
REDIS_IMAGE = (
    "docker.io/library/redis:7.4.10-alpine3.21"
    "@sha256:e7723ff73d963f5cc6d9c4643ea3d989527a402a319239054e9472a7fb9219a2"
)
EVAL_IMAGE = "registry.example.com/triton-eval@sha256:" + "a" * 64
EVALUATOR_BUILD = EVAL_IMAGE.rsplit("@", 1)[1]
CAPABILITY_TOKEN = "preflight-seeded-secret-token"


class FakeDocker:
    """Dispatches canned JSON for the exact argv shapes preflight issues."""

    def __init__(
        self,
        *,
        redis_digest_ok=True,
        eval_digest_ok=True,
        rendered_overrides=None,
        rendered_env=None,
        fail_commands=(),
    ):
        self.redis_digest_ok = redis_digest_ok
        self.eval_digest_ok = eval_digest_ok
        self.rendered_overrides = rendered_overrides or {}
        self.rendered_env = rendered_env or {}
        self.fail_commands = fail_commands
        self.calls = []

    def __call__(self, argv):
        self.calls.append(list(argv))
        for bad in self.fail_commands:
            if bad in " ".join(argv):
                return 1, "", "docker failed"
        if argv[:3] == ["docker", "image", "inspect"]:
            image = argv[-1]
            if "redis" in image:
                repo_digests = (
                    [f"redis@sha256:{REDIS_IMAGE.split('@sha256:')[1]}"]
                    if self.redis_digest_ok
                    else ["redis@sha256:" + "0" * 64]
                )
                return 0, json.dumps(
                    {"RepoDigests": repo_digests,
                     "Os": "linux", "Architecture": "arm64"}
                ), ""
            repo_digests = (
                [EVAL_IMAGE]
                if self.eval_digest_ok
                else ["registry.example.com/triton-eval@sha256:" + "0" * 64]
            )
            return 0, json.dumps(
                {"RepoDigests": repo_digests,
                 "Os": "linux", "Architecture": "arm64"}
            ), ""
        if argv[-2:] == ["config", "--format"] or "config" in argv:
            return 0, json.dumps(self._rendered()), ""
        return 1, "", f"unexpected command {' '.join(argv[:3])}"

    def _rendered(self):
        services = {
            "triton-eval": {
                "image": EVAL_IMAGE,
                "network_mode": "host",
                "environment": {
                    "TRITON_EVAL_IMAGE": EVAL_IMAGE,
                    "TRITON_EVAL_API_HOST": INSTANCE_HOST,
                    "TRITON_EVAL_API_PORT": "11907",
                    "TRITON_EVAL_REDIS_URL": "redis://127.0.0.1:6379/7",
                    "TRITON_EVAL_REDIS_PREFIX": "triton-eval:v2.1",
                    "TRITON_EVAL_CARDS": "0",
                    "TRITON_EVAL_EVALUATOR_BUILD": EVALUATOR_BUILD,
                    "TRITON_EVAL_CAPABILITY_SECRET_PATH": "/run/secrets/triton-eval",
                    "TRITON_EVAL_RELEASE_MODE": "release",
                    "TRITON_EVAL_RELEASE_MANIFEST": CONTAINER_MANIFEST,
                },
                "volumes": [
                    {"type": "bind", "source": str(self.secret),
                     "target": "/run/secrets/triton-eval",
                     "read_only": True},
                    {"type": "bind", "source": str(self.manifest),
                     "target": CONTAINER_MANIFEST,
                     "read_only": True},
                ],
            },
            "redis": {
                "image": REDIS_IMAGE,
                "network_mode": "host",
                "command": [
                    "redis-server", "--bind", "127.0.0.1", "--port", "6379",
                    "--save", "", "--appendonly", "yes", "--appendfsync", "always",
                    "--aof-use-rdb-preamble", "yes",
                ],
                "volumes": [
                    {"source": "triton-eval-redis-v2", "target": "/data",
                     "read_only": False},
                ],
            },
        }
        rendered = {
            "services": services,
            "volumes": {"triton-eval-redis-v2": {}},
        }
        if self.rendered_env:
            services["triton-eval"]["environment"].update(self.rendered_env)
        rendered.update(self.rendered_overrides)
        return rendered

    secret = "/run/secrets/source"
    manifest = "/srv/release-manifest.json"


def base_env(tmp_path, **overrides):
    secret = _write_secret(tmp_path)
    env = {
        "TRITON_EVAL_API_HOST": INSTANCE_HOST,
        "TRITON_EVAL_API_PORT": "11907",
        "TRITON_EVAL_CAPABILITY_SECRET_PATH": str(secret),
        "TRITON_EVAL_IMAGE": EVAL_IMAGE,
        "TRITON_EVAL_EVALUATOR_BUILD": EVALUATOR_BUILD,
        "TRITON_EVAL_COMPOSE_FILE": "docker/compose.yaml",
        "TRITON_EVAL_ENV_FILE": "/dev/null",
    }
    env.update(overrides)
    env = {k: v for k, v in env.items() if v is not None}
    if "TRITON_EVAL_RELEASE_MANIFEST_HOST" not in overrides:
        env["TRITON_EVAL_RELEASE_MANIFEST_HOST"] = str(
            write_manifest(tmp_path, **_binding(env))
        )
    return env


def _binding(env):
    """Bind the manifest to this deployment's build; unbindable inputs keep the
    defaults, since preflight rejects them before reaching the manifest."""
    build = env.get("TRITON_EVAL_EVALUATOR_BUILD", EVALUATOR_BUILD)
    image = env.get("TRITON_EVAL_IMAGE", EVAL_IMAGE)
    if versioning.BUILD_PATTERN.fullmatch(build) is None or "@sha256:" not in image:
        return {}
    return {"build": build, "image": image}


def write_manifest(tmp_path, *, build=EVALUATOR_BUILD, image=EVAL_IMAGE, payload=None):
    """A release manifest bound to the image this deployment claims to run."""
    path = tmp_path / "release-manifest.json"
    if payload is not None:
        path.write_bytes(payload)
        return path
    manifest = versioning.build_manifest(
        service_version=versioning.service_version(),
        client_version="0.4.0",
        source_digest=versioning.source_digest(),
        dependencies=versioning.declared_dependencies(),
        evaluator_build=build,
        image_reference=image,
    )
    path.write_bytes(versioning.canonical_manifest(manifest) + b"\n")
    return path


def _write_secret(tmp_path):
    secret = tmp_path / "capability.json"
    secret.write_text(json.dumps({
        "capabilities": [{
            "token": CAPABILITY_TOKEN, "caller_id": "trial-1",
            "caller_class": "trial", "scopes": ["trial:submit", "trial:read"],
        }]
    }), encoding="utf-8")
    secret.chmod(0o600)
    return secret


def make_runner(tmp_path, **kwargs):
    runner = FakeDocker(**kwargs)
    runner.secret = tmp_path / "capability.json"
    runner.manifest = tmp_path / "release-manifest.json"
    return runner


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_default_compose_path_is_independent_of_cwd(tmp_path, monkeypatch):
    runner = make_runner(tmp_path)
    env = base_env(tmp_path, TRITON_EVAL_COMPOSE_FILE=None)
    monkeypatch.chdir(tmp_path)
    inspect_deployment(env, runner)
    assert runner.calls[0] == [
        "docker", "compose", "--project-directory", str(PROJECT_ROOT),
        "--env-file", "/dev/null", "-f", str(PROJECT_ROOT / "docker/compose.yaml"),
        "config", "--format", "json",
    ]


def test_valid_deployment_passes_and_returns_config(tmp_path):
    runner = make_runner(tmp_path)
    config = inspect_deployment(base_env(tmp_path), runner)
    assert isinstance(config, DeploymentConfig)
    assert config.api_host == ipaddress.IPv4Address(INSTANCE_HOST)
    assert config.redis_image == REDIS_IMAGE
    assert config.evaluator_build == EVALUATOR_BUILD
    assert config.api_port == 11907
    assert not (tmp_path / "release").exists()


def test_preflight_never_shows_docker_or_pulls(tmp_path):
    runner = make_runner(tmp_path)
    inspect_deployment(base_env(tmp_path), runner)
    for call in runner.calls:
        assert "pull" not in call
        assert "shell" not in " ".join(call)
        assert call[0] == "docker"


# ---------------------------------------------------------------------------
# Network and URL checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("address", ["10.2.3.4", "192.168.1.12", INSTANCE_HOST])
def test_independent_instance_private_addresses_need_no_sandbox_network(tmp_path, address):
    runner = make_runner(tmp_path, rendered_env={"TRITON_EVAL_API_HOST": address})
    config = inspect_deployment(base_env(tmp_path, TRITON_EVAL_API_HOST=address), runner)
    assert str(config.api_host) == address
    assert not any(call[:3] == ["docker", "network", "inspect"] for call in runner.calls)


@pytest.mark.parametrize("address", ["127.0.0.1", "0.0.0.0", "8.8.8.8", "::", "host"])
def test_non_private_or_ambiguous_binds_are_rejected(tmp_path, address):
    with pytest.raises(PreflightError):
        inspect_deployment(base_env(tmp_path, TRITON_EVAL_API_HOST=address),
                           make_runner(tmp_path))


def test_rendered_instance_must_match_requested_bind(tmp_path):
    with pytest.raises(PreflightError, match="configured instance address"):
        inspect_deployment(base_env(tmp_path, TRITON_EVAL_API_HOST="10.2.3.4"),
                           make_runner(tmp_path))


@pytest.mark.parametrize("url", [
    "redis://10.0.0.2:6379/7", "redis://127.0.0.1:6380/7",
    "redis://user:password@127.0.0.1:6379/7", "redis://127.0.0.1:6379/7?db=0",
    "redis://127.0.0.1:6379/7#fragment", "redis://127.0.0.1:6379/-1",
    "unix:///tmp/redis.sock", "redis://127.0.0.1:6379/db", "",
])
def test_cache_store_must_be_the_validated_local_redis(tmp_path, url):
    runner = make_runner(tmp_path, rendered_env={"TRITON_EVAL_REDIS_URL": url})
    with pytest.raises(PreflightError, match="Redis"):
        inspect_deployment(base_env(tmp_path, TRITON_EVAL_REDIS_URL=url), runner)


def test_rendered_redis_database_must_match_requested_database(tmp_path):
    runner = make_runner(tmp_path, rendered_env={"TRITON_EVAL_REDIS_URL": "redis://127.0.0.1:6379/3"})
    with pytest.raises(PreflightError, match="Redis"):
        inspect_deployment(base_env(tmp_path), runner)


@pytest.mark.parametrize("location", ["configured", "rendered"])
def test_redis_namespace_is_fixed_for_v21(tmp_path, location):
    change = {"TRITON_EVAL_REDIS_PREFIX": "other-instance"}
    runner = make_runner(tmp_path, rendered_env=change if location == "rendered" else {})
    env = base_env(tmp_path, **(change if location == "configured" else {}))
    with pytest.raises(PreflightError, match="namespace"):
        inspect_deployment(env, runner)


@pytest.mark.parametrize("cards", ["", " ", "0,0", "0,00", "0,", "-1", "0 1", "a", "０"])
def test_cards_must_be_nonempty_unique_decimal_device_ids(tmp_path, cards):
    runner = make_runner(tmp_path, rendered_env={"TRITON_EVAL_CARDS": cards})
    with pytest.raises(PreflightError, match="cards"):
        inspect_deployment(base_env(tmp_path, TRITON_EVAL_CARDS=cards), runner)


def test_rendered_cards_must_match_configured_instance(tmp_path):
    runner = make_runner(tmp_path, rendered_env={"TRITON_EVAL_CARDS": "1,3"})
    with pytest.raises(PreflightError, match="cards"):
        inspect_deployment(base_env(tmp_path, TRITON_EVAL_CARDS="0,2"), runner)


def test_explicit_local_database_and_unique_card_list_are_allowed(tmp_path):
    values = {"TRITON_EVAL_REDIS_URL": "redis://127.0.0.1:6379/3", "TRITON_EVAL_CARDS": "1,3"}
    inspect_deployment(base_env(tmp_path, **values), make_runner(tmp_path, rendered_env=values))


@pytest.mark.parametrize("setting, value", [
    ("TRITON_EVAL_CANARY_BUDGET_SECONDS", "0"),
    ("TRITON_EVAL_CARD_RECOVERY_MAX_STEP", "4"),
    ("TRITON_EVAL_ROLLOUT_CALLER_IDS", "not-json"),
    ("TRITON_EVAL_CARD_AFFINITY_AVOIDANCE", "not-a-bool"),
    ("TRITON_EVAL_WORKER_DRIVER_VERSION", ""),
    ("TRITON_EVAL_CANARY_INTERVAL_SECONDS", "-1"),
    ("TRITON_EVAL_SESSION_MAX_INFLIGHT", "0"),
    ("TRITON_EVAL_CPU_WORKERS", "-1"),
    ("TRITON_EVAL_CPU_WORKERS", "33"),
    ("TRITON_EVAL_CPU_WORKERS", ""),
])
def test_invalid_rendered_runtime_options_are_rejected_before_image_inspection(
    tmp_path, setting, value,
):
    runner = make_runner(tmp_path, rendered_env={setting: value})
    with pytest.raises(PreflightError, match="runtime settings"):
        inspect_deployment(base_env(tmp_path), runner)
    assert not any(call[:3] == ["docker", "image", "inspect"] for call in runner.calls)


def test_runtime_settings_validation_ignores_unrendered_host_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_EVAL_CANARY_BUDGET_SECONDS", "invalid-host-only-value")
    monkeypatch.setenv("TRITON_EVAL_ROLLOUT_CALLER_IDS", "also-invalid-host-value")
    inspect_deployment(base_env(tmp_path), make_runner(tmp_path))


@pytest.mark.parametrize("target", ["/run/secrets/triton-eval"])
def test_mounted_source_must_be_the_validated_host_path(tmp_path, target):
    runner = make_runner(tmp_path)
    rendered = runner._rendered()
    for mount in rendered["services"]["triton-eval"]["volumes"]:
        if mount["target"] == target:
            mount["source"] = str(tmp_path / "unvalidated-source")
    runner.rendered_overrides = rendered
    with pytest.raises(PreflightError, match="source"):
        inspect_deployment(base_env(tmp_path), runner)


@pytest.mark.parametrize("setting", ["TRITON_EVAL_CAPABILITY_SECRET_PATH"])
def test_runtime_security_paths_must_match_validated_mounts(tmp_path, setting):
    runner = make_runner(tmp_path, rendered_env={setting: "/unvalidated-path"})
    with pytest.raises(PreflightError, match="runtime"):
        inspect_deployment(base_env(tmp_path), runner)


# ---------------------------------------------------------------------------
# Rendered Compose checks
# ---------------------------------------------------------------------------


def test_rendered_non_host_network_is_fatal(tmp_path):
    runner = make_runner(
        tmp_path,
        rendered_overrides={
            "services": {
                "triton-eval": {"network_mode": "bridge"},
                "redis": {"network_mode": "host"},
            }
        },
    )
    env = base_env(tmp_path)
    with pytest.raises(PreflightError):
        inspect_deployment(env, runner)


def test_non_loopback_or_wrong_digest_redis_is_fatal(tmp_path):
    runner = make_runner(
        tmp_path,
        rendered_overrides={
            "services": {
                "triton-eval": FakeDocker()._rendered()["services"]["triton-eval"],
                "redis": {
                    "image": "redis:7.4.10-alpine3.21",
                    "network_mode": "host",
                    "command": ["redis-server", "--bind", "0.0.0.0"],
                    "volumes": [],
                },
            }
        },
    )
    with pytest.raises(PreflightError):
        inspect_deployment(base_env(tmp_path), runner)


def test_missing_aof_or_appendfsync_is_fatal(tmp_path):
    rendered_redis = FakeDocker()._rendered()["services"]["redis"]
    for bad_command in (
        ["redis-server", "--bind", "127.0.0.1", "--save", "", "--appendonly", "no"],
        ["redis-server", "--bind", "127.0.0.1", "--appendonly", "yes"],
        ["redis-server", "--bind", "127.0.0.1", "--appendonly", "yes",
         "--appendfsync", "everysec", "--save", ""],
    ):
        runner = make_runner(
            tmp_path,
            rendered_overrides={
                "services": {
                    "triton-eval": FakeDocker()._rendered()["services"][
                        "triton-eval"
                    ],
                    "redis": rendered_redis | {"command": bad_command},
                }
            },
        )
        with pytest.raises(PreflightError):
            inspect_deployment(base_env(tmp_path), runner)


def test_missing_redis_volume_is_fatal(tmp_path):
    runner = make_runner(
        tmp_path,
        rendered_overrides={
            "services": {
                "triton-eval": FakeDocker()._rendered()["services"][
                    "triton-eval"
                ],
                "redis": FakeDocker()._rendered()["services"]["redis"]
                | {"volumes": []},
            }
        },
    )
    with pytest.raises(PreflightError):
        inspect_deployment(base_env(tmp_path), runner)


def test_wrong_redis_repo_digest_is_fatal(tmp_path):
    runner = make_runner(tmp_path, redis_digest_ok=False)
    with pytest.raises(PreflightError):
        inspect_deployment(base_env(tmp_path), runner)


def test_mutable_or_missing_eval_image_is_fatal(tmp_path):
    for bad in (
        {"TRITON_EVAL_IMAGE": "registry.example.com/triton-eval:latest"},
        {"TRITON_EVAL_IMAGE": "registry.example.com/triton-eval@sha256:" + "b" * 64},
        {"TRITON_EVAL_EVALUATOR_BUILD": "registry.example.com/other@sha256:" + "a" * 64},
        {"TRITON_EVAL_EVALUATOR_BUILD": None},
    ):
        runner = make_runner(tmp_path, eval_digest_ok=True)
        with pytest.raises(PreflightError):
            inspect_deployment(base_env(tmp_path, **bad), runner)


def test_reviewed_git_build_is_accepted(tmp_path):
    runner = make_runner(
        tmp_path, rendered_env={"TRITON_EVAL_EVALUATOR_BUILD": "git:" + "c" * 40}
    )
    config = inspect_deployment(
        base_env(tmp_path, TRITON_EVAL_EVALUATOR_BUILD="git:" + "c" * 40), runner
    )
    assert config.evaluator_build == "git:" + "c" * 40


@pytest.mark.parametrize("build", [EVAL_IMAGE, "sha256:" + "b" * 64])
def test_production_build_must_be_the_image_digest_without_registry_name(tmp_path, build):
    runner = make_runner(tmp_path, rendered_env={"TRITON_EVAL_EVALUATOR_BUILD": build})
    with pytest.raises(PreflightError, match="evaluator build"):
        inspect_deployment(base_env(tmp_path, TRITON_EVAL_EVALUATOR_BUILD=build), runner)


@pytest.mark.parametrize("image", [None, "registry.example.com/triton-eval@sha256:" + "b" * 64])
def test_runtime_image_identity_must_match_the_image_being_deployed(tmp_path, image):
    runner = make_runner(tmp_path, rendered_env={"TRITON_EVAL_IMAGE": image})
    with pytest.raises(PreflightError, match="runtime image"):
        inspect_deployment(base_env(tmp_path), runner)


def test_absent_local_eval_image_digest_is_fatal(tmp_path):
    runner = make_runner(tmp_path, eval_digest_ok=False)
    with pytest.raises(PreflightError):
        inspect_deployment(base_env(tmp_path), runner)


def test_writable_secret_mount_is_fatal(tmp_path):
    good = make_runner(tmp_path)._rendered()
    volumes = [
        mount | ({"read_only": False} if mount["target"] == "/run/secrets/triton-eval" else {})
        for mount in good["services"]["triton-eval"]["volumes"]
    ]
    runner = make_runner(
        tmp_path,
        rendered_overrides={
            "services": good["services"]
            | {"triton-eval": good["services"]["triton-eval"] | {"volumes": volumes}}
        },
    )
    with pytest.raises(PreflightError):
        inspect_deployment(base_env(tmp_path), runner)


def test_docker_socket_mount_is_fatal(tmp_path):
    good = make_runner(tmp_path)._rendered()
    volumes = good["services"]["triton-eval"]["volumes"] + [
        {"source": "/var/run/docker.sock", "target": "/var/run/docker.sock",
         "read_only": True}
    ]
    runner = make_runner(
        tmp_path,
        rendered_overrides={
            "services": good["services"]
            | {"triton-eval": good["services"]["triton-eval"] | {"volumes": volumes}}
        },
    )
    with pytest.raises(PreflightError):
        inspect_deployment(base_env(tmp_path), runner)


def _without_manifest_mount(tmp_path, **overrides):
    """Render the eval service with its release manifest mount replaced."""
    good = make_runner(tmp_path)._rendered()
    service = good["services"]["triton-eval"]
    volumes = [v for v in service["volumes"] if v["target"] != CONTAINER_MANIFEST]
    if overrides:
        volumes.append({"type": "bind", "source": str(tmp_path / "release-manifest.json"),
                        "target": CONTAINER_MANIFEST, "read_only": True} | overrides)
    return {
        "services": good["services"] | {"triton-eval": service | {"volumes": volumes}}
    }


@pytest.mark.parametrize("rendered_env", [
    {"TRITON_EVAL_RELEASE_MODE": "dev"},
    {"TRITON_EVAL_RELEASE_MANIFEST": "/opt/triton-eval/other.json"},
])
def test_deployed_service_must_verify_the_release_at_startup(tmp_path, rendered_env):
    runner = make_runner(tmp_path, rendered_env=rendered_env)
    with pytest.raises(PreflightError):
        inspect_deployment(base_env(tmp_path), runner)


@pytest.mark.parametrize("overrides", [
    {},
    {"read_only": False},
    {"source": "/elsewhere/release-manifest.json"},
])
def test_release_manifest_must_be_the_validated_read_only_mount(tmp_path, overrides):
    runner = make_runner(
        tmp_path, rendered_overrides=_without_manifest_mount(tmp_path, **overrides)
    )
    with pytest.raises(PreflightError):
        inspect_deployment(base_env(tmp_path), runner)


@pytest.mark.parametrize("manifest_build, manifest_image", [
    ("sha256:" + "b" * 64, EVAL_IMAGE),
    (EVALUATOR_BUILD, "registry.example.com/triton-eval@sha256:" + "b" * 64),
])
def test_release_manifest_must_bind_the_deployed_artifact(
    tmp_path, manifest_build, manifest_image,
):
    manifest = write_manifest(tmp_path, build=manifest_build, image=manifest_image)
    with pytest.raises(PreflightError, match="must bind the deployed"):
        inspect_deployment(
            base_env(tmp_path, TRITON_EVAL_RELEASE_MANIFEST_HOST=str(manifest)),
            make_runner(tmp_path),
        )


@pytest.mark.parametrize("payload", [b"", b"{}", b'{"schema_version": "1.0"}'])
def test_unusable_release_manifest_is_fatal(tmp_path, payload):
    manifest = write_manifest(tmp_path, payload=payload)
    with pytest.raises(PreflightError, match="release manifest failed validation"):
        inspect_deployment(
            base_env(tmp_path, TRITON_EVAL_RELEASE_MANIFEST_HOST=str(manifest)),
            make_runner(tmp_path),
        )


def test_missing_release_manifest_input_is_fatal(tmp_path):
    with pytest.raises(PreflightError, match="missing deployment inputs"):
        inspect_deployment(
            base_env(tmp_path, TRITON_EVAL_RELEASE_MANIFEST_HOST=None), make_runner(tmp_path)
        )


def test_preflight_subprocess_failure_is_fatal(tmp_path):
    runner = make_runner(tmp_path, fail_commands=("compose",))
    with pytest.raises(PreflightError):
        inspect_deployment(base_env(tmp_path), runner)


# ---------------------------------------------------------------------------
# Secret checks
# ---------------------------------------------------------------------------


def test_missing_or_unreadable_secret_is_fatal(tmp_path):
    runner = make_runner(tmp_path)
    runner.secret = tmp_path / "nope.json"
    with pytest.raises(PreflightError):
        inspect_deployment(
            base_env(tmp_path, TRITON_EVAL_CAPABILITY_SECRET_PATH=str(tmp_path / "nope.json")),
            runner,
        )


def test_secret_mode_with_group_or_world_bits_is_fatal(tmp_path):
    secret = tmp_path / "loose-capability.json"
    secret.write_text(json.dumps({
        "capabilities": [{
            "token": "tok", "caller_id": "trial-1", "caller_class": "trial",
            "scopes": ["trial:submit", "trial:read"],
        }]
    }), encoding="utf-8")
    secret.chmod(0o640)
    runner = make_runner(tmp_path)
    runner.secret = secret
    with pytest.raises(PreflightError):
        inspect_deployment(
            base_env(tmp_path, TRITON_EVAL_CAPABILITY_SECRET_PATH=str(secret)), runner
        )


def test_duplicate_capability_configuration_is_fatal(tmp_path):
    secret = tmp_path / "duplicate-capability.json"
    entry = {
        "token": "tok", "caller_id": "trial-1", "caller_class": "trial",
        "scopes": ["trial:submit", "trial:read"],
    }
    secret.write_text(json.dumps({"capabilities": [entry, entry]}), encoding="utf-8")
    secret.chmod(0o600)
    runner = make_runner(tmp_path)
    runner.secret = secret
    with pytest.raises(PreflightError):
        inspect_deployment(
            base_env(tmp_path, TRITON_EVAL_CAPABILITY_SECRET_PATH=str(secret)), runner
        )


def test_failures_never_reveal_seeded_secret(tmp_path, capsys):
    runner = make_runner(tmp_path)
    with pytest.raises(PreflightError):
        inspect_deployment(
            base_env(tmp_path, TRITON_EVAL_API_HOST="bad"), runner
        )
    captured = capsys.readouterr()
    assert CAPABILITY_TOKEN not in captured.out + captured.err


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_help_exits_zero_without_docker(tmp_path, capsys):
    import scripts.deployment_preflight as preflight

    assert preflight.main(["--help"]) == 0


def test_main_prints_machine_readable_instance_address_on_success(
    tmp_path, monkeypatch, capsys
):
    import scripts.deployment_preflight as preflight

    runner = make_runner(tmp_path)
    monkeypatch.setattr(preflight, "real_runner", lambda: runner)
    monkeypatch.setattr(
        preflight, "deployment_environment", lambda: base_env(tmp_path)
    )
    assert preflight.main([]) == 0
    out = capsys.readouterr().out
    assert json.loads(out)["api_host"] == INSTANCE_HOST
