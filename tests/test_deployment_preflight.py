"""Deployment preflight: pure-stdlib, fakeable Docker checks, fail closed."""
import ipaddress
import json

import pytest

from scripts.deployment_preflight import (
    DeploymentConfig,
    PreflightError,
    inspect_deployment,
)

GATEWAY = "172.18.0.1"
REDIS_IMAGE = (
    "docker.io/library/redis:7.4.10-alpine3.21"
    "@sha256:e7723ff73d963f5cc6d9c4643ea3d989527a402a319239054e9472a7fb9219a2"
)
EVAL_IMAGE = "registry.example.com/triton-eval@sha256:" + "a" * 64
CAPABILITY_TOKEN = "preflight-seeded-secret-token"
REFERENCE_SOURCE = "import torch  # preflight-seeded-reference"


class FakeDocker:
    """Dispatches canned JSON for the exact argv shapes preflight issues."""

    def __init__(
        self,
        *,
        gateway=GATEWAY,
        networks=True,
        redis_digest_ok=True,
        eval_digest_ok=True,
        rendered_overrides=None,
        rendered_env=None,
        platform_available=True,
        fail_commands=(),
    ):
        self.gateway = gateway
        self.networks = networks
        self.redis_digest_ok = redis_digest_ok
        self.eval_digest_ok = eval_digest_ok
        self.rendered_overrides = rendered_overrides or {}
        self.rendered_env = rendered_env or {}
        self.platform_available = platform_available
        self.fail_commands = fail_commands
        self.calls = []

    def __call__(self, argv):
        self.calls.append(list(argv))
        for bad in self.fail_commands:
            if bad in " ".join(argv):
                return 1, "", "docker failed"
        if argv[:3] == ["docker", "network", "inspect"]:
            if not self.networks:
                return 1, "", "network not found"
            if self.gateway is None:
                return 0, json.dumps([_network([])]), ""
            if isinstance(self.gateway, list):
                return 0, json.dumps([_network(self.gateway)]), ""
            return 0, json.dumps([_network([self.gateway])]), ""
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
                [EVAL_IMAGE.replace("registry.example.com/", "registry.example.com/")]
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
                    "TRITON_EVAL_API_HOST": GATEWAY,
                    "TRITON_EVAL_API_PORT": "11907",
                    "TRITON_EVAL_EVALUATOR_BUILD": EVAL_IMAGE,
                    "TRITON_EVAL_CAPABILITY_FILE": "/run/secrets/triton-eval",
                    "TRITON_EVAL_CATALOG_ROOT": "/srv/triton-eval-release",
                    "TRITON_EVAL_CATALOG_MANIFEST": "/srv/triton-eval-release/catalog.json",
                },
                "volumes": [
                    {"source": str(self.secret), "target": "/run/secrets/triton-eval",
                     "read_only": True},
                    {"source": str(self.catalog), "target": "/srv/triton-eval-release",
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
    catalog = "/srv/release-source"


def _network(gateways):
    return {
        "Name": "kernel-agent-sandbox-net",
        "IPAM": [{"Config": [{"Gateway": gateway} for gateway in gateways]}],
    }


def base_env(tmp_path, **overrides):
    secret = _write_secret(tmp_path)
    catalog_root = _write_catalog(tmp_path)
    env = {
        "TRITON_EVAL_API_HOST": GATEWAY,
        "TRITON_EVAL_API_PORT": "11907",
        "PROXY_HOST": GATEWAY,
        "EVALUATION_URL": f"http://{GATEWAY}:11907",
        "NO_PROXY": f"localhost,{GATEWAY},127.0.0.1",
        "TRITON_EVAL_CAPABILITY_FILE": str(secret),
        "TRITON_EVAL_CATALOG_ROOT": str(catalog_root),
        "TRITON_EVAL_CATALOG_MANIFEST": str(catalog_root / "catalog.json"),
        "TRITON_EVAL_IMAGE": EVAL_IMAGE,
        "TRITON_EVAL_EVALUATOR_BUILD": EVAL_IMAGE,
        "TRITON_EVAL_COMPOSE_FILE": "docker-compose.yml",
        "TRITON_EVAL_ENV_FILE": "/dev/null",
    }
    env.update(overrides)
    env = {k: v for k, v in env.items() if v is not None}
    return env


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


def _write_catalog(tmp_path):
    root = tmp_path / "release"
    (root / "references").mkdir(parents=True, exist_ok=True)
    reference = root / "references" / "addmm-1.py"
    reference.write_text(REFERENCE_SOURCE, encoding="utf-8")
    from triton_evaluation.api.contract_artifacts import sha256_hex

    (root / "catalog.json").write_text(json.dumps({
        "schema_version": "1.0",
        "entries": [{
            "manifest_id": "release-1", "problem_id": "addmm-1",
            "reference_digest": sha256_hex(REFERENCE_SOURCE.encode("utf-8")),
            "reference_path": "references/addmm-1.py",
        }],
    }), encoding="utf-8")
    return root


def make_runner(tmp_path, **kwargs):
    private = tmp_path / "runner-files"
    private.mkdir(exist_ok=True)
    runner = FakeDocker(**kwargs)
    runner.secret = _write_secret(private)
    runner.catalog = _write_catalog(private)
    return runner


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_valid_deployment_passes_and_returns_config(tmp_path):
    runner = make_runner(tmp_path)
    config = inspect_deployment(base_env(tmp_path), runner)
    assert isinstance(config, DeploymentConfig)
    assert config.gateway == ipaddress.IPv4Address(GATEWAY)
    assert config.redis_image == REDIS_IMAGE
    assert config.evaluator_build == EVAL_IMAGE
    assert config.api_port == 11907


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


def test_missing_network_is_fatal(tmp_path):
    runner = make_runner(tmp_path, networks=False)
    with pytest.raises(PreflightError):
        inspect_deployment(base_env(tmp_path), runner)


@pytest.mark.parametrize("gateways", [[], [GATEWAY, "172.19.0.1"], ["2001:db8::1"]])
def test_bad_gateway_sets_are_fatal(tmp_path, gateways):
    runner = make_runner(tmp_path, gateway=gateways if gateways else None)
    env = base_env(tmp_path)
    if gateways == ["2001:db8::1"]:
        pytest.skip("ipv6 gateway json handled as string form")
    with pytest.raises(PreflightError):
        inspect_deployment(env, runner)


def test_gateway_must_match_all_host_inputs(tmp_path):
    for bad in (
        {"TRITON_EVAL_API_HOST": "127.0.0.1"},
        {"TRITON_EVAL_API_HOST": "0.0.0.0"},
        {"PROXY_HOST": "127.0.0.1"},
        {"EVALUATION_URL": "http://127.0.0.1:11907"},
        {"EVALUATION_URL": f"http://{GATEWAY}:11907/path"},
        {"EVALUATION_URL": f"http://user@{GATEWAY}:11907"},
        {"TRITON_EVAL_API_PORT": "11908"},
    ):
        runner = make_runner(tmp_path)
        with pytest.raises(PreflightError):
            inspect_deployment(base_env(tmp_path, **bad), runner)


def test_no_proxy_must_contain_gateway_as_complete_entry(tmp_path):
    runner = make_runner(tmp_path)
    with pytest.raises(PreflightError):
        inspect_deployment(
            base_env(tmp_path, NO_PROXY=f"localhost,{GATEWAY}0,127.0.0.1"), runner
        )
    with pytest.raises(PreflightError):
        inspect_deployment(base_env(tmp_path, NO_PROXY=""), runner)


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


def test_absent_local_eval_image_digest_is_fatal(tmp_path):
    runner = make_runner(tmp_path, eval_digest_ok=False)
    with pytest.raises(PreflightError):
        inspect_deployment(base_env(tmp_path), runner)


def test_writable_secret_or_catalog_mount_is_fatal(tmp_path):
    good = FakeDocker()._rendered()
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
    good = FakeDocker()._rendered()
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


def test_preflight_subprocess_failure_is_fatal(tmp_path):
    runner = make_runner(tmp_path, fail_commands=("network inspect",))
    with pytest.raises(PreflightError):
        inspect_deployment(base_env(tmp_path), runner)


# ---------------------------------------------------------------------------
# Secret/catalog checks
# ---------------------------------------------------------------------------


def test_missing_or_unreadable_secret_is_fatal(tmp_path):
    runner = make_runner(tmp_path)
    with pytest.raises(PreflightError):
        inspect_deployment(
            base_env(tmp_path, TRITON_EVAL_CAPABILITY_FILE=str(tmp_path / "nope.json")),
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
    with pytest.raises(PreflightError):
        inspect_deployment(
            base_env(tmp_path, TRITON_EVAL_CAPABILITY_FILE=str(secret)), runner
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
    with pytest.raises(PreflightError):
        inspect_deployment(
            base_env(tmp_path, TRITON_EVAL_CAPABILITY_FILE=str(secret)), runner
        )


def test_catalog_duplicate_triple_is_fatal(tmp_path):
    root = _write_catalog(tmp_path / "mutated")
    payload = json.loads((root / "catalog.json").read_text())
    payload["entries"].append(dict(payload["entries"][0]))
    (root / "catalog.json").write_text(json.dumps(payload), encoding="utf-8")
    runner = make_runner(tmp_path)
    with pytest.raises(PreflightError):
        inspect_deployment(
            base_env(tmp_path, TRITON_EVAL_CATALOG_ROOT=str(root),
                     TRITON_EVAL_CATALOG_MANIFEST=str(root / "catalog.json")),
            runner,
        )


def test_catalog_digest_mismatch_is_fatal(tmp_path):
    root = _write_catalog(tmp_path / "mutated")
    reference = root / "references" / "addmm-1.py"
    reference.write_text(REFERENCE_SOURCE + "\n# mutated", encoding="utf-8")
    runner = make_runner(tmp_path)
    with pytest.raises(PreflightError):
        inspect_deployment(
            base_env(tmp_path, TRITON_EVAL_CATALOG_ROOT=str(root),
                     TRITON_EVAL_CATALOG_MANIFEST=str(root / "catalog.json")),
            runner,
        )


def test_catalog_traversal_escape_is_fatal(tmp_path):
    root = _write_catalog(tmp_path / "mutated")
    outside = tmp_path / "outside.py"
    outside.write_text(REFERENCE_SOURCE, encoding="utf-8")
    payload = json.loads((root / "catalog.json").read_text())
    payload["entries"][0]["reference_path"] = "../outside.py"
    (root / "catalog.json").write_text(json.dumps(payload), encoding="utf-8")
    runner = make_runner(tmp_path)
    with pytest.raises(PreflightError):
        inspect_deployment(
            base_env(tmp_path, TRITON_EVAL_CATALOG_ROOT=str(root),
                     TRITON_EVAL_CATALOG_MANIFEST=str(root / "catalog.json")),
            runner,
        )


def test_failures_never_reveal_seeded_secret_or_reference(tmp_path, capsys):
    runner = make_runner(tmp_path)
    with pytest.raises(PreflightError):
        inspect_deployment(
            base_env(tmp_path, EVALUATION_URL="http://bad:11907"), runner
        )
    captured = capsys.readouterr()
    assert CAPABILITY_TOKEN not in captured.out + captured.err
    assert REFERENCE_SOURCE not in captured.out + captured.err


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_help_exits_zero_without_docker(tmp_path, capsys):
    import scripts.deployment_preflight as preflight

    assert preflight.main(["--help"]) == 0


def test_main_prints_machine_readable_gateway_on_success(
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
    assert json.loads(out)["gateway"] == GATEWAY
