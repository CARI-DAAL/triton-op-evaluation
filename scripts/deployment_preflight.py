#!/usr/bin/env python
"""Validate one independent evaluation instance before starting Compose.

Validate an explicit private IPv4 bind, immutable images, mounted capabilities
and catalog, and durable local Redis. Consumer sandbox networking and endpoint
selection belong to the consumer and are not deployment inputs here.
"""
import argparse
import ipaddress
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

API_PORT = 11907
REDIS_IMAGE = (
    "docker.io/library/redis:7.4.10-alpine3.21"
    "@sha256:e7723ff73d963f5cc6d9c4643ea3d989527a402a319239054e9472a7fb9219a2"
)
# EXEC acknowledgement under --appendfsync always is the first-2xx durability
# boundary: an accepted evaluation survives a container crash.
REDIS_COMMAND = [
    "redis-server", "--bind", "127.0.0.1", "--port", "6379", "--save", "",
    "--appendonly", "yes", "--appendfsync", "always",
    "--aof-use-rdb-preamble", "yes",
]
REDIS_VOLUME = "triton-eval-redis-v2"
REDIS_URL = "redis://127.0.0.1:6379/7"
REDIS_PREFIX = "triton-eval:v2.1"
_REQUIRED_ENV = (
    "TRITON_EVAL_API_HOST", "TRITON_EVAL_API_PORT",
    "TRITON_EVAL_CAPABILITY_SECRET_PATH", "TRITON_EVAL_CATALOG_ROOT",
    "TRITON_EVAL_CATALOG_MANIFEST_PATH", "TRITON_EVAL_IMAGE",
    "TRITON_EVAL_EVALUATOR_BUILD",
)


CommandRunner = Callable[[list[str]], tuple[int, str, str]]


class PreflightError(Exception):
    """A redacted deployment failure naming only the failing condition."""


@dataclass(frozen=True)
class DeploymentConfig:
    api_host: ipaddress.IPv4Address
    api_port: int
    capability_file: Path
    catalog_root: Path
    catalog_manifest: Path
    redis_image: str
    evaluator_build: str


def inspect_deployment(
    env: Mapping[str, str], runner: CommandRunner
) -> DeploymentConfig:
    """Validate rendered runtime settings and deployment assets before startup."""
    missing = [name for name in _REQUIRED_ENV if not env.get(name)]
    if missing:
        raise PreflightError(f"missing deployment inputs: {','.join(missing)}")

    api_host = _validate_bind(env)

    rendered = _render_compose(env, runner)
    _validate_redis_service(rendered)
    _validate_eval_service(rendered, env, api_host)
    _validate_secret_and_catalog(env, runner)

    _verify_local_image(runner, REDIS_IMAGE)
    _verify_local_image(runner, env["TRITON_EVAL_IMAGE"])

    return DeploymentConfig(
        api_host=api_host,
        api_port=API_PORT,
        capability_file=Path(env["TRITON_EVAL_CAPABILITY_SECRET_PATH"]),
        catalog_root=Path(env["TRITON_EVAL_CATALOG_ROOT"]),
        catalog_manifest=Path(env["TRITON_EVAL_CATALOG_MANIFEST_PATH"]),
        redis_image=REDIS_IMAGE,
        evaluator_build=env["TRITON_EVAL_EVALUATOR_BUILD"],
    )


def _run(runner: CommandRunner, argv: list[str]) -> str:
    code, stdout, stderr = runner(argv)
    if code != 0:
        raise PreflightError(f"command failed: {argv[0]} {argv[1]} {argv[2]}")
    return stdout


def _validate_bind(env: Mapping[str, str]) -> ipaddress.IPv4Address:
    if env["TRITON_EVAL_API_PORT"] != str(API_PORT):
        raise PreflightError("TRITON_EVAL_API_PORT must be the production port")
    try:
        address = ipaddress.IPv4Address(env["TRITON_EVAL_API_HOST"])
    except ValueError:
        raise PreflightError("TRITON_EVAL_API_HOST must be a private IPv4 address") from None
    private_networks = (
        ipaddress.IPv4Network("10.0.0.0/8"),
        ipaddress.IPv4Network("172.16.0.0/12"),
        ipaddress.IPv4Network("192.168.0.0/16"),
    )
    if not any(address in network for network in private_networks):
        raise PreflightError("TRITON_EVAL_API_HOST must be an explicit private IPv4 address")
    return address


def _render_compose(env: Mapping[str, str], runner: CommandRunner) -> dict:
    compose_file = env.get("TRITON_EVAL_COMPOSE_FILE", "docker-compose.yml")
    env_file = env.get("TRITON_EVAL_ENV_FILE", "/dev/null")
    payload = _run(
        runner,
        ["docker", "compose", "--env-file", env_file, "-f", compose_file,
         "config", "--format", "json"],
    )
    try:
        return json.loads(payload)
    except ValueError:
        raise PreflightError("compose config did not render JSON") from None


def _validate_redis_service(rendered: dict) -> None:
    redis = rendered.get("services", {}).get("redis")
    if not isinstance(redis, dict):
        raise PreflightError("compose must define the redis service")
    if redis.get("image") != REDIS_IMAGE:
        raise PreflightError("redis image must be the pinned index digest")
    if redis.get("network_mode") != "host":
        raise PreflightError("redis must stay on the host network")
    command = redis.get("command") or []
    if list(command) != REDIS_COMMAND:
        raise PreflightError("redis command must enable durable AOF persistence")
    volumes = redis.get("volumes") or []
    if not any(
        v.get("source") == REDIS_VOLUME and v.get("target") == "/data" for v in volumes
    ):
        raise PreflightError("redis must persist into the named data volume")
    if REDIS_VOLUME not in (rendered.get("volumes") or {}):
        raise PreflightError("compose must declare the redis data volume")


def _validate_eval_service(rendered: dict, env: Mapping[str, str], api_host) -> None:
    service = rendered.get("services", {}).get("triton-eval")
    if not isinstance(service, dict):
        raise PreflightError("compose must define the eval service")
    if service.get("network_mode") != "host":
        raise PreflightError("eval service must stay on the host network")
    image = env["TRITON_EVAL_IMAGE"]
    if "@sha256:" not in image:
        raise PreflightError("eval image must be a digest reference")
    digest = image.split("@sha256:")[-1]
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise PreflightError("eval image must be an immutable sha256 reference")
    if service.get("image") != image:
        raise PreflightError("rendered eval image must match the deployment input")
    settings_env = service.get("environment") or {}
    if settings_env.get("TRITON_EVAL_IMAGE") != image:
        raise PreflightError("runtime image identity must match the immutable image")
    build = env["TRITON_EVAL_EVALUATOR_BUILD"]
    git_sha = build.removeprefix("git:")
    is_reviewed_git_build = (
        build.startswith("git:") and len(git_sha) == 40
        and all(char in "0123456789abcdef" for char in git_sha)
    )
    if build != image.rsplit("@", 1)[1] and not is_reviewed_git_build:
        raise PreflightError("evaluator build must equal the immutable image's sha256 digest")
    if settings_env.get("TRITON_EVAL_EVALUATOR_BUILD") != build:
        raise PreflightError("rendered evaluator build must match the deployment input")
    if settings_env.get("TRITON_EVAL_API_HOST") != str(api_host):
        raise PreflightError("rendered API host must equal the configured instance address")
    if settings_env.get("TRITON_EVAL_API_PORT") != str(API_PORT):
        raise PreflightError("rendered API port must be the production port")
    _validate_instance_settings(settings_env, env)
    _validate_mounts(service.get("volumes") or [], settings_env, env)
    _validate_runtime_settings(settings_env)


def _validate_runtime_settings(settings_env: Mapping[str, str]) -> None:
    """Use the runtime parser with only the environment this container receives."""
    try:
        from triton_evaluation.config import Settings
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from triton_evaluation.config import Settings

    from pydantic_settings import EnvSettingsSource

    class RenderedSettings(Settings):
        @classmethod
        def settings_customise_sources(
            cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings,
        ):
            source = EnvSettingsSource(settings_cls)
            source.env_vars = {name.lower(): value for name, value in settings_env.items()}
            return (source,)

    try:
        RenderedSettings()
    except Exception:
        raise PreflightError("rendered runtime settings failed validation") from None


def _validate_instance_settings(settings_env: Mapping[str, str], env: Mapping[str, str]) -> None:
    redis_url = env.get("TRITON_EVAL_REDIS_URL", REDIS_URL)
    if re.fullmatch(r"redis://127\.0\.0\.1:6379/(0|[1-9][0-9]*)", redis_url) is None:
        raise PreflightError("Redis URL must use the local managed Redis without credentials")
    if settings_env.get("TRITON_EVAL_REDIS_URL") != redis_url:
        raise PreflightError("rendered Redis URL must equal the configured local database")
    if (env.get("TRITON_EVAL_REDIS_PREFIX", REDIS_PREFIX) != REDIS_PREFIX
            or settings_env.get("TRITON_EVAL_REDIS_PREFIX") != REDIS_PREFIX):
        raise PreflightError("Redis namespace must be triton-eval:v2.1")

    cards = env.get("TRITON_EVAL_CARDS", "0")
    if re.fullmatch(r"[0-9]+(?:,[0-9]+)*", cards) is None:
        raise PreflightError("cards must be a nonempty comma-separated decimal device list")
    device_ids = [int(card) for card in cards.split(",")]
    if len(set(device_ids)) != len(device_ids):
        raise PreflightError("cards must contain unique device IDs")
    if settings_env.get("TRITON_EVAL_CARDS") != cards:
        raise PreflightError("rendered cards must equal the configured device list")


def _validate_mounts(volumes: list[dict], settings_env: Mapping[str, str],
                     env: Mapping[str, str]) -> None:
    runtime_paths = {
        "TRITON_EVAL_CAPABILITY_SECRET_PATH": "/run/secrets/triton-eval",
        "TRITON_EVAL_CATALOG_ROOT": "/srv/triton-eval-release",
        "TRITON_EVAL_CATALOG_MANIFEST_PATH": "/srv/triton-eval-release/catalog.json",
    }
    if any(settings_env.get(name) != path for name, path in runtime_paths.items()):
        raise PreflightError("runtime secret and catalog paths must match the validated mounts")
    catalog_root = Path(env["TRITON_EVAL_CATALOG_ROOT"]).resolve()
    if Path(env["TRITON_EVAL_CATALOG_MANIFEST_PATH"]).resolve() != catalog_root / "catalog.json":
        raise PreflightError("catalog manifest must be catalog.json inside the mounted catalog")

    for volume in volumes:
        if "docker.sock" in str(volume.get("source", "")):
            raise PreflightError("the Docker socket must not be mounted")
    mounts = {v.get("target"): v for v in volumes}
    for name in ("TRITON_EVAL_CAPABILITY_SECRET_PATH", "TRITON_EVAL_CATALOG_ROOT"):
        target = runtime_paths[name]
        mount = mounts.get(target)
        if mount is None:
            raise PreflightError(f"required read-only mount missing: {target}")
        if mount.get("read_only") is not True:
            raise PreflightError(f"mount must be read-only: {target}")
        source = mount.get("source")
        if (mount.get("type") != "bind" or not isinstance(source, str)
                or Path(source).resolve() != Path(env[name]).resolve()):
            raise PreflightError(f"mount source must equal the validated host path: {target}")


def _validate_secret_and_catalog(env: Mapping[str, str], runner) -> None:
    try:
        from triton_evaluation.catalog import ReleaseCatalog
        from triton_evaluation.security import CapabilityRegistry
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from triton_evaluation.catalog import ReleaseCatalog
        from triton_evaluation.security import CapabilityRegistry

    try:
        CapabilityRegistry.from_secret_file(Path(env["TRITON_EVAL_CAPABILITY_SECRET_PATH"]))
    except Exception:
        raise PreflightError("capability secret failed validation") from None
    try:
        ReleaseCatalog.load(
            Path(env["TRITON_EVAL_CATALOG_ROOT"]),
            Path(env["TRITON_EVAL_CATALOG_MANIFEST_PATH"]),
        )
    except Exception:
        raise PreflightError("release catalog failed validation") from None


def _verify_local_image(runner: CommandRunner, reference: str) -> None:
    payload = _run(runner, ["docker", "image", "inspect", "--format", "json",
                            reference])
    try:
        info = json.loads(payload)
        digests = info.get("RepoDigests") or []
    except ValueError:
        raise PreflightError("image inspect did not render JSON") from None
    pinned = reference.split("@")[-1] if "@" in reference else None
    if pinned is None or not any(digests) or pinned not in " ".join(digests):
        raise PreflightError("local image digest does not match the pinned reference")


def real_runner() -> CommandRunner:
    def _runner(argv: list[str]) -> tuple[int, str, str]:
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
        return result.returncode, result.stdout, result.stderr

    return _runner


def deployment_environment() -> dict[str, str]:
    return dict(os.environ)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true",
        help="print the validated instance bind as machine-readable JSON",
    )
    try:
        parser.parse_args(argv)
    except SystemExit as exit_request:
        return int(exit_request.code or 0)
    try:
        config = inspect_deployment(deployment_environment(), real_runner())
    except PreflightError as error:
        print(f"deployment preflight failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"api_host": str(config.api_host)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
