#!/usr/bin/env python
"""Fail-closed deployment preflight for the durable evaluation service.

Pure stdlib. Runs exactly three Docker read-only commands — ``network
inspect``, ``image inspect``, ``compose config`` — parses their JSON, and
never invokes a shell or pulls an image. Exits 0 only when starting this
exact rendered deployment is safe; on success prints a machine-readable
``{"gateway": ...}`` line for the Compose wrapper to export.
"""
import argparse
import ipaddress
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

NETWORK_NAME = "kernel-agent-sandbox-net"
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
_REQUIRED_ENV = (
    "TRITON_EVAL_API_HOST", "TRITON_EVAL_API_PORT", "PROXY_HOST",
    "EVALUATION_URL", "NO_PROXY", "TRITON_EVAL_CAPABILITY_FILE",
    "TRITON_EVAL_CATALOG_ROOT", "TRITON_EVAL_CATALOG_MANIFEST",
    "TRITON_EVAL_IMAGE", "TRITON_EVAL_EVALUATOR_BUILD",
)

CommandRunner = Callable[[list[str]], tuple[int, str, str]]


class PreflightError(Exception):
    """A redacted deployment failure naming only the failing condition."""


@dataclass(frozen=True)
class DeploymentConfig:
    network_name: str
    gateway: ipaddress.IPv4Address
    api_host: ipaddress.IPv4Address
    proxy_host: ipaddress.IPv4Address
    api_port: int
    evaluation_url: str
    capability_file: Path
    catalog_root: Path
    catalog_manifest: Path
    redis_image: str
    evaluator_build: str


def inspect_deployment(
    env: Mapping[str, str], runner: CommandRunner
) -> DeploymentConfig:
    """Perform every pure-stdlib and fakeable Docker preflight check."""
    missing = [name for name in _REQUIRED_ENV if not env.get(name)]
    if missing:
        raise PreflightError(f"missing deployment inputs: {','.join(missing)}")

    gateway = _inspect_gateway(runner)
    _validate_hosts_and_url(env, gateway)

    rendered = _render_compose(env, runner)
    _validate_redis_service(rendered)
    _validate_eval_service(rendered, env, gateway)
    _validate_secret_and_catalog(env, runner)

    _verify_local_image(runner, REDIS_IMAGE)
    _verify_local_image(runner, env["TRITON_EVAL_IMAGE"])

    return DeploymentConfig(
        network_name=NETWORK_NAME,
        gateway=gateway,
        api_host=gateway,
        proxy_host=gateway,
        api_port=API_PORT,
        evaluation_url=env["EVALUATION_URL"],
        capability_file=Path(env["TRITON_EVAL_CAPABILITY_FILE"]),
        catalog_root=Path(env["TRITON_EVAL_CATALOG_ROOT"]),
        catalog_manifest=Path(env["TRITON_EVAL_CATALOG_MANIFEST"]),
        redis_image=REDIS_IMAGE,
        evaluator_build=env["TRITON_EVAL_EVALUATOR_BUILD"],
    )


def _run(runner: CommandRunner, argv: list[str]) -> str:
    code, stdout, stderr = runner(argv)
    if code != 0:
        raise PreflightError(f"command failed: {argv[0]} {argv[1]} {argv[2]}")
    return stdout


def _inspect_gateway(runner: CommandRunner) -> ipaddress.IPv4Address:
    payload = _run(runner, ["docker", "network", "inspect", NETWORK_NAME])
    try:
        network = json.loads(payload)[0]
        configs = network["IPAM"][0]["Config"]
        gateways = [entry["Gateway"] for entry in configs if entry.get("Gateway")]
    except (ValueError, KeyError, IndexError, TypeError):
        raise PreflightError("network inspect produced no usable gateway") from None
    ipv4 = []
    for text in gateways:
        try:
            address = ipaddress.ip_address(text)
        except ValueError:
            continue
        if isinstance(address, ipaddress.IPv4Address):
            ipv4.append(address)
    if len(ipv4) != 1:
        raise PreflightError("network must expose exactly one IPv4 gateway")
    return ipv4[0]


def _validate_hosts_and_url(env: Mapping[str, str], gateway) -> None:
    if env["TRITON_EVAL_API_PORT"] != str(API_PORT):
        raise PreflightError("TRITON_EVAL_API_PORT must be the production port")
    for name in ("TRITON_EVAL_API_HOST", "PROXY_HOST"):
        try:
            address = ipaddress.IPv4Address(env[name])
        except ValueError:
            raise PreflightError(f"{name} must be an IPv4 address") from None
        if address != gateway or env[name] != str(gateway):
            raise PreflightError(f"{name} must equal the inspected gateway")
        if address.is_loopback or address.is_unspecified or address.is_multicast:
            raise PreflightError(f"{name} must be the bridge gateway")

    url = urlsplit(env["EVALUATION_URL"])
    if (
        url.scheme != "http" or url.hostname != str(gateway)
        or url.username is not None or url.password is not None
        or url.path not in ("", "/") or url.query or url.fragment
        or url.port != API_PORT
        or env["EVALUATION_URL"] != f"http://{gateway}:{API_PORT}"
    ):
        raise PreflightError("EVALUATION_URL must be exactly the gateway URL")

    entries = [entry.strip() for entry in env["NO_PROXY"].split(",")]
    if str(gateway) not in entries:
        raise PreflightError("NO_PROXY must contain the gateway as a complete entry")


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


def _validate_eval_service(rendered: dict, env: Mapping[str, str], gateway) -> None:
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
    build = env["TRITON_EVAL_EVALUATOR_BUILD"]
    git_sha = build.removeprefix("git:")
    is_reviewed_git_build = (
        build.startswith("git:") and len(git_sha) == 40
        and all(char in "0123456789abcdef" for char in git_sha)
    )
    if build != image and not is_reviewed_git_build:
        raise PreflightError("evaluator build must equal the immutable image")
    if settings_env.get("TRITON_EVAL_EVALUATOR_BUILD") != build:
        raise PreflightError("rendered evaluator build must match the deployment input")
    if settings_env.get("TRITON_EVAL_API_HOST") != str(gateway):
        raise PreflightError("rendered API host must equal the inspected gateway")
    if settings_env.get("TRITON_EVAL_API_PORT") != str(API_PORT):
        raise PreflightError("rendered API port must be the production port")

    volumes = service.get("volumes") or []
    for volume in volumes:
        if "docker.sock" in str(volume.get("source", "")):
            raise PreflightError("the Docker socket must not be mounted")
    mounts = {v.get("target"): v for v in volumes}
    for target in ("/run/secrets/triton-eval", "/srv/triton-eval-release"):
        mount = mounts.get(target)
        if mount is None:
            raise PreflightError(f"required read-only mount missing: {target}")
        if mount.get("read_only") is not True:
            raise PreflightError(f"mount must be read-only: {target}")


def _validate_secret_and_catalog(env: Mapping[str, str], runner) -> None:
    try:
        from triton_evaluation.catalog import ReleaseCatalog
        from triton_evaluation.security import CapabilityRegistry
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from triton_evaluation.catalog import ReleaseCatalog
        from triton_evaluation.security import CapabilityRegistry

    try:
        CapabilityRegistry.from_secret_file(Path(env["TRITON_EVAL_CAPABILITY_FILE"]))
    except Exception:
        raise PreflightError("capability secret failed validation") from None
    try:
        ReleaseCatalog.load(
            Path(env["TRITON_EVAL_CATALOG_ROOT"]),
            Path(env["TRITON_EVAL_CATALOG_MANIFEST"]),
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
        help="print the validated gateway as machine-readable JSON",
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
    print(json.dumps({"gateway": str(config.gateway)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
