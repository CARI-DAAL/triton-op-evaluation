#!/usr/bin/env python
"""Generate, bind and verify this repository's release manifest.

The service release number comes from pyproject.toml; the client release number
comes from the plugin manifests and the standalone CLI, which must agree. The
manifest describes the sources and declared dependencies of one release and is
generated deterministically, so the same inputs always produce the same bytes.

Build identity is bound after the artifact exists -- a container's digest is
only known once the image is built, so the external manifest references it
instead of the image containing its own digest.

    generate     write the manifest for the current checkout
    bind         bind a built image digest or reviewed source build
    verify       check the installed artifacts against a manifest
    consistency  check the declared version sources against each other
"""
import argparse
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PLUGIN_MANIFESTS = (
    "plugins/triton-eval/plugin.json",
    "plugins/triton-eval/.claude-plugin/plugin.json",
    "plugins/triton-eval/.codex-plugin/plugin.json",
)
CLI_SCRIPT = "plugins/triton-eval/skills/triton-eval/scripts/triton_eval.py"
PACKAGE = "triton_evaluation"

_RELEASE_TAG = re.compile(r"^v?([0-9A-Za-z][0-9A-Za-z.+_-]{0,63})$")


def _import_versioning():
    try:
        import triton_evaluation.versioning as versioning
    except ImportError:
        sys.path.insert(0, str(PROJECT_ROOT))
        import triton_evaluation.versioning as versioning
    return versioning


# ---------------------------------------------------------------------------
# Declared version sources
# ---------------------------------------------------------------------------


def _project_table(text: str) -> str:
    """Return the ``[project]`` table body.

    Only two declarations are needed from it, and the supported Python range
    starts before tomllib, so the table is read directly.
    """
    marker = "[project]"
    if marker not in text:
        raise ValueError("pyproject.toml has no [project] table")
    body = text.split(marker, 1)[1]
    following = re.search(r"^\[", body, re.MULTILINE)
    return body[: following.start()] if following else body


def pyproject_version(root: Path) -> str:
    table = _project_table((root / "pyproject.toml").read_text(encoding="utf-8"))
    found = re.search(r'^version = "([^"]+)"', table, re.MULTILINE)
    if found is None:
        raise ValueError("pyproject.toml declares no project version")
    return found.group(1)


def pyproject_dependencies(root: Path) -> dict[str, str]:
    """Pinned runtime dependencies; optional extras are deployment choices."""
    table = _project_table((root / "pyproject.toml").read_text(encoding="utf-8"))
    block = re.search(r"^dependencies = \[(.*?)\]", table, re.MULTILINE | re.DOTALL)
    if block is None:
        raise ValueError("pyproject.toml declares no dependencies")
    pins = {}
    for requirement in re.findall(r'"([^"]+)"', block.group(1)):
        name, separator, version = requirement.partition("==")
        if not separator:
            raise ValueError(f"runtime dependency is not pinned: {requirement}")
        pins[name] = version
    return pins


def plugin_versions(root: Path) -> dict[str, str]:
    """Declared client version per plugin manifest."""
    return {
        relative: json.loads((root / relative).read_text(encoding="utf-8"))["version"]
        for relative in PLUGIN_MANIFESTS
    }


def cli_version(root: Path) -> str:
    """The standalone CLI's own version literal, read without importing it."""
    source = (root / CLI_SCRIPT).read_text(encoding="utf-8")
    found = re.search(r'^__version__ = "([^"]+)"', source, re.MULTILINE)
    if found is None:
        raise ValueError("the client CLI declares no __version__")
    return found.group(1)


def client_version(root: Path) -> str:
    """The one client release number; disagreement is a release defect."""
    declared = set(plugin_versions(root).values()) | {cli_version(root)}
    if len(declared) != 1:
        raise ValueError("plugin manifests and CLI declare different client versions")
    return declared.pop()


def release_tags(runner=None) -> tuple[str, ...]:
    """Release tags pointing at HEAD; no tag means this commit is not a release."""
    runner = runner or _git
    output = runner(["tag", "--points-at", "HEAD"])
    if output is None:
        return ()
    return tuple(line.strip() for line in output.splitlines() if line.strip())


def source_revision(runner=None) -> str | None:
    runner = runner or _git
    output = runner(["rev-parse", "HEAD"])
    if output is None:
        return None
    revision = output.strip()
    return revision if re.fullmatch(r"[0-9a-f]{40}", revision) else None


def _git(arguments: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *arguments],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def declaration_mismatches(root: Path, *, tags: Sequence[str] = ()) -> tuple[str, ...]:
    """Reasons the declared release identities disagree with each other."""
    versioning = _import_versioning()
    reasons = []
    declared = pyproject_version(root)
    installed = versioning.service_version()
    if installed == versioning.UNKNOWN_VERSION:
        reasons.append("service-metadata-absent")
    elif installed != declared:
        reasons.append("installed-service-version-mismatch")
    if versioning.declared_dependencies() != pyproject_dependencies(root):
        reasons.append("installed-dependency-pins-mismatch")
    manifests = plugin_versions(root)
    if len(set(manifests.values())) != 1:
        reasons.append("plugin-manifest-version-mismatch")
    if cli_version(root) not in set(manifests.values()):
        reasons.append("cli-client-version-mismatch")
    for tag in tags:
        matched = _RELEASE_TAG.fullmatch(tag)
        if matched is not None and matched.group(1) != declared:
            reasons.append("release-tag-mismatch")
            break
    return tuple(reasons)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def generate(root: Path, *, revision: str | None) -> dict:
    """Build the manifest for this checkout, cross-checked against the install."""
    versioning = _import_versioning()
    mismatches = declaration_mismatches(root)
    if mismatches:
        raise ValueError("declared versions disagree: " + ",".join(mismatches))
    return versioning.build_manifest(
        service_version=pyproject_version(root),
        client_version=client_version(root),
        source_revision=revision,
        source_digest=versioning.digest_sources(root / PACKAGE),
        dependencies=pyproject_dependencies(root),
    )


def _emit(manifest: dict, output: Path | None) -> None:
    versioning = _import_versioning()
    payload = versioning.canonical_manifest(manifest) + b"\n"
    if output is None:
        sys.stdout.buffer.write(payload)
    else:
        output.write_bytes(payload)


def _verify(manifest_path: Path) -> tuple[int, dict]:
    versioning = _import_versioning()
    try:
        from triton_evaluation.config import Settings
    except ImportError:
        sys.path.insert(0, str(PROJECT_ROOT))
        from triton_evaluation.config import Settings

    build = Settings(release_mode="dev", release_manifest=None).evaluator_build
    identity = versioning.release_identity(
        mode="release", manifest_path=manifest_path, evaluator_build=build,
    )
    report = {
        "verified": identity.verified,
        "unverified_reasons": list(identity.reasons),
        "service_version": identity.service_version,
        "source_digest": identity.source_digest,
        "manifest_digest": identity.manifest_digest,
    }
    return (0 if identity.verified else 1), report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    builder = commands.add_parser("generate", help="write the manifest for this checkout")
    builder.add_argument("--output", type=Path)
    builder.add_argument(
        "--source-revision",
        help="40-hex commit; defaults to HEAD, and stays unknown outside a checkout",
    )

    binder = commands.add_parser("bind", help="bind the built artifact's identity")
    binder.add_argument("--manifest", type=Path, required=True)
    binder.add_argument("--output", type=Path)
    binder.add_argument("--image", help="immutable name@sha256:<digest> reference")
    binder.add_argument(
        "--build",
        help="reviewed source build git:<40-hex>; defaults to the image digest",
    )

    checker = commands.add_parser("verify", help="check installed artifacts against a manifest")
    checker.add_argument("--manifest", type=Path, required=True)

    commands.add_parser("consistency", help="check the declared version sources")

    args = parser.parse_args(argv)
    versioning = _import_versioning()
    try:
        if args.command == "generate":
            revision = args.source_revision
            if revision is None:
                revision = source_revision()
            _emit(generate(PROJECT_ROOT, revision=revision), args.output)
            return 0
        if args.command == "bind":
            if args.image is None and args.build is None:
                parser.error("bind requires an image reference or a reviewed build")
            manifest = versioning.load_manifest(args.manifest)
            build = args.build or "sha256:" + args.image.rsplit("@sha256:", 1)[-1]
            bound = versioning.bind_build(
                manifest, evaluator_build=build, image_reference=args.image,
            )
            _emit(bound, args.output)
            return 0
        if args.command == "verify":
            status, report = _verify(args.manifest)
            print(json.dumps(report, sort_keys=True))
            return status
        reasons = declaration_mismatches(PROJECT_ROOT, tags=release_tags())
        print(json.dumps({"consistent": not reasons, "mismatches": list(reasons)},
                         sort_keys=True))
        return 1 if reasons else 0
    except (versioning.ManifestError, ValueError, OSError, KeyError) as error:
        print(f"release manifest command failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
