"""Unified release identity for the evaluation service.

One module owns every published identity -- the service release number, the
supported HTTP contracts, the persisted storage format, the build -- plus the
release-manifest format and the rule that decides whether a startup path may
call itself verified. The API, the workers and the release tooling share these
definitions, so a declared identity always describes implemented behaviour.

Identities this release does not yet have stay explicitly empty instead of
being guessed: the negotiated API contract and the immutable evaluation
protocol arrive with their own tickets.
"""
import hashlib
import logging
import re
from dataclasses import dataclass
from functools import cache
from importlib import metadata
from pathlib import Path

from .serialization import canonical_json_bytes, load_canonical, sha256_hex

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA_VERSION = "1.0"
DISTRIBUTION = "triton-evaluation"

# Reported when the running code has no installed distribution metadata: an
# explicitly unknown release is never a plausible-looking version number.
UNKNOWN_VERSION = "0+uninstalled"

# The frozen current wire behaviour. Exactly one contract is implemented, so
# exactly one is advertised.
API_VERSIONS = ("legacy",)
DEFAULT_API_VERSION = "legacy"

# The persisted format is the unversioned durable layout this release reads and
# writes. Naming it "legacy" records today's decoder, not a compatibility
# promise about formats that do not exist yet.
STORAGE_READ_VERSIONS = ("legacy",)
STORAGE_WRITE_VERSION = "legacy"

# The canonical evaluation-protocol digest is not built yet; an absent identity
# is published as absent.
DEFAULT_PROTOCOL = None

# Behaviour callers can rely on today, nothing planned.
CAPABILITIES = (
    "evaluation-submit",
    "evaluation-recovery",
    "evaluation-progress",
    "operator-metrics",
    "version-discovery",
)

# The placeholder build of an unconfigured deployment; never a release build.
ZERO_BUILD = "git:" + "0" * 40

BUILD_PATTERN = re.compile(r"^(sha256:[0-9a-f]{64}|git:[0-9a-f]{40})$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_VERSION = re.compile(r"^[0-9A-Za-z][0-9A-Za-z.+_-]{0,63}$")
_IMAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}@sha256:[0-9a-f]{64}$")
_PACKAGE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class ManifestError(Exception):
    """An unusable release manifest, named by a stable reason code."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class ReleaseVerificationError(Exception):
    """Release-mode startup refused: the running artifacts are not verifiable."""


@dataclass(frozen=True)
class ReleaseIdentity:
    """What a running process can prove about the release it is serving."""

    mode: str
    verified: bool
    reasons: tuple[str, ...]
    service_version: str
    evaluator_build: str
    source_revision: str | None
    source_digest: str
    manifest_digest: str | None


# ---------------------------------------------------------------------------
# Installed-artifact identity
# ---------------------------------------------------------------------------


@cache
def service_version() -> str:
    """The installed distribution version: the one published service release."""
    try:
        return metadata.version(DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return UNKNOWN_VERSION


def digest_sources(root: Path) -> str:
    """Digest the Python sources under ``root``.

    Deterministic over sorted POSIX-relative paths and file bytes, so the same
    sources digest identically in a checkout and in site-packages. Byte
    compilation output is not source and never enters the digest.
    """
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_hex(path.read_bytes()).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


@cache
def source_digest() -> str:
    """Digest of the sources this process is actually running."""
    return digest_sources(Path(__file__).resolve().parent)


def declared_dependencies() -> dict[str, str]:
    """Runtime dependency pins declared by the installed distribution.

    Optional extras are deployment choices, not part of the release identity;
    only the unconditional ``name==version`` pins are returned.
    """
    pins: dict[str, str] = {}
    for requirement in metadata.requires(DISTRIBUTION) or ():
        if ";" in requirement:
            continue
        name, separator, version = requirement.partition("==")
        if separator and _PACKAGE_NAME.fullmatch(name.strip()):
            pins[name.strip()] = version.strip()
    return pins


def installed_dependencies(names) -> dict[str, str]:
    """Installed version of each named dependency; absent ones say so."""
    found = {}
    for name in names:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = "absent"
    return found


# ---------------------------------------------------------------------------
# Release manifest
# ---------------------------------------------------------------------------


def build_manifest(
    *,
    service_version: str,
    client_version: str,
    source_digest: str,
    dependencies: dict[str, str],
    source_revision: str | None = None,
    evaluator_build: str | None = None,
    image_reference: str | None = None,
) -> dict:
    """Assemble a release manifest; the same inputs always produce the same one.

    Build identity stays optional here: a container's digest only exists once
    the image is built, so it is bound afterwards by :func:`bind_build` rather
    than by asking an artifact to contain its own digest.
    """
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "service_version": service_version,
        "client_version": client_version,
        "source_revision": source_revision,
        "source_digest": source_digest,
        "dependencies": dict(dependencies),
        "evaluator_build": evaluator_build,
        "image_reference": image_reference,
        "api_versions": list(API_VERSIONS),
        "default_api_version": DEFAULT_API_VERSION,
        "storage": {
            "read_versions": list(STORAGE_READ_VERSIONS),
            "write_version": STORAGE_WRITE_VERSION,
        },
        "default_protocol": DEFAULT_PROTOCOL,
        "capabilities": list(CAPABILITIES),
    }
    return validate_manifest(manifest)


def bind_build(
    manifest: dict, *, evaluator_build: str, image_reference: str | None = None
) -> dict:
    """Return the manifest with its final build identity bound."""
    bound = {**manifest, "evaluator_build": evaluator_build,
             "image_reference": image_reference}
    return validate_manifest(bound)


def canonical_manifest(manifest: dict) -> bytes:
    """Canonical manifest bytes: one release description, one serialization."""
    return canonical_json_bytes(manifest)


def manifest_digest(manifest: dict) -> str:
    return sha256_hex(canonical_manifest(manifest))


def validate_manifest(manifest: object) -> dict:
    """Validate a manifest's exact shape, rejecting unknown or malformed fields."""
    if not isinstance(manifest, dict):
        raise ManifestError("manifest-malformed")
    expected = {
        "schema_version", "service_version", "client_version", "source_revision",
        "source_digest", "dependencies", "evaluator_build", "image_reference",
        "api_versions", "default_api_version", "storage", "default_protocol",
        "capabilities",
    }
    if set(manifest) != expected:
        raise ManifestError("manifest-malformed")
    storage = manifest["storage"]
    if not isinstance(storage, dict) or set(storage) != {"read_versions", "write_version"}:
        raise ManifestError("manifest-malformed")
    checks = (
        manifest["schema_version"] == MANIFEST_SCHEMA_VERSION,
        _matches(_VERSION, manifest["service_version"]),
        _matches(_VERSION, manifest["client_version"]),
        _matches(_REVISION, manifest["source_revision"], optional=True),
        _matches(_HEX64, manifest["source_digest"]),
        _is_pins(manifest["dependencies"]),
        _matches(BUILD_PATTERN, manifest["evaluator_build"], optional=True),
        _matches(_IMAGE, manifest["image_reference"], optional=True),
        _is_names(manifest["api_versions"]),
        _matches(_VERSION, manifest["default_api_version"]),
        _is_names(storage["read_versions"]),
        _matches(_VERSION, storage["write_version"]),
        _matches(_HEX64, manifest["default_protocol"], optional=True),
        _is_names(manifest["capabilities"]),
    )
    if not all(checks):
        raise ManifestError("manifest-malformed")
    return manifest


def load_manifest(path: Path) -> dict:
    """Read and validate an external release manifest."""
    try:
        payload = Path(path).read_bytes()
    except OSError:
        raise ManifestError("manifest-unreadable") from None
    try:
        manifest = load_canonical(payload)
    except ValueError:
        raise ManifestError("manifest-malformed") from None
    return validate_manifest(manifest)


def _matches(pattern: re.Pattern[str], value: object, *, optional: bool = False) -> bool:
    if value is None:
        return optional
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _is_names(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(_matches(_VERSION, item) for item in value)
    )


def _is_pins(value: object) -> bool:
    return isinstance(value, dict) and all(
        _matches(_PACKAGE_NAME, name) and _matches(_VERSION, pin)
        for name, pin in value.items()
    )


# ---------------------------------------------------------------------------
# Startup verification
# ---------------------------------------------------------------------------


def manifest_mismatches(manifest: dict, *, evaluator_build: str) -> tuple[str, ...]:
    """Reasons the running artifacts disagree with this manifest."""
    reasons = []
    if manifest["service_version"] != service_version():
        reasons.append("service-version-mismatch")
    if manifest["source_digest"] != source_digest():
        reasons.append("source-digest-mismatch")
    if installed_dependencies(manifest["dependencies"]) != manifest["dependencies"]:
        reasons.append("dependency-mismatch")
    if manifest["evaluator_build"] is None:
        reasons.append("evaluator-build-unbound")
    elif manifest["evaluator_build"] != evaluator_build:
        reasons.append("evaluator-build-mismatch")
    if tuple(manifest["api_versions"]) != API_VERSIONS:
        reasons.append("api-versions-mismatch")
    if (
        tuple(manifest["storage"]["read_versions"]) != STORAGE_READ_VERSIONS
        or manifest["storage"]["write_version"] != STORAGE_WRITE_VERSION
    ):
        reasons.append("storage-versions-mismatch")
    if manifest["default_protocol"] != DEFAULT_PROTOCOL:
        reasons.append("protocol-mismatch")
    if tuple(manifest["capabilities"]) != CAPABILITIES:
        reasons.append("capabilities-mismatch")
    return tuple(reasons)


def release_identity(
    *, mode: str, manifest_path: Path | None, evaluator_build: str
) -> ReleaseIdentity:
    """Resolve what this process can prove, without deciding whether to serve."""
    reasons = []
    manifest = None
    if service_version() == UNKNOWN_VERSION:
        reasons.append("service-metadata-absent")
    if manifest_path is None:
        reasons.append("manifest-absent")
    else:
        try:
            manifest = load_manifest(manifest_path)
        except ManifestError as error:
            reasons.append(error.reason)
    if manifest is not None:
        reasons.extend(manifest_mismatches(manifest, evaluator_build=evaluator_build))
    if evaluator_build == ZERO_BUILD:
        reasons.append("zero-evaluator-build")
    return ReleaseIdentity(
        mode=mode,
        verified=not reasons,
        reasons=tuple(reasons),
        service_version=service_version(),
        evaluator_build=evaluator_build,
        source_revision=None if manifest is None else manifest["source_revision"],
        source_digest=source_digest(),
        manifest_digest=None if manifest is None else manifest_digest(manifest),
    )


def unverified_identity(evaluator_build: str = ZERO_BUILD) -> ReleaseIdentity:
    """The development identity of a process that verified nothing.

    Test fixtures and in-process app assembly get this instead of an implied
    release identity; it reads no manifest and claims nothing.
    """
    return ReleaseIdentity(
        mode="dev",
        verified=False,
        reasons=("not-verified-at-startup",),
        service_version=service_version(),
        evaluator_build=evaluator_build,
        source_revision=None,
        source_digest=source_digest(),
        manifest_digest=None,
    )


def startup_identity(settings) -> ReleaseIdentity:
    """Verify the release on a formal startup path.

    Release mode refuses to serve anything it cannot verify. Development mode
    -- the default -- serves and says out loud which identities are unverified;
    a release-mode failure never falls back to it.
    """
    identity = release_identity(
        mode=settings.release_mode,
        manifest_path=settings.release_manifest,
        evaluator_build=settings.evaluator_build,
    )
    if identity.verified:
        logger.info(
            "release %s verified (manifest %s)",
            identity.service_version, identity.manifest_digest,
        )
        return identity
    if identity.mode == "release":
        raise ReleaseVerificationError(
            "release startup refused; unverified: " + ",".join(identity.reasons)
        )
    logger.warning(
        "development startup with unverified release identity: %s",
        ",".join(identity.reasons),
    )
    return identity
