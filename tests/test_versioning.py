"""Release identity: one version module, one manifest, verified startup.

The identities declared here must describe implemented behaviour, so these
tests also guard against announcing a contract or storage format that does not
exist yet.
"""
import json
import shutil

import pytest

from scripts import release_manifest
from triton_evaluation import versioning
from triton_evaluation.config import Settings

ROOT = release_manifest.PROJECT_ROOT
IMAGE = "registry.example.com/triton-eval@sha256:" + "a" * 64
BUILD = "sha256:" + "a" * 64
GIT_BUILD = "git:" + "c" * 40


@pytest.fixture
def manifest():
    return release_manifest.generate(ROOT, revision="d" * 40)


def bound(manifest, **overrides):
    return versioning.bind_build(
        manifest, evaluator_build=overrides.pop("evaluator_build", BUILD),
        image_reference=overrides.pop("image_reference", IMAGE),
    )


# ---------------------------------------------------------------------------
# One service release number, one client release number
# ---------------------------------------------------------------------------


def test_one_service_version_is_reported_by_package_manifest_and_openapi(tmp_path, manifest):
    from test_api import build_app

    declared = release_manifest.pyproject_version(ROOT)
    assert versioning.service_version() == declared
    assert manifest["service_version"] == declared
    app = build_app(tmp_path, None)
    assert app.app.openapi()["info"]["version"] == declared


def test_one_client_version_is_reported_by_every_manifest_and_the_offline_cli(manifest):
    versions = set(release_manifest.plugin_versions(ROOT).values())
    assert len(versions) == 1
    assert release_manifest.cli_version(ROOT) == versions.pop()
    assert manifest["client_version"] == release_manifest.client_version(ROOT)


def test_declared_version_sources_agree_with_the_installed_distribution():
    assert release_manifest.declaration_mismatches(ROOT) == ()
    assert versioning.declared_dependencies() == release_manifest.pyproject_dependencies(ROOT)


@pytest.mark.parametrize("tag, mismatched", [
    ("v" + release_manifest.pyproject_version(ROOT), False),
    (release_manifest.pyproject_version(ROOT), False),
    ("not-a-release-tag/1", False),
    ("v99.99.99", True),
])
def test_release_tags_on_head_must_name_the_declared_version(tag, mismatched):
    reasons = release_manifest.declaration_mismatches(ROOT, tags=[tag])
    assert ("release-tag-mismatch" in reasons) is mismatched


def test_client_version_drift_between_manifests_is_a_release_defect(tmp_path):
    shutil.copytree(ROOT / "plugins", tmp_path / "plugins")
    shutil.copy(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    drifted = tmp_path / "plugins/triton-eval/.claude-plugin/plugin.json"
    payload = json.loads(drifted.read_text())
    payload["version"] = "9.9.9"
    drifted.write_text(json.dumps(payload))
    assert "plugin-manifest-version-mismatch" in release_manifest.declaration_mismatches(tmp_path)
    with pytest.raises(ValueError, match="different client versions"):
        release_manifest.client_version(tmp_path)


# ---------------------------------------------------------------------------
# Deterministic generation, late build binding
# ---------------------------------------------------------------------------


def test_same_inputs_generate_the_same_canonical_manifest():
    first = release_manifest.generate(ROOT, revision="d" * 40)
    second = release_manifest.generate(ROOT, revision="d" * 40)
    assert versioning.canonical_manifest(first) == versioning.canonical_manifest(second)
    assert versioning.manifest_digest(first) == versioning.manifest_digest(second)
    assert release_manifest.generate(ROOT, revision=None)["source_revision"] is None


def test_the_build_is_bound_after_the_artifact_exists(manifest):
    assert manifest["evaluator_build"] is None
    assert manifest["image_reference"] is None
    container = bound(manifest)
    assert container["evaluator_build"] == BUILD
    assert container["image_reference"] == IMAGE
    host = versioning.bind_build(manifest, evaluator_build=GIT_BUILD)
    assert (host["evaluator_build"], host["image_reference"]) == (GIT_BUILD, None)
    # Binding changes nothing else about the described release.
    for key in set(manifest) - {"evaluator_build", "image_reference"}:
        assert container[key] == manifest[key]


def test_canonical_bytes_ignore_key_order(manifest):
    reordered = dict(reversed(list(manifest.items())))
    assert versioning.canonical_manifest(reordered) == versioning.canonical_manifest(manifest)


def test_manifest_round_trips_through_a_file(tmp_path, manifest):
    path = tmp_path / "release-manifest.json"
    path.write_bytes(versioning.canonical_manifest(bound(manifest)) + b"\n")
    assert versioning.load_manifest(path) == bound(manifest)


@pytest.mark.parametrize("payload", [
    b"", b"not json", b"[]", b"{}", b'{"schema_version": "1.0"}',
])
def test_unusable_manifests_are_refused(tmp_path, payload):
    path = tmp_path / "release-manifest.json"
    path.write_bytes(payload)
    with pytest.raises(versioning.ManifestError):
        versioning.load_manifest(path)


def test_absent_manifest_file_is_refused(tmp_path):
    with pytest.raises(versioning.ManifestError, match="manifest-unreadable"):
        versioning.load_manifest(tmp_path / "missing.json")


@pytest.mark.parametrize("field, value", [
    ("schema_version", "2.0"),
    ("service_version", ""),
    ("source_digest", "not-a-digest"),
    ("source_revision", "HEAD"),
    ("evaluator_build", "sha256:zz"),
    ("image_reference", "triton-eval:latest"),
    ("dependencies", {"fastapi": None}),
    ("api_versions", []),
    ("storage", {"read_versions": ["legacy"]}),
    ("capabilities", "version-discovery"),
    ("default_protocol", "pending"),
])
def test_malformed_manifest_fields_are_refused(manifest, field, value):
    with pytest.raises(versioning.ManifestError, match="manifest-malformed"):
        versioning.validate_manifest({**manifest, field: value})


def test_unknown_manifest_fields_are_refused(manifest):
    with pytest.raises(versioning.ManifestError):
        versioning.validate_manifest({**manifest, "protocol_lock": "3.0"})


# ---------------------------------------------------------------------------
# Only implemented identities are declared
# ---------------------------------------------------------------------------


def test_no_unimplemented_contract_protocol_or_storage_compatibility(manifest):
    assert manifest["api_versions"] == ["legacy"]
    assert manifest["default_api_version"] == "legacy"
    assert manifest["storage"] == {"read_versions": ["legacy"], "write_version": "legacy"}
    assert manifest["default_protocol"] is None
    assert "3.0" not in json.dumps(manifest)
    assert versioning.DEFAULT_PROTOCOL is None


# ---------------------------------------------------------------------------
# Startup verification: drift, zero build, modes
# ---------------------------------------------------------------------------


def write_manifest(tmp_path, manifest):
    path = tmp_path / "release-manifest.json"
    path.write_bytes(versioning.canonical_manifest(manifest) + b"\n")
    return path


def release_settings(tmp_path, manifest, *, build=BUILD, mode="release"):
    return Settings(
        release_mode=mode, release_manifest=write_manifest(tmp_path, manifest),
        evaluator_build=build,
    )


def test_matching_artifacts_verify_on_a_formal_startup_path(tmp_path, manifest):
    settings = release_settings(tmp_path, bound(manifest))
    identity = versioning.startup_identity(settings)
    assert identity.verified and identity.reasons == ()
    assert identity.mode == "release"
    assert identity.service_version == versioning.service_version()
    assert identity.source_digest == versioning.source_digest()
    assert identity.source_revision == "d" * 40
    assert identity.manifest_digest == versioning.manifest_digest(bound(manifest))


@pytest.mark.parametrize("field, value, reason", [
    ("service_version", "99.0.0", "service-version-mismatch"),
    ("source_digest", "b" * 64, "source-digest-mismatch"),
    ("api_versions", ["3.0"], "api-versions-mismatch"),
    ("default_protocol", "e" * 64, "protocol-mismatch"),
    ("capabilities", ["evaluation-submit"], "capabilities-mismatch"),
    ("evaluator_build", "sha256:" + "f" * 64, "evaluator-build-mismatch"),
])
def test_tampered_manifest_fails_release_startup_with_a_locatable_reason(
    tmp_path, manifest, field, value, reason,
):
    settings = release_settings(tmp_path, {**bound(manifest), field: value})
    with pytest.raises(versioning.ReleaseVerificationError, match=reason):
        versioning.startup_identity(settings)


@pytest.mark.parametrize("storage, reason", [
    ({"read_versions": ["legacy", "3.0"], "write_version": "legacy"},
     "storage-versions-mismatch"),
    ({"read_versions": ["legacy"], "write_version": "3.0"}, "storage-versions-mismatch"),
])
def test_unsupported_storage_claims_fail_release_startup(tmp_path, manifest, storage, reason):
    settings = release_settings(tmp_path, {**bound(manifest), "storage": storage})
    with pytest.raises(versioning.ReleaseVerificationError, match=reason):
        versioning.startup_identity(settings)


def test_tampered_dependency_identity_fails_release_startup(tmp_path, manifest):
    dependencies = dict(manifest["dependencies"])
    dependencies["fastapi"] = "0.0.1"
    settings = release_settings(tmp_path, {**bound(manifest), "dependencies": dependencies})
    with pytest.raises(versioning.ReleaseVerificationError, match="dependency-mismatch"):
        versioning.startup_identity(settings)


def test_changed_sources_change_the_digest_they_are_verified_against(tmp_path):
    copied = tmp_path / "triton_evaluation"
    shutil.copytree(ROOT / "triton_evaluation", copied)
    before = versioning.digest_sources(copied)
    assert before == versioning.source_digest()
    (copied / "versioning.py").write_text(
        (copied / "versioning.py").read_text() + "\n# drift\n", encoding="utf-8"
    )
    assert versioning.digest_sources(copied) != before


def test_byte_compilation_output_is_not_source(tmp_path):
    copied = tmp_path / "triton_evaluation"
    shutil.copytree(ROOT / "triton_evaluation", copied)
    cache = copied / "__pycache__"
    cache.mkdir(exist_ok=True)
    (cache / "versioning.cpython-310.py").write_text("# not source\n", encoding="utf-8")
    assert versioning.digest_sources(copied) == versioning.source_digest()


def test_unbound_build_cannot_verify_a_release(tmp_path, manifest):
    settings = release_settings(tmp_path, manifest)
    with pytest.raises(versioning.ReleaseVerificationError, match="evaluator-build-unbound"):
        versioning.startup_identity(settings)


def test_all_zero_build_cannot_verify_a_release(tmp_path, manifest):
    settings = release_settings(
        tmp_path, bound(manifest, evaluator_build=versioning.ZERO_BUILD),
        build=versioning.ZERO_BUILD,
    )
    with pytest.raises(versioning.ReleaseVerificationError, match="zero-evaluator-build"):
        versioning.startup_identity(settings)


def test_release_mode_requires_a_manifest_and_never_degrades_to_dev():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="release mode requires"):
        Settings(release_mode="release")
    with pytest.raises(ValidationError, match="absolute without traversal"):
        Settings(release_mode="release", release_manifest="relative/manifest.json")


def test_development_startup_serves_and_names_its_unverified_identities(caplog):
    identity = versioning.startup_identity(Settings())
    assert identity.mode == "dev"
    assert identity.verified is False
    assert "manifest-absent" in identity.reasons
    assert "zero-evaluator-build" in identity.reasons
    assert "unverified release identity" in caplog.text


def test_in_process_assembly_claims_no_release_identity():
    identity = versioning.unverified_identity()
    assert (identity.mode, identity.verified) == ("dev", False)
    assert identity.manifest_digest is None
    assert identity.source_revision is None
    assert identity.reasons == ("not-verified-at-startup",)


# ---------------------------------------------------------------------------
# Formal startup paths
# ---------------------------------------------------------------------------


@pytest.fixture
def unverifiable(tmp_path, manifest, monkeypatch):
    """A release-mode deployment whose manifest does not describe it."""
    settings = release_settings(
        tmp_path, bound(manifest, evaluator_build="sha256:" + "f" * 64),
    )
    monkeypatch.setattr(Settings, "from_env", lambda: settings)
    return settings


def start_api():
    """Load the module uvicorn serves, the way uvicorn loads it."""
    import runpy

    return runpy.run_module("scripts.run_api", run_name="release_verification_test")


def test_api_startup_refuses_an_unverifiable_release(unverifiable, monkeypatch):
    import redis

    monkeypatch.setattr(
        redis.Redis, "from_url", lambda *a, **k: pytest.fail("verification must come first")
    )
    with pytest.raises(versioning.ReleaseVerificationError, match="evaluator-build-mismatch"):
        start_api()


def test_worker_startup_refuses_an_unverifiable_release(unverifiable, monkeypatch):
    from scripts import run_worker

    monkeypatch.setattr("sys.argv", ["run_worker.py", "--cpu"])
    monkeypatch.setattr(run_worker, "set_process_title", lambda: None)
    monkeypatch.setattr(
        run_worker, "build_worker", lambda *a, **k: pytest.fail("verification must come first")
    )
    with pytest.raises(versioning.ReleaseVerificationError, match="evaluator-build-mismatch"):
        run_worker.main()


def test_verified_release_startup_builds_the_api_with_its_identity(
    tmp_path, manifest, monkeypatch,
):
    import fakeredis
    import redis

    from triton_evaluation.service.security import CapabilityRegistry

    build = "git:" + "c" * 40
    settings = release_settings(
        tmp_path, bound(manifest, evaluator_build=build), build=build,
    ).model_copy(update={"capability_secret_path": tmp_path / "registry.json"})
    monkeypatch.setattr(Settings, "from_env", lambda: settings)
    monkeypatch.setattr(redis.Redis, "from_url", lambda *a, **k: fakeredis.FakeRedis())
    monkeypatch.setattr(CapabilityRegistry, "from_secret_file", lambda _path: object())
    identity = start_api()["app"].state.context.release
    assert identity.verified and identity.mode == "release"
    assert identity.evaluator_build == build


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def test_generate_bind_verify_round_trip_from_the_command_line(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("TRITON_EVAL_EVALUATOR_BUILD", GIT_BUILD)
    generated = tmp_path / "generated.json"
    assert release_manifest.main(
        ["generate", "--source-revision", "d" * 40, "--output", str(generated)]
    ) == 0
    bound_path = tmp_path / "bound.json"
    assert release_manifest.main(
        ["bind", "--manifest", str(generated), "--build", GIT_BUILD,
         "--output", str(bound_path)]
    ) == 0
    assert release_manifest.main(["verify", "--manifest", str(bound_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == {
        "verified": True,
        "unverified_reasons": [],
        "service_version": versioning.service_version(),
        "source_digest": versioning.source_digest(),
        "manifest_digest": versioning.manifest_digest(
            versioning.load_manifest(bound_path)
        ),
    }


def test_command_line_verify_reports_drift_without_raising(tmp_path, capsys):
    manifest = bound(release_manifest.generate(ROOT, revision=None))
    path = write_manifest(tmp_path, {**manifest, "source_digest": "b" * 64})
    assert release_manifest.main(["verify", "--manifest", str(path)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["verified"] is False
    assert "source-digest-mismatch" in report["unverified_reasons"]


def test_command_line_consistency_reports_agreement(capsys):
    assert release_manifest.main(["consistency"]) == 0
    assert json.loads(capsys.readouterr().out) == {"consistent": True, "mismatches": []}


def test_command_line_refuses_an_unusable_manifest(tmp_path, capsys):
    path = tmp_path / "release-manifest.json"
    path.write_bytes(b"{}")
    assert release_manifest.main(["bind", "--manifest", str(path), "--image", IMAGE]) == 1
    assert "failed" in capsys.readouterr().err


def test_command_line_bind_requires_an_artifact_identity(tmp_path, manifest):
    path = write_manifest(tmp_path, manifest)
    with pytest.raises(SystemExit) as exit_request:
        release_manifest.main(["bind", "--manifest", str(path)])
    assert exit_request.value.code == 2
