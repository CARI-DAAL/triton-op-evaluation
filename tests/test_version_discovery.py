"""Version discovery over HTTP and from the standalone client.

Callers identify the running release before submitting anything, so discovery
must answer while no worker can serve and must not leak deployment settings.
"""
import json
import subprocess
import sys

import pytest
from test_api import (
    OBSERVER_TOKEN,
    TRIAL_TOKEN,
    build_app,
    make_test_settings,
    post,
    trial_headers,
)
from test_cli_plugin import SCRIPT, TOKEN, server

from triton_evaluation import versioning
from triton_evaluation.api.dependencies import AppContext
from triton_evaluation.config import Settings

pytestmark = pytest.mark.anyio


def auth(token=TRIAL_TOKEN):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def app(tmp_path):
    return build_app(tmp_path, None)


async def test_version_reports_the_running_release_to_any_authenticated_caller(app):
    async with app.http as client:
        for token in (TRIAL_TOKEN, OBSERVER_TOKEN):
            response = await client.get("/version", headers=auth(token))
            assert response.status_code == 200
            document = response.json()
    assert document["schema_version"] == "2.1"
    assert document["service_version"] == versioning.service_version()
    assert document["api_versions"] == ["legacy"]
    assert document["default_api_version"] == "legacy"
    assert document["evaluator_build"] == app.settings.evaluator_build
    assert document["storage"] == {"read_versions": ["legacy"], "write_version": "legacy"}
    assert document["default_protocol"] is None
    assert document["capabilities"] == list(versioning.CAPABILITIES)


async def test_version_requires_a_capability_and_hides_deployment_configuration(app):
    async with app.http as client:
        assert (await client.get("/version")).status_code == 401
        assert (
            await client.get("/version", headers={"Authorization": "Bearer wrong"})
        ).status_code == 401
        document = (await client.get("/version", headers=auth())).json()
    serialized = json.dumps(document)
    assert app.settings.redis_url not in serialized
    assert str(app.settings.capability_secret_path) not in serialized
    assert app.settings.worker_toolchain_version not in serialized
    assert set(document["release"]) == {
        "mode", "verified", "unverified_reasons", "manifest_digest",
        "source_revision", "source_digest",
    }


async def test_identity_is_discoverable_while_no_worker_can_serve(app):
    async with app.http as client:
        # Admission has no compatible worker: submission fails, discovery does not.
        refused = await post(client, headers=trial_headers())
        assert refused.status_code == 503
        assert (await client.get("/readyz")).status_code == 503
        assert (await client.get("/version", headers=auth())).status_code == 200


async def test_in_process_assembly_reports_an_unverified_development_identity(app):
    async with app.http as client:
        release = (await client.get("/version", headers=auth())).json()["release"]
    assert release["mode"] == "dev"
    assert release["verified"] is False
    assert release["unverified_reasons"] == ["not-verified-at-startup"]
    assert release["manifest_digest"] is None
    assert release["source_digest"] == versioning.source_digest()


async def test_verified_startup_identity_is_reported_as_verified(tmp_path):
    from scripts import release_manifest

    manifest = versioning.bind_build(
        release_manifest.generate(release_manifest.PROJECT_ROOT, revision="d" * 40),
        evaluator_build="git:" + "c" * 40,
    )
    path = tmp_path / "release-manifest.json"
    path.write_bytes(versioning.canonical_manifest(manifest) + b"\n")
    settings = make_test_settings(tmp_path).model_copy(
        update={"release_mode": "release", "release_manifest": path,
                "evaluator_build": "git:" + "c" * 40},
    )
    app = build_app(tmp_path, None, settings=settings)
    app.app.state.context = AppContext(
        instance=app.app.state.context.instance,
        registry=app.app.state.context.registry,
        release=versioning.startup_identity(settings),
    )
    async with app.http as client:
        document = (await client.get("/version", headers=auth())).json()
    assert document["release"] == {
        "mode": "release",
        "verified": True,
        "unverified_reasons": [],
        "manifest_digest": versioning.manifest_digest(manifest),
        "source_revision": "d" * 40,
        "source_digest": versioning.source_digest(),
    }
    assert document["evaluator_build"] == "git:" + "c" * 40


async def test_live_spec_documents_version_discovery(app):
    async with app.http as client:
        document = (await client.get("/openapi.json")).json()
        response = await client.get("/version", headers=auth())
    operation = document["paths"]["/version"]["get"]
    assert operation["security"] == [{"CapabilityBearer": []}]
    schema = operation["responses"]["200"]["content"]["application/json"]["schema"]
    assert schema
    from jsonschema import Draft202012Validator

    Draft202012Validator(
        {**schema, "components": document["components"]}
    ).validate(response.json())
    for status in (401, 403, 413, 500):
        assert operation["responses"][str(status)]["content"]["application/json"]["schema"]


# ---------------------------------------------------------------------------
# Standalone client
# ---------------------------------------------------------------------------


def run_cli(*arguments, cwd=None):
    return subprocess.run(
        [sys.executable, "-I", "-S", str(SCRIPT), *arguments],
        cwd=cwd, capture_output=True, text=True, timeout=10,
    )


def version_document(**overrides):
    return {
        "schema_version": "2.1",
        "service_version": "0.1.0",
        "api_versions": ["legacy"],
        "default_api_version": "legacy",
        "evaluator_build": versioning.ZERO_BUILD,
        "storage": {"read_versions": ["legacy"], "write_version": "legacy"},
        "default_protocol": None,
        "capabilities": list(versioning.CAPABILITIES),
        "release": {
            "mode": "dev", "verified": False, "unverified_reasons": ["manifest-absent"],
            "manifest_digest": None, "source_revision": None, "source_digest": "a" * 64,
        },
    } | overrides


def test_copied_client_reports_its_own_version_offline(tmp_path):
    copied = tmp_path / "triton_eval.py"
    copied.write_bytes(SCRIPT.read_bytes())
    result = run_cli("--version", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    declared = json.loads(result.stdout)
    assert declared == {"client_version": "0.4.0"}
    isolated = subprocess.run(
        [sys.executable, "-I", "-S", str(copied), "--version"],
        cwd=tmp_path, capture_output=True, text=True, timeout=10,
    )
    assert json.loads(isolated.stdout) == declared


def test_client_queries_the_service_version_with_its_capability(tmp_path):
    token = tmp_path / "capability"
    token.write_text(TOKEN + "\n", encoding="utf-8")
    token.chmod(0o600)
    with server(lambda calls: (200, version_document(), {})) as (url, calls):
        result = run_cli(
            "version", "--url", url, "--capability-file", str(token), cwd=tmp_path
        )
    assert result.returncode == 0, result.stderr
    method, path, headers, _ = calls[0]
    assert (method, path) == ("GET", "/version")
    assert headers["Authorization"] == f"Bearer {TOKEN}"
    document = json.loads(result.stdout)
    assert document["client_version"] == "0.4.0"
    assert document["service"]["default_api_version"] == "legacy"
    assert TOKEN not in result.stdout and TOKEN not in result.stderr


def test_client_tolerates_added_fields_but_not_a_missing_contract(tmp_path):
    token = tmp_path / "capability"
    token.write_text(TOKEN + "\n", encoding="utf-8")
    token.chmod(0o600)
    with server(lambda calls: (200, version_document(future_field=[1]), {})) as (url, _):
        accepted = run_cli(
            "version", "--url", url, "--capability-file", str(token), cwd=tmp_path
        )
    assert accepted.returncode == 0, accepted.stderr
    broken = version_document()
    del broken["api_versions"]
    with server(lambda calls: (200, broken, {})) as (url, _):
        refused = run_cli(
            "version", "--url", url, "--capability-file", str(token), cwd=tmp_path
        )
    assert refused.returncode == 2
    assert json.loads(refused.stdout)["error"]["code"] == "INVALID_RESPONSE"


def test_client_version_command_requires_a_capability_file(tmp_path):
    with server(lambda calls: (200, version_document(), {})) as (url, calls):
        result = run_cli("version", "--url", url, cwd=tmp_path)
    assert result.returncode == 2
    assert json.loads(result.stdout)["error"]["code"] == "INVALID_INPUT"
    assert calls == []


def test_settings_accept_the_release_inputs_from_the_environment(monkeypatch, tmp_path):
    manifest = tmp_path / "release-manifest.json"
    monkeypatch.setenv("TRITON_EVAL_RELEASE_MODE", "release")
    monkeypatch.setenv("TRITON_EVAL_RELEASE_MANIFEST", str(manifest))
    settings = Settings.from_env()
    assert settings.release_mode == "release"
    assert settings.release_manifest == manifest
