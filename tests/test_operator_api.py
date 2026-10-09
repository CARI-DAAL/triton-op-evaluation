"""Operator tools share durable submission, capability auth, and recovery."""

import asyncio

import pytest
from test_api import (
    OBSERVER_TOKEN,
    TRIAL_TOKEN,
    build_app,
    request_payload,
    seed_worker,
    trial_headers,
)
from test_durable_http import claim_and_complete

pytestmark = pytest.mark.anyio


@pytest.fixture
def app(tmp_path, release_catalog_files):
    return build_app(tmp_path, release_catalog_files)


@pytest.mark.parametrize("path,checks", [
    ("/evaluations", ["correctness"]),
    ("/verify", ["correctness"]),
    ("/benchmark", ["correctness", "performance"]),
    ("/profile", ["profile"]),
])
async def test_all_entrypoints_submit_and_recover_same_resource(app, path, checks):
    seed_worker(app)
    async with app.http as client:
        first = await client.post(path, json=request_payload(), headers=trial_headers())
        assert first.status_code == 202, first.text
        replay = await client.post(path, json=request_payload(), headers=trial_headers())
        assert replay.json() == first.json()
        recovered = await client.get(first.headers["Location"], headers=trial_headers())
        assert recovered.json() == first.json()
    identity = app.store._r.hgetall(
        f"{app.settings.redis_prefix}:resource:{first.json()['evaluation_id']}"
    )
    import json

    assert json.loads(identity[b"canonical_request"])["checks"] == checks


async def test_missing_optional_headers_create_independent_resources(app):
    seed_worker(app)
    async with app.http as client:
        headers = {"Authorization": f"Bearer {TRIAL_TOKEN}"}
        first = await client.post("/verify", json=request_payload(), headers=headers)
        second = await client.post("/verify", json=request_payload(), headers=headers)
    assert first.status_code == second.status_code == 202
    assert first.json()["evaluation_id"] != second.json()["evaluation_id"]


async def test_key_without_session_replays_across_tool_and_evaluations(app):
    seed_worker(app)
    headers = {"Authorization": f"Bearer {TRIAL_TOKEN}", "Idempotency-Key": "retry"}
    async with app.http as client:
        first = await client.post("/verify", json=request_payload(), headers=headers)
        replay = await client.post("/evaluations", json=request_payload(), headers=headers)
    assert first.status_code == replay.status_code == 202
    assert first.json()["evaluation_id"] == replay.json()["evaluation_id"]


async def test_tool_wait_returns_terminal_and_get_recovers_it(app):
    seed_worker(app)
    async with app.http as client:
        first = await client.post("/verify", json=request_payload(), headers=trial_headers())
        pending = asyncio.create_task(client.post(
            "/verify?wait_seconds=5", json=request_payload(), headers=trial_headers(),
        ))
        await asyncio.sleep(0.05)
        claim_and_complete(app)
        completed = await pending
        recovered = await client.get(first.headers["Location"], headers=trial_headers())
    assert completed.status_code == 200
    assert completed.json()["state"] == "completed"
    assert recovered.json() == completed.json()


@pytest.mark.parametrize("path", ["/evaluations", "/verify", "/benchmark", "/profile"])
async def test_all_entrypoints_require_capability_submit_scope(app, path):
    async with app.http as client:
        missing = await client.post(path, json=request_payload())
        observer = await client.post(path, json=request_payload(), headers={
            "Authorization": f"Bearer {OBSERVER_TOKEN}",
        })
    assert missing.status_code == 401
    assert observer.status_code == 403


@pytest.mark.parametrize("field,value", [
    ("purpose", "grade"), ("task_id", "old-task"), ("token", "old-token"),
    ("grade_context", {}), ("reference_code", "REF"), ("kernel_code", "IMPL"),
    ("torch_code", "REF"), ("cases", [{"inputs": []}]),
])
async def test_retired_request_fields_are_rejected(app, field, value):
    async with app.http as client:
        response = await client.post(
            "/verify", json=request_payload(**{field: value}), headers=trial_headers(),
        )
    assert response.status_code == 422


async def test_health_alias_has_same_readiness_as_readyz(app):
    async with app.http as client:
        assert (await client.get("/health")).status_code == 503
        seed_worker(app)
        seed_worker(app, execution_class="cpu")
        health = await client.get("/health")
        ready = await client.get("/readyz")
    assert health.status_code == ready.status_code == 200
    assert health.json() == ready.json() == {"ready": True}


async def test_profile_options_use_endpoint_checks_and_evaluation_rejects_mismatch(app):
    seed_worker(app)
    body = request_payload(profile={"mode": "deep"})
    async with app.http as client:
        profile = await client.post("/profile", json=body, headers=trial_headers())
        invalid = await client.post("/evaluations", json=body, headers=trial_headers(key="invalid"))
    assert profile.status_code == 202, profile.text
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "INVALID_REQUEST"
    assert "REF" not in invalid.text


async def test_mounted_tool_uses_its_declared_checks(app):
    import json

    import httpx
    from fastapi import FastAPI

    seed_worker(app)
    parent = FastAPI()
    parent.mount("/operator", app.app)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=parent), base_url="http://test",
    ) as client:
        response = await client.post(
            "/operator/benchmark", json=request_payload(), headers=trial_headers(),
        )
        assert response.headers["Location"].startswith("/operator/evaluations/")
        recovered = await client.get(response.headers["Location"], headers=trial_headers())
        assert recovered.json() == response.json()
    assert response.status_code == 202, response.text
    canonical = app.store._r.hget(
        f"{app.settings.redis_prefix}:resource:{response.json()['evaluation_id']}",
        "canonical_request",
    )
    assert json.loads(canonical)["checks"] == ["correctness", "performance"]
