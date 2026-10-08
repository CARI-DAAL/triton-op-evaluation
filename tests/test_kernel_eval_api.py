"""HTTP adapter tests: authentication, decoding, session mapping and wire responses."""
import asyncio
import json

import pytest
from native_eval_helpers import TASK_ID, TOKEN, official_catalog, outcome
from test_api import algorithms, build_app, make_test_settings, seed_worker, trial_headers

from triton_evaluation.kernel_evaluation import TOOL_CALLER
from triton_evaluation.models import WorkerIdentity, new_worker_id

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def build_native_app(tmp_path, **overrides):
    files, task = official_catalog(tmp_path)
    settings = make_test_settings(tmp_path)
    settings = type(settings).model_validate({
        **settings.model_dump(), "kernel_eval_token": TOKEN,
        "kernel_eval_enabled": True, "kernel_eval_auto_diagnostics": False, **overrides,
    })
    app = build_app(tmp_path, files, settings=settings)
    app.task = task
    return app


@pytest.fixture
def native_app(tmp_path):
    return build_native_app(tmp_path)


def payload(**overrides):
    return {"task_id": TASK_ID, "kernel_code": "KERNEL", "session": "trial-1", **overrides}


def headers(key=None):
    return {"Authorization": f"Bearer {TOKEN}", **({"Idempotency-Key": key} if key else {})}


async def complete_one(app, request_task, name="verify_correct"):
    worker = WorkerIdentity(new_worker_id(), app.settings.evaluator_build, algorithms(app.settings))
    for _ in range(500):
        lease = await asyncio.to_thread(app.store.claim_next, worker)
        if lease is not None:
            result = outcome(name, app.settings)
            assert await asyncio.to_thread(app.store.commit_terminal, lease, "completed", result)
            return lease
        if request_task.done():
            response = await request_task
            pytest.fail(f"request finished before durable work was admitted: {response.text}")
        await asyncio.sleep(0.005)
    request_task.cancel()
    pytest.fail("test worker could not claim a request")


@pytest.mark.parametrize("route", ["verify", "benchmark", "profile"])
async def test_single_token_and_legacy_refusals(native_app, route):
    app = native_app
    async with app.http as client:
        unauthorized = await client.post(f"/{route}", content="not json")
        assert (unauthorized.status_code, unauthorized.json()) == (401, {"error": "unauthorized"})
        v21_token = await client.post(f"/{route}", json=payload(), headers=trial_headers())
        assert v21_token.status_code == 401
        malformed = await client.post(f"/{route}", content="not json", headers=headers())
        assert (malformed.status_code, malformed.json()) == (400, {"error": "invalid json"})
        reference = await client.post(
            f"/{route}", json=payload(reference_code="ATTACK"), headers=headers()
        )
        assert (reference.status_code, reference.json()) == (
            400,
            {"error": "reference fields are forbidden"},
        )
        missing = await client.post(f"/{route}", json={"task_id": TASK_ID}, headers=headers())
        assert missing.json() == {"error": "task_id and kernel_code are required"}
        mismatch = await client.post(f"/{route}", json=payload(task_id="other"), headers=headers())
        assert (mismatch.status_code, mismatch.json()) == (403, {"error": "task_id mismatch"})
    assert not app.store._r.keys(f"{app.settings.redis_prefix}:kernel-eval:budget:*")


@pytest.mark.parametrize("changes", [{"kernel_eval_enabled": False}, {"kernel_eval_token": ""}])
async def test_disabled_or_empty_token_fails_closed(native_app, changes):
    from pydantic import SecretStr

    for name, value in changes.items():
        setattr(native_app.settings, name, SecretStr(value) if name.endswith("token") else value)
    async with native_app.http as client:
        response = await client.post("/verify", json=payload(), headers=headers())
        health = await client.get("/health")
    assert response.status_code == health.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


@pytest.mark.parametrize(
    "route,name,checks",
    [
        ("verify", "verify_correct", ["correctness"]),
        ("benchmark", "benchmark_measured", ["correctness", "performance"]),
        ("profile", "profile_complete", ["profile"]),
    ],
)
async def test_native_catalog_admission_polling_and_owned_replay(native_app, route, name, checks):
    app = native_app
    seed_worker(app)
    async with app.http as client:
        pending = asyncio.create_task(
            client.post(f"/{route}", json=payload(), headers=headers("call-1"))
        )
        lease = await complete_one(app, pending, name)
        first = await pending
        replay = await client.post(f"/{route}", json=payload(), headers=headers("call-1"))
        invisible = await client.get(f"/evaluations/{lease.evaluation_id}", headers=trial_headers())
        direct_token = await client.get(f"/evaluations/{lease.evaluation_id}", headers=headers())
    canonical = json.loads(lease.canonical_request)
    assert canonical["reference_code"] == app.task["reference_source"]
    assert canonical["op_name"] == "Demo"
    assert canonical["purpose"] == "trial"
    assert canonical["checks"] == checks
    if route == "benchmark":
        assert canonical["performance"] == {
            "warmup": 5, "repeats": 20, "method": "profiler", "case_indices": None,
        }
    assert first.status_code == replay.status_code == 200
    assert first.json() == replay.json()
    assert first.json()["eval_budget"] == "1/18"
    assert first.json()["request_status"] == "completed"
    assert invisible.status_code == 404
    assert direct_token.status_code == 401  # the tool token is confined to its facade
    assert app.store.get_visible(lease.evaluation_id, TOOL_CALLER).state == "completed"
    assert (
        app.store.claim_next(
            WorkerIdentity(new_worker_id(), app.settings.evaluator_build, algorithms(app.settings))
        )
        is None
    )


async def test_no_session_uses_legacy_remote_and_explicit_trial_is_separate(tmp_path):
    app = build_native_app(tmp_path, kernel_eval_max_turns=1)
    seed_worker(app)
    body = payload()
    body.pop("session")
    async with app.http as client:
        pending = asyncio.create_task(
            client.post("/verify", json=body, headers=headers("legacy-1"))
        )
        await complete_one(app, pending)
        await pending
        blocked = await client.post("/verify", json=body, headers=headers("legacy-2"))
        explicit = asyncio.create_task(
            client.post("/verify", json=payload(), headers=headers("legacy-2"))
        )
        await complete_one(app, explicit)
        independent = await explicit
    assert blocked.json()["request_status"] == "budget_exhausted"
    assert independent.json()["request_status"] == "completed"



async def test_session_header_mapping_and_conflict(native_app):
    seed_worker(native_app)
    body = {"task_id": TASK_ID, "kernel_code": "KERNEL"}
    session_headers = {**headers("header-session"), "X-Session-ID": "from-header"}
    async with native_app.http as client:
        pending = asyncio.create_task(client.post("/verify", json=body, headers=session_headers))
        await complete_one(native_app, pending)
        first = await pending
        replay = await client.post(
            "/verify", json={**body, "session": "from-header"}, headers=headers("header-session")
        )
        mismatch = await client.post("/verify", json=payload(), headers=session_headers)
    assert first.json() == replay.json()
    assert (mismatch.status_code, mismatch.json()) == (400, {"error": "invalid session"})


async def test_operation_conflict_maps_to_http_409(native_app):
    # Lack of workers still reserves a durable turn before readiness rejection.
    async with native_app.http as client:
        await client.post("/verify", json=payload(), headers=headers("same-operation"))
        response = await client.post(
            "/verify", json=payload(kernel_code="CHANGED"), headers=headers("same-operation")
        )
    assert (response.status_code, response.json()) == (409, {"error": "idempotency conflict"})
