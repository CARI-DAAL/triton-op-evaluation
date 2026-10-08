"""Durable HTTP endpoint tests (POST/GET semantics).

Shared helpers build a complete test application over fakeredis with a fake
clock and a real capability secret file. The other
HTTP suites import these helpers.
"""
import json
from pathlib import Path

import fakeredis
import httpx
import pytest
from conftest import FakeClock, make_completed_result

from triton_evaluation.api import create_app
from triton_evaluation.api.contract import ErrorEnvelope, EvaluationResource
from triton_evaluation.api.dependencies import AppContext
from triton_evaluation.config import Settings
from triton_evaluation.service.broker import EvaluationStore
from triton_evaluation.service.instance import EvaluationInstance
from triton_evaluation.service.models import AlgorithmVersions, WorkerIdentity, new_worker_id
from triton_evaluation.service.security import CapabilityRegistry

pytestmark = pytest.mark.anyio

TRIAL_TOKEN = "trial-token-1"
TRIAL_TOKEN_2 = "trial-token-2"
GRADE_TOKEN = "grade-token-1"
OBSERVER_TOKEN = "observer-token-1"


def make_test_settings(tmp_path):
    return Settings(
        engine_dir=str(tmp_path),
        cann_fingerprint="cann-test",
        heartbeat_interval_seconds=0.05,
        queue_poll_interval_seconds=0.05,
        api_poll_interval_seconds=0.05,
        maintenance_interval_seconds=5.0,
        capability_secret_path=str(tmp_path / "capabilities.json"),
        # Startup canary off: e2e protocol tests predate card health.
        card_canary_on_startup=False,
    )


def write_secret(path, capabilities, mode=0o600):
    path.write_text(json.dumps({"capabilities": capabilities}), encoding="utf-8")
    path.chmod(mode)
    return path


def default_capabilities():
    return [
        {
            "token": TRIAL_TOKEN,
            "caller_id": "trial-caller",
            "caller_class": "trial",
            "scopes": ["trial:submit", "trial:read"],
        },
        {
            "token": TRIAL_TOKEN_2,
            "caller_id": "trial-caller-2",
            "caller_class": "trial",
            "scopes": ["trial:submit", "trial:read"],
        },
        {
            "token": GRADE_TOKEN,
            "caller_id": "grade-caller",
            "caller_class": "trusted",
            "scopes": ["grade:submit", "grade:read", "observer:read"],
        },
        {
            "token": OBSERVER_TOKEN,
            "caller_id": "observer-caller",
            "caller_class": "observer",
            "scopes": ["observer:read"],
        },
    ]


def algorithms(settings):
    return AlgorithmVersions(
        correctness=settings.correctness_algorithm_version,
        decoy=settings.decoy_algorithm_version,
        benchmark=settings.benchmark_algorithm_version,
        profiler=settings.profiler_algorithm_version,
    )


def seed_worker(app, *, execution_class="npu"):
    """Heartbeat one compatible worker so new admissions pass readiness."""
    app.store.heartbeat(
        WorkerIdentity(new_worker_id(), app.settings.evaluator_build,
                       algorithms(app.settings), execution_class=execution_class)
    )


class BuiltApp:
    def __init__(self, app, store, clock, http, settings):
        self.app = app
        self.store = store
        self.clock = clock
        self.http = http
        self.settings = settings


def build_app(tmp_path, catalog_files, clock=None, settings=None):
    settings = settings or make_test_settings(tmp_path)
    write_secret(Path(settings.capability_secret_path), default_capabilities())
    server = fakeredis.FakeServer()
    client = fakeredis.FakeStrictRedis(server=server)
    clock = clock or FakeClock()
    from triton_evaluation.service.metrics import RedisMetrics

    metrics = RedisMetrics(client, settings.redis_prefix)
    store = EvaluationStore(client, settings, clock=clock, metrics=metrics)
    registry = CapabilityRegistry.from_secret_file(Path(settings.capability_secret_path))


    instance = EvaluationInstance(
        settings=settings, store=store,
        metrics=metrics,
    )
    context = AppContext(
        instance=instance,
        registry=registry,
    )
    app = create_app(context)
    transport = httpx.ASGITransport(app=app)
    http = httpx.AsyncClient(transport=transport, base_url="http://test")
    return BuiltApp(app, store, clock, http, settings)


def trial_headers(session="session-1", key="key-1", request_id=None):
    headers = {
        "Authorization": f"Bearer {TRIAL_TOKEN}",
        "X-Session-ID": session,
        "Idempotency-Key": key,
        "Content-Type": "application/json",
    }
    if request_id is not None:
        headers["X-Request-ID"] = request_id
    return headers


def grade_headers(session="session-g", key="key-g"):
    return {
        "Authorization": f"Bearer {GRADE_TOKEN}",
        "X-Session-ID": session,
        "Idempotency-Key": key,
        "Content-Type": "application/json",
    }


def request_payload(**overrides):
    payload = {
        "torch_code": "REF",
        "triton_code": "IMPL",
        "cases": [{"inputs": []}],
        "checks": ["correctness"],
    }
    payload.update(overrides)
    return payload


def internal_request(**overrides):
    from triton_evaluation.api.operator import OperatorRequest
    from triton_evaluation.evaluators.operator import to_evaluation_request

    return to_evaluation_request(OperatorRequest.model_validate(request_payload(**overrides)))


def raw_json(payload):
    return json.dumps(payload).encode()


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def app(tmp_path, release_catalog_files):
    return build_app(tmp_path, release_catalog_files)


async def post(client, payload=None, headers=None, **kwargs):
    return await client.post(
        "/evaluations",
        content=raw_json(payload if payload is not None else request_payload()),
        headers=headers if headers is not None else trial_headers(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# POST semantics
# ---------------------------------------------------------------------------


async def test_post_creates_queued_resource_with_location(app):
    seed_worker(app)
    async with app.http as client:
        response = await post(client)
    assert response.status_code == 202
    resource = EvaluationResource.model_validate_json(response.content)
    assert resource.state == "queued"
    assert response.headers["Location"] == f"/evaluations/{resource.evaluation_id}"
    assert resource.evaluation_id.startswith("ev_")


async def test_post_replay_same_key_returns_same_resource(app):
    seed_worker(app)
    async with app.http as client:
        first = await post(client)
        second = await post(client)
    assert first.status_code == second.status_code == 202
    assert first.json()["evaluation_id"] == second.json()["evaluation_id"]


async def test_terminal_replay_returns_200(app, make_command):
    from conftest import make_caller

    seed_worker(app)
    command = make_command(
        request=internal_request(),
        caller=make_caller("trial-caller", "trial", ("trial:submit", "trial:read"))
    )
    created = app.store.create_or_replay(command)
    lease = app.store.claim_next(
        WorkerIdentity(new_worker_id(), app.settings.evaluator_build,
                       algorithms(app.settings))
    )
    app.store.commit_terminal(lease, "completed", make_completed_result(app.settings))

    async with app.http as client:
        response = await post(client)
    assert response.status_code == 200
    assert response.json()["evaluation_id"] == created.resource.evaluation_id
    assert response.json()["state"] == "completed"


# ---------------------------------------------------------------------------
# GET semantics
# ---------------------------------------------------------------------------


async def test_get_returns_queued_resource_and_retry_after(app):
    seed_worker(app)
    async with app.http as client:
        created = await post(client)
        evaluation_id = created.json()["evaluation_id"]
        response = await client.get(
            f"/evaluations/{evaluation_id}", headers=trial_headers()
        )
    assert response.status_code == 200
    assert response.headers.get("Retry-After") == "1"
    assert response.json()["state"] == "queued"


async def test_get_terminal_resource_has_no_retry_after(app, make_command):
    from conftest import make_caller

    command = make_command(
        request=internal_request(),
        caller=make_caller("trial-caller", "trial", ("trial:submit", "trial:read"))
    )
    created = app.store.create_or_replay(command)
    lease = app.store.claim_next(
        WorkerIdentity(new_worker_id(), app.settings.evaluator_build,
                       algorithms(app.settings))
    )
    app.store.commit_terminal(lease, "completed", make_completed_result(app.settings))
    async with app.http as client:
        response = await client.get(
            f"/evaluations/{created.resource.evaluation_id}",
            headers=trial_headers(),
        )
    assert response.status_code == 200
    assert "Retry-After" not in response.headers
    assert response.json()["state"] == "completed"


async def test_unknown_id_returns_404_envelope(app):
    async with app.http as client:
        response = await client.get(
            f"/evaluations/ev_{'0' * 32}", headers=trial_headers()
        )
    assert response.status_code == 404
    envelope = ErrorEnvelope.model_validate_json(response.content)
    assert envelope.error.code == "NOT_FOUND"
    assert envelope.error.phase == "request"
    assert envelope.error.retryable is False
