"""End-to-end durable flow: real app, real store, stubbed engine evaluator."""
import asyncio
from pathlib import Path

import pytest
from conftest import make_caller, make_completed_result
from test_api import (
    algorithms,
    build_app,
    post,
    seed_worker,
    trial_headers,
)

from triton_evaluation.api.contract import EvaluationResource
from triton_evaluation.worker import Worker

pytestmark = pytest.mark.anyio


class CompletingEvaluator:
    def __init__(self, settings):
        self.settings = settings

    def run(self, request, work_dir, device_id, deadline_at, lease_valid,
            execution_environment, pinned, progress, *, precheck=None):
        return make_completed_result(self.settings)


@pytest.fixture
def app(tmp_path, release_catalog_files):
    return build_app(tmp_path, release_catalog_files)


def run_worker_once(app):
    worker = Worker(
        app.store, CompletingEvaluator(app.settings), device_id=0,
        settings=app.settings, clock=app.clock, sleep=lambda _s: None,
    )
    assert worker.run_once() is True


async def test_full_durable_round_trip(app):
    seed_worker(app)
    async with app.http as client:
        created = await post(client)
        evaluation_id = created.json()["evaluation_id"]
        assert created.status_code == 202

        # Worker claims the admitted resource and fenced-commits a result.
        await asyncio.to_thread(run_worker_once, app)

        recovered = await client.get(
            f"/evaluations/{evaluation_id}", headers=trial_headers()
        )
        assert recovered.status_code == 200
        assert recovered.json()["state"] == "completed"
        assert recovered.json()["result"]["correctness"]["status"] == "succeeded"

        # Same key + same request replays the terminal resource as 200.
        replay = await post(client)
        assert replay.status_code == 200
        assert replay.json()["evaluation_id"] == evaluation_id


async def test_api_object_recreation_recovers_by_id(app):
    """A new API process over the same store data serves the same ID."""
    seed_worker(app)
    async with app.http as client:
        created = await post(client)
        evaluation_id = created.json()["evaluation_id"]
    await asyncio.to_thread(run_worker_once, app)

    # Recreate the app object over the SAME fakeredis server + clock.
    second = _rebuild(app)
    async with second.http as client:
        recovered = await client.get(
            f"/evaluations/{evaluation_id}", headers=trial_headers()
        )
    assert recovered.status_code == 200
    assert recovered.json()["state"] == "completed"


def _rebuild(app):
    import httpx

    from triton_evaluation.api import create_app
    from triton_evaluation.broker import EvaluationStore
    from triton_evaluation.metrics import RedisMetrics
    from triton_evaluation.security import CapabilityRegistry

    metrics = RedisMetrics(app.store._r, app.settings.redis_prefix)
    store = EvaluationStore(app.store._r, app.settings, clock=app.clock,
                            metrics=metrics)
    from triton_evaluation.api.dependencies import AppContext
    from triton_evaluation.instance import EvaluationInstance

    context = AppContext(
        instance=EvaluationInstance(
            settings=app.settings, store=store,
            catalog=app.app.state.context.instance.catalog, metrics=metrics,
            artifacts_verified=True,
        ),
        registry=CapabilityRegistry.from_secret_file(
            Path(app.settings.capability_secret_path)
        ),
    )
    application = create_app(context)
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=application),
                             base_url="http://test")
    return type("B", (), {"http": http, "store": store, "clock": app.clock,
                          "settings": app.settings})()


async def test_worker_replacement_after_lease_expiry(app):
    seed_worker(app)
    async with app.http as client:
        created = await post(client)
        evaluation_id = created.json()["evaluation_id"]

    # First worker claims, then dies (no commit). The lease expires.
    lease = app.store.claim_next(
        _identity(app)
    )
    assert lease is not None
    app.clock.advance_ms(app.settings.lease_seconds * 1000 + 1)
    assert app.store.recover_expired_leases().requeued == 1

    # A replacement worker commits; the abandoned token cannot mutate.
    await asyncio.to_thread(run_worker_once, app)
    resource = app.store.get_visible(evaluation_id, _trial_caller())
    assert resource.state == "completed"
    assert app.store.commit_terminal(
        lease, "failed", make_completed_result(app.settings)
    ) is False
    assert app.store.get_visible(evaluation_id, _trial_caller()).revision \
        == resource.revision


def _identity(app):
    from triton_evaluation.models import WorkerIdentity, new_worker_id

    return WorkerIdentity(new_worker_id(), app.settings.evaluator_build,
                          algorithms(app.settings))


def _trial_caller():
    return make_caller("trial-caller", "trial", ("trial:submit", "trial:read"))


async def test_deadline_terminalizes_without_any_worker(app):
    seed_worker(app)  # admission passes; nothing ever claims the resource
    async with app.http as client:
        created = await post(client)
        evaluation_id = created.json()["evaluation_id"]
    app.clock.advance_ms(3600 * 1000 + 1)
    app.store.expire_due_resources(limit=10)
    resource = app.store.get_visible(evaluation_id, _trial_caller())
    assert resource.state == "timed_out"
    assert resource.result.correctness.error.code == "DEADLINE_EXCEEDED"


async def test_cross_caller_hiding_end_to_end(app):
    seed_worker(app)
    async with app.http as client:
        created = await post(client)
        evaluation_id = created.json()["evaluation_id"]
        other = await client.get(
            f"/evaluations/{evaluation_id}",
            headers=trial_headers() | {"Authorization": "Bearer trial-token-2"},
        )
        owner = await client.get(
            f"/evaluations/{evaluation_id}", headers=trial_headers()
        )
    assert other.status_code == 404
    assert owner.status_code == 200
    assert EvaluationResource.model_validate_json(
        owner.content
    ).evaluation_id == evaluation_id
