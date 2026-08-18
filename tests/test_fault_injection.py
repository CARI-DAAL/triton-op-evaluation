"""Failpoint tests: crashes around EXEC ack never strand an unqueryable ID."""

import pytest
import redis
from conftest import make_caller, make_completed_result
from test_api import algorithms, build_app, post, seed_worker, trial_headers

from triton_evaluation import broker

pytestmark = pytest.mark.anyio


@pytest.fixture
def app(tmp_path, release_catalog_files):
    return build_app(tmp_path, release_catalog_files)


@pytest.fixture(autouse=True)
def _restore_failpoints():
    yield
    broker._create_pre_exec = broker._no_op_failpoint
    broker._create_post_exec = broker._no_op_failpoint
    broker._commit_pre_exec = broker._no_op_failpoint
    broker._commit_post_exec = broker._no_op_failpoint


class _Crash(Exception):
    """Simulated process death at a failpoint."""


def _crash(_evaluation_id):
    raise _Crash


def _tolerant_client(app):
    import httpx

    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app.app, raise_app_exceptions=False),
        base_url="http://test",
    )


def _trial_caller():
    return make_caller("trial-caller", "trial", ("trial:submit", "trial:read"))


async def test_crash_before_admission_exec_accepts_nothing(app, monkeypatch):
    broker._create_pre_exec = _crash
    seed_worker(app)
    async with _tolerant_client(app) as client:
        response = await post(client)
    assert response.status_code == 500
    body = response.json()
    assert "evaluation_id" not in body  # an ErrorEnvelope never supplies an ID
    # No resource or queue entry exists at all.
    assert app.store._r.zcard(f"{app.settings.redis_prefix}:state:queued") == 0
    assert app.store._r.get(f"{app.settings.redis_prefix}:queue-sequence") is None


async def test_crash_after_admission_exec_leaves_recoverable_id(app, monkeypatch):
    broker._create_post_exec = _crash
    seed_worker(app)
    async with _tolerant_client(app) as client:
        response = await post(client)
    assert response.status_code == 500
    # The resource WAS accepted durably: same key + same request recovers it.
    broker._create_post_exec = broker._no_op_failpoint
    async with app.http as client:
        replay = await post(client)
    assert replay.status_code == 202
    evaluation_id = replay.json()["evaluation_id"]
    assert app.store.get_visible(evaluation_id, _trial_caller()).state == "queued"


async def test_crash_before_terminal_commit_keeps_resource_running(app):
    seed_worker(app)
    async with app.http as client:
        created = await post(client)
        evaluation_id = created.json()["evaluation_id"]

    from triton_evaluation.models import WorkerIdentity, new_worker_id

    lease = app.store.claim_next(
        WorkerIdentity(new_worker_id(), app.settings.evaluator_build,
                       algorithms(app.settings))
    )
    broker._commit_pre_exec = _crash
    with pytest.raises(_Crash):
        app.store.commit_terminal(lease, "completed",
                                  make_completed_result(app.settings))
    assert app.store.get_visible(evaluation_id, _trial_caller()).state == "running"

    # The replacement attempt (after lease expiry) commits the winner.
    app.clock.advance_ms(app.settings.lease_seconds * 1000 + 1)
    app.store.recover_expired_leases()
    broker._commit_pre_exec = broker._no_op_failpoint
    lease2 = app.store.claim_next(
        WorkerIdentity(new_worker_id(), app.settings.evaluator_build,
                       algorithms(app.settings))
    )
    assert app.store.commit_terminal(
        lease2, "completed", make_completed_result(app.settings)
    ) is True
    resource = app.store.get_visible(evaluation_id, _trial_caller())
    assert resource.state == "completed"


async def test_crash_after_terminal_commit_is_recoverable_not_lost(app):
    seed_worker(app)
    async with app.http as client:
        created = await post(client)
        evaluation_id = created.json()["evaluation_id"]

    from triton_evaluation.models import WorkerIdentity, new_worker_id

    lease = app.store.claim_next(
        WorkerIdentity(new_worker_id(), app.settings.evaluator_build,
                       algorithms(app.settings))
    )
    broker._commit_post_exec = _crash
    with pytest.raises(_Crash):
        app.store.commit_terminal(lease, "completed",
                                  make_completed_result(app.settings))
    # The commit DID land: the resource is queryable and terminal.
    resource = app.store.get_visible(evaluation_id, _trial_caller())
    assert resource.state == "completed"
    # A late second commit from the same token cannot rewrite the winner.
    broker._commit_post_exec = broker._no_op_failpoint
    assert app.store.commit_terminal(
        lease, "failed", make_completed_result(app.settings)
    ) is False
    assert app.store.get_visible(evaluation_id, _trial_caller()).revision \
        == resource.revision


async def test_redis_loss_during_commit_fails_closed(app):
    seed_worker(app)
    async with app.http as client:
        created = await post(client)
        evaluation_id = created.json()["evaluation_id"]

        class _BrokenRedis:
            def __getattr__(self, name):
                raise redis.exceptions.ConnectionError("connection lost")

        original_client = app.store._r
        app.store._r = _BrokenRedis()
        try:
            with pytest.raises(redis.exceptions.ConnectionError):
                app.store.expire_due_resources(limit=5)
        finally:
            app.store._r = original_client

        # The resource is untouched and still recoverable by ID.
        resource = app.store.get_visible(evaluation_id, _trial_caller())
        assert resource.state == "queued"
        view = await client.get(f"/evaluations/{evaluation_id}",
                                headers=trial_headers())
    assert view.status_code == 200


def test_duplicated_ready_delivery_yields_one_claim(app, make_command):
    """A duplicated ready member cannot deliver the resource twice."""
    evaluation_id = _sync_create(app, make_command)
    ready_key = f"{app.settings.redis_prefix}:ready"
    app.store._r.zadd(ready_key, {evaluation_id: 999})  # duplicate delivery

    first = app.store.claim_next(_identity(app))
    assert first is not None and first.evaluation_id == evaluation_id
    # The duplicate member is gone with the claim: nothing claims twice.
    assert app.store.claim_next(_identity(app)) is None
    assert app.store.get_visible(evaluation_id, _trial_caller()).state == "running"


def test_terminal_cas_race_has_exactly_one_winner(app, make_command):
    """Both racing tokens try to commit; exactly one wins and it stands."""
    evaluation_id = _sync_create(app, make_command)
    stale = app.store.claim_next(_identity(app))
    assert stale is not None
    # The stale lease expires; the reaper requeues with a higher token.
    app.clock.advance_ms(app.settings.lease_seconds * 1000 + 1)
    assert app.store.recover_expired_leases().requeued == 1
    fresh = app.store.claim_next(_identity(app))
    assert fresh is not None
    assert fresh.fencing_token > stale.fencing_token

    # Fresh commits first (the fence's own now/deadline window); the stale
    # token's late commit is rejected and can never rewrite the winner.
    assert app.store.commit_terminal(
        fresh, "completed", make_completed_result(app.settings)
    ) is True
    assert app.store.commit_terminal(
        stale, "failed", make_completed_result(app.settings)
    ) is False
    winner = app.store.get_visible(evaluation_id, _trial_caller())
    assert winner.state == "completed"
    # And the mirrored ordering: nothing rewrites the terminal result either way.
    assert app.store.commit_terminal(
        fresh, "timed_out", make_completed_result(app.settings)
    ) is False
    assert app.store.get_visible(evaluation_id, _trial_caller()).revision         == winner.revision


def _identity(app):
    from triton_evaluation.models import WorkerIdentity, new_worker_id

    return WorkerIdentity(new_worker_id(), app.settings.evaluator_build,
                          algorithms(app.settings))


def _sync_create(app, make_command):
    """Create one resource through the store for direct protocol tests."""
    command = make_command(caller=_trial_caller())
    return app.store.create_or_replay(command).resource.evaluation_id
