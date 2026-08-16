"""Durable waiting, recovery, and disconnect semantics."""
import asyncio
import json

import pytest
from conftest import make_completed_result
from test_api import algorithms, post, request_payload, seed_worker, trial_headers

from triton_evaluation.api.contract import EvaluationResource
from triton_evaluation.api.routes import wait_until_terminal
from triton_evaluation.models import WorkerIdentity, new_worker_id

pytestmark = pytest.mark.anyio


@pytest.fixture
def app(tmp_path, release_catalog_files):
    from test_api import build_app

    return build_app(tmp_path, release_catalog_files)


def _http_trial_caller():
    from conftest import make_caller

    return make_caller("trial-caller", "trial", ("trial:submit", "trial:read"))


def claim_and_complete(app):
    lease = app.store.claim_next(
        WorkerIdentity(new_worker_id(), app.settings.evaluator_build,
                       algorithms(app.settings))
    )
    assert lease is not None
    assert app.store.commit_terminal(
        lease, "completed", make_completed_result(app.settings)
    )


async def test_wait_seconds_returns_200_when_terminal_in_budget(app):
    seed_worker(app)
    async with app.http as client:
        await post(client)
        claim_and_complete(app)
        response = await client.post(
            "/evaluations",
            content=json.dumps(request_payload()).encode(),
            headers=trial_headers(),
            params={"wait_seconds": "5"},
        )
    assert response.status_code == 200
    assert response.json()["state"] == "completed"


async def test_wait_expiry_returns_202_without_state_mutation(app):
    seed_worker(app)
    async with app.http as client:
        response = await post(client, params={"wait_seconds": "0"})
    assert response.status_code == 202
    resource = EvaluationResource.model_validate_json(response.content)
    assert resource.state == "queued"  # waiting never terminalizes or cancels
    from conftest import make_caller

    assert app.store.get_visible(
        resource.evaluation_id,
        make_caller("trial-caller", "trial", ("trial:submit", "trial:read")),
    ).state == "queued"


async def test_disconnect_budget_is_not_a_task_timeout(app, make_command):
    resource = app.store.create_or_replay(make_command(caller=_http_trial_caller())).resource
    final = await wait_until_terminal(
        app.store,
        _http_trial_caller(),
        resource.evaluation_id,
        budget_seconds=0.05,
        poll_seconds=0.01,
        is_disconnected=lambda: True,  # client hangs up on the first poll
    )
    assert final is not None and final.state == "queued"


async def test_wait_helper_returns_terminal_resource(app, make_command):
    resource = app.store.create_or_replay(make_command(caller=_http_trial_caller())).resource
    task = asyncio.get_running_loop().create_task(
        asyncio.to_thread(claim_and_complete, app)
    )
    final = await wait_until_terminal(
        app.store,
        _http_trial_caller(),
        resource.evaluation_id,
        budget_seconds=5,
        poll_seconds=0.01,
        is_disconnected=lambda: False,
    )
    await task
    assert final.state == "completed"


async def test_ambiguous_post_recovered_by_same_key(app):
    """No response was obtained; the same key+request safely replays."""
    seed_worker(app)
    async with app.http as client:
        first = await post(client)
        # Simulate the client never seeing the response: same key, same request.
        replay = await post(client)
        assert replay.status_code == 202
        assert replay.json()["evaluation_id"] == first.json()["evaluation_id"]
        # And GET recovery against the persisted ID.
        got = await client.get(
            f"/evaluations/{first.json()['evaluation_id']}", headers=trial_headers()
        )
        assert got.status_code == 200
