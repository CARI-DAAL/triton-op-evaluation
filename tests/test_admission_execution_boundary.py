"""Cached verdicts never bypass durable admission and leased execution."""
import pytest
from conftest import make_caller
from test_api import build_app, post, request_payload, seed_worker

from triton_evaluation import verdict_cache

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize("checks", [
    ["compile"], ["correctness"], ["compile", "correctness"], ["correctness", "profile"],
])
async def test_cached_verdict_still_admits_queued_work(tmp_path, release_catalog_files, checks):
    app = build_app(tmp_path, release_catalog_files)
    execution_class = "cpu" if checks == ["compile"] else "npu"
    seed_worker(app, execution_class=execution_class)
    payload = request_payload(kernel_code="def broken(: pass", checks=checks)
    digest = verdict_cache.digest_for(
        payload["kernel_code"], "triton_ascend_impl", app.settings.cann_fingerprint,
    )
    verdict_cache.put(app.store, digest, "compile_error", 60)
    async with app.http as client:
        response = await post(client, payload)
    assert response.status_code == 202, response.text
    assert response.json()["state"] == "queued"
    assert response.json()["result"] is None
    assert app.store._r.zcard(
        f"{app.settings.redis_prefix}:ready:{execution_class}:interactive"
    ) == 1


# ---------------------------------------------------------------------------
# Poison quarantine is non-retryable
# ---------------------------------------------------------------------------


def _poison_terminal(store, settings, clock, make_command, make_worker, request,
                     idempotency_key):
    cmd = make_command(request=request, idempotency_key=idempotency_key)
    store.create_or_replay(cmd)
    for _ in range(2):
        store.claim_next(make_worker(), card=7)
        clock.advance_ms(settings.lease_seconds * 1000 + 1)
        counts = store.recover_expired_leases()
        if counts.failed:
            break
        clock.advance_ms(1)  # re-arm the replacement claim for the next death
    return store.get_visible(cmd.evaluation_id, make_caller())


def test_poison_quarantine_code_is_non_retryable(store, settings, clock,
                                                 make_command, make_worker):
    from conftest import make_request

    v21 = _poison_terminal(store, settings, clock, make_command, make_worker,
                           make_request(schema_version="2.1"), "key-21")
    assert v21.schema_version == "2.1"
    assert v21.state == "failed"
    assert v21.result.correctness.error.code == "SUSPECTED_POISON_CANDIDATE"
    assert v21.result.correctness.error.retryable is False



# ---------------------------------------------------------------------------
# Terminal HTTP response bodies validate against the published schema
# ---------------------------------------------------------------------------


async def test_v21_terminal_http_bodies_validate_against_published_schema(
    tmp_path, release_catalog_files, make_command
):
    import jsonschema
    from conftest import make_completed_result
    from test_api import (
        WorkerIdentity,
        algorithms,
        build_app,
        make_test_settings,
        new_worker_id,
        post,
        seed_worker,
        trial_headers,
    )
    from test_contract_v21 import published_v21_resource_validator

    app = build_app(tmp_path, release_catalog_files, settings=make_test_settings(tmp_path))
    seed_worker(app)
    command = make_command(
        caller=make_caller("trial-caller", "trial", ("trial:submit", "trial:read"))
    )
    app.store.create_or_replay(command)
    lease = app.store.claim_next(
        WorkerIdentity(new_worker_id(), app.settings.evaluator_build,
                       algorithms(app.settings))
    )
    app.store.commit_terminal(lease, "completed",
                              make_completed_result(app.settings))

    validator = published_v21_resource_validator()
    async with app.http as client:
        # an idempotent replay of the terminal answers 200 immediately
        replayed = await post(client)
        assert replayed.status_code == 200, replayed.text
        validator.validate(replayed.json())
        evaluation_id = replayed.json()["evaluation_id"]
        # and GET returns the identical wire shape
        got = await client.get(f"/evaluations/{evaluation_id}",
                               headers=trial_headers())
        assert got.status_code == 200
        validator.validate(got.json())
    with pytest.raises(jsonschema.ValidationError):
        validator.validate({"schema_version": "2.1", "result": {"compile": None}})
