"""Admission creates recoverable work before any candidate execution."""
from dataclasses import replace

import pytest
from conftest import make_caller, make_request
from test_api import (
    algorithms,
    build_app,
    make_test_settings,
    post,
    request_payload,
    seed_worker,
    trial_headers,
)

from triton_evaluation.service.models import WorkerIdentity, new_worker_id


def cpu_worker(settings):
    return WorkerIdentity(
        new_worker_id(), settings.evaluator_build, algorithms(settings),
        execution_class="cpu",
    )


@pytest.mark.anyio
async def test_admission_replay_conflict_and_rejection_never_execute_candidate(
    tmp_path, release_catalog_files,
):
    settings = make_test_settings(tmp_path).model_copy(update={
        "trial_rate_limit_per_minute": 1,
    })
    app = build_app(tmp_path, release_catalog_files, settings=settings)
    seed_worker(app)
    marker = tmp_path / "candidate-executed"
    payload = request_payload(triton_code=(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "class ModelNew: pass\n"
    ))
    async with app.http as client:
        first = await post(client, payload)
        replay = await post(client, payload)
        conflict = await post(client, {**payload, "triton_code": "class Other: pass"})
        limited = await post(client, payload, headers=trial_headers(key="second"))
    assert [r.status_code for r in (first, replay, conflict, limited)] == [202, 202, 409, 429]
    assert first.json()["evaluation_id"] == replay.json()["evaluation_id"]
    assert not marker.exists(), "candidate execution must belong to a leased Attempt"


@pytest.mark.anyio
async def test_cpu_capacity_accepts_compile_but_not_npu_checks(tmp_path, release_catalog_files):
    app = build_app(tmp_path, release_catalog_files)
    app.store.heartbeat(cpu_worker(app.settings))
    async with app.http as client:
        compile_response = await post(client, request_payload(checks=["compile"]))
        correctness_response = await post(client, headers=trial_headers(key="npu"))
    assert compile_response.status_code == 202, compile_response.text
    assert compile_response.json()["state"] == "queued"
    assert compile_response.json()["result"] is None
    assert correctness_response.status_code == 503
    assert correctness_response.json()["error"]["code"] == "NOT_READY"


@pytest.mark.anyio
async def test_compile_obeys_normal_inflight_limit(tmp_path, release_catalog_files):
    settings = make_test_settings(tmp_path).model_copy(update={"session_max_inflight": 1})
    app = build_app(tmp_path, release_catalog_files, settings=settings)
    app.store.heartbeat(cpu_worker(settings))
    async with app.http as client:
        first = await post(client, request_payload(checks=["compile"]))
        second = await post(client, request_payload(checks=["compile"]),
                            headers=trial_headers(key="second"))
    assert first.status_code == 202, first.text
    assert second.status_code == 413, second.text
    assert second.json()["error"]["code"] == "LIMIT_EXCEEDED"


def test_cpu_claim_and_recovery_keep_the_same_execution_class(
    store, settings, clock, make_command, make_worker,
):
    command = make_command(request=make_request(checks=["compile"]))
    store.create_or_replay(command)
    assert store.claim_next(make_worker(), card=3) is None
    first = store.claim_next(cpu_worker(settings), card=3)
    assert first is not None
    stored = store._r.hgetall(f"{settings.redis_prefix}:resource:{command.evaluation_id}")
    assert stored[b"attempt_card"] == b""
    assert stored[b"execution_class"] == b"cpu"
    clock.advance_ms(settings.lease_seconds * 1000 + 1)
    assert store.recover_expired_leases().requeued == 1
    assert store.claim_next(make_worker(), card=3) is None
    second = store.claim_next(cpu_worker(settings))
    assert second is not None and second.evaluation_id == first.evaluation_id
    assert second.attempt_id != first.attempt_id
    assert store.get_visible(command.evaluation_id, make_caller()).state == "running"


def test_claim_requires_the_admitted_build_and_algorithms(store, make_command, make_worker):
    command = make_command()
    store.create_or_replay(command)
    worker = make_worker()
    assert store.claim_next(replace(worker, evaluator_build="different-build")) is None
    assert store.claim_next(replace(
        worker, algorithms=replace(worker.algorithms, correctness="different-version"),
    )) is None
    assert store.claim_next(worker).evaluation_id == command.evaluation_id


def test_cpu_deadline_removes_dispatch_and_releases_quota(
    store, settings, clock, make_command,
):
    command = make_command(request=make_request(checks=["compile"], deadline_seconds=30))
    store.create_or_replay(command)
    clock.advance_ms(30_001)
    assert store.expire_due_resources() == 1
    assert store.claim_next(cpu_worker(settings)) is None
    resource = store.get_visible(command.evaluation_id, make_caller())
    assert resource.state == "timed_out"
    assert resource.result.compile.status == "timeout"
    assert store._r.zcard(f"{settings.redis_prefix}:ready:cpu:interactive") == 0
    # A fresh admission in the same session has capacity after the deadline.
    assert store.create_or_replay(make_command(
        request=make_request(checks=["compile"]), idempotency_key="after-deadline",
    )).decision == "created"


def test_recovery_keeps_grade_binding_when_canonical_source_is_corrupt(
    store, settings, clock, make_command,
):
    from conftest import make_grade_caller, make_grade_request

    caller = make_grade_caller()
    request = make_grade_request(checks=["compile"], deadline_seconds=30)
    command = make_command(caller=caller, request=request)
    store.create_or_replay(command)
    store._r.hset(
        f"{settings.redis_prefix}:resource:{command.evaluation_id}",
        "canonical_request", b"not valid JSON",
    )
    lease = store.claim_next(cpu_worker(settings))
    assert lease.requested_checks == ("compile",)
    assert lease.pinned.manifest_id == request.grade_context.manifest_id
    clock.advance_ms(30_001)
    assert store.expire_due_resources() == 1
    resource = store.get_visible(command.evaluation_id, caller)
    assert resource.result.compile.status == "timeout"
    assert resource.result.provenance.manifest_id == request.grade_context.manifest_id
    assert resource.result.provenance.problem_id == request.grade_context.problem_id
    assert resource.result.provenance.grade_eligible is False
