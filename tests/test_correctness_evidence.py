"""Opt-in correctness reuse stays owner-bound, fresh, complete and source-exact."""

import json
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import make_caller, make_completed_result, make_provenance
from pydantic import ValidationError
from test_api import algorithms, build_app, post, request_payload, seed_worker, trial_headers
from test_evaluation_evaluator import ENVIRONMENT, StubCheck, load
from test_operator_inputs import request as operator_request

from triton_evaluation.api.contract import (
    CorrectnessCoverage,
    CorrectnessResult,
    EvaluationRequest,
)
from triton_evaluation.evaluators.base import EngineCheckResult, LeaseLostError
from triton_evaluation.evaluators.evaluation import EvaluationEvaluator
from triton_evaluation.operator_input import to_evaluation_request
from triton_evaluation.runtime.correctness_evidence import (
    CorrectnessEvidenceError,
    validate_handoff,
    write_handoff,
)
from triton_evaluation.runtime.worker import Worker
from triton_evaluation.service.instance import EvaluationInstance
from triton_evaluation.service.models import PinnedProvenance, WorkerIdentity, new_worker_id


@pytest.fixture
def source(store, settings, clock, make_command, make_worker):
    # Keep the engine handoff's wall clock on the same timeline as this store.
    clock.now_ms = int(time.time() * 1000)
    public = operator_request(checks=["correctness"])
    request = to_evaluation_request(public)
    command = make_command(request=request)
    store.create_or_replay(command)
    lease = store.claim_next(make_worker(), card=0)
    result = make_completed_result(settings).model_copy(update={
        "provenance": make_provenance(settings, execution_environment=ENVIRONMENT),
    })
    assert store.commit_terminal(lease, "completed", result)
    return SimpleNamespace(
        public=public, request=request, command=command, lease=lease,
        resource=store.get_visible(command.evaluation_id, make_caller()),
        pinned=lease.pinned,
    )


def target_request(source, **overrides):
    payload = source.request.model_dump(mode="json")
    payload.update(checks=["correctness", "performance"],
                   correctness_from=source.resource.evaluation_id)
    payload.update(overrides)
    return EvaluationRequest.model_validate(payload)


def validate(store, source, **overrides):
    return store.validate_correctness_source(
        target_request(source, **overrides), make_caller(), source.pinned,
    )


def overwrite_resource(store, source, **updates):
    resource = source.resource.model_copy(update=updates)
    store._r.hset(store._key(f"resource:{resource.evaluation_id}"),
                  "resource", resource.model_dump_json())


def test_matching_source_reuses_across_different_checks_and_timing_params(store, source):
    evidence = validate(store, source, performance={"repeats": 3, "warmup": 0})
    assert evidence.evaluation_id == source.resource.evaluation_id
    assert evidence.correctness.reused_from == source.resource.evaluation_id
    assert evidence.correctness.outcome == "correct"


@pytest.mark.parametrize("field,value", [
    ("reference_code", "different reference"), ("kernel_code", "different candidate"),
    ("op_name", "different name"), ("triton_impl_name", "other_impl"), ("non_compute", True),
])
def test_changed_subject_refuses_reuse(store, source, field, value):
    with pytest.raises(CorrectnessEvidenceError, match="submitted subject"):
        validate(store, source, **{field: value})


@pytest.mark.parametrize("mutation", ["seed", "tolerance", "workload"])
def test_workload_generator_identity_includes_seed_tolerance_and_inputs(store, source, mutation):
    payload = source.public.model_dump(mode="json")
    payload["correctness_from"] = source.resource.evaluation_id
    if mutation == "seed":
        payload["seed"] += 1
    elif mutation == "tolerance":
        payload["workloads"][0]["tolerance"]["max_atol"] = 0.25
    else:
        payload["workloads"][0]["inputs"]["x"]["shape"] = [8]
    changed = to_evaluation_request(type(source.public).model_validate(payload))
    with pytest.raises(CorrectnessEvidenceError, match="submitted subject"):
        store.validate_correctness_source(changed, make_caller(), source.pinned)


@pytest.mark.parametrize("kind", ["build", "correctness", "decoy"])
def test_changed_evaluator_protocol_refuses_reuse(store, source, kind):
    pinned = source.pinned
    if kind == "build":
        pinned = replace(pinned, evaluator_build="git:" + "1" * 40)
    else:
        pinned = replace(pinned, algorithms=replace(pinned.algorithms, **{kind: "changed"}))
    with pytest.raises(CorrectnessEvidenceError, match="evaluator protocol"):
        store.validate_correctness_source(target_request(source), make_caller(), pinned)


def test_absent_invisible_and_observer_source_have_same_error(store, source):
    observer = make_caller("other", scopes=("trial:submit", "observer:read"))
    assert store.get_visible(source.resource.evaluation_id, observer) is not None
    errors = []
    for request, caller in [
        (target_request(source, correctness_from="ev_" + "f" * 32), make_caller()),
        (target_request(source), make_caller("other")),
        (target_request(source), observer),
        (target_request(source), make_caller(scopes=("trial:submit",))),
    ]:
        with pytest.raises(CorrectnessEvidenceError) as error:
            store.validate_correctness_source(request, caller, source.pinned)
        errors.append(str(error.value))
    assert len(set(errors)) == 1


def test_expired_source_is_refused(store, source, clock, settings):
    clock.advance_ms(settings.max_recovery_age_seconds * 1000 + 1)
    with pytest.raises(CorrectnessEvidenceError, match="freshness"):
        validate(store, source)


def test_incomplete_and_reused_observations_cannot_be_sources(store, source):
    incomplete = CorrectnessResult(
        outcome="incomplete", passed=False, passed_cases=1, total_cases=2, failures=[],
        coverage=CorrectnessCoverage(
            selected_case_indices=[0], evaluated_case_indices=[0], stop_reason="complete",
        ),
    )
    for data in [incomplete, source.resource.result.correctness.data.model_copy(update={
        "reused_from": "ev_" + "a" * 32,
    })]:
        result = source.resource.result.model_copy(update={
            "correctness": source.resource.result.correctness.model_copy(update={"data": data}),
        })
        overwrite_resource(store, source, result=result)
        with pytest.raises(CorrectnessEvidenceError, match="original full-task"):
            validate(store, source)


def test_corrupted_stored_canonical_source_is_refused(store, source):
    store._r.hset(store._key(f"resource:{source.resource.evaluation_id}"),
                  "canonical_request", b"{}")
    with pytest.raises(CorrectnessEvidenceError):
        validate(store, source)


def test_source_requires_populated_execution_environment(store, source):
    result = source.resource.result.model_copy(update={
        "provenance": source.resource.result.provenance.model_copy(update={
            "execution_environment": None,
        }),
    })
    overwrite_resource(store, source, result=result)
    with pytest.raises(CorrectnessEvidenceError, match="environment"):
        validate(store, source)


_INCOMPLETE_ENVIRONMENT_FIELDS = [
    ("hardware_model", ""), ("hardware_count", 0), ("driver_version", ""),
    ("runtime_version", " "), ("torch_version", None), ("torch_npu_version", None),
    ("triton_version", None), ("toolchain_version", ""),
]


@pytest.mark.parametrize("field,value", _INCOMPLETE_ENVIRONMENT_FIELDS)
def test_source_with_unknown_identity_is_not_reusable_even_when_runtime_matches(
    store, source, make_command, make_worker, field, value,
):
    incomplete = ENVIRONMENT.model_copy(update={field: value})
    result = source.resource.result.model_copy(update={
        "provenance": source.resource.result.provenance.model_copy(update={
            "execution_environment": incomplete,
        }),
    })
    overwrite_resource(store, source, result=result)
    with pytest.raises(CorrectnessEvidenceError, match="environment"):
        store.validate_correctness_source(target_request(source), make_caller(), source.pinned)
    request, lease = destination_lease(store, source, make_command, make_worker)
    with pytest.raises(CorrectnessEvidenceError, match="environment"):
        store.correctness_for_attempt(
            request, lease, execution_environment=incomplete, card=0,
        )


@pytest.mark.parametrize("field,value", _INCOMPLETE_ENVIRONMENT_FIELDS)
def test_admitted_reuse_refuses_runtime_with_missing_identity(
    store, source, make_command, make_worker, field, value,
):
    store.validate_correctness_source(target_request(source), make_caller(), source.pinned)
    request, lease = destination_lease(store, source, make_command, make_worker)
    with pytest.raises(CorrectnessEvidenceError, match="environment"):
        store.correctness_for_attempt(
            request, lease, execution_environment=ENVIRONMENT.model_copy(update={field: value}),
            card=0,
        )


def test_running_or_queued_source_is_not_complete_evidence(store, source, make_command):
    command = make_command(request=source.request, idempotency_key="unfinished-source")
    store.create_or_replay(command)
    with pytest.raises(CorrectnessEvidenceError):
        validate(store, source, correctness_from=command.evaluation_id)


def destination_lease(store, source, make_command, make_worker):
    request = target_request(source)
    store.create_or_replay(make_command(request=request, idempotency_key="reuse"))
    return request, store.claim_next(make_worker(), card=0)


def test_attempt_resolution_checks_environment_card_and_fence(
    store, source, make_command, make_worker,
):
    request, lease = destination_lease(store, source, make_command, make_worker)
    evidence = store.correctness_for_attempt(
        request, lease, execution_environment=ENVIRONMENT, card=0,
    )
    assert evidence.correctness.reused_from == source.resource.evaluation_id
    with pytest.raises(CorrectnessEvidenceError, match="environment"):
        store.correctness_for_attempt(
            request, lease, execution_environment=ENVIRONMENT.model_copy(update={
                "driver_version": "changed",
            }), card=0,
        )
    with pytest.raises(CorrectnessEvidenceError, match="different device"):
        store.correctness_for_attempt(request, lease, execution_environment=ENVIRONMENT, card=1)
    store._r.hset(store._key(f"resource:{lease.evaluation_id}"), "attempt_id", "replacement")
    with pytest.raises(LeaseLostError):
        store.correctness_for_attempt(request, lease, execution_environment=ENVIRONMENT, card=0)


def test_missing_optional_legacy_card_binding_does_not_block_reuse(
    store, source, make_command, make_worker,
):
    store._r.hdel(store._key(f"resource:{source.resource.evaluation_id}"), "attempt_card")
    request, lease = destination_lease(store, source, make_command, make_worker)
    assert store.correctness_for_attempt(
        request, lease, execution_environment=ENVIRONMENT, card=1,
    ).evaluation_id == source.resource.evaluation_id


def test_explicit_reuse_affinity_skips_other_card_without_blocking_normal_work(
    store, source, make_command, make_worker,
):
    reuse = make_command(request=target_request(source), idempotency_key="reuse-card")
    ordinary = make_command(request=source.request, idempotency_key="ordinary-card")
    store.create_or_replay(reuse)
    store.create_or_replay(ordinary)
    ordinary_lease = store.claim_next(make_worker(), card=1)
    assert ordinary_lease.evaluation_id == ordinary.evaluation_id
    assert store.claim_next(make_worker(), card=1) is None
    reuse_lease = store.claim_next(make_worker(), card=0)
    assert reuse_lease.evaluation_id == reuse.evaluation_id


@pytest.mark.parametrize("blocker_count", [10, 21])
def test_affinity_backlog_pages_preserve_eligible_fifo_and_interactive_priority(
    store, source, settings, make_command, make_worker, monkeypatch, blocker_count,
):
    monkeypatch.setattr(settings, "trial_max_inflight", 40)
    monkeypatch.setattr(settings, "session_max_inflight", 40)
    older_bulk = make_command(idempotency_key="older-bulk", traffic_class="rollout")
    assert store.create_or_replay(older_bulk).decision == "created"
    blockers = []
    for index in range(blocker_count):
        command = make_command(request=target_request(source), idempotency_key=f"blocked-{index}")
        assert store.create_or_replay(command).decision == "created"
        blockers.append(command)
    ordinary = [make_command(idempotency_key=f"ordinary-{index}") for index in range(2)]
    for command in ordinary:
        assert store.create_or_replay(command).decision == "created"
    ready_key = store._key("ready:npu:interactive")
    stale_id = "ev_" + "f" * 32
    # A missing member at the page boundary must not shift past eligible work.
    before_boundary = store._r.zscore(ready_key, blockers[8].evaluation_id)
    store._r.zadd(ready_key, {stale_id: before_boundary + 0.5})
    worker = make_worker()
    assert store.claim_next(worker, card=1).evaluation_id == ordinary[0].evaluation_id
    assert store.claim_next(worker, card=1).evaluation_id == ordinary[1].evaluation_id
    assert store._r.zscore(ready_key, stale_id) is None
    assert store._r.zcard(ready_key) == blocker_count
    assert store.claim_next(worker, card=1).evaluation_id == older_bulk.evaluation_id
    assert store.claim_next(worker, card=0).evaluation_id == blockers[0].evaluation_id
    assert store._r.zcard(ready_key) == blocker_count - 1


def test_paged_scan_retries_if_queue_order_changes_before_claim(
    store, source, settings, make_command, make_worker, monkeypatch,
):
    monkeypatch.setattr(settings, "trial_max_inflight", 20)
    monkeypatch.setattr(settings, "session_max_inflight", 20)
    for index in range(10):
        command = make_command(request=target_request(source), idempotency_key=f"blocked-{index}")
        assert store.create_or_replay(command).decision == "created"
    ordinary = make_command(idempotency_key="ordinary")
    moved_earlier = make_command(idempotency_key="moved-earlier")
    for command in (ordinary, moved_earlier):
        assert store.create_or_replay(command).decision == "created"
    install_claim = store._install_claim
    attempts = []

    def reorder_before_commit(pipeline, **kwargs):
        attempts.append(kwargs["member"].decode())
        if len(attempts) == 1:
            store._r.zadd(kwargs["ready_key"], {moved_earlier.evaluation_id: 0})
        return install_claim(pipeline, **kwargs)

    monkeypatch.setattr(store, "_install_claim", reorder_before_commit)
    lease = store.claim_next(make_worker(), card=1)
    assert lease.evaluation_id == moved_earlier.evaluation_id
    assert attempts == [ordinary.evaluation_id, moved_earlier.evaluation_id]
    assert store.get_visible(ordinary.evaluation_id, make_caller()).state == "queued"


def test_expired_proof_beyond_affinity_page_can_be_claimed_and_refused(
    store, source, settings, clock, make_command, make_worker, monkeypatch,
):
    monkeypatch.setattr(settings, "trial_max_inflight", 20)
    monkeypatch.setattr(settings, "session_max_inflight", 20)
    clock.advance_ms((settings.max_recovery_age_seconds - 10) * 1000)
    worker = make_worker()
    fresh = make_command(request=source.request, idempotency_key="fresh-source")
    assert store.create_or_replay(fresh).decision == "created"
    fresh_lease = store.claim_next(worker, card=0)
    assert store.commit_terminal(fresh_lease, "completed", source.resource.result)
    for index in range(10):
        request = target_request(source, correctness_from=fresh.evaluation_id)
        command = make_command(request=request, idempotency_key=f"fresh-blocked-{index}")
        assert store.create_or_replay(command).decision == "created"
    expiring_request = target_request(source)
    expiring = make_command(request=expiring_request, idempotency_key="expiring-after-page")
    assert store.create_or_replay(expiring).decision == "created"
    clock.advance_ms(11_000)
    lease = store.claim_next(worker, card=1)
    assert lease.evaluation_id == expiring.evaluation_id
    with pytest.raises(CorrectnessEvidenceError, match="freshness"):
        store.correctness_for_attempt(
            expiring_request, lease, execution_environment=ENVIRONMENT, card=1,
        )


@pytest.mark.parametrize("missing", [True, False])
def test_unavailable_proof_releases_affinity_and_worker_can_reject(
    store, source, settings, clock, make_command, make_worker, missing,
):
    # Queue close to source expiry so the destination deadline remains alive.
    clock.advance_ms((settings.max_recovery_age_seconds - 10) * 1000)
    request = target_request(source)
    command = make_command(request=request, idempotency_key="unavailable-proof")
    store.create_or_replay(command)
    if missing:
        store._r.delete(store._key(f"resource:{source.resource.evaluation_id}"))
    else:
        clock.advance_ms(11_000)
    lease = store.claim_next(make_worker(), card=1)
    assert lease.evaluation_id == command.evaluation_id
    with pytest.raises(CorrectnessEvidenceError):
        store.correctness_for_attempt(request, lease, execution_environment=ENVIRONMENT, card=1)


def test_handoff_checks_exact_files_and_freshness(store, source, tmp_path):
    evidence = validate(store, source)
    path = write_handoff(tmp_path, evidence)
    reference = tmp_path / "reference.py"
    candidate = tmp_path / "candidate.py"
    reference.write_text(source.request.reference_code)
    candidate.write_text(source.request.kernel_code)
    kwargs = dict(reference_path=reference, candidate_path=candidate,
                  triton_impl_name=source.request.triton_impl_name)
    assert validate_handoff(path, **kwargs) == 1
    candidate.write_text("changed")
    with pytest.raises(CorrectnessEvidenceError):
        validate_handoff(path, **kwargs)
    candidate.write_text(source.request.kernel_code)
    expired = evidence.model_copy(update={
        "observed_at": evidence.observed_at - timedelta(days=3),
        "expires_at": evidence.expires_at - timedelta(days=3),
    })
    write_handoff(tmp_path, expired)
    with pytest.raises(CorrectnessEvidenceError):
        validate_handoff(path, **kwargs)


class EvidenceBenchmark:
    def __init__(self):
        self.calls = 0

    def run(self, request, work_dir, *args):
        self.calls += 1
        proof = json.loads((Path(work_dir) / "correctness_evidence.json").read_text())
        assert proof["evaluation_id"] == request.correctness_from
        assert proof["correctness"]["reused_from"] == request.correctness_from
        assert not (Path(work_dir) / "verify_result.json").exists()
        raw = load("benchmark-event-timing.json")
        raw.update(
            total_cases=1, passed_cases=1, failed_cases=0,
            per_shape_results=raw["per_shape_results"][:1], speedup_vs_torch=2.0,
        )
        raw["implementation"]["avg_latency_ms"] = 0.5
        return EngineCheckResult(
            status="succeeded", raw=raw, error=None,
        )


@pytest.mark.parametrize("environment_valid", [True, False])
def test_worker_reuses_zero_verify_and_publishes_typed_outcome(
    store, settings, clock, source, make_command, monkeypatch, environment_valid,
):
    request = target_request(source)
    command = make_command(request=request, idempotency_key="reuse-worker")
    store.create_or_replay(command)
    verify = StubCheck(None)
    benchmark = EvidenceBenchmark()
    environment = ENVIRONMENT if environment_valid else ENVIRONMENT.model_copy(update={
        "runtime_version": "changed",
    })
    monkeypatch.setattr("triton_evaluation.runtime.worker.build_execution_environment",
                        lambda settings: environment)
    worker = Worker(
        store, EvaluationEvaluator(verify=verify, benchmark=benchmark, profile=None),
        device_id=0, settings=settings, clock=clock, sleep=lambda _: None,
    )
    assert worker.run_once()
    result = store.get_visible(command.evaluation_id, make_caller()).result
    assert verify.calls == 0
    if environment_valid:
        assert result.correctness.status == "succeeded"
        assert result.correctness.data.reused_from == source.resource.evaluation_id
        assert result.performance.status == "succeeded"
        assert benchmark.calls == 1
    else:
        assert result.correctness.status == "failed"
        assert result.correctness.error.code == "CORRECTNESS_EVIDENCE_INVALID"
        assert result.correctness.error.phase == "request"
        assert result.correctness.error.retryable is False
        assert result.performance.status == "skipped"
        assert benchmark.calls == 0


def test_admission_replay_is_recoverable_after_source_expiry(
    store, settings, clock, source, make_worker,
):
    store.heartbeat(make_worker())
    instance = EvaluationInstance(settings=settings, store=store)
    request = target_request(source)
    admitted = instance.submit(request, make_caller(), session_id="reuse", idempotency_key="reuse")
    assert admitted.decision == "created"
    clock.advance_ms(settings.max_recovery_age_seconds * 1000 + 1)
    replay = instance.submit(request, make_caller(), session_id="reuse", idempotency_key="reuse")
    assert replay.decision == "replayed"
    assert replay.resource.evaluation_id == admitted.resource.evaluation_id
    with pytest.raises(CorrectnessEvidenceError):
        instance.submit(request, make_caller(), session_id="reuse", idempotency_key="new")


def test_expiry_while_queued_fails_without_verification(
    store, settings, clock, source, make_worker, monkeypatch,
):
    clock.advance_ms((settings.max_recovery_age_seconds - 10) * 1000)
    store.heartbeat(make_worker())
    instance = EvaluationInstance(settings=settings, store=store)
    admitted = instance.submit(
        target_request(source), make_caller(), session_id="reuse", idempotency_key="expiry",
    )
    clock.advance_ms(11_000)
    monkeypatch.setattr("triton_evaluation.runtime.worker.build_execution_environment",
                        lambda settings: ENVIRONMENT)
    verify = StubCheck(None)
    benchmark = EvidenceBenchmark()
    worker = Worker(
        store, EvaluationEvaluator(verify=verify, benchmark=benchmark, profile=None),
        device_id=0, settings=settings, clock=clock, sleep=lambda _: None,
    )
    assert worker.run_once()
    result = store.get_visible(admitted.resource.evaluation_id, make_caller()).result
    assert result.correctness.error.code == "CORRECTNESS_EVIDENCE_INVALID"
    assert result.correctness.error.retryable is False
    assert verify.calls == benchmark.calls == 0


def test_request_rejects_grade_partial_and_missing_correctness_reuse(source):
    for overrides in [
        {"checks": ["profile"], "performance": None},
        {"correctness": {"case_indices": [0]}},
        {"purpose": "grade"},
    ]:
        with pytest.raises(ValidationError, match="correctness_from"):
            target_request(source, **overrides)


def test_default_request_and_outcome_omit_reuse_markers(source):
    assert "correctness_from" not in source.request.model_dump(mode="json", exclude_none=False)
    assert "reused_from" not in source.resource.result.correctness.data.model_dump(
        mode="json", exclude_none=False,
    )
    pinned = PinnedProvenance.for_request(
        target_request(source), evaluator_build=source.pinned.evaluator_build,
        algorithms=source.pinned.algorithms,
    )
    assert pinned == source.pinned


@pytest.mark.anyio
async def test_http_explicit_reuse_admits_without_executing_code_and_replays(
    tmp_path, release_catalog_files,
):
    app = build_app(tmp_path, release_catalog_files)
    seed_worker(app)
    marker = tmp_path / "candidate-executed"
    payload = request_payload(
        checks=["correctness"],
        triton_code=f"from pathlib import Path\nPath({str(marker)!r}).touch()\n",
    )
    async with app.http as client:
        first = await post(client, payload)
        assert first.status_code == 202
        source_id = first.json()["evaluation_id"]
        lease = app.store.claim_next(WorkerIdentity(
            new_worker_id(), app.settings.evaluator_build, algorithms(app.settings),
        ), card=0)
        result = make_completed_result(app.settings).model_copy(update={
            "provenance": make_provenance(app.settings, execution_environment=ENVIRONMENT),
        })
        assert app.store.commit_terminal(lease, "completed", result)
        reuse_payload = {
            **payload, "checks": ["correctness", "performance"], "correctness_from": source_id,
        }
        headers = trial_headers(key="reuse-http")
        admitted = await post(client, reuse_payload, headers=headers)
        replay = await post(client, reuse_payload, headers=headers)
        unknown = await post(client, {
            **reuse_payload, "correctness_from": "ev_" + "f" * 32,
        }, headers=trial_headers(key="unknown-proof"))
    assert [admitted.status_code, replay.status_code, unknown.status_code] == [202, 202, 422]
    assert admitted.json()["evaluation_id"] == replay.json()["evaluation_id"]
    assert unknown.json()["error"]["code"] == "INVALID_REQUEST"
    assert unknown.json()["error"]["retryable"] is False
    assert not marker.exists()
