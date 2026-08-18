"""Resource state machine: allowed transitions, visibility invariants, synthesis.

The only legal transitions are ``queued -> running``, ``queued ->
failed|timed_out``, and ``running -> completed|failed|timed_out``. Terminal
state and terminal results are immutable, the visible revision is monotonic,
``updated_at`` tracks each visible mutation, and ``started_at`` is set by the
first attempt only.
"""
import dataclasses
import json
import re
from datetime import datetime, timezone

from conftest import (
    make_caller,
    make_completed_result,
    make_correctness_success,
    make_grade_caller,
    make_grade_request,
    make_request,
)

from triton_evaluation.api.contract_artifacts import sha256_hex
from triton_evaluation.broker import MAX_ATTEMPTS


def _key(settings, suffix):
    return f"{settings.redis_prefix}:{suffix}"


def _now(clock):
    return datetime.fromtimestamp(clock.now_ms / 1000, tz=timezone.utc)


def _create_and_claim(store, make_command, make_worker, clock, advance_ms=5_000, **kwargs):
    command = make_command(**kwargs)
    store.create_or_replay(command)
    clock.advance_ms(advance_ms)
    lease = store.claim_next(make_worker())
    assert lease is not None
    return command, lease


def test_claim_transitions_queued_to_running(
    store, make_command, make_worker, clock, settings, redis_client
):
    command = make_command()
    store.create_or_replay(command)
    created_ms = clock.now_ms
    clock.advance_ms(5_000)

    lease = store.claim_next(make_worker())

    assert lease is not None
    assert lease.evaluation_id == command.evaluation_id
    assert re.fullmatch(r"at_[0-9a-f]{32}", lease.attempt_id)
    assert lease.fencing_token == 1
    assert lease.expected_revision == 2
    assert lease.deadline_at_ms == created_ms + make_request().deadline_seconds * 1000
    assert lease.canonical_request == command.admission.canonical_request
    assert lease.evaluator_build == settings.evaluator_build
    assert lease.algorithms == command.algorithms

    visible = store.get_visible(command.evaluation_id, make_caller())
    assert visible.state == "running"
    assert visible.revision == 2
    assert visible.started_at == _now(clock)
    assert visible.updated_at == _now(clock)
    assert visible.result is None
    assert visible.finished_at is None

    member = command.evaluation_id.encode()
    assert redis_client.zrange(_key(settings, "ready"), 0, -1) == []
    assert redis_client.zscore(_key(settings, "state:queued"), member) is None
    assert redis_client.zscore(_key(settings, "state:running"), member) == float(
        lease.deadline_at_ms
    )
    lease_expiry_ms = clock.now_ms + settings.lease_seconds * 1000
    assert redis_client.zscore(_key(settings, "running-leases"), member) == float(
        lease_expiry_ms
    )


def test_claim_returns_none_when_queue_is_empty(store, make_worker):
    assert store.claim_next(make_worker()) is None


def test_claim_never_starts_a_past_deadline_delivery(
    store, make_command, make_worker, clock
):
    command = make_command(request=make_request(deadline_seconds=10))
    store.create_or_replay(command)
    clock.advance_ms(11_000)

    assert store.claim_next(make_worker()) is None
    visible = store.get_visible(command.evaluation_id, make_caller())
    assert visible.state == "queued"
    assert visible.revision == 1


def test_renewal_extends_lease_without_visible_mutation(
    store, make_command, make_worker, clock, settings, redis_client
):
    command, lease = _create_and_claim(store, make_command, make_worker, clock)
    clock.advance_ms(10_000)

    assert store.renew_lease(lease) is True

    member = command.evaluation_id.encode()
    expected = clock.now_ms + settings.lease_seconds * 1000
    assert redis_client.zscore(_key(settings, "running-leases"), member) == float(expected)
    visible = store.get_visible(command.evaluation_id, make_caller())
    assert visible.state == "running"
    assert visible.revision == 2
    assert visible.updated_at == visible.started_at


def test_internal_progress_writes_stay_private(
    store, make_command, make_worker, clock, settings, redis_client
):
    command, lease = _create_and_claim(store, make_command, make_worker, clock)

    assert store.set_active_check(lease, "correctness") is True
    assert store.save_check_outcome(lease, "correctness", make_correctness_success()) is True

    stored = redis_client.hgetall(_key(settings, f"resource:{command.evaluation_id}"))
    assert stored[b"active_check"] == b"correctness"
    outcomes = json.loads(stored[b"check_outcomes"])
    assert outcomes["correctness"]["status"] == "succeeded"
    assert outcomes["correctness"]["data"]["outcome"] == "correct"

    visible = store.get_visible(command.evaluation_id, make_caller())
    assert visible.state == "running"
    assert visible.revision == 2
    assert visible.result is None
    assert visible.finished_at is None


def test_commit_completed_is_one_atomic_terminal_transition(
    store, make_command, make_worker, clock, settings, redis_client
):
    command, lease = _create_and_claim(store, make_command, make_worker, clock)
    store.save_check_outcome(lease, "correctness", make_correctness_success())
    clock.advance_ms(20_000)

    result = make_completed_result(settings)
    assert store.commit_terminal(lease, "completed", result) is True

    visible = store.get_visible(command.evaluation_id, make_caller())
    assert visible.state == "completed"
    assert visible.revision == 3
    assert visible.finished_at == _now(clock)
    assert visible.updated_at == _now(clock)
    assert visible.result == result

    member = command.evaluation_id.encode()
    expiry_ms = clock.now_ms + settings.retention_seconds * 1000
    assert redis_client.zscore(_key(settings, "state:completed"), member) == float(expiry_ms)
    assert redis_client.zscore(_key(settings, "state:running"), member) is None
    assert redis_client.zscore(_key(settings, "deadlines"), member) is None
    assert redis_client.zscore(_key(settings, "running-leases"), member) is None
    assert redis_client.zrange(_key(settings, "ready"), 0, -1) == []

    retention = settings.retention_seconds
    resource_key = _key(settings, f"resource:{command.evaluation_id}")
    assert redis_client.ttl(resource_key) == retention
    idempotency_key = redis_client.hget(resource_key, "idempotency_key").decode()
    assert redis_client.ttl(idempotency_key) == retention

    caller_digest = sha256_hex(b"caller-1")
    session_digest = redis_client.hget(resource_key, "session_digest").decode()
    assert redis_client.get(_key(settings, f"inflight:caller:trial:{caller_digest}")) == b"0"
    assert redis_client.get(_key(settings, f"inflight:session:{session_digest}")) == b"0"


def test_commit_failed_from_running(store, make_command, make_worker, clock, settings):
    command, lease = _create_and_claim(store, make_command, make_worker, clock)
    result = make_completed_result(settings)

    assert store.commit_terminal(lease, "failed", result) is True

    visible = store.get_visible(command.evaluation_id, make_caller())
    assert visible.state == "failed"
    assert visible.revision == 3
    assert visible.result == result


def test_commit_timed_out_from_running(store, make_command, make_worker, clock, settings):
    command, lease = _create_and_claim(store, make_command, make_worker, clock)
    result = make_completed_result(settings)

    assert store.commit_terminal(lease, "timed_out", result) is True

    visible = store.get_visible(command.evaluation_id, make_caller())
    assert visible.state == "timed_out"
    assert visible.revision == 3


def test_terminal_state_is_immutable(store, make_command, make_worker, clock, settings):
    command, lease = _create_and_claim(store, make_command, make_worker, clock)
    result = make_completed_result(settings)
    assert store.commit_terminal(lease, "completed", result) is True
    before = store.get_visible(command.evaluation_id, make_caller())
    clock.advance_ms(5_000)

    assert store.commit_terminal(lease, "failed", result) is False
    assert store.renew_lease(lease) is False
    assert store.set_active_check(lease, "correctness") is False
    assert store.save_check_outcome(lease, "correctness", make_correctness_success()) is False
    assert store.expire_due_resources() == 0
    counts = store.recover_expired_leases()
    assert counts.requeued == 0
    assert counts.failed == 0

    after = store.get_visible(command.evaluation_id, make_caller())
    assert after == before


def test_revision_is_monotonic_and_updated_at_tracks_mutations(
    store, make_command, make_worker, clock, settings
):
    command = make_command()
    store.create_or_replay(command)
    accepted = store.get_visible(command.evaluation_id, make_caller())
    assert (accepted.revision, accepted.updated_at) == (1, _now(clock))

    clock.advance_ms(7_000)
    lease = store.claim_next(make_worker())
    running = store.get_visible(command.evaluation_id, make_caller())
    assert running.revision == 2
    assert running.updated_at == _now(clock)
    assert running.updated_at > accepted.updated_at

    clock.advance_ms(7_000)
    assert store.commit_terminal(lease, "completed", make_completed_result(settings)) is True
    terminal = store.get_visible(command.evaluation_id, make_caller())
    assert terminal.revision == 3
    assert terminal.updated_at == _now(clock)
    assert terminal.updated_at > running.updated_at


def test_replacement_attempt_keeps_first_started_at(
    store, make_command, make_worker, clock, settings
):
    command, first = _create_and_claim(store, make_command, make_worker, clock)
    started_at = store.get_visible(command.evaluation_id, make_caller()).started_at

    clock.advance_ms(settings.lease_seconds * 1000 + 1)
    counts = store.recover_expired_leases()
    assert counts.requeued == 1
    second = store.claim_next(make_worker())
    assert second is not None
    assert second.attempt_id != first.attempt_id
    assert second.fencing_token == first.fencing_token + 1
    assert second.expected_revision == 2

    visible = store.get_visible(command.evaluation_id, make_caller())
    assert visible.state == "running"
    assert visible.revision == 2
    assert visible.started_at == started_at
    assert visible.updated_at == started_at

    clock.advance_ms(3_000)
    assert store.commit_terminal(second, "completed", make_completed_result(settings)) is True
    terminal = store.get_visible(command.evaluation_id, make_caller())
    assert terminal.started_at == started_at
    assert terminal.finished_at == _now(clock)


def test_queued_deadline_commits_synthesized_timed_out_result(
    store, make_command, clock, settings
):
    request = make_request(
        checks=["correctness", "performance", "profile"], deadline_seconds=10
    )
    command = make_command(request=request)
    store.create_or_replay(command)
    clock.advance_ms(11_000)

    assert store.expire_due_resources() == 1

    visible = store.get_visible(command.evaluation_id, make_caller())
    assert visible.state == "timed_out"
    assert visible.revision == 2
    assert visible.started_at is None
    assert visible.finished_at == _now(clock)
    assert visible.updated_at == _now(clock)

    result = visible.result
    correctness = result.correctness
    assert correctness.status == "timeout"
    assert correctness.error.code == "DEADLINE_EXCEEDED"
    assert correctness.error.phase == "deadline"
    assert correctness.error.check == "correctness"
    assert correctness.error.retryable is False

    performance = result.performance
    assert performance.status == "skipped"
    assert performance.reason.code == "RESOURCE_DEADLINE"
    assert performance.reason.depends_on == "correctness"

    profile = result.profile
    assert profile.status == "skipped"
    assert profile.reason.code == "RESOURCE_DEADLINE"
    assert profile.reason.depends_on is None

    provenance = result.provenance
    assert provenance.evaluator_build == settings.evaluator_build
    assert provenance.correctness_algorithm == settings.correctness_algorithm_version
    assert provenance.decoy_algorithm == settings.decoy_algorithm_version
    assert provenance.benchmark_algorithm == settings.benchmark_algorithm_version
    assert provenance.profiler_algorithm == settings.profiler_algorithm_version
    assert provenance.grade_eligible is False
    assert provenance.manifest_id is None
    assert provenance.problem_id is None
    assert provenance.execution_environment is None


def test_running_deadline_synthesis_retains_persisted_outcomes(
    store, make_command, make_worker, clock
):
    request = make_request(checks=["correctness", "performance"], deadline_seconds=60)
    command, lease = _create_and_claim(
        store, make_command, make_worker, clock, request=request
    )
    store.set_active_check(lease, "correctness")
    store.save_check_outcome(lease, "correctness", make_correctness_success())
    store.set_active_check(lease, "performance")
    clock.advance_ms(61_000)

    assert store.expire_due_resources() == 1

    visible = store.get_visible(command.evaluation_id, make_caller())
    assert visible.state == "timed_out"
    assert visible.started_at is not None
    result = visible.result
    assert result.correctness.status == "succeeded"
    assert result.correctness.data.outcome == "correct"
    assert result.performance.status == "timeout"
    assert result.performance.error.code == "DEADLINE_EXCEEDED"
    assert result.performance.error.check == "performance"
    assert result.profile is None  # unrequested checks stay null
    assert result.provenance.execution_environment is None


def test_grade_deadline_result_pins_manifest_provenance(
    store, make_command, clock, settings
):
    command = make_command(
        caller=make_grade_caller(),
        request=make_grade_request(deadline_seconds=10),
    )
    store.create_or_replay(command)
    clock.advance_ms(11_000)

    assert store.expire_due_resources() == 1

    visible = store.get_visible(command.evaluation_id, make_grade_caller())
    provenance = visible.result.provenance
    assert provenance.grade_eligible is False
    assert provenance.manifest_id == "release-1"
    assert provenance.problem_id == "addmm-1"


def test_inflight_counters_never_go_below_zero(
    store, make_command, make_worker, clock, settings, redis_client
):
    command, lease = _create_and_claim(store, make_command, make_worker, clock)
    # Simulate a counter already drained by a duplicated maintenance pass.
    caller_digest = sha256_hex(b"caller-1")
    redis_client.set(_key(settings, f"inflight:caller:trial:{caller_digest}"), 0)

    assert store.commit_terminal(lease, "completed", make_completed_result(settings)) is True

    assert redis_client.get(_key(settings, f"inflight:caller:trial:{caller_digest}")) == b"0"


def test_fenced_writes_record_a_metric(
    store, make_command, make_worker, clock, settings, redis_client
):
    command, lease = _create_and_claim(store, make_command, make_worker, clock)
    stale = dataclasses.replace(lease, fencing_token=lease.fencing_token + 1)

    assert store.renew_lease(stale) is False
    assert store.set_active_check(stale, "correctness") is False
    assert store.save_check_outcome(stale, "correctness", make_correctness_success()) is False
    assert store.commit_terminal(stale, "completed", make_completed_result(settings)) is False

    metric = redis_client.get(_key(settings, "metrics:fenced-write-rejected"))
    assert int(metric) >= 4


def test_worker_and_attempt_ids_are_process_random(make_worker):
    # Attempt and worker IDs are random tokens; they never encode host/card
    # identity, and MAX_ATTEMPTS leaves room for at least one replacement.
    worker = make_worker()
    assert re.fullmatch(r"wk_[0-9a-f]{32}", worker.worker_id)
    assert make_worker().worker_id != worker.worker_id
    assert MAX_ATTEMPTS >= 2
