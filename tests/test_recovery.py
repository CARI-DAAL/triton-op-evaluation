"""Recovery: deadlines, worker crashes, reconnects, heartbeats, index cleanup."""
import fakeredis
from conftest import (
    make_caller,
    make_completed_result,
    make_correctness_success,
    make_request,
)

from triton_evaluation.api.contract import EvaluationResource
from triton_evaluation.api.contract_artifacts import sha256_hex
from triton_evaluation.service.broker import MAX_ATTEMPTS, EvaluationStore
from triton_evaluation.service.models import AlgorithmVersions, WorkerCompatibility


def _key(settings, suffix):
    return f"{settings.redis_prefix}:{suffix}"


def _fresh_store(redis_server, settings, clock):
    return EvaluationStore(
        fakeredis.FakeStrictRedis(server=redis_server), settings, clock=clock
    )


def _compatibility(worker):
    return WorkerCompatibility(
        evaluator_build=worker.evaluator_build, algorithms=worker.algorithms
    )


def redis_client_hgetall(store, evaluation_id):
    return store._r.hgetall(f"triton-eval:v2.1:resource:{evaluation_id}")


def test_queued_deadline_times_out_without_an_attempt(
    store, make_command, clock, settings, redis_client
):
    command = make_command(request=make_request(deadline_seconds=10))
    store.create_or_replay(command)
    clock.advance_ms(11_000)

    assert store.expire_due_resources() == 1

    visible = store.get_visible(command.evaluation_id, make_caller())
    assert visible.state == "timed_out"
    assert visible.started_at is None
    assert visible.result.correctness.status == "timeout"
    assert visible.result.correctness.error.code == "DEADLINE_EXCEEDED"
    # Delivery after the deadline never starts an evaluator.
    assert store.expire_due_resources() == 0
    member = command.evaluation_id.encode()
    assert redis_client.zscore(_key(settings, "deadlines"), member) is None
    assert redis_client.zrange(_key(settings, "ready:npu:interactive"), 0, -1) == []


def test_running_deadline_times_out_and_raises_the_fence(
    store, make_command, make_worker, clock, settings, redis_client
):
    command = make_command()
    store.create_or_replay(command)
    lease = store.claim_next(make_worker())
    store.save_check_outcome(lease, "correctness", make_correctness_success())
    clock.advance_ms(901_000)

    assert store.expire_due_resources() == 1

    visible = store.get_visible(command.evaluation_id, make_caller())
    assert visible.state == "timed_out"
    assert visible.result.correctness.status == "succeeded"  # retained outcome
    # The pre-deadline attempt is fenced: no late write can land.
    assert store.renew_lease(lease) is False
    assert store.commit_terminal(lease, "completed", make_completed_result(settings)) is False
    assert redis_client.zscore(
        _key(settings, "running-leases"), command.evaluation_id.encode()
    ) is None


def test_worker_crash_below_attempt_limit_requeues_and_reruns(
    store, make_command, make_worker, clock, settings, redis_client
):
    command = make_command()
    store.create_or_replay(command)
    first = store.claim_next(make_worker())
    store.set_active_check(first, "correctness")
    store.save_check_outcome(first, "correctness", make_correctness_success())
    clock.advance_ms(settings.lease_seconds * 1000 + 1)

    counts = store.recover_expired_leases()
    assert (counts.requeued, counts.failed) == (1, 0)

    # Public state stays running; exactly one fresh ready member exists.
    visible = store.get_visible(command.evaluation_id, make_caller())
    assert visible.state == "running"
    assert visible.revision == 2
    assert redis_client.zrange(_key(settings, "ready:npu:interactive"), 0, -1) == [
        command.evaluation_id.encode()
    ]

    second = store.claim_next(make_worker())
    assert second is not None
    assert second.fencing_token == first.fencing_token + 1
    # Replacement clears partial progress: the pure evaluation reruns cleanly.
    stored = redis_client.hgetall(_key(settings, f"resource:{command.evaluation_id}"))
    assert stored[b"check_outcomes"] == b"{}"
    assert stored[b"active_check"] == b""

    assert store.commit_terminal(second, "completed", make_completed_result(settings)) is True
    terminal = store.get_visible(command.evaluation_id, make_caller())
    assert terminal.state == "completed"
    assert terminal.result.correctness.status == "succeeded"


def test_worker_crash_at_attempt_limit_commits_worker_lost(
    store, make_command, make_worker, clock, settings, redis_client
):
    command = make_command()
    store.create_or_replay(command)
    lease = store.claim_next(make_worker())
    assert lease is not None
    for _ in range(MAX_ATTEMPTS - 1):
        clock.advance_ms(settings.lease_seconds * 1000 + 1)
        counts = store.recover_expired_leases()
        assert (counts.requeued, counts.failed) == (1, 0)
        lease = store.claim_next(make_worker())
        assert lease is not None
    clock.advance_ms(settings.lease_seconds * 1000 + 1)

    counts = store.recover_expired_leases()
    assert (counts.requeued, counts.failed) == (0, 1)

    visible = store.get_visible(command.evaluation_id, make_caller())
    assert visible.state == "failed"
    outcome = visible.result.correctness
    assert outcome.status == "failed"
    assert outcome.error.code == "WORKER_LOST"
    assert outcome.error.phase == "execution"
    assert outcome.error.check == "correctness"
    assert outcome.error.retryable is True
    assert visible.result.performance is None
    assert visible.result.profile is None

    # Terminal cleanup ran as part of the same transition.
    caller_digest = sha256_hex(b"caller-1")
    assert redis_client.get(_key(settings, f"inflight:caller:trial:{caller_digest}")) == b"0"
    retention = settings.retention_seconds
    assert redis_client.ttl(_key(settings, f"resource:{command.evaluation_id}")) == retention
    # No further reaping or claiming is possible.
    assert store.claim_next(make_worker()) is None
    assert store.recover_expired_leases().failed == 0


def test_worker_lost_result_retains_committed_check_outcomes(
    store, make_command, make_worker, clock, settings
):
    request = make_request(checks=["correctness", "performance"])
    command = make_command(request=request)
    store.create_or_replay(command)
    lease = store.claim_next(make_worker())
    for _ in range(MAX_ATTEMPTS - 1):
        clock.advance_ms(settings.lease_seconds * 1000 + 1)
        store.recover_expired_leases()
        lease = store.claim_next(make_worker())
    # The final attempt durably committed correctness, then died mid-performance.
    store.set_active_check(lease, "correctness")
    store.save_check_outcome(lease, "correctness", make_correctness_success())
    store.set_active_check(lease, "performance")
    clock.advance_ms(settings.lease_seconds * 1000 + 1)

    counts = store.recover_expired_leases()
    assert (counts.requeued, counts.failed) == (0, 1)

    result = store.get_visible(command.evaluation_id, make_caller()).result
    assert result.correctness.status == "succeeded"
    assert result.correctness.data.outcome == "correct"
    assert result.performance.status == "failed"
    assert result.performance.error.code == "WORKER_LOST"
    assert result.performance.error.check == "performance"
    assert result.profile is None


def test_duplicate_ready_member_for_a_running_resource_is_removed(
    store, make_command, make_worker, clock, settings, redis_client
):
    command = make_command()
    store.create_or_replay(command)
    lease = store.claim_next(make_worker())
    assert lease is not None
    # A duplicated queue delivery for an already-claimed resource.
    redis_client.zadd(_key(settings, "ready:npu:interactive"), {command.evaluation_id: 99})

    assert store.claim_next(make_worker()) is None

    assert redis_client.zrange(_key(settings, "ready:npu:interactive"), 0, -1) == []
    visible = store.get_visible(command.evaluation_id, make_caller())
    assert visible.state == "running"
    assert visible.revision == 2
    # The live lease is untouched.
    assert store.renew_lease(lease) is True


def test_redis_reconnect_preserves_resources_and_leases(
    redis_server, settings, clock, store, make_command, make_worker
):
    command = make_command()
    store.create_or_replay(command)
    lease = store.claim_next(make_worker())

    # A reconnecting worker/API gets a fresh client over the same data.
    reconnected = _fresh_store(redis_server, settings, clock)
    assert reconnected.renew_lease(lease) is True
    result = make_completed_result(settings)
    assert reconnected.commit_terminal(lease, "completed", result) is True

    visible = store.get_visible(command.evaluation_id, make_caller())
    assert visible.state == "completed"
    assert visible.result == result


def test_api_process_recreation_recovers_against_the_same_data(
    redis_server, settings, clock, store, make_command, make_worker
):
    command = make_command()
    store.create_or_replay(command)

    # A recreated API process rebuilds the store; queued work is still claimed.
    recreated = _fresh_store(redis_server, settings, clock)
    visible = recreated.get_visible(command.evaluation_id, make_caller())
    assert visible.state == "queued"
    assert visible.revision == 1

    lease = recreated.claim_next(make_worker())
    assert lease is not None
    assert recreated.commit_terminal(lease, "completed", make_completed_result(settings))
    assert recreated.expire_due_resources() == 0
    assert recreated.recover_expired_leases() == store.recover_expired_leases()


def test_heartbeat_readiness_requires_fresh_compatible_workers(
    store, make_worker, clock, settings, redis_client
):
    worker = make_worker()
    compatibility = _compatibility(worker)
    assert store.has_fresh_compatible_worker(compatibility) is False

    store.heartbeat(worker)
    assert store.has_fresh_compatible_worker(compatibility) is True

    # Incompatible evaluator build or any algorithm version does not count.
    foreign = make_worker()
    store.heartbeat(foreign)
    wrong_build = WorkerCompatibility(
        evaluator_build="git:" + "f" * 40, algorithms=worker.algorithms
    )
    assert store.has_fresh_compatible_worker(wrong_build) is False
    wrong_algorithms = WorkerCompatibility(
        evaluator_build=worker.evaluator_build,
        algorithms=AlgorithmVersions(
            correctness="correctness-v2",
            decoy=worker.algorithms.decoy,
            benchmark=worker.algorithms.benchmark,
            profiler=worker.algorithms.profiler,
        ),
    )
    assert store.has_fresh_compatible_worker(wrong_algorithms) is False

    # Stale heartbeats are pruned during readiness checks and never counted.
    clock.advance_ms(settings.lease_seconds * 1000 + 1)
    assert store.has_fresh_compatible_worker(compatibility) is False
    assert redis_client.hgetall(_key(settings, "worker-heartbeats")) == {}
    assert redis_client.zrange(_key(settings, "worker-heartbeat-expiry"), 0, -1) == []


def test_terminal_index_purge_removes_only_expired_resources(
    store, make_command, make_worker, clock, settings, redis_client
):
    command = make_command()
    store.create_or_replay(command)
    lease = store.claim_next(make_worker())
    assert store.commit_terminal(lease, "completed", make_completed_result(settings)) is True
    member = command.evaluation_id.encode()
    resource_key = _key(settings, f"resource:{command.evaluation_id}")

    # Score is in the future: not selected even if the hash vanished early.
    redis_client.delete(resource_key)
    assert store.purge_expired_terminal_indexes() == 0
    assert redis_client.zscore(_key(settings, "state:completed"), member) is not None

    # Past the retention expiry the member is removed only because the hash is
    # gone; a concurrent purge pass is idempotent.
    clock.advance_ms(settings.retention_seconds * 1000 + 1)
    assert store.purge_expired_terminal_indexes() == 1
    assert store.purge_expired_terminal_indexes() == 0
    assert redis_client.zscore(_key(settings, "state:completed"), member) is None


def test_terminal_index_purge_never_deletes_a_live_resources_index(
    store, make_command, make_worker, clock, settings, redis_client
):
    command = make_command()
    store.create_or_replay(command)
    lease = store.claim_next(make_worker())
    assert store.commit_terminal(lease, "completed", make_completed_result(settings)) is True

    clock.advance_ms(settings.retention_seconds * 1000 + 1)
    # The resource hash still exists (TTL expiry is lazy in the fake server
    # only when accessed; here the hash is genuinely present), so its index
    # member must survive cleanup.
    assert store.purge_expired_terminal_indexes() == 0
    member = command.evaluation_id.encode()
    assert redis_client.zscore(_key(settings, "state:completed"), member) is not None


def test_reap_counts_suspicious_death_and_records_failed_card(store, redis_client,
                                                              clock, settings,
                                                              make_command, make_worker):
    cmd = make_command()
    store.create_or_replay(cmd)
    worker = make_worker()
    store.claim_next(worker, card=7)
    clock.advance_ms(settings.lease_seconds * 1000 + 1)

    counts = store.recover_expired_leases()
    assert counts.requeued == 1
    key = f"triton-eval:v2.1:resource:{cmd.evaluation_id}"
    assert redis_client.hget(key, "suspicious_deaths") == b"1"
    assert redis_client.hget(key, "last_failed_card") == b"7"


def test_second_suspicious_death_quarantines_as_poison(store, clock, settings,
                                                            make_command, make_worker):
    cmd = make_command()
    store.create_or_replay(cmd)
    worker = make_worker()
    for _ in range(2):
        store.claim_next(worker, card=7)
        clock.advance_ms(settings.lease_seconds * 1000 + 1)
        counts = store.recover_expired_leases()
        if counts.failed:
            break
        # re-arm the replacement claim for the next death
        clock.advance_ms(1)
    stored = redis_client_hgetall(store, cmd.evaluation_id)
    assert stored[b"suspicious_deaths"] == b"2"
    resource = EvaluationResource.model_validate_json(stored[b"resource"])
    assert resource.state == "failed"
    assert resource.result.correctness.status == "failed"
    assert resource.result.correctness.error.code == "SUSPECTED_POISON_CANDIDATE"
    assert "poison" in resource.result.correctness.error.message


def test_clean_failure_does_not_count_suspicious_death(store, redis_client, clock,
                                                       settings, make_command,
                                                       make_worker):
    """A committed terminal before lease expiry removes the lease — the reaper
    never sees the attempt, so no suspicious death is counted (spec §11)."""
    cmd = make_command()
    store.create_or_replay(cmd)
    lease = store.claim_next(make_worker(), card=7)
    assert store.commit_terminal(lease, "completed", make_completed_result(settings))
    clock.advance_ms(settings.lease_seconds * 1000 + 1)
    store.recover_expired_leases()
    key = f"triton-eval:v2.1:resource:{cmd.evaluation_id}"
    assert redis_client.hget(key, "suspicious_deaths") == b"0"
    assert redis_client.hget(key, "last_failed_card") is None
