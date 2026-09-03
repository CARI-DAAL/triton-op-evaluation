"""Durable resource admission: creation shape, FIFO ordering, keyspace, TTLs."""
import re
from datetime import datetime, timedelta, timezone

import pytest
from conftest import make_caller, make_completed_result, make_request

from triton_evaluation.api.contract import EvaluationResource
from triton_evaluation.api.contract_artifacts import sha256_hex
from triton_evaluation.broker import EvaluationStore, StorageError, _digest_text
from triton_evaluation.config import Settings
from triton_evaluation.models import new_evaluation_id


def _key(settings, suffix):
    return f"{settings.redis_prefix}:{suffix}"


def _idem_key(settings, caller_id, idempotency_key):
    digest = sha256_hex(f"{caller_id}\0{idempotency_key}".encode())
    return _key(settings, f"idempotency:{digest}")


def test_new_evaluation_id_format():
    evaluation_id = new_evaluation_id()
    assert re.fullmatch(r"ev_[0-9a-f]{32}", evaluation_id)
    assert new_evaluation_id() != evaluation_id


def test_create_returns_initial_queued_resource(store, make_command, clock):
    command = make_command()
    result = store.create_or_replay(command)

    assert result.decision == "created"
    resource = result.resource
    accepted_at = datetime.fromtimestamp(clock.now_ms / 1000, tz=timezone.utc)
    assert resource.schema_version == "2.0"
    assert resource.evaluation_id == command.evaluation_id
    assert resource.revision == 1
    assert resource.state == "queued"
    assert resource.created_at == accepted_at
    assert resource.updated_at == accepted_at
    assert resource.started_at is None
    assert resource.finished_at is None
    assert resource.deadline_at == accepted_at + timedelta(
        seconds=make_request().deadline_seconds
    )
    assert resource.request_digest == command.admission.request_digest
    assert resource.reference_digest == command.admission.reference_digest
    assert resource.result is None


def test_resource_hash_stores_exact_canonical_bytes_and_metadata(
    store, redis_client, settings, make_command, clock
):
    command = make_command()
    result = store.create_or_replay(command)

    stored = redis_client.hgetall(_key(settings, f"resource:{command.evaluation_id}"))
    assert stored[b"canonical_request"] == command.admission.canonical_request
    assert EvaluationResource.model_validate_json(stored[b"resource"]) == result.resource
    assert stored[b"owner"] == b"caller-1"
    assert stored[b"purpose"] == b"trial"
    assert stored[b"session_digest"] == sha256_hex(b"caller-1\0session-1").encode()
    assert stored[b"idempotency_key"] == _idem_key(settings, "caller-1", "key-1").encode()
    assert stored[b"request_digest"] == command.admission.request_digest.encode()
    assert stored[b"reference_digest"] == command.admission.reference_digest.encode()
    assert stored[b"requested_checks"] == b'["correctness"]'
    assert stored[b"evaluator_build"] == settings.evaluator_build.encode()
    assert stored[b"correctness_algorithm"] == settings.correctness_algorithm_version.encode()
    assert stored[b"decoy_algorithm"] == settings.decoy_algorithm_version.encode()
    assert stored[b"benchmark_algorithm"] == settings.benchmark_algorithm_version.encode()
    assert stored[b"profiler_algorithm"] == settings.profiler_algorithm_version.encode()
    assert stored[b"attempt_count"] == b"0"
    assert stored[b"created_at_ms"] == str(clock.now_ms).encode()
    assert stored[b"updated_at_ms"] == str(clock.now_ms).encode()
    expected_deadline_ms = clock.now_ms + make_request().deadline_seconds * 1000
    assert stored[b"deadline_ms"] == str(expected_deadline_ms).encode()


def test_fifo_sequence_numbers_are_monotonic(store, redis_client, settings, make_command):
    ids = []
    for index in range(3):
        command = make_command(idempotency_key=f"key-{index}", session_id=f"s-{index}")
        assert store.create_or_replay(command).decision == "created"
        ids.append(command.evaluation_id)

    assert redis_client.get(_key(settings, "queue-sequence")) == b"3"
    entries = redis_client.zrange(_key(settings, "ready:exec:interactive"), 0, -1, withscores=True)
    assert entries == [(ids[0].encode(), 1.0), (ids[1].encode(), 2.0), (ids[2].encode(), 3.0)]


def test_state_and_deadline_indexes_track_queued_resource(
    store, redis_client, settings, make_command, clock
):
    command = make_command()
    store.create_or_replay(command)

    deadline_ms = float(clock.now_ms + make_request().deadline_seconds * 1000)
    member = command.evaluation_id.encode()
    assert redis_client.zscore(_key(settings, "deadlines"), member) == deadline_ms
    assert redis_client.zscore(_key(settings, "state:queued"), member) == deadline_ms
    # Indexes carry only evaluation IDs — never request source.
    assert redis_client.zrange(_key(settings, "deadlines"), 0, -1) == [member]
    assert redis_client.zrange(_key(settings, "state:queued"), 0, -1) == [member]


def test_nonterminal_keys_have_no_expiry(store, redis_client, settings, make_command, clock):
    command = make_command()
    store.create_or_replay(command)

    caller_digest = sha256_hex(b"caller-1")
    session_digest = sha256_hex(b"caller-1\0session-1")
    minute = clock.now_ms // 60_000
    no_expiry_keys = [
        _key(settings, f"resource:{command.evaluation_id}"),
        _idem_key(settings, "caller-1", "key-1"),
        _key(settings, "queue-sequence"),
        _key(settings, f"inflight:caller:trial:{caller_digest}"),
        _key(settings, f"inflight:session:{session_digest}"),
    ]
    for key in no_expiry_keys:
        assert redis_client.ttl(key) == -1, key
    rate_key = _key(settings, f"rate:trial:{caller_digest}:{minute}")
    assert redis_client.ttl(rate_key) == 120


def test_idempotency_record_mapping_to_missing_resource_is_storage_failure(
    store, redis_client, settings, make_command
):
    command = make_command()
    redis_client.hset(
        _idem_key(settings, "caller-1", "key-1"),
        mapping={
            "evaluation_id": "ev_" + "f" * 32,
            "request_digest": command.admission.request_digest,
        },
    )

    with pytest.raises(StorageError):
        store.create_or_replay(command)


def test_store_rejects_retention_below_recovery_window(redis_client, clock):
    settings = Settings.model_construct(
        redis_prefix="triton-eval:v2",
        retention_seconds=60,
        max_recovery_age_seconds=120,
    )
    with pytest.raises(ValueError, match="retention"):
        EvaluationStore(redis_client, settings, clock=clock)


def test_default_clock_comes_from_redis_time(redis_client, settings, make_command):
    store = EvaluationStore(redis_client, settings)
    before = datetime.now(timezone.utc)
    result = store.create_or_replay(make_command())
    after = datetime.now(timezone.utc)

    assert result.decision == "created"
    # The store clock truncates to epoch milliseconds; allow that rounding.
    assert before - timedelta(milliseconds=1) <= result.resource.created_at <= after
    assert result.resource.deadline_at - result.resource.created_at == timedelta(seconds=900)


def test_terminal_commit_releases_rollout_inflight(
    store, redis_client, make_command, make_worker, settings,
):
    cmd = make_command(
        caller=make_caller(caller_id="rl-1"),
        session_id="s1",
        idempotency_key="k1",
        traffic_class="rollout",
    )
    store.create_or_replay(cmd)
    rollout_key = _key(settings, f"inflight:caller:rollout:{_digest_text('rl-1')}")
    assert int(redis_client.get(rollout_key)) == 1

    lease = store.claim_next(make_worker())
    assert lease is not None
    assert store.commit_terminal(lease, "completed", make_completed_result(settings))
    assert int(redis_client.get(rollout_key)) == 0


def test_terminal_release_derives_class_when_field_missing(
    store, redis_client, make_command, make_worker, settings,
):
    """Pre-deploy resources lack traffic_class; purpose-derived fallback releases
    the right bucket (durable Redis volume outlives the deploy)."""
    cmd = make_command()  # trial caller-1
    store.create_or_replay(cmd)
    redis_client.hdel(_key(settings, f"resource:{cmd.evaluation_id}"), "traffic_class")

    lease = store.claim_next(make_worker())
    assert store.commit_terminal(lease, "completed", make_completed_result(settings))
    trial_key = _key(settings, f"inflight:caller:trial:{_digest_text('caller-1')}")
    assert int(redis_client.get(trial_key)) == 0


def _queue_members(redis_client):
    return {
        "interactive": set(
            redis_client.zrange("triton-eval:v2:ready:exec:interactive", 0, -1)
        ),
        "bulk": set(redis_client.zrange("triton-eval:v2:ready:exec:bulk", 0, -1)),
        "legacy": set(redis_client.zrange("triton-eval:v2:ready", 0, -1)),
    }


def _rollout_command(make_command, n):
    return make_command(caller=make_caller(caller_id="rl-1"), session_id=f"s-{n}",
                        idempotency_key=f"k-{n}", traffic_class="rollout")


def test_rollout_routes_to_bulk_queue(store, redis_client, make_command):
    cmd = _rollout_command(make_command, 1)
    store.create_or_replay(cmd)
    queues = _queue_members(redis_client)
    assert cmd.evaluation_id.encode() in queues["bulk"]
    assert not queues["interactive"] and not queues["legacy"]


def test_trial_routes_to_interactive_queue(store, redis_client, make_command):
    cmd = make_command()
    store.create_or_replay(cmd)
    queues = _queue_members(redis_client)
    assert cmd.evaluation_id.encode() in queues["interactive"]
    assert not queues["bulk"]


def test_interactive_claim_precedes_older_bulk(store, redis_client, make_command,
                                               make_worker):
    older_bulk = _rollout_command(make_command, 1)
    store.create_or_replay(older_bulk)
    newer_interactive = make_command(session_id="s-2", idempotency_key="k-2")
    store.create_or_replay(newer_interactive)

    worker = make_worker()
    first = store.claim_next(worker)
    assert first.evaluation_id == newer_interactive.evaluation_id
    second = store.claim_next(worker)
    assert second.evaluation_id == older_bulk.evaluation_id


def test_bulk_drains_in_fifo_order_when_interactive_empty(store, make_command,
                                                          make_worker):
    cmds = [_rollout_command(make_command, n) for n in (1, 2)]
    for cmd in cmds:
        store.create_or_replay(cmd)
    worker = make_worker()
    assert store.claim_next(worker).evaluation_id == cmds[0].evaluation_id
    assert store.claim_next(worker).evaluation_id == cmds[1].evaluation_id
    assert store.claim_next(worker) is None


def test_recovery_requeues_rollout_to_bulk(store, redis_client, clock, settings,
                                           make_command, make_worker):
    cmd = _rollout_command(make_command, 1)
    store.create_or_replay(cmd)
    store.claim_next(make_worker())
    clock.advance_ms(settings.lease_seconds * 1000 + 1)
    counts = store.recover_expired_leases()
    assert counts.requeued == 1
    queues = _queue_members(redis_client)
    assert cmd.evaluation_id.encode() in queues["bulk"]
    assert not queues["interactive"]


def test_legacy_ready_zset_still_drains(store, redis_client, make_command,
                                        make_worker):
    """Pre-deploy members of the old ready zset remain claimable (durable volume)."""
    cmd = make_command()
    store.create_or_replay(cmd)
    member = cmd.evaluation_id.encode()
    redis_client.zrem("triton-eval:v2:ready:exec:interactive", member)
    redis_client.zadd("triton-eval:v2:ready", {member: 1})
    lease = store.claim_next(make_worker())
    assert lease is not None and lease.evaluation_id == cmd.evaluation_id
