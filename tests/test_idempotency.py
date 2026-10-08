"""Idempotent admission: replay, conflicts, canonical equivalence, quotas, races."""
import threading
from concurrent.futures import ThreadPoolExecutor

import fakeredis
from conftest import make_caller, make_grade_caller, make_grade_request, make_request

from triton_evaluation.api.contract_artifacts import sha256_hex
from triton_evaluation.config import Settings
from triton_evaluation.service.broker import EvaluationStore, _digest_text


def _key(settings, suffix):
    return f"{settings.redis_prefix}:{suffix}"


def _idem_key(settings, caller_id, idempotency_key):
    digest = sha256_hex(f"{caller_id}\0{idempotency_key}".encode())
    return _key(settings, f"idempotency:{digest}")


def _store_with(redis_client, clock, **overrides):
    return EvaluationStore(redis_client, Settings(**overrides), clock=clock)


def test_same_key_same_request_replays_without_new_writes(
    store, redis_client, settings, make_command
):
    first = store.create_or_replay(make_command())
    assert first.decision == "created"

    replay = store.create_or_replay(make_command())

    assert replay.decision == "replayed"
    assert replay.resource == first.resource
    assert replay.resource.evaluation_id == first.resource.evaluation_id
    assert redis_client.get(_key(settings, "queue-sequence")) == b"1"
    assert redis_client.zcard(_key(settings, "ready:npu:interactive")) == 1
    caller_digest = sha256_hex(b"caller-1")
    session_digest = sha256_hex(b"caller-1\0session-1")
    assert redis_client.get(_key(settings, f"inflight:caller:trial:{caller_digest}")) == b"1"
    assert redis_client.get(_key(settings, f"inflight:session:{session_digest}")) == b"1"


def test_idempotency_record_stores_only_evaluation_id_and_digest(
    store, redis_client, settings, make_command
):
    command = make_command()
    store.create_or_replay(command)

    record = redis_client.hgetall(_idem_key(settings, "caller-1", "key-1"))
    assert record == {
        b"evaluation_id": command.evaluation_id.encode(),
        b"request_digest": command.admission.request_digest.encode(),
    }
    assert redis_client.ttl(_idem_key(settings, "caller-1", "key-1")) == -1


def test_same_key_different_request_conflicts(store, make_command):
    assert store.create_or_replay(make_command()).decision == "created"

    conflict = store.create_or_replay(make_command(request=make_request(op_name="other")))

    assert conflict.decision == "conflict"
    assert conflict.resource is None


def test_same_key_across_callers_creates_independent_resources(
    store, redis_client, settings, make_command
):
    first = store.create_or_replay(make_command(caller=make_caller("caller-1")))
    second = store.create_or_replay(make_command(caller=make_caller("caller-2")))

    assert first.decision == "created"
    assert second.decision == "created"
    assert first.resource.evaluation_id != second.resource.evaluation_id
    assert redis_client.zcard(_key(settings, "ready:npu:interactive")) == 2


def test_replay_compares_exact_canonical_bytes_not_digest_alone(
    store, redis_client, settings, make_command
):
    command = make_command()
    assert store.create_or_replay(command).decision == "created"
    # A record whose stored request digest matches but whose stored canonical
    # bytes differ is a conflict, never a replay.
    redis_client.hset(
        _key(settings, f"resource:{command.evaluation_id}"),
        "canonical_request",
        command.admission.canonical_request + b" ",
    )

    result = store.create_or_replay(command)

    assert result.decision == "conflict"
    assert result.resource is None


def test_defaulted_params_canonical_equivalence(store, make_command):
    explicit = make_request(
        checks=["correctness", "performance"], performance={"warmup": 5, "repeats": 50}
    )
    implicit = make_request(checks=["correctness", "performance"])
    first = make_command(request=explicit)
    replay = make_command(request=implicit)
    assert first.admission.canonical_request == replay.admission.canonical_request

    assert store.create_or_replay(first).decision == "created"
    result = store.create_or_replay(replay)

    assert result.decision == "replayed"
    assert result.resource.evaluation_id == first.evaluation_id


def test_check_order_canonical_equivalence(store, make_command):
    forward = make_command(request=make_request(checks=["correctness", "performance"]))
    reversed_ = make_command(request=make_request(checks=["performance", "correctness"]))
    assert forward.admission.canonical_request == reversed_.admission.canonical_request

    assert store.create_or_replay(forward).decision == "created"
    assert store.create_or_replay(reversed_).decision == "replayed"


def test_case_index_order_canonical_equivalence(store, make_command):
    ascending = make_command(
        request=make_request(checks=["correctness", "profile"], profile={"case_indices": [0, 2]})
    )
    shuffled = make_command(
        request=make_request(checks=["correctness", "profile"], profile={"case_indices": [2, 0]})
    )
    assert ascending.admission.canonical_request == shuffled.admission.canonical_request

    assert store.create_or_replay(ascending).decision == "created"
    assert store.create_or_replay(shuffled).decision == "replayed"


def test_rate_limit_rejects_new_requests(redis_client, clock, make_command):
    store = _store_with(redis_client, clock, trial_rate_limit_per_minute=2)
    assert store.create_or_replay(make_command(idempotency_key="k1")).decision == "created"
    assert store.create_or_replay(make_command(idempotency_key="k2")).decision == "created"

    limited = store.create_or_replay(make_command(idempotency_key="k3"))

    assert limited.decision == "rate_limited"
    assert limited.resource is None
    caller_digest = sha256_hex(b"caller-1")
    rate_keys = list(redis_client.scan_iter(match=f"*:rate:trial:{caller_digest}:*"))
    assert len(rate_keys) == 1
    assert redis_client.get(rate_keys[0]) == b"2"


def test_replay_bypasses_exhausted_rate_limit(redis_client, clock, make_command):
    store = _store_with(redis_client, clock, trial_rate_limit_per_minute=1)
    command = make_command(idempotency_key="k1")
    assert store.create_or_replay(command).decision == "created"
    assert store.create_or_replay(make_command(idempotency_key="k2")).decision == "rate_limited"

    replay = store.create_or_replay(make_command(idempotency_key="k1"))

    assert replay.decision == "replayed"
    assert replay.resource.evaluation_id == command.evaluation_id


def test_conflict_and_replay_do_not_charge_quota(redis_client, clock, make_command):
    store = _store_with(redis_client, clock, trial_rate_limit_per_minute=2)
    assert store.create_or_replay(make_command(idempotency_key="k1")).decision == "created"
    replay = store.create_or_replay(make_command(idempotency_key="k1"))
    assert replay.decision == "replayed"
    conflict = store.create_or_replay(
        make_command(idempotency_key="k1", request=make_request(op_name="other"))
    )
    assert conflict.decision == "conflict"

    # Only the initial create consumed quota; one more new request still fits.
    assert store.create_or_replay(make_command(idempotency_key="k2")).decision == "created"
    assert store.create_or_replay(make_command(idempotency_key="k3")).decision == "rate_limited"


def test_caller_inflight_limit(redis_client, clock, make_command):
    store = _store_with(redis_client, clock, trial_max_inflight=2)
    first = store.create_or_replay(make_command(idempotency_key="k1", session_id="s1"))
    second = store.create_or_replay(make_command(idempotency_key="k2", session_id="s2"))
    assert first.decision == "created"
    assert second.decision == "created"

    limited = store.create_or_replay(make_command(idempotency_key="k3", session_id="s3"))

    assert limited.decision == "limit_exceeded"
    assert limited.resource is None


def test_session_inflight_limit(redis_client, clock, make_command):
    store = _store_with(redis_client, clock, session_max_inflight=1)
    assert store.create_or_replay(make_command(idempotency_key="k1")).decision == "created"

    limited = store.create_or_replay(make_command(idempotency_key="k2"))
    assert limited.decision == "limit_exceeded"

    other_session = store.create_or_replay(make_command(idempotency_key="k3", session_id="s2"))
    assert other_session.decision == "created"


def test_session_limits_are_per_caller(redis_client, clock, make_command):
    store = _store_with(redis_client, clock, session_max_inflight=1)
    first = store.create_or_replay(make_command(caller=make_caller("caller-1")))
    assert first.decision == "created"

    other_caller = store.create_or_replay(make_command(caller=make_caller("caller-2")))

    assert other_caller.decision == "created"


def test_trial_and_trusted_quotas_are_independent(
    redis_client, clock, settings, make_command
):
    store = _store_with(
        redis_client, clock, trial_max_inflight=1, trusted_max_inflight=1
    )
    assert store.create_or_replay(make_command(idempotency_key="k1")).decision == "created"
    assert store.create_or_replay(make_command(idempotency_key="k2")).decision == "limit_exceeded"

    grade = store.create_or_replay(
        make_command(caller=make_grade_caller(), request=make_grade_request())
    )

    assert grade.decision == "created"
    trusted_digest = sha256_hex(b"grader-1")
    assert (
        redis_client.get(_key(settings, f"inflight:caller:trusted:{trusted_digest}")) == b"1"
    )
    # The trial caller never touched the trusted rate bucket.
    trial_digest = sha256_hex(b"caller-1")
    pattern = f"*:rate:trusted:{trial_digest}:*"
    assert list(redis_client.scan_iter(match=pattern)) == []


def test_observer_grade_submit_consumes_trusted_capacity(
    redis_client, clock, settings, make_command
):
    observer = make_caller("observer-1", "observer", ("observer:read", "grade:submit"))
    store = EvaluationStore(redis_client, settings, clock=clock)

    result = store.create_or_replay(
        make_command(caller=observer, request=make_grade_request())
    )

    assert result.decision == "created"
    observer_digest = sha256_hex(b"observer-1")
    assert (
        redis_client.get(_key(settings, f"inflight:caller:trusted:{observer_digest}"))
        == b"1"
    )
    assert list(
        redis_client.scan_iter(match=f"*:inflight:caller:trial:{observer_digest}")
    ) == []


def test_sixteen_concurrent_identical_creates_create_exactly_one_resource(
    redis_server, settings, clock, make_command
):
    barrier = threading.Barrier(16)

    def race(_):
        client = fakeredis.FakeStrictRedis(server=redis_server)
        store = EvaluationStore(client, settings, clock=clock)
        barrier.wait(timeout=10)
        return store.create_or_replay(make_command())

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(race, range(16)))

    decisions = [result.decision for result in results]
    assert decisions.count("created") == 1
    assert decisions.count("replayed") == 15
    assert len({result.resource.evaluation_id for result in results}) == 1

    client = fakeredis.FakeStrictRedis(server=redis_server)
    assert client.get(_key(settings, "queue-sequence")) == b"1"
    assert client.zcard(_key(settings, "ready:npu:interactive")) == 1
    caller_digest = sha256_hex(b"caller-1")
    session_digest = sha256_hex(b"caller-1\0session-1")
    assert client.get(_key(settings, f"inflight:caller:trial:{caller_digest}")) == b"1"
    assert client.get(_key(settings, f"inflight:session:{session_digest}")) == b"1"
    rate_keys = list(client.scan_iter(match=f"*:rate:trial:{caller_digest}:*"))
    assert len(rate_keys) == 1
    assert client.get(rate_keys[0]) == b"1"


def test_rollout_inflight_limit_is_independent(redis_client, clock, make_command):
    store = _store_with(redis_client, clock, rollout_max_inflight=1)
    rollout_caller = make_caller(caller_id="rl-runner-1")

    first = make_command(caller=rollout_caller, session_id="s-rl-1",
                         idempotency_key="k-rl-1", traffic_class="rollout")
    assert store.create_or_replay(first).decision == "created"

    blocked = make_command(caller=rollout_caller, session_id="s-rl-2",
                           idempotency_key="k-rl-2", traffic_class="rollout")
    assert store.create_or_replay(blocked).decision == "limit_exceeded"

    trial = make_command()  # separate bucket, unaffected
    assert store.create_or_replay(trial).decision == "created"

    rollout_key = "triton-eval:v2.1:inflight:caller:rollout:" + _digest_text("rl-runner-1")
    assert int(redis_client.get(rollout_key)) == 1
    trial_key = "triton-eval:v2.1:inflight:caller:trial:" + _digest_text("caller-1")
    assert int(redis_client.get(trial_key)) == 1


def test_rollout_rate_limit_uses_rollout_bucket(redis_client, clock, make_command):
    store = _store_with(redis_client, clock, rollout_rate_limit_per_minute=1)
    caller = make_caller(caller_id="rl-runner-1")
    first = make_command(caller=caller, session_id="s1", idempotency_key="k1",
                         traffic_class="rollout")
    assert store.create_or_replay(first).decision == "created"
    second = make_command(caller=caller, session_id="s2", idempotency_key="k2",
                          traffic_class="rollout")
    assert store.create_or_replay(second).decision == "rate_limited"
