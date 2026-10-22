"""Fencing races: exactly one terminal winner, stale writes always rejected.

Races run on real threads against one shared fake Redis server, each thread
with its own connection and store — the same shape as production workers and
API maintenance loops.
"""
import dataclasses
import threading
from concurrent.futures import ThreadPoolExecutor

import fakeredis
from conftest import make_caller, make_completed_result, make_correctness_success

from triton_evaluation.service.broker import EvaluationStore


def _key(settings, suffix):
    return f"{settings.redis_prefix}:{suffix}"


def _fresh_store(redis_server, settings, clock):
    return EvaluationStore(
        fakeredis.FakeStrictRedis(server=redis_server), settings, clock=clock
    )


def _create_and_claim(store, make_command, make_worker, clock, **kwargs):
    command = make_command(**kwargs)
    store.create_or_replay(command)
    clock.advance_ms(5_000)
    lease = store.claim_next(make_worker())
    assert lease is not None
    return command, lease


def test_duplicate_claim_race_yields_exactly_one_lease(
    redis_server, settings, clock, store, make_command, make_worker
):
    command = make_command()
    store.create_or_replay(command)
    barrier = threading.Barrier(8)
    worker = make_worker()

    def race(_):
        contender = _fresh_store(redis_server, settings, clock)
        barrier.wait(timeout=10)
        return contender.claim_next(worker)

    with ThreadPoolExecutor(max_workers=8) as pool:
        leases = list(pool.map(race, range(8)))

    winners = [lease for lease in leases if lease is not None]
    assert len(winners) == 1
    assert winners[0].evaluation_id == command.evaluation_id
    assert winners[0].fencing_token == 1

    visible = store.get_visible(command.evaluation_id, make_caller())
    assert visible.state == "running"
    assert visible.revision == 2


def test_renewal_rejects_wrong_attempt_token_and_revision(
    store, make_command, make_worker, clock
):
    _, lease = _create_and_claim(store, make_command, make_worker, clock)

    wrong_attempt = dataclasses.replace(lease, attempt_id="at_" + "0" * 32)
    wrong_token = dataclasses.replace(lease, fencing_token=lease.fencing_token + 1)
    wrong_revision = dataclasses.replace(
        lease, expected_revision=lease.expected_revision + 1
    )
    assert store.renew_lease(wrong_attempt) is False
    assert store.renew_lease(wrong_token) is False
    assert store.renew_lease(wrong_revision) is False
    # The intact lease still renews.
    assert store.renew_lease(lease) is True


def test_renewal_rejects_writes_past_the_resource_deadline(
    store, make_command, make_worker, clock, settings
):
    _, lease = _create_and_claim(store, make_command, make_worker, clock)
    clock.advance_ms(lease.deadline_at_ms - clock.now_ms + 1000)

    assert store.renew_lease(lease) is False
    assert store.set_active_check(lease, "correctness") is False
    assert store.save_check_outcome(lease, "correctness", make_correctness_success()) is False


def test_expired_lease_is_replaced_and_old_attempt_loses_writes(
    store, make_command, make_worker, clock, settings
):
    command, first = _create_and_claim(store, make_command, make_worker, clock)
    clock.advance_ms(settings.lease_seconds * 1000 + 1)

    counts = store.recover_expired_leases()
    assert (counts.requeued, counts.failed) == (1, 0)

    # The old attempt is fenced immediately, before any replacement claims.
    assert store.renew_lease(first) is False
    assert store.commit_terminal(first, "completed", make_completed_result(settings)) is False

    second = store.claim_next(make_worker())
    assert second is not None
    assert second.fencing_token == first.fencing_token + 1
    assert second.attempt_id != first.attempt_id

    visible = store.get_visible(command.evaluation_id, make_caller())
    assert visible.state == "running"
    assert visible.revision == 2

    clock.advance_ms(2_000)
    assert store.commit_terminal(second, "completed", make_completed_result(settings)) is True
    terminal = store.get_visible(command.evaluation_id, make_caller())
    assert terminal.state == "completed"
    assert terminal.revision == 3


def test_old_attempt_finishing_before_replacement_wins(
    store, make_command, make_worker, clock, settings
):
    command, lease = _create_and_claim(store, make_command, make_worker, clock)

    assert store.commit_terminal(lease, "completed", make_completed_result(settings)) is True

    clock.advance_ms(settings.lease_seconds * 1000 + 1)
    counts = store.recover_expired_leases()
    assert (counts.requeued, counts.failed) == (0, 0)
    assert store.claim_next(make_worker()) is None

    visible = store.get_visible(command.evaluation_id, make_caller())
    assert visible.state == "completed"
    assert visible.revision == 3


def test_two_terminal_commits_have_exactly_one_winner(
    redis_server, settings, clock, store, make_command, make_worker
):
    command, lease = _create_and_claim(store, make_command, make_worker, clock)
    result = make_completed_result(settings)
    barrier = threading.Barrier(2)

    def race(_):
        contender = _fresh_store(redis_server, settings, clock)
        barrier.wait(timeout=10)
        return contender.commit_terminal(lease, "completed", result)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(race, range(2)))

    assert outcomes.count(True) == 1
    assert outcomes.count(False) == 1

    visible = store.get_visible(command.evaluation_id, make_caller())
    assert visible.state == "completed"
    assert visible.revision == 3
    assert visible.result == result

    # No terminal rewrite: a later stale commit keeps losing.
    assert store.commit_terminal(lease, "failed", result) is False
    assert store.get_visible(command.evaluation_id, make_caller()) == visible


def test_deadline_watchdog_racing_worker_commit_has_one_winner(
    redis_server, settings, clock, store, make_command, make_worker
):
    command, lease = _create_and_claim(store, make_command, make_worker, clock)
    result = make_completed_result(settings)
    clock.advance_ms(lease.deadline_at_ms - clock.now_ms + 1000)
    barrier = threading.Barrier(2)

    def worker_commit():
        contender = _fresh_store(redis_server, settings, clock)
        barrier.wait(timeout=10)
        return contender.commit_terminal(lease, "completed", result)

    def watchdog():
        contender = _fresh_store(redis_server, settings, clock)
        barrier.wait(timeout=10)
        return contender.expire_due_resources()

    with ThreadPoolExecutor(max_workers=2) as pool:
        committed, expired = pool.map(lambda f: f(), [worker_commit, watchdog])

    assert (committed is True) != (expired == 1)

    visible = store.get_visible(command.evaluation_id, make_caller())
    assert visible.revision == 3
    if committed:
        assert visible.state == "completed"
        assert visible.result.correctness.status == "succeeded"
    else:
        assert visible.state == "timed_out"
        assert visible.result.correctness.status == "timeout"
        assert visible.result.correctness.error.code == "DEADLINE_EXCEEDED"

    # The loser stays fenced forever; the terminal result is never rewritten.
    assert store.commit_terminal(lease, "completed", result) is False
    assert store.expire_due_resources() == 0
    assert store.get_visible(command.evaluation_id, make_caller()) == visible


def test_fenced_writes_increment_the_rejected_metric(
    redis_server, settings, clock, store, make_command, make_worker
):
    command, lease = _create_and_claim(store, make_command, make_worker, clock)
    clock.advance_ms(settings.lease_seconds * 1000 + 1)
    store.recover_expired_leases()
    store.claim_next(make_worker())

    metric_key = _key(settings, "metrics:fenced-write-rejected")
    client = fakeredis.FakeStrictRedis(server=redis_server)
    baseline = int(client.get(metric_key) or 0)

    assert store.commit_terminal(lease, "completed", make_completed_result(settings)) is False
    assert int(client.get(metric_key) or 0) > baseline
