"""Attempt card provenance through the claim/recovery interface."""
import json

import pytest
import redis
from conftest import make_caller, make_completed_result

from triton_evaluation.broker import EvaluationStore


class _ExecEventRedis:
    """Redis adapter injecting an event before/after a selected transaction."""

    def __init__(self, client, event, *, when="after", transaction=1):
        self.client = client
        self.event = event
        self.when = when
        self.transaction = transaction
        self.executions = 0

    def __getattr__(self, name):
        return getattr(self.client, name)

    def pipeline(self, *args, **kwargs):
        pipeline = self.client.pipeline(*args, **kwargs)
        execute = pipeline.execute

        def execute_with_event(*args, **kwargs):
            self.executions += 1
            event = self.event if self.executions == self.transaction else None
            if event is not None and self.when == "before":
                event()
            result = execute(*args, **kwargs)
            if event is not None and self.when == "after":
                event()
            return result

        pipeline.execute = execute_with_event
        return pipeline


def test_claim_records_card_for_recovery_affinity(
    store, settings, clock, make_command, make_worker,
):
    command = make_command()
    store.create_or_replay(command)
    first = store.claim_next(make_worker(), card=3)
    assert first is not None
    clock.advance_ms(settings.lease_seconds * 1000 + 1)
    assert store.recover_expired_leases().requeued == 1

    assert store.claim_next(make_worker(), card=3, avoid_card=3) is None
    replacement = store.claim_next(make_worker(), card=7, avoid_card=7)
    assert replacement.evaluation_id == command.evaluation_id
    assert replacement.attempt_id != first.attempt_id


@pytest.mark.parametrize("when, transaction", [("after", 1), ("before", 2)])
def test_delayed_old_claim_cannot_overwrite_replacement_card(
    redis_client, settings, clock, make_command, make_worker, when, transaction,
):
    settings = settings.model_copy(update={"poison_death_threshold": 3})
    store = EvaluationStore(redis_client, settings, clock=clock)
    command = make_command()
    store.create_or_replay(command)
    replacements = []

    def replace_before_old_claim_returns():
        clock.advance_ms(settings.lease_seconds * 1000 + 1)
        assert store.recover_expired_leases().requeued == 1
        replacements.append(store.claim_next(make_worker(), card=3))

    delayed_store = EvaluationStore(
        _ExecEventRedis(
            redis_client, replace_before_old_claim_returns,
            when=when, transaction=transaction,
        ),
        settings, clock=clock,
    )
    stale = delayed_store.claim_next(make_worker(), card=7)
    assert stale.attempt_id != replacements[0].attempt_id

    clock.advance_ms(settings.lease_seconds * 1000 + 1)
    assert store.recover_expired_leases().requeued == 1
    # The last death belongs to card 3, even though card 7's claim finished last.
    assert store.claim_next(make_worker(), card=3, avoid_card=3) is None
    assert store.claim_next(make_worker(), card=7, avoid_card=7) is not None


def test_advisory_redis_outage_preserves_the_committed_claim(
    store, redis_client, redis_server, settings, clock, make_command, make_worker,
):
    command = make_command()
    store.create_or_replay(command)

    def disconnect():
        redis_server.connected = False

    interrupted = EvaluationStore(
        _ExecEventRedis(redis_client, disconnect), settings, clock=clock,
    )
    try:
        lease = interrupted.claim_next(make_worker(), card=7)
    finally:
        redis_server.connected = True
    assert lease is not None
    assert store.get_visible(command.evaluation_id, make_caller()).state == "running"
    assert store.renew_lease(lease)
    assert store.commit_terminal(lease, "completed", make_completed_result(settings))


def test_redis_outage_before_claim_commit_is_not_swallowed(
    store, redis_client, redis_server, settings, clock, make_command, make_worker,
):
    command = make_command()
    store.create_or_replay(command)

    def disconnect():
        redis_server.connected = False

    interrupted = EvaluationStore(
        _ExecEventRedis(redis_client, disconnect, when="before"), settings, clock=clock,
    )
    try:
        with pytest.raises(redis.ConnectionError):
            interrupted.claim_next(make_worker(), card=7)
    finally:
        redis_server.connected = True
    assert store.get_visible(command.evaluation_id, make_caller()).state == "queued"


def test_missing_replacement_card_after_redis_outage_does_not_quarantine(
    store, redis_client, redis_server, settings, clock, make_command, make_worker,
):
    command = make_command()
    store.create_or_replay(command)
    store.claim_next(make_worker(), card=7)
    clock.advance_ms(settings.lease_seconds * 1000 + 1)
    assert store.recover_expired_leases().requeued == 1

    def disconnect():
        redis_server.connected = False

    interrupted = EvaluationStore(
        _ExecEventRedis(redis_client, disconnect), settings, clock=clock,
    )
    try:
        assert interrupted.claim_next(make_worker(), card=3) is not None
    finally:
        redis_server.connected = True
    clock.advance_ms(settings.lease_seconds * 1000 + 1)
    counts = store.recover_expired_leases()
    assert (counts.requeued, counts.failed) == (1, 0)
    assert store.get_visible(command.evaluation_id, make_caller()).state == "running"


@pytest.mark.parametrize("remove_resource", [False, True])
def test_late_provenance_does_not_change_terminal_or_recreate_expired_resource(
    store, redis_client, settings, clock, make_command, make_worker, remove_resource,
):
    command = make_command()
    store.create_or_replay(command)
    terminal = []

    def expire_before_provenance():
        clock.advance_ms(901_000)
        assert store.expire_due_resources() == 1
        terminal.append(store.get_visible(command.evaluation_id, make_caller()))
        if remove_resource:
            # Model retention expiry while the original worker is suspended.
            redis_client.delete(f"{settings.redis_prefix}:resource:{command.evaluation_id}")

    delayed = EvaluationStore(
        _ExecEventRedis(redis_client, expire_before_provenance), settings, clock=clock,
    )
    assert delayed.claim_next(make_worker(), card=7) is not None
    assert store.get_visible(command.evaluation_id, make_caller()) == (
        None if remove_resource else terminal[0]
    )


def test_card_unaware_replacement_does_not_inherit_prior_attempt_provenance(
    store, settings, clock, make_command, make_worker,
):
    command = make_command()
    store.create_or_replay(command)
    store.claim_next(make_worker(), card=7)
    clock.advance_ms(settings.lease_seconds * 1000 + 1)
    assert store.recover_expired_leases().requeued == 1

    replacement = store.claim_next(make_worker())
    assert replacement is not None
    clock.advance_ms(settings.lease_seconds * 1000 + 1)
    counts = store.recover_expired_leases()
    assert (counts.requeued, counts.failed) == (1, 0)
    assert store.get_visible(command.evaluation_id, make_caller()).state == "running"


@pytest.mark.parametrize("card", [3, None])
def test_legacy_worker_late_write_cannot_change_new_attempt_attribution(
    store, redis_client, settings, clock, make_command, make_worker, card,
):
    command = make_command()
    store.create_or_replay(command)
    store.claim_next(make_worker(), card=card)
    # A pre-upgrade worker only knows the unscoped advisory HSET. This must
    # neither overwrite a current card nor supply one for a card-unaware claim.
    redis_client.hset(
        f"{settings.redis_prefix}:resource:{command.evaluation_id}", "claimed_card", "7"
    )
    clock.advance_ms(settings.lease_seconds * 1000 + 1)
    assert store.recover_expired_leases().requeued == 1
    if card == 3:
        assert store.claim_next(make_worker(), card=3, avoid_card=3) is None
    assert store.claim_next(make_worker(), card=7, avoid_card=7) is not None


def test_pre_upgrade_running_attempt_keeps_legacy_card_attribution(
    store, redis_client, settings, clock, make_command, make_worker,
):
    command = make_command()
    store.create_or_replay(command)
    store.claim_next(make_worker(), card=7)
    # This is the resource shape written by the pre-upgrade claim path.
    key = f"{settings.redis_prefix}:resource:{command.evaluation_id}"
    redis_client.hdel(key, "attempt_card")
    clock.advance_ms(settings.lease_seconds * 1000 + 1)
    assert store.recover_expired_leases().requeued == 1
    assert store.claim_next(make_worker(), card=7, avoid_card=7) is None
    assert store.claim_next(make_worker(), card=3, avoid_card=3) is not None


@pytest.mark.parametrize("binding", [
    b"",
    b"invalid JSON",
    b"[]",
    {"attempt_id": "at_old", "card": 7},
    {"card": True},
    {"card": -1},
])
def test_unusable_scoped_provenance_never_falls_back_to_legacy_card(
    store, redis_client, settings, clock, make_command, make_worker, binding,
):
    command = make_command()
    store.create_or_replay(command)
    lease = store.claim_next(make_worker(), card=7)
    if isinstance(binding, dict):
        binding = json.dumps({"attempt_id": lease.attempt_id, **binding})
    redis_client.hset(
        f"{settings.redis_prefix}:resource:{command.evaluation_id}", "attempt_card", binding
    )
    clock.advance_ms(settings.lease_seconds * 1000 + 1)
    assert store.recover_expired_leases().requeued == 1
    assert store.claim_next(make_worker(), card=7, avoid_card=7) is not None
