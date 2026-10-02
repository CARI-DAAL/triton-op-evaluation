"""Durable budget semantics across API instances, sessions and concurrent calls."""

from concurrent.futures import ThreadPoolExecutor

import pytest
import redis

from triton_evaluation.broker import StorageError
from triton_evaluation.kernel_eval_budget import EvalBudgetRegistry


def registry(client, maximum=18):
    return EvalBudgetRegistry(client, prefix="test", maximum=maximum, retention_seconds=86400)


def test_all_actions_share_trial_budget_and_replay_survives_restart(redis_client):
    first = registry(redis_client, maximum=2)
    assert first.reserve("caller", "trial", "verify", "digest-v").turn == 1
    assert first.reserve("caller", "trial", "benchmark", "digest-b").turn == 2
    restarted = registry(redis_client, maximum=2)
    replay = restarted.reserve("caller", "trial", "verify", "digest-v")
    assert (replay.turn, replay.used, replay.decision) == (1, 1, "replayed")
    blocked = restarted.reserve("caller", "trial", "profile", "digest-p")
    assert (blocked.turn, blocked.used, blocked.decision) == (None, 2, "budget_exhausted")
    assert restarted.reserve("caller", "trial-2", "profile", "digest-p").turn == 1


def test_conflicting_key_does_not_consume_another_turn(redis_client):
    budget = registry(redis_client)
    budget.reserve("caller", "trial", "key", "original")
    assert budget.reserve("caller", "trial", "key", "changed").decision == "conflict"
    assert budget.reserve("caller", "trial", "new-key", "changed").turn == 2


def test_concurrent_api_processes_cannot_exceed_eighteen(redis_client):
    def reserve(index):
        return registry(redis_client).reserve("caller", "trial", str(index), str(index))

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(reserve, range(48)))
    accepted = [result.turn for result in results if result.decision == "reserved"]
    assert sorted(accepted) == list(range(1, 19))
    assert sum(result.decision == "budget_exhausted" for result in results) == 30


def test_storage_failure_has_no_memory_fallback(redis_client, monkeypatch):
    def fail():
        raise redis.ConnectionError("private endpoint")

    monkeypatch.setattr(redis_client, "pipeline", fail)
    with pytest.raises(StorageError):
        registry(redis_client).reserve("caller", "trial", "key", "digest")
