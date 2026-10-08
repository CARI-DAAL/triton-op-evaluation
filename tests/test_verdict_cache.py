"""Advisory cache scope, atomic publication, precedence, and expiry."""
import importlib.metadata
import sys
import time
from unittest import mock

import pytest
import redis

from triton_evaluation.runtime import verdict_cache


def test_digest_is_stable_and_scoped():
    digest = verdict_cache.digest_for("code-a", "impl-a", "cann-a")
    assert digest == verdict_cache.digest_for("code-a", "impl-a", "cann-a")
    assert digest != verdict_cache.digest_for("code-b", "impl-a", "cann-a")
    assert digest != verdict_cache.digest_for("code-a", "impl-b", "cann-a")
    assert digest != verdict_cache.digest_for("code-a", "impl-a", "cann-b")


@pytest.mark.parametrize("package", ["torch", "torch-npu", "triton", "triton-ascend"])
def test_runtime_upgrade_changes_digest(package):
    with mock.patch.object(importlib.metadata, "version", return_value="1.0"):
        original = verdict_cache.digest_for("code", "impl", "cann")
    with mock.patch.object(
        importlib.metadata, "version",
        side_effect=lambda name: "2.0" if name == package else "1.0",
    ):
        upgraded = verdict_cache.digest_for("code", "impl", "cann")
    assert original != upgraded


def test_missing_cann_identity_disables_cache_without_touching_redis(store):
    digest = verdict_cache.digest_for("code", "impl", "")
    assert digest is None
    with mock.patch.object(store._r, "hgetall") as read:
        with mock.patch.object(store._r, "pipeline") as write:
            assert verdict_cache.get(store, digest) is None
            assert verdict_cache.put(store, digest, "compile_error", 60) is None
    read.assert_not_called()
    write.assert_not_called()


def test_python_upgrade_changes_digest():
    original = verdict_cache.digest_for("code", "impl", "cann")
    with mock.patch.object(sys, "version", "different Python runtime"):
        upgraded = verdict_cache.digest_for("code", "impl", "cann")
    assert original != upgraded


def test_put_get_roundtrip_and_expiry(store, redis_client, settings):
    digest = verdict_cache.digest_for("k", "i", "cann")
    assert verdict_cache.get(store, digest) is None
    verdict_cache.put(store, digest, "compile_error", settings.verdict_cache_ttl_seconds)
    assert verdict_cache.get(store, digest) == "compile_error"
    key = f"{settings.redis_prefix}:verdict:{digest}"
    assert redis_client.ttl(key) > 0
    # fakeredis key expiry runs on real time, not the store's FakeClock:
    # jump the wall clock past the TTL for the read instead.
    future = time.time() + settings.verdict_cache_ttl_seconds + 1
    with mock.patch("time.time", return_value=future):
        assert verdict_cache.get(store, digest) is None  # expired


@pytest.mark.parametrize("first", ["compile_error", "precheck_passed"])
def test_negative_takes_precedence_in_either_publication_order(store, first):
    second = "precheck_passed" if first == "compile_error" else "compile_error"
    verdict_cache.put(store, "candidate", first, 60)
    assert verdict_cache.put(store, "candidate", second, 60) == "compile_error"
    assert verdict_cache.get(store, "candidate") == "compile_error"


def test_late_pass_does_not_extend_negative_expiry(store, redis_client, settings):
    verdict_cache.put(store, "candidate", "compile_error", 60)
    verdict_cache.put(store, "candidate", "precheck_passed", 3600)
    assert 0 < redis_client.ttl(f"{settings.redis_prefix}:verdict:candidate") <= 60


def test_negative_published_during_positive_transaction_wins(store):
    original_pipeline = store._r.pipeline

    def racing_pipeline(*args, **kwargs):
        pipeline = original_pipeline(*args, **kwargs)
        execute = pipeline.execute

        def publish_negative_then_execute(*args, **kwargs):
            # Another admission commits a definitive failure after this
            # publisher read the record, but before its transaction executes.
            with mock.patch.object(store._r, "pipeline", original_pipeline):
                verdict_cache.put(store, "candidate", "compile_error", 60)
            return execute(*args, **kwargs)

        pipeline.execute = publish_negative_then_execute
        return pipeline

    with mock.patch.object(store._r, "pipeline", racing_pipeline):
        assert verdict_cache.put(store, "candidate", "precheck_passed", 60) == "compile_error"
    assert verdict_cache.get(store, "candidate") == "compile_error"


def test_cache_failure_does_not_block_caller_or_leave_partial_record(store):
    with mock.patch.object(store._r, "hgetall", side_effect=redis.ConnectionError("down")):
        assert verdict_cache.get(store, "candidate") is None
    pipeline = store._r.pipeline()
    with mock.patch.object(pipeline, "execute", side_effect=redis.ConnectionError("down")):
        with mock.patch.object(store._r, "pipeline", return_value=pipeline):
            assert verdict_cache.put(store, "candidate", "compile_error", 60) is None
    assert verdict_cache.get(store, "candidate") is None


@pytest.mark.parametrize("value", [b"\xff", b"unknown", b"compiled"])
def test_invalid_record_is_a_cache_miss(store, redis_client, settings, value):
    redis_client.hset(f"{settings.redis_prefix}:verdict:candidate", mapping={"verdict": value})
    assert verdict_cache.get(store, "candidate") is None
