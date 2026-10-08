"""Advisory cache scope, atomic publication, precedence, and expiry."""
import importlib.metadata
import sys
import time
from unittest import mock

import pytest
import redis

from triton_evaluation.runtime import verdict_cache


@pytest.fixture
def cache(redis_client, settings):
    return verdict_cache.VerdictCache(redis_client, prefix=settings.redis_prefix)



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


def test_missing_cann_identity_disables_cache_without_touching_redis(cache, redis_client):
    digest = verdict_cache.digest_for("code", "impl", "")
    assert digest is None
    with mock.patch.object(redis_client, "hgetall") as read:
        with mock.patch.object(redis_client, "pipeline") as write:
            assert cache.get(digest) is None
            assert cache.put(digest, "compile_error", 60) is None
    read.assert_not_called()
    write.assert_not_called()


def test_python_upgrade_changes_digest():
    original = verdict_cache.digest_for("code", "impl", "cann")
    with mock.patch.object(sys, "version", "different Python runtime"):
        upgraded = verdict_cache.digest_for("code", "impl", "cann")
    assert original != upgraded


def test_put_get_roundtrip_and_expiry(cache, redis_client, settings):
    digest = verdict_cache.digest_for("k", "i", "cann")
    assert cache.get(digest) is None
    cache.put(digest, "compile_error", settings.verdict_cache_ttl_seconds)
    assert cache.get(digest) == "compile_error"
    key = f"{settings.redis_prefix}:verdict:{digest}"
    assert redis_client.ttl(key) > 0
    # fakeredis key expiry runs on real time, not the cache's FakeClock:
    # jump the wall clock past the TTL for the read instead.
    future = time.time() + settings.verdict_cache_ttl_seconds + 1
    with mock.patch("time.time", return_value=future):
        assert cache.get(digest) is None  # expired


@pytest.mark.parametrize("first", ["compile_error", "precheck_passed"])
def test_negative_takes_precedence_in_either_publication_order(cache, first):
    second = "precheck_passed" if first == "compile_error" else "compile_error"
    cache.put("candidate", first, 60)
    assert cache.put("candidate", second, 60) == "compile_error"
    assert cache.get("candidate") == "compile_error"


def test_late_pass_does_not_extend_negative_expiry(cache, redis_client, settings):
    cache.put("candidate", "compile_error", 60)
    cache.put("candidate", "precheck_passed", 3600)
    assert 0 < redis_client.ttl(f"{settings.redis_prefix}:verdict:candidate") <= 60


def test_negative_published_during_positive_transaction_wins(cache, redis_client):
    original_pipeline = redis_client.pipeline

    def racing_pipeline(*args, **kwargs):
        pipeline = original_pipeline(*args, **kwargs)
        execute = pipeline.execute

        def publish_negative_then_execute(*args, **kwargs):
            # Another admission commits a definitive failure after this
            # publisher read the record, but before its transaction executes.
            with mock.patch.object(redis_client, "pipeline", original_pipeline):
                cache.put("candidate", "compile_error", 60)
            return execute(*args, **kwargs)

        pipeline.execute = publish_negative_then_execute
        return pipeline

    with mock.patch.object(redis_client, "pipeline", racing_pipeline):
        assert cache.put("candidate", "precheck_passed", 60) == "compile_error"
    assert cache.get("candidate") == "compile_error"


def test_cache_failure_does_not_block_caller_or_leave_partial_record(cache, redis_client):
    with mock.patch.object(redis_client, "hgetall", side_effect=redis.ConnectionError("down")):
        assert cache.get("candidate") is None
    pipeline = redis_client.pipeline()
    with mock.patch.object(pipeline, "execute", side_effect=redis.ConnectionError("down")):
        with mock.patch.object(redis_client, "pipeline", return_value=pipeline):
            assert cache.put("candidate", "compile_error", 60) is None
    assert cache.get("candidate") is None


@pytest.mark.parametrize("value", [b"\xff", b"unknown", b"compiled"])
def test_invalid_record_is_a_cache_miss(cache, redis_client, settings, value):
    redis_client.hset(f"{settings.redis_prefix}:verdict:candidate", mapping={"verdict": value})
    assert cache.get("candidate") is None


def test_digest_preserves_existing_nul_delimited_utf8_bytes():
    with mock.patch.object(sys, "version", "python-test"):
        with mock.patch.object(importlib.metadata, "version", return_value="1.0"):
            assert verdict_cache.digest_for("code\0source", "impl", "cann") == (
                "fe07af47b590deeb3eb4469564e141b6a05481b5d3b3dd529f1dad18bdc0d5ca"
            )
