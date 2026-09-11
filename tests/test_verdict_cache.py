"""Verdict cache: digest stability, get/put round-trip, worker feedback writer."""
import time
from unittest import mock

from triton_evaluation import verdict_cache
from triton_evaluation.broker import _scoped_digest


def test_digest_is_stable_and_scoped():
    d1 = verdict_cache.digest_for("code-a", "triton_ascend_impl", "")
    d2 = verdict_cache.digest_for("code-a", "triton_ascend_impl", "")
    d3 = verdict_cache.digest_for("code-b", "triton_ascend_impl", "")
    assert d1 == d2 and d1 != d3
    assert d1 == _scoped_digest(
        "code-a", "triton_ascend_impl",
        verdict_cache._triton_ascend_version(), "", "ascend-npu",
    )


def test_put_get_roundtrip_and_expiry(store, redis_client, settings):
    digest = verdict_cache.digest_for("k", "i", "")
    assert verdict_cache.get(store, digest) is None
    verdict_cache.put(store, digest, "compile_error", settings.verdict_cache_ttl_seconds)
    assert verdict_cache.get(store, digest) == "compile_error"
    key = f"triton-eval:v2:verdict:{digest}"
    assert redis_client.ttl(key) > 0
    # fakeredis key expiry runs on real time, not the store's FakeClock:
    # jump the wall clock past the TTL for the read instead.
    future = time.time() + settings.verdict_cache_ttl_seconds + 1
    with mock.patch("time.time", return_value=future):
        assert verdict_cache.get(store, digest) is None  # expired


def test_get_put_swallow_redis_errors(store):
    with mock.patch.object(store._r, "hgetall", side_effect=Exception("down")):
        assert verdict_cache.get(store, "x") is None
    with mock.patch.object(store._r, "hset", side_effect=Exception("down")):
        verdict_cache.put(store, "x", "compile_error", 60)  # must not raise
