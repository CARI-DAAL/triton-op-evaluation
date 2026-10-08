import json
from concurrent.futures import ThreadPoolExecutor

import fakeredis
import pytest
import redis

from triton_evaluation.runtime.baseline_cache import BaselineCache, BaselineSnapshot


@pytest.fixture
def cache():
    return BaselineCache(fakeredis.FakeRedis(), prefix="test")


def key(cache, code="reference with cases and seed", **kwargs):
    return cache.key_for(code, method=kwargs.get("method", "event"),
                         warmup=kwargs.get("warmup", 5), repeats=kwargs.get("repeats", 10))


def test_cold_partial_and_warm(cache):
    address = key(cache)
    assert cache.read(address) is None
    cache.publish(address, 3, {2: 2.5})
    assert cache.read(address) == BaselineSnapshot(3, {2: 2.5})
    cache.publish(address, 3, {1: 1, 2: 99, 3: 3})
    assert cache.read(address) == BaselineSnapshot(3, {1: 1.0, 2: 2.5, 3: 3.0})
    assert cache._client.ttl(address) == -1


def test_identity_and_scope(cache):
    original = key(cache)
    assert original == key(cache)
    for changed in (key(cache, "different source"), key(cache, "different cases/seed"),
                    key(cache, method="wall"), key(cache, warmup=6), key(cache, repeats=11)):
        assert changed != original
    assert key(BaselineCache(cache._client, prefix="other")) != original
    assert key(BaselineCache(cache._client, prefix="test", namespace="v2")) != original
    assert key(BaselineCache(cache._client, prefix="a:b", namespace="c")) != key(
        BaselineCache(cache._client, prefix="a", namespace="b:c"))


@pytest.mark.parametrize("total,values", [
    (True, {1: 1}), (0, {1: 1}), (129, {1: 1}), (1.0, {1: 1}),
    (2, {True: 1}), (2, {0: 1}), (2, {3: 1}), (2, {"1": 1}),
    (2, {1: True}), (2, {1: 0}), (2, {1: -1}),
    (2, {1: float("nan")}), (2, {1: float("inf")}), (2, {1: "1"}),
])
def test_invalid_publish_is_noop(cache, total, values):
    cache.publish(key(cache), total, values)
    assert cache.read(key(cache)) is None


@pytest.mark.parametrize("mapping", [
    {"total_cases": "garbage"}, {"total_cases": "true"}, {"total_cases": "129"},
    {"total_cases": "2", "case:0": "1"}, {"total_cases": "2", "case:3": "1"},
    {"total_cases": "2", "case:01": "1"}, {"total_cases": "2", "case:1": "NaN"},
    {"total_cases": "2", "case:1": "Infinity"}, {"total_cases": "2", "case:1": "true"},
    {"total_cases": "2", "case:1": "0"}, {"total_cases": "2", "unknown": "1"},
    {"case:1": "1"},
])
def test_malformed_record_is_miss(cache, mapping):
    cache._client.hset(key(cache), mapping=mapping)
    assert cache.read(key(cache)) is None


def test_total_mismatch_does_not_publish(cache):
    cache.publish(key(cache), 2, {1: 1})
    cache.publish(key(cache), 3, {2: 2})
    assert cache.read(key(cache)) == BaselineSnapshot(2, {1: 1.0})


def test_concurrent_publish_preserves_winner_and_total(cache):
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda value: cache.publish(key(cache), 2, {1: value}), range(1, 17)))
    first = cache.read(key(cache)).latencies[1]
    cache.publish(key(cache), 2, {1: 100, 2: 2})
    assert cache.read(key(cache)) == BaselineSnapshot(2, {1: first, 2: 2.0})
    other_key = key(cache, "different")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(cache.publish, other_key, total, {total: total}) for total in (2, 3)]
        for future in futures:
            future.result()
    snapshot = cache.read(other_key)
    assert snapshot in (BaselineSnapshot(2, {2: 2.0}), BaselineSnapshot(3, {3: 3.0}))
    assert json.loads(cache._client.hget(other_key, "total_cases")) == snapshot.total_cases


def test_redis_errors_are_fail_open_and_redacted(caplog):
    class BrokenRedis:
        def hgetall(self, key):
            raise redis.ConnectionError("private source and credential")

        def pipeline(self):
            raise redis.ConnectionError("private source and credential")

    cache = BaselineCache(BrokenRedis(), prefix="test")
    assert cache.read(key(cache)) is None
    assert cache.publish(key(cache), 1, {1: 1}) is None
    assert "ConnectionError" in caplog.text
    assert "private source" not in caplog.text


def test_key_uses_canonical_protocol_and_conventional_prefix(cache):
    import hashlib

    from triton_evaluation.serialization import canonical_json_bytes

    protocol = {
        "namespace": "torch-baseline-v1",
        "reference_sha256": hashlib.sha256(b"reference with cases and seed").hexdigest(),
        "method": "event", "warmup": 5, "repeats": 10,
        "sample_count": 1, "cv_threshold": 0.3, "clear_l2_cache": False,
    }
    digest = hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
    assert key(cache) == f"test:baseline:{digest}"


def test_decoded_redis_responses():
    cache = BaselineCache(fakeredis.FakeRedis(decode_responses=True), prefix="test")
    cache.publish(key(cache), 1, {1: 0.25})
    assert cache.read(key(cache)) == BaselineSnapshot(1, {1: 0.25})


def test_invalid_publish_does_not_change_existing_record(cache):
    cache.publish(key(cache), 2, {1: 1})
    cache.publish(key(cache), 2, {1: 99, 2: float("nan")})
    assert cache.read(key(cache)) == BaselineSnapshot(2, {1: 1.0})


def test_redis_wrong_type_is_fail_open(cache):
    cache._client.set(key(cache), "not a hash")
    assert cache.read(key(cache)) is None
    cache.publish(key(cache), 1, {1: 1})
    assert cache._client.get(key(cache)) == b"not a hash"
