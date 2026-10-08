"""Production app assembly provides observability without borrowing the durable client."""

import runpy
from unittest import mock

import fakeredis
import redis

from triton_evaluation.config import Settings
from triton_evaluation.service.security import CapabilityRegistry


def test_api_assembles_shared_metrics_with_bounded_connection(monkeypatch, tmp_path):
    settings = Settings(capability_secret_path=tmp_path / "registry.json")
    durable, advisory = fakeredis.FakeRedis(), fakeredis.FakeRedis()
    factory = mock.Mock(side_effect=[durable, advisory])
    monkeypatch.setattr(redis.Redis, "from_url", factory)
    monkeypatch.setattr(Settings, "from_env", lambda: settings)
    monkeypatch.setattr(CapabilityRegistry, "from_secret_file", lambda _path: object())
    assembled = runpy.run_module("scripts.run_api", run_name="api_composition_test")
    instance = assembled["app"].state.context.instance
    assert instance.metrics is instance.store._metrics
    instance.metrics.increment("worker_heartbeats_total", {})
    key = f"{settings.redis_prefix}:metrics:data:worker_heartbeats_total"
    assert advisory.hget(key, "total") == b"1"
    assert not durable.exists(key)
    assert factory.call_args_list[0].kwargs == {"decode_responses": False}
    options = factory.call_args_list[1].kwargs
    assert options["socket_timeout"] == options["socket_connect_timeout"] == 0.25
    assert options["retry"].get_retries() == 0
