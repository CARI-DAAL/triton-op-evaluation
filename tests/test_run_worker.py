"""Worker command line selects one execution resource without opening another."""
from types import SimpleNamespace
from unittest import mock

import pytest

from scripts import run_worker


def test_cpu_cli_never_holds_an_npu(monkeypatch):
    worker = SimpleNamespace(settings=object(), serve_forever=mock.Mock())
    build = mock.Mock(return_value=worker)
    monkeypatch.setattr("sys.argv", ["run_worker.py", "--cpu"])
    monkeypatch.setattr(run_worker, "build_worker", build)
    monkeypatch.setattr(run_worker, "hold_device", mock.Mock(side_effect=AssertionError))
    monkeypatch.setattr(run_worker, "set_process_title", lambda: None)
    monkeypatch.setattr(run_worker, "_install_signal_handlers", lambda: "stop")
    run_worker.main()
    build.assert_called_once_with(None)
    worker.serve_forever.assert_called_once_with(stop_event="stop")


@pytest.mark.parametrize("options", [[], ["--cpu", "--card", "0"], ["--card", "-1"]])
def test_worker_cli_requires_exactly_one_valid_execution_resource(monkeypatch, options):
    monkeypatch.setattr("sys.argv", ["run_worker.py", *options])
    with pytest.raises(SystemExit) as error:
        run_worker.main()
    assert error.value.code == 2


def test_worker_assembles_independent_dependencies_and_shared_runner(monkeypatch, tmp_path):
    import fakeredis

    from triton_evaluation.config import Settings

    settings = Settings(
        redis_url="redis://localhost:6380/2", redis_prefix="test-owner",
        baseline_cache_namespace="changed-hardware", engine_dir=str(tmp_path),
        reference_device_policy="npu-strict", card_canary_on_startup=False,
    )
    durable, advisory = fakeredis.FakeRedis(), fakeredis.FakeRedis()
    redis_factory = mock.Mock(side_effect=[durable, advisory])
    monkeypatch.setattr(run_worker.redis.Redis, "from_url", redis_factory)
    monkeypatch.setenv("TRITON_EVAL_REFERENCE_DEVICE_POLICY", "cpu-first")
    worker = run_worker.build_worker(3, settings)
    assert redis_factory.call_args_list[0] == mock.call(settings.redis_url, decode_responses=False)
    cache_options = redis_factory.call_args_list[1].kwargs
    assert cache_options["socket_connect_timeout"] == 0.25
    assert cache_options["socket_timeout"] == 0.25
    assert cache_options["retry"].get_retries() == 0
    assert cache_options["retry_on_timeout"] is False
    assert cache_options["retry_on_error"] == []
    cache = worker.evaluator.benchmark.baseline_cache
    key = cache.key_for("torch", method="event", warmup=5, repeats=50)
    cache.publish(key, 1, {1: 2.0})
    assert advisory.exists(key)
    assert not durable.exists(key)
    worker.verdict_cache.put("candidate", "compile_error", 60)
    assert advisory.exists("test-owner:verdict:candidate")
    assert not durable.exists("test-owner:verdict:candidate")
    worker.metrics.increment("worker_attempts_total", {})
    assert advisory.hget("test-owner:metrics:data:worker_attempts_total", "total") == b"1"
    assert not durable.exists("test-owner:metrics:data:worker_attempts_total")
    assert worker.store._metrics is worker.metrics
    assert worker._health._metrics is worker.metrics
    for check in (worker.evaluator.verify, worker.evaluator.benchmark, worker.evaluator.profile):
        assert check.engine_dir == str(tmp_path)
        assert check.runner is worker.script_runner
    assert worker.evaluator.verify.reference_device_policy == "npu-strict"
    captured = worker.script_runner.keywords["environment"]
    assert captured["TRITON_EVAL_REFERENCE_DEVICE_POLICY"] == "npu-strict"
    assert captured["TRITON_EVAL_ENGINE_DIR"] == str(tmp_path)
    monkeypatch.setenv("TRITON_EVAL_REFERENCE_DEVICE_POLICY", "changed-later")
    assert captured["TRITON_EVAL_REFERENCE_DEVICE_POLICY"] == "npu-strict"


def test_baseline_cache_namespace_can_be_set_from_environment(monkeypatch):
    from triton_evaluation.config import Settings

    monkeypatch.setenv("TRITON_EVAL_BASELINE_CACHE_NAMESPACE", "runtime-2026")
    assert Settings().baseline_cache_namespace == "runtime-2026"


def test_scalar_baseline_reuse_has_explicit_algorithm_identity():
    from triton_evaluation.config import Settings

    assert Settings().benchmark_algorithm_version == "geometric-mean-baseline-v2"
