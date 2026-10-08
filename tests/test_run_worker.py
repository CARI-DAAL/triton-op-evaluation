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


def test_worker_injects_separate_bounded_advisory_cache(monkeypatch):
    from triton_evaluation.config import Settings

    settings = Settings(redis_url="redis://localhost:6380/2", redis_prefix="test-owner",
                        baseline_cache_namespace="changed-hardware")
    durable, advisory = object(), object()
    redis_factory = mock.Mock(side_effect=[durable, advisory])
    store_factory = mock.Mock(return_value=object())
    evaluator_factory = mock.Mock(return_value=object())
    worker_factory = mock.Mock(return_value=object())
    monkeypatch.setattr(run_worker.redis.Redis, "from_url", redis_factory)
    monkeypatch.setattr(run_worker, "EvaluationStore", store_factory)
    monkeypatch.setattr(run_worker, "EvaluationEvaluator", evaluator_factory)
    monkeypatch.setattr(run_worker, "Worker", worker_factory)
    run_worker.build_worker(3, settings)
    store_factory.assert_called_once_with(durable, settings)
    assert redis_factory.call_args_list[0] == mock.call(settings.redis_url, decode_responses=False)
    cache_options = redis_factory.call_args_list[1].kwargs
    assert cache_options["socket_connect_timeout"] == 0.25
    assert cache_options["socket_timeout"] == 0.25
    assert cache_options["retry"].get_retries() == 0
    assert cache_options["retry_on_timeout"] is False
    assert cache_options["retry_on_error"] == []
    cache = evaluator_factory.call_args.kwargs["baseline_cache"]
    assert cache._client is advisory
    expected = run_worker.BaselineCache(advisory, prefix="test-owner", namespace="changed-hardware")
    assert cache.key_for("torch", method="event", warmup=5, repeats=50) == expected.key_for(
        "torch", method="event", warmup=5, repeats=50,
    )
    assert worker_factory.call_args.args == (
        store_factory.return_value, evaluator_factory.return_value,
    )


def test_evaluation_forwards_optional_baseline_cache():
    from triton_evaluation.evaluators.evaluation import EvaluationEvaluator

    cache = object()
    evaluator = EvaluationEvaluator("unused", baseline_cache=cache)
    assert evaluator._benchmark_evaluator().baseline_cache is cache
    assert EvaluationEvaluator("unused")._benchmark_evaluator().baseline_cache is None


def test_baseline_cache_namespace_can_be_set_from_environment(monkeypatch):
    from triton_evaluation.config import Settings

    monkeypatch.setenv("TRITON_EVAL_BASELINE_CACHE_NAMESPACE", "runtime-2026")
    assert Settings().baseline_cache_namespace == "runtime-2026"


def test_scalar_baseline_reuse_has_explicit_algorithm_identity():
    from triton_evaluation.config import Settings

    assert Settings().benchmark_algorithm_version == "geometric-mean-baseline-v2"
