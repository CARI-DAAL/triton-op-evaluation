"""Only isolated reference measurements may populate shared baseline timings."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import fakeredis
import pytest

from triton_evaluation.api.contract import EvaluationRequest
from triton_evaluation.evaluators import benchmark as runner
from triton_evaluation.evaluators.base import LeaseLostError, ScriptOutcome
from triton_evaluation.runtime.baseline_cache import BaselineCache


@pytest.fixture
def harness(monkeypatch, tmp_path):
    client = fakeredis.FakeRedis()
    cache = BaselineCache(client, prefix="test")
    evaluator = runner.BenchmarkEvaluator("/engine", baseline_cache=cache)
    calls = []
    control = {}

    def execute(cmd, cwd, device, deadline, lease, **kwargs):
        calls.append((cmd, cwd, deadline, kwargs["budget_seconds"]))
        indices = [1, 2, 3]
        if "--case-indices" in cmd:
            indices = list(map(int, cmd[cmd.index("--case-indices") + 1].split(",")))
        if "--reference-only" in cmd:
            assert Path(cwd) != tmp_path
            assert len(list(Path(cwd).iterdir())) == 1
            assert next(Path(cwd).iterdir()).name.endswith("_torch.py")
            result = {
                "total_cases": 3,
                "case_indices": indices,
                "reference_latencies_ms": {str(index): float(index) for index in indices},
                "measurement_method": "event",
                "warmup": 5,
                "repeats": 20,
            }
            result.update(control.get("envelope", {}))
            if control.get("fence"):
                control["lease"] = False
            if control.get("expire"):
                control["expired"] = True
            if control.get("producer_exit"):
                return ScriptOutcome(control["producer_exit"], "", "", False)
            if control.get("timeout"):
                return ScriptOutcome(-1, "", "", True, "budget")
        else:
            injected = (
                json.loads(cmd[cmd.index("--baseline-timings-json") + 1])
                if "--baseline-timings-json" in cmd
                else None
            )
            indices = list(map(int, injected)) if injected is not None else indices
            result = {
                "per_shape_results": [
                    {
                        "case_idx": index,
                        "status": "pass",
                        "framework": {
                            "avg_latency_ms": control.get("candidate_latency", float(index)),
                            "baseline_cache_hit": True,
                        },
                    }
                    for index in indices
                ]
            }
        Path(cmd[cmd.index("--output") + 1]).write_text(json.dumps(result))
        return ScriptOutcome(0, "", "", False)

    monkeypatch.setattr(runner, "run_script", execute)
    control["lease"] = True

    def run(*, enabled=True, selected=None, candidate="candidate", repeats=20):
        request = EvaluationRequest(
            schema_version="2.1",
            op_name="test",
            reference_code="reference",
            kernel_code=candidate,
            checks=["correctness", "performance"],
            performance={
                "method": "event",
                "warmup": 5,
                "repeats": repeats,
                "use_baseline_cache": enabled,
                "case_indices": selected,
            },
        )
        return evaluator.run(
            request,
            tmp_path,
            0,
            datetime.now(timezone.utc) + timedelta(minutes=10),
            lease_valid=lambda: control["lease"],
        )

    return run, cache, calls, control, client, evaluator


def test_cold_warm_and_candidate_independent(harness):
    run, cache, calls, control, client, evaluator = harness
    first = run()
    assert first.status == "succeeded"
    assert len(calls) == 2
    assert all(
        not case["framework"]["baseline_cache_hit"] for case in first.raw["per_shape_results"]
    )
    second = run(candidate="different candidate")
    assert second.status == "succeeded"
    assert len(calls) == 3
    assert all(case["framework"]["baseline_cache_hit"] for case in second.raw["per_shape_results"])
    assert calls[0][2] == calls[1][2]
    assert calls[0][3] >= calls[1][3] - 1


def test_partial_cache_measures_missing_cases_only(harness):
    run, cache, calls, control, client, evaluator = harness
    assert run(selected=[1]).status == "succeeded"
    assert run().status == "succeeded"
    producer = calls[2][0]
    assert producer[producer.index("--case-indices") + 1] == "1,3"
    final = run().raw["per_shape_results"]
    assert all(case["framework"]["baseline_cache_hit"] for case in final)


def test_bypass_does_not_read_or_write_cache(harness, monkeypatch):
    run, cache, calls, control, client, evaluator = harness
    for name in ("key_for", "read", "publish"):
        monkeypatch.setattr(cache, name, lambda *_a, **_k: pytest.fail("cache accessed"))
    result = run(enabled=False)
    assert result.status == "succeeded"
    assert len(calls) == 1
    assert "--disable-reference-cache" in calls[0][0]
    assert "--baseline-timings-json" not in calls[0][0]
    assert all(
        not case["framework"]["baseline_cache_hit"] for case in result.raw["per_shape_results"]
    )


def test_absent_cache_uses_fresh_combined_path(harness):
    run, cache, calls, control, client, evaluator = harness
    evaluator.baseline_cache = None
    assert run().status == "succeeded"
    assert len(calls) == 1
    assert "--reference-only" not in calls[0][0]
    assert client.dbsize() == 0


@pytest.mark.parametrize(
    "envelope",
    [
        {"warmup": 9},
        {"total_cases": 2},
        {"case_indices": [1]},
        {"reference_latencies_ms": {"1": 1, "2": 2, "3": float("nan")}},
    ],
)
def test_bad_reference_envelope_never_publishes(harness, envelope):
    run, cache, calls, control, client, evaluator = harness
    control["envelope"] = envelope
    assert run().status == "succeeded"
    assert client.dbsize() == 0
    assert len(calls) == 2
    assert "--reference-only" not in calls[-1][0]
    assert "--baseline-timings-json" not in calls[-1][0]


def test_lease_loss_prevents_publish_and_candidate(harness):
    run, cache, calls, control, client, evaluator = harness
    control["fence"] = True
    with pytest.raises(LeaseLostError):
        run()
    assert client.dbsize() == 0
    assert len(calls) == 1


def test_reference_timeout_prevents_publish(harness):
    run, cache, calls, control, client, evaluator = harness
    control["timeout"] = True
    assert run().status == "timeout"
    assert client.dbsize() == 0


def test_candidate_denominator_cannot_poison_cache(harness):
    run, cache, calls, control, client, evaluator = harness
    control["candidate_latency"] = 99.0
    assert run().status == "failed"
    key = cache.key_for("reference", method="event", warmup=5, repeats=20)
    assert cache.read(key).latencies == {1: 1, 2: 2, 3: 3}
    control.pop("candidate_latency")
    assert run().status == "succeeded"
    assert len(calls) == 3


def test_deadline_expiry_prevents_publish(harness, monkeypatch):
    run, cache, calls, control, client, evaluator = harness
    remaining = runner.remaining_seconds
    monkeypatch.setattr(
        runner,
        "remaining_seconds",
        lambda deadline: 0 if control.get("expired") else remaining(deadline),
    )
    control["expire"] = True
    assert run().status == "timeout"
    assert client.dbsize() == 0
    assert len(calls) == 1


def test_parent_stamp_reaches_public_performance_shape():
    from triton_evaluation.evaluators.shaping import shape_performance

    fixture = Path(__file__).parent / "fixtures" / "engine" / "benchmark-event-timing.json"
    raw = json.loads(fixture.read_text())
    runner.BenchmarkEvaluator._stamp_result(raw, {"1": 1.0, "2": 1.0, "3": 1.0}, {2}, None)
    result = shape_performance(raw)
    assert [case.baseline_cache_hit for case in result.per_case] == [False, True]


def test_failed_reference_producer_runs_fresh_candidate(harness):
    run, cache, calls, control, client, evaluator = harness
    control["producer_exit"] = 1
    assert run().status == "succeeded"
    assert len(calls) == 2
    assert "--baseline-timings-json" not in calls[-1][0]
    assert client.dbsize() == 0


@pytest.mark.parametrize("operation", ["hgetall", "pipeline"])
def test_redis_fault_is_advisory_at_evaluator(harness, monkeypatch, operation):
    import redis

    run, cache, calls, control, client, evaluator = harness

    def unavailable(*_args, **_kwargs):
        raise redis.ConnectionError("unavailable")

    monkeypatch.setattr(client, operation, unavailable)
    result = run()
    assert result.status == "succeeded"
    assert len(calls) == 2
    assert all(
        not case["framework"]["baseline_cache_hit"] for case in result.raw["per_shape_results"]
    )
    assert client.dbsize() == (1 if operation == "hgetall" else 0)


@pytest.mark.parametrize("repeats", [1, 2])
def test_real_unstable_reference_measurement_falls_back(harness, monkeypatch, repeats):
    import sys
    from types import SimpleNamespace

    from triton_evaluation.engine import benchmark as engine

    run, cache, calls, control, client, evaluator = harness
    fake_torch = SimpleNamespace(
        device=lambda _: "npu",
        manual_seed=lambda _: None,
        npu=SimpleNamespace(manual_seed=lambda _: None),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(engine, "_ensure_npu_ambient_shared", lambda *_a, **_k: True)
    monkeypatch.setattr(engine, "cleanup_npu_memory", lambda: None)
    monkeypatch.setattr(engine, "resolve_inputs", lambda *_: [[1], [2], [3]])
    module = SimpleNamespace(
        Model=lambda: SimpleNamespace(to=lambda _: object()),
        get_init_inputs=lambda: [],
    )
    monkeypatch.setattr(engine.importlib, "import_module", lambda _: module)
    # Use the actual event stability gate: fewer than three samples produce None.
    monkeypatch.setattr(
        engine,
        "_measure_framework",
        lambda _m, _i, cfg, *_a: (
            {},
            engine._stable_latency([1.0] * cfg.repeats, cfg.cv_threshold),
            0,
            None,
        ),
    )
    original_execute = runner.run_script

    def execute(cmd, cwd, device, deadline, lease, **kwargs):
        if "--reference-only" in cmd:
            calls.append((cmd, cwd, deadline, kwargs["budget_seconds"]))
            cfg = engine.BenchmarkConfig("task", cwd, repeats=repeats, measurement_method="event")
            with pytest.raises(ValueError):
                engine.benchmark_reference_only(cfg)
            return ScriptOutcome(1, "", "", False)
        return original_execute(cmd, cwd, device, deadline, lease, **kwargs)

    monkeypatch.setattr(runner, "run_script", execute)
    result = run(repeats=repeats)
    assert result.status == "succeeded"
    assert len(calls) == 2
    assert "--baseline-timings-json" not in calls[-1][0]
    assert client.dbsize() == 0


def test_reference_invocation_exception_falls_back_without_publish(harness, monkeypatch):
    run, cache, calls, control, client, evaluator = harness
    execute = runner.run_script

    def unavailable(cmd, *args, **kwargs):
        if "--reference-only" in cmd:
            raise OSError("reference engine launch failed")
        return execute(cmd, *args, **kwargs)

    monkeypatch.setattr(runner, "run_script", unavailable)
    result = run()
    assert result.status == "succeeded"
    assert len(calls) == 1
    assert "--baseline-timings-json" not in calls[0][0]
    assert client.dbsize() == 0
