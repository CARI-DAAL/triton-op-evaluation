"""Joint process checkpoints preserve evidence and baseline/lease boundaries."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import fakeredis
import pytest

from triton_evaluation.api.contract import EvaluationRequest
from triton_evaluation.evaluators import benchmark as evaluator_module
from triton_evaluation.evaluators.base import LeaseLostError, ScriptOutcome
from triton_evaluation.evaluators.benchmark import BenchmarkEvaluator
from triton_evaluation.runtime.baseline_cache import BaselineCache


def request(implementation="comparison", **kwargs):
    return EvaluationRequest(
        schema_version="2.1", op_name="joint", reference_code="pass",
        kernel_code="" if implementation == "torch" else "pass", implementation=implementation,
        checks=["correctness", "performance", "profile"]
        if implementation == "comparison" else ["performance", "profile"],
        performance={"case_indices": [0]},
        profile={"mode": "trace", "case_indices": [1]},
        **kwargs,
    )


def deadline():
    return datetime.now(timezone.utc) + timedelta(minutes=5)


def write(cmd, flag, content):
    Path(cmd[cmd.index(flag) + 1]).write_text(json.dumps(content))


def timing_raw(implementation="comparison"):
    if implementation == "comparison":
        return {"per_shape_results": [
            {"case_idx": 1, "status": "pass", "framework": {"avg_latency_ms": 1.0}}
        ]}
    return {"implementation": implementation, "correctness_verified": False}


@pytest.mark.parametrize("implementation", ["comparison", "torch", "triton"])
def test_one_candidate_subprocess_publishes_timing_before_trace_finishes(tmp_path, implementation):
    called, saved = [], []

    def run(cmd, cwd, device, end, lease, **kwargs):
        called.append(cmd)
        assert Path(cmd[1]).name == "joint_measurement.py"
        assert cmd[cmd.index("--case-indices") + 1] == "1"
        assert cmd[cmd.index("--trace-case-indices") + 1] == "2"
        if implementation != "comparison":
            assert cmd[cmd.index("--implementation") + 1] == implementation
        assert lease()
        assert saved == []
        write(cmd, "--output", timing_raw(implementation))
        assert lease()
        assert len(saved) == 1 and saved[0].status == "succeeded"
        # Additional polls must not publish the same evidence twice.
        assert lease()
        assert len(saved) == 1
        write(cmd, "--trace-output", {"mode": "trace"})
        return ScriptOutcome(0, "", "", False)

    evaluator = BenchmarkEvaluator("engine", runner=run)
    performance, trace = evaluator.run_with_trace(
        request(implementation), tmp_path, 0, deadline(), on_performance=saved.append,
    )
    assert len(called) == 1
    assert saved == [performance] and performance is saved[0]
    assert trace.status == "succeeded"


@pytest.mark.parametrize("timeout", [False, True])
def test_completed_timing_survives_later_trace_failure_or_timeout(tmp_path, timeout):
    saved = []

    def run(cmd, cwd, device, end, lease, **kwargs):
        write(cmd, "--output", timing_raw())
        assert lease()
        return ScriptOutcome(-1, "", "", True, "budget") if timeout else ScriptOutcome(
            1, "", "trace failed", False
        )

    performance, trace = BenchmarkEvaluator("engine", runner=run).run_with_trace(
        request(), tmp_path, 0, deadline(), on_performance=saved.append,
    )
    assert performance.status == "succeeded" and saved == [performance]
    assert trace.status == ("timeout" if timeout else "failed")


def test_no_checkpoint_makes_both_checks_timeout(tmp_path):
    def run(*args, **kwargs):
        return ScriptOutcome(-1, "", "", True, "budget")

    performance, trace = BenchmarkEvaluator("engine", runner=run).run_with_trace(
        request(), tmp_path, 0, deadline(),
    )
    assert performance.status == trace.status == "timeout"


def test_independent_trace_can_succeed_after_timing_has_no_result(tmp_path):
    def run(cmd, *args, **kwargs):
        write(cmd, "--trace-output", {"mode": "trace"})
        return ScriptOutcome(0, "", "", False)

    performance, trace = BenchmarkEvaluator("engine", runner=run).run_with_trace(
        request(), tmp_path, 0, deadline(),
    )
    assert performance.status == "failed" and trace.status == "succeeded"


def test_fenced_checkpoint_is_never_published(tmp_path):
    active, saved = [True], []

    def run(cmd, cwd, device, end, lease, **kwargs):
        write(cmd, "--output", timing_raw())
        active[0] = False
        assert not lease()
        return ScriptOutcome(-1, "", "", True, "fenced")

    with pytest.raises(LeaseLostError):
        BenchmarkEvaluator("engine", runner=run).run_with_trace(
            request(), tmp_path, 0, deadline(), lease_valid=lambda: active[0],
            on_performance=saved.append,
        )
    assert saved == []


def test_rejected_progress_write_propagates_to_runner_for_process_cleanup(tmp_path):
    def run(cmd, cwd, device, end, lease, **kwargs):
        write(cmd, "--output", timing_raw())
        lease()
        pytest.fail("fenced progress cannot continue")

    def rejected(_):
        raise LeaseLostError("progress fenced")

    with pytest.raises(LeaseLostError, match="progress fenced"):
        BenchmarkEvaluator("engine", runner=run).run_with_trace(
            request(), tmp_path, 0, deadline(), on_performance=rejected,
        )


def test_reference_baseline_producer_remains_isolated_from_joint_runner(tmp_path):
    commands = []
    cache = BaselineCache(fakeredis.FakeRedis(), prefix="joint-test")

    def run(cmd, cwd, device, end, lease, **kwargs):
        commands.append(cmd)
        if "--reference-only" in cmd:
            assert Path(cmd[1]).name == "benchmark.py"
            assert "--trace-output" not in cmd
            assert Path(cwd) != tmp_path
            assert [path.name for path in Path(cwd).iterdir()] == [
                cmd[cmd.index("--op_name") + 1] + "_torch.py"
            ]
            write(cmd, "--output", {
                "total_cases": 2, "case_indices": [1], "measurement_method": "event",
                "warmup": 5, "repeats": 50, "reference_latencies_ms": {"1": 1.0},
            })
        else:
            assert Path(cmd[1]).name == "joint_measurement.py"
            assert json.loads(cmd[cmd.index("--baseline-timings-json") + 1]) == {"1": 1.0}
            write(cmd, "--output", timing_raw())
            write(cmd, "--trace-output", {"mode": "trace"})
        return ScriptOutcome(0, "", "", False)

    evaluator = BenchmarkEvaluator("engine", runner=run, baseline_cache=cache)
    first, _ = evaluator.run_with_trace(request(), tmp_path, 0, deadline())
    second, _ = evaluator.run_with_trace(request(), tmp_path, 0, deadline())
    assert len(commands) == 3  # Cold reference + two candidate executions.
    assert first.raw["per_shape_results"][0]["framework"]["baseline_cache_hit"] is False
    assert second.raw["per_shape_results"][0]["framework"]["baseline_cache_hit"] is True


@pytest.mark.parametrize("implementation", ["torch", "triton"])
def test_single_source_joint_execution_never_touches_baseline_cache(tmp_path, implementation):
    class ForbiddenCache:
        def __getattr__(self, name):
            pytest.fail("single-source request consulted baseline cache")

    def run(cmd, *args, **kwargs):
        assert "--baseline-timings-json" not in cmd
        assert "--reference-only" not in cmd
        write(cmd, "--output", timing_raw(implementation))
        write(cmd, "--trace-output", {"mode": "trace"})
        return ScriptOutcome(0, "", "", False)

    performance, trace = BenchmarkEvaluator(
        "engine", runner=run, baseline_cache=ForbiddenCache(),
    ).run_with_trace(request(implementation), tmp_path, 0, deadline())
    assert performance.status == trace.status == "succeeded"


def test_standalone_trace_does_not_request_timing_or_correctness_gate(tmp_path):
    commands = []

    def run(cmd, *args, **kwargs):
        commands.append(cmd)
        assert "--trace-only" in cmd and "--output" not in cmd
        assert "--case-indices" not in cmd
        assert cmd[cmd.index("--trace-case-indices") + 1] == "2"
        assert cmd[cmd.index("--measurement-method") + 1] == "profiler"
        write(cmd, "--trace-output", {"mode": "trace"})
        return ScriptOutcome(0, "", "", False)

    result = BenchmarkEvaluator("engine", runner=run).run_trace(
        request(), tmp_path, 0, deadline(),
    )
    assert result.status == "succeeded" and len(commands) == 1


def test_repeated_invocation_does_not_reuse_a_stale_checkpoint(tmp_path):
    invocations = []

    def run(cmd, *args, **kwargs):
        path = cmd[cmd.index("--output") + 1]
        invocations.append(path)
        if len(invocations) == 1:
            write(cmd, "--output", timing_raw())
        write(cmd, "--trace-output", {"mode": "trace"})
        return ScriptOutcome(0, "", "", False)

    evaluator = BenchmarkEvaluator("engine", runner=run)
    first, _ = evaluator.run_with_trace(request(), tmp_path, 0, deadline())
    second, _ = evaluator.run_with_trace(request(), tmp_path, 0, deadline())
    assert first.status == "succeeded" and second.status == "failed"
    assert invocations[0] != invocations[1]


def test_trace_uses_remaining_resource_deadline_after_timing_phase_budget(tmp_path, monkeypatch):
    resource_end = deadline()
    expired = [False]

    def remaining(end):
        return -1 if expired[0] and end < resource_end else 200

    monkeypatch.setattr(evaluator_module, "remaining_seconds", remaining)

    def run(cmd, cwd, device, end, lease, **kwargs):
        assert end == resource_end
        write(cmd, "--output", timing_raw())
        assert lease()
        expired[0] = True
        assert lease(), "completed timing must not limit the later trace phase"
        write(cmd, "--trace-output", {"mode": "trace"})
        return ScriptOutcome(0, "", "", False)

    evaluator = BenchmarkEvaluator("engine", timeout_seconds=10, runner=run)
    performance, trace = evaluator.run_with_trace(
        request(), tmp_path, 0, resource_end,
    )
    assert performance.status == trace.status == "succeeded"


def test_timing_phase_budget_stops_shared_child_without_automatically_rerunning_trace(
    tmp_path, monkeypatch,
):
    resource_end = deadline()
    expired, commands = [False], []
    monkeypatch.setattr(
        evaluator_module, "remaining_seconds",
        lambda end: -1 if expired[0] and end < resource_end else 200,
    )

    def run(cmd, cwd, device, end, lease, **kwargs):
        commands.append(cmd)
        expired[0] = True
        lease()
        pytest.fail("timing phase deadline must interrupt runner")

    evaluator = BenchmarkEvaluator("engine", timeout_seconds=10, runner=run)
    performance, trace = evaluator.run_with_trace(
        request(), tmp_path, 0, resource_end,
    )
    assert performance.status == "timeout"
    assert trace.status == "failed" and trace.error.code == "SHARED_TIMING_TIMEOUT"
    assert len(commands) == 1


def test_rejected_shaped_timing_stops_trace_without_claiming_lease_loss(tmp_path):
    saved = []

    def publish(result):
        saved.append(result)
        return False

    def run(cmd, cwd, device, end, lease, **kwargs):
        write(cmd, "--output", timing_raw())
        lease()
        pytest.fail("rejected performance should stop shared trace")

    performance, trace = BenchmarkEvaluator("engine", runner=run).run_with_trace(
        request(), tmp_path, 0, deadline(), on_performance=publish,
    )
    assert saved == [performance]
    assert trace.status == "failed" and trace.error.code == "PRIOR_CHECK_FAILED"


@pytest.mark.parametrize("joint", [False, True])
def test_explicit_correctness_handoff_only_reaches_candidate_command(tmp_path, joint):
    commands = []
    cache = BaselineCache(fakeredis.FakeRedis(), prefix="handoff-test")
    reused = request().model_copy(update={"correctness_from": "ev_" + "1" * 32})

    def run(cmd, cwd, device, end, lease, **kwargs):
        commands.append(cmd)
        if "--reference-only" in cmd:
            assert "--correctness-evidence" not in cmd
            write(cmd, "--output", {
                "total_cases": 2, "case_indices": [1], "measurement_method": "event",
                "warmup": 5, "repeats": 50, "reference_latencies_ms": {"1": 1.0},
            })
        else:
            assert cmd[cmd.index("--correctness-evidence") + 1] == str(
                tmp_path / "correctness_evidence.json"
            )
            write(cmd, "--output", timing_raw())
            if joint:
                write(cmd, "--trace-output", {"mode": "trace"})
        return ScriptOutcome(0, "", "", False)

    evaluator = BenchmarkEvaluator("engine", runner=run, baseline_cache=cache)
    result = (evaluator.run_with_trace if joint else evaluator.run)(
        reused, tmp_path, 0, deadline(),
    )
    performance = result[0] if joint else result
    assert performance.status == "succeeded" and len(commands) == 2


def test_real_process_publishes_checkpoint_while_trace_is_still_running(tmp_path, monkeypatch):
    from triton_evaluation.evaluators import base

    # Exercise the actual subprocess poller without requiring an NPU or an
    # installed deployment sandbox. The child cannot finish until its timing
    # callback is observed and acknowledged by the parent.
    monkeypatch.setattr(base, "isolate_command", lambda command, cwd, device, env: (command, env))
    engine, work = tmp_path / "engine", tmp_path / "work"
    engine.mkdir()
    work.mkdir()
    (engine / "joint_measurement.py").write_text(
        "import json, sys, time\n"
        "from pathlib import Path\n"
        "def output(flag): return Path(sys.argv[sys.argv.index(flag) + 1])\n"
        "output('--output').write_text(json.dumps({\n"
        "    'per_shape_results': [{'case_idx': 1, 'status': 'pass',\n"
        "                           'framework': {'avg_latency_ms': 1.0}}]}))\n"
        "for _ in range(100):\n"
        "    if Path('progress-accepted').exists(): break\n"
        "    time.sleep(0.02)\n"
        "else: raise RuntimeError('parent did not publish timing during trace')\n"
        "output('--trace-output').write_text(json.dumps({'mode': 'trace'}))\n"
    )
    saved = []

    def publish(result):
        saved.append(result)
        (work / "progress-accepted").touch()

    performance, trace = BenchmarkEvaluator(str(engine)).run_with_trace(
        request(), work, 0, deadline(), on_performance=publish,
    )
    assert saved == [performance]
    assert performance.status == trace.status == "succeeded"


@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("reuse", [False, True])
def test_expired_handoff_is_not_retried_as_infrastructure_failure(tmp_path, joint, reuse):
    requested = request()
    if reuse:
        requested = requested.model_copy(update={"correctness_from": "ev_" + "1" * 32})

    def run(cmd, *args, **kwargs):
        assert ("--correctness-evidence" in cmd) is reuse
        return ScriptOutcome(2, "", "", False)

    evaluator = BenchmarkEvaluator("engine", runner=run)
    result = (evaluator.run_with_trace if joint else evaluator.run)(
        requested, tmp_path, 0, deadline(),
    )
    performance = result[0] if joint else result
    assert performance.status == "failed"
    if reuse:
        assert performance.error.code == "CORRECTNESS_EVIDENCE_INVALID"
        assert performance.error.phase == "request" and performance.error.retryable is False
    else:
        assert performance.error.retryable is True and "verify gate" in performance.error.message


@pytest.mark.parametrize("joint", [False, True])
def test_reused_correctness_materializes_sources_for_the_real_engine_gate(
    tmp_path, settings, joint,
):
    from types import SimpleNamespace

    from conftest import make_correctness_success, make_provenance
    from test_evaluation_evaluator import ENVIRONMENT

    from triton_evaluation.engine.benchmark import check_correctness_gate
    from triton_evaluation.runtime.correctness_evidence import (
        CorrectnessEvidence,
        CorrectnessSubject,
        write_handoff,
    )

    evidence_id = "ev_" + "a" * 32
    requested = request().model_copy(update={
        "correctness_from": evidence_id,
        "reference_code": "def run(x): return x + 1\n",
        "kernel_code": "def run(x): return x + 1.0\n",
    })
    now = datetime.now(timezone.utc)
    evidence = CorrectnessEvidence(
        evaluation_id=evidence_id,
        subject=CorrectnessSubject.from_request(requested),
        correctness=make_correctness_success(2, 2).data.model_copy(update={
            "reused_from": evidence_id,
        }),
        provenance=make_provenance(settings, execution_environment=ENVIRONMENT),
        observed_at=now - timedelta(seconds=1), expires_at=now + timedelta(minutes=5),
    )
    write_handoff(tmp_path, evidence)
    assert [path.name for path in tmp_path.iterdir()] == ["correctness_evidence.json"]
    checked = []

    def run(cmd, *args, **kwargs):
        def flag(name):
            return cmd[cmd.index(name) + 1]

        stem = flag("--op_name")
        assert (tmp_path / f"{stem}_torch.py").read_text() == requested.reference_code
        assert (tmp_path / f"{stem}_{requested.triton_impl_name}.py").read_text() == (
            requested.kernel_code
        )
        checked.append(check_correctness_gate(SimpleNamespace(
            op_name=stem, verify_dir=flag("--verify_dir"),
            triton_impl_name=flag("--triton_impl_name"),
            correctness_evidence=flag("--correctness-evidence"),
            correctness_non_compute=False,
        )))
        assert not (tmp_path / "verify_result.json").exists()
        write(cmd, "--output", timing_raw())
        if joint:
            write(cmd, "--trace-output", {"mode": "trace"})
        return ScriptOutcome(0, "", "", False)

    evaluator = BenchmarkEvaluator("engine", runner=run)
    result = (evaluator.run_with_trace if joint else evaluator.run)(
        requested, tmp_path, 0, deadline(),
    )
    performance = result[0] if joint else result
    assert performance.status == "succeeded" and checked == [2]
