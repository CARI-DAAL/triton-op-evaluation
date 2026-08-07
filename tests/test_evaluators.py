import json
import sys
from pathlib import Path

from triton_evaluation.evaluators import benchmark as benchmark_mod
from triton_evaluation.evaluators import build_registry
from triton_evaluation.evaluators import profile as profile_mod
from triton_evaluation.evaluators import verify as verify_mod
from triton_evaluation.evaluators.base import (
    ScriptOutcome,
    materialize,
    read_json,
    run_script,
    tail,
)
from triton_evaluation.evaluators.benchmark import BenchmarkEvaluator
from triton_evaluation.evaluators.profile import ProfileEvaluator
from triton_evaluation.evaluators.verify import VerifyEvaluator
from triton_evaluation.models import JobSpec, TaskStatus


def test_materialize_writes_named_files(tmp_path):
    spec = JobSpec(op_name="softmax", reference_code="REF", kernel_code="IMPL")
    materialize(spec, str(tmp_path))
    assert (tmp_path / "softmax_torch.py").read_text() == "REF"
    assert (tmp_path / "softmax_triton_ascend_impl.py").read_text() == "IMPL"


def test_run_script_captures_exit_and_env(tmp_path):
    script = tmp_path / "echo_env.py"
    script.write_text(
        "import os, sys\n"
        "sys.stdout.write(os.environ.get('ASCEND_RT_VISIBLE_DEVICES',''))\n"
        "sys.exit(7)\n"
    )
    outcome = run_script([sys.executable, str(script)], str(tmp_path), device_id=3, timeout=30)
    assert outcome.exit_code == 7
    assert outcome.stdout == "3"
    assert outcome.timed_out is False


def test_run_script_times_out(tmp_path):
    script = tmp_path / "sleep.py"
    script.write_text("import time\ntime.sleep(30)\n")
    outcome = run_script([sys.executable, str(script)], str(tmp_path), device_id=0, timeout=1)
    assert outcome.timed_out is True


def test_read_json_missing_returns_none(tmp_path):
    assert read_json(str(tmp_path / "nope.json")) is None


def test_tail_truncates():
    assert tail("x" * 5000, limit=100).endswith("x" * 50)


FIXTURES = Path(__file__).parent / "fixtures"


class _BenchHarness(BenchmarkEvaluator):
    """Point benchmark.py at the fake script instead of the real engine script."""

    def _script_path(self):
        return str(FIXTURES / "fake_benchmark.py")


class _MissingBench(BenchmarkEvaluator):
    """A benchmark whose script does not exist — simulates benchmark blowing up."""

    def _script_path(self):
        return str(FIXTURES / "does_not_exist.py")


class _ProfileHarness(ProfileEvaluator):
    def _script_path(self):
        return str(FIXTURES / "fake_profile.py")


class _VerifyHarness(VerifyEvaluator):
    """Point verify.py + the chained benchmark at the fake scripts."""

    def _script_path(self):
        return str(FIXTURES / "fake_verify.py")

    def _benchmark_evaluator(self):
        return _BenchHarness("unused")


class _VerifyFlakyBenchHarness(_VerifyHarness):
    def _benchmark_evaluator(self):
        return _MissingBench("unused")


def _spec(measure_speed=False):
    return JobSpec(
        op_name="softmax", reference_code="R", kernel_code="K", measure_speed=measure_speed
    )


def test_benchmark_uses_300_second_hard_timeout(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, cwd, device_id, timeout):
        seen["timeout"] = timeout
        return ScriptOutcome(-1, "", "", True)

    monkeypatch.setattr(benchmark_mod, "run_script", fake_run)

    result = BenchmarkEvaluator("unused").run(
        _spec(),
        str(tmp_path),
        device_id=0,
    )

    assert result.status == TaskStatus.TIMEOUT.value
    assert seen["timeout"] == 300


def _profile_spec(**overrides):
    values = {
        "op_name": "softmax",
        "reference_code": "R",
        "kernel_code": "K",
        "timeout": 10,
        "profile_mode": "op",
        "case_idx": None,
        "profile_launch_count": None,
    }
    values.update(overrides)
    return JobSpec(**values)


def test_profile_evaluator_returns_direct_result(tmp_path, monkeypatch):
    monkeypatch.setenv("MSPROF_BIN", sys.executable)
    result = _ProfileHarness("unused").run(
        _profile_spec(),
        str(tmp_path),
        device_id=5,
    )
    assert result.status == "SUCCEEDED"
    assert result.result["profile_mode"] == "op"
    assert result.result["selected_cases"] == 2
    assert result.result["visible_device"] == "5"


def test_profile_evaluator_passes_single_case_and_launch_count(tmp_path, monkeypatch):
    monkeypatch.setenv("MSPROF_BIN", sys.executable)
    result = _ProfileHarness("unused").run(
        _profile_spec(case_idx=1, profile_launch_count=3),
        str(tmp_path),
        device_id=2,
    )
    assert result.status == "SUCCEEDED"
    assert result.result["requested_case_idx"] == 1
    assert result.result["selected_cases"] == 1


def test_profile_evaluator_missing_msprof_is_failed(tmp_path, monkeypatch):
    monkeypatch.setenv("MSPROF_BIN", str(tmp_path / "missing-msprof"))
    result = _ProfileHarness("unused").run(
        _profile_spec(),
        str(tmp_path),
        device_id=0,
    )
    assert result.status == "FAILED"
    assert "MSPROF_BIN is not executable" in result.error


def test_profile_evaluator_rejects_invalid_pass_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("MSPROF_BIN", sys.executable)
    monkeypatch.setenv("MSPROF_TIMEOUT_SEC", "zero")
    result = _ProfileHarness("unused").run(
        _profile_spec(),
        str(tmp_path),
        device_id=0,
    )
    assert result.status == "FAILED"
    assert "MSPROF_TIMEOUT_SEC must be a positive integer" in result.error


def test_profile_evaluator_nonzero_exit_keeps_valid_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("MSPROF_BIN", sys.executable)
    monkeypatch.setenv("FAKE_PROFILE_EXIT", "2")
    result = _ProfileHarness("unused").run(
        _profile_spec(),
        str(tmp_path),
        device_id=0,
    )
    assert result.status == "FAILED"
    assert result.result["summary_text"] == "fake profile"


def test_profile_evaluator_rejects_invalid_result_json(tmp_path, monkeypatch):
    monkeypatch.setenv("MSPROF_BIN", sys.executable)
    monkeypatch.setenv("FAKE_PROFILE_BAD_JSON", "1")
    result = _ProfileHarness("unused").run(
        _profile_spec(),
        str(tmp_path),
        device_id=0,
    )
    assert result.status == "FAILED"
    assert result.result is None


def test_profile_timeout_keeps_last_snapshot(tmp_path, monkeypatch):
    snapshot = {
        "profile_mode": "deep",
        "timed_out": False,
        "per_case_results": [{"case_idx": 0, "status": "complete"}],
    }
    monkeypatch.setenv("MSPROF_BIN", sys.executable)
    monkeypatch.setattr(
        profile_mod,
        "run_script",
        lambda *args, **kwargs: ScriptOutcome(124, "", "deadline", False),
    )
    monkeypatch.setattr(profile_mod, "read_json", lambda path: dict(snapshot))
    result = ProfileEvaluator("unused").run(
        _profile_spec(profile_mode="deep"),
        str(tmp_path),
        device_id=0,
    )
    assert result.status == "TIMEOUT"
    assert result.result["timed_out"] is True
    assert result.result["per_case_results"][0]["case_idx"] == 0


def test_profile_timeout_without_snapshot_is_still_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("MSPROF_BIN", sys.executable)
    monkeypatch.setattr(
        profile_mod,
        "run_script",
        lambda *args, **kwargs: ScriptOutcome(-1, "", "", True),
    )
    monkeypatch.setattr(profile_mod, "read_json", lambda path: None)
    result = ProfileEvaluator("unused").run(
        _profile_spec(profile_mode="deep"),
        str(tmp_path),
        device_id=0,
    )
    assert result.status == "TIMEOUT"
    assert result.result is None


def test_verify_pass_is_succeeded(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_VERIFY_MODE", "pass")
    res = _VerifyHarness("unused").run(_spec(), str(tmp_path), device_id=1)
    assert res.status == "SUCCEEDED"
    assert res.result["passed_cases"] == 2


def test_verify_fail_is_still_succeeded(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_VERIFY_MODE", "fail")
    res = _VerifyHarness("unused").run(_spec(), str(tmp_path), device_id=1)
    assert res.status == "SUCCEEDED"
    assert res.result["passed_cases"] == 1  # wrong kernel, but job ran fine


def test_verify_crash_no_json_is_failed(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_VERIFY_MODE", "crash")
    res = _VerifyHarness("unused").run(_spec(), str(tmp_path), device_id=1)
    assert res.status == "FAILED"
    assert "boom" in res.error


def test_measure_speed_false_attaches_no_benchmark(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_VERIFY_MODE", "pass")
    res = _VerifyHarness("unused").run(_spec(measure_speed=False), str(tmp_path), device_id=1)
    assert res.status == "SUCCEEDED"
    assert res.result["passed_cases"] == 2
    assert "benchmark" not in res.result


def test_measure_speed_pass_attaches_benchmark(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_VERIFY_MODE", "pass")
    res = _VerifyHarness("unused").run(_spec(measure_speed=True), str(tmp_path), device_id=1)
    assert res.status == "SUCCEEDED"
    assert res.result["passed_cases"] == 2
    assert res.result["benchmark"]["speedup_vs_torch"] == 2.0


def test_measure_speed_verify_fail_attaches_no_benchmark(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_VERIFY_MODE", "fail")
    res = _VerifyHarness("unused").run(_spec(measure_speed=True), str(tmp_path), device_id=1)
    assert res.status == "SUCCEEDED"
    assert res.result["passed_cases"] == 1
    assert "benchmark" not in res.result


def test_measure_speed_benchmark_flake_keeps_verify_succeeded(tmp_path, monkeypatch):
    # verify passed, but the benchmark subprocess blows up — job stays SUCCEEDED
    # with the verify verdict and simply no benchmark sub-object.
    monkeypatch.setenv("FAKE_VERIFY_MODE", "pass")
    res = _VerifyFlakyBenchHarness("unused").run(
        _spec(measure_speed=True), str(tmp_path), device_id=1
    )
    assert res.status == "SUCCEEDED"
    assert res.result["passed_cases"] == 2
    assert "benchmark" not in res.result


def test_registry_has_verify_profile_and_evaluation(tmp_path):
    registry = build_registry(str(tmp_path))
    assert set(registry) == {"verify", "profile", "evaluation"}


def test_verify_step0_decoy_short_circuits(monkeypatch, tmp_path):
    ev = verify_mod.VerifyEvaluator(engine_dir="/engine")
    # validate step: regression detected (exit 1, JSON on stdout); verify.py must NOT run.
    calls = []

    def fake_run(cmd, cwd, device_id, timeout):
        calls.append(cmd)
        if cmd[1].endswith("validate_triton_impl.py"):
            return ScriptOutcome(
                1,
                json.dumps({
                    "valid": False,
                    "regression_type": 1,
                    "checks": {},
                    "suggestion": "no @triton.jit",
                }),
                "",
                False,
            )
        raise AssertionError("verify.py must not run on a decoy")

    monkeypatch.setattr(verify_mod, "run_script", fake_run)
    res = ev.run(_spec(measure_speed=True), str(tmp_path), device_id=0)
    assert res.status == TaskStatus.SUCCEEDED.value
    assert res.result["decoy"] is True
    assert res.result["regression_type"] == 1
    assert "benchmark" not in res.result
    assert len(calls) == 1  # validate only


def test_verify_step0_clean_runs_verify(monkeypatch, tmp_path):
    ev = verify_mod.VerifyEvaluator(engine_dir="/engine")

    def fake_run(cmd, cwd, device_id, timeout):
        if cmd[1].endswith("validate_triton_impl.py"):
            return ScriptOutcome(0, json.dumps({"valid": True}), "", False)
        return ScriptOutcome(0, "", "", False)  # verify.py

    monkeypatch.setattr(verify_mod, "run_script", fake_run)
    monkeypatch.setattr(verify_mod, "read_json", lambda p: {"op_name": "softmax",
        "total_cases": 3, "passed_cases": 3, "failed_cases": 0, "failures": []})
    res = ev.run(_spec(), str(tmp_path), device_id=0)
    assert res.status == TaskStatus.SUCCEEDED.value
    assert res.result["passed_cases"] == 3
    assert res.result["validate"] == {"valid": True}
