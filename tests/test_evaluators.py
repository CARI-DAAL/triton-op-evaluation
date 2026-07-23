import sys
from pathlib import Path

import pytest
from triton_evaluation.models import JobSpec
from triton_evaluation.evaluators.base import materialize, run_script, read_json, tail
from triton_evaluation.evaluators.verify import VerifyEvaluator


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


class _VerifyHarness(VerifyEvaluator):
    """Point verify.py at the fake script instead of the real skill script."""
    def _script_path(self):
        return str(FIXTURES / "fake_verify.py")


def _spec():
    return JobSpec(op_name="softmax", reference_code="R", kernel_code="K")


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


from triton_evaluation.evaluators.benchmark import BenchmarkEvaluator, verify_json_name
from triton_evaluation.evaluators import build_registry


class _BenchHarness(BenchmarkEvaluator):
    def _script_path(self):
        return str(FIXTURES / "fake_benchmark.py")


def test_verify_json_name_mapping():
    assert verify_json_name("triton_ascend_impl") == "verify_result.json"
    assert verify_json_name("triton_optimized") == "verify_result_optimized.json"


def test_benchmark_with_seed_passes_gate(tmp_path):
    spec = _spec()
    spec.verify_result_seed = {"total_cases": 2, "passed_cases": 2}
    res = _BenchHarness("unused").run(spec, str(tmp_path), device_id=2)
    assert res.status == "SUCCEEDED"
    assert res.result["speedup_vs_torch"] == 2.0


def test_benchmark_without_seed_is_rejected(tmp_path):
    spec = _spec()  # no seed, no verify_not_required
    res = _BenchHarness("unused").run(spec, str(tmp_path), device_id=2)
    assert res.status == "REJECTED"


def test_benchmark_verify_not_required_skips_gate(tmp_path):
    spec = _spec()
    spec.verify_not_required = True
    res = _BenchHarness("unused").run(spec, str(tmp_path), device_id=2)
    assert res.status == "SUCCEEDED"


def test_registry_has_both(tmp_path):
    reg = build_registry(str(tmp_path))
    assert reg["verify"].name == "verify"
    assert reg["benchmark"].name == "benchmark"
