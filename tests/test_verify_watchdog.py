"""Whole-run watchdog fix: identifiable engine death + deploy-tunable budgets.

Covers the K3 intermittent-failure root cause: the engine's outer watchdog used
to die as exit=1 with no result file, which the evaluator misclassified as a
retryable "no usable result" execution failure. The watchdog must now die as
exit=3 and map to DEADLINE_EXCEEDED, and both budgets come from Settings.
"""
from datetime import datetime, timedelta, timezone

import pytest
from conftest import make_request

import triton_evaluation.evaluators.verify as verify_mod
from triton_evaluation.evaluators.base import ScriptOutcome
from triton_evaluation.evaluators.verify import VerifyEvaluator


def _deadline(seconds=120):
    return datetime.now(timezone.utc) + timedelta(seconds=seconds)


def _fake_run_script(seen, *, exit_code=0, outcome="exited"):
    def fake(cmd, cwd, device_id, deadline_at=None, lease_valid=None,
             *, budget_seconds=None):
        seen["cmd"] = list(cmd)
        seen["budget_seconds"] = budget_seconds
        return ScriptOutcome(exit_code, "", "", outcome != "exited", outcome)

    return fake


def test_watchdog_exit_code_maps_to_deadline_exceeded(tmp_path, monkeypatch):
    """Engine exit=3 (whole-run watchdog) must not become a retryable
    EVALUATOR_EXECUTION_FAILED; it is DEADLINE_EXCEEDED, retryable=False."""
    seen = {}
    monkeypatch.setattr(
        verify_mod, "run_script", _fake_run_script(seen, exit_code=3)
    )
    monkeypatch.setattr(verify_mod, "read_json", lambda path, **_kwargs: None)

    evaluator = VerifyEvaluator("unused")
    result = evaluator.run(make_request(), tmp_path, 0, _deadline())

    assert result.status == "timeout"
    assert result.error.code == "DEADLINE_EXCEEDED"
    assert result.error.retryable is False
    assert "whole-run watchdog" in result.error.message


def test_other_exit_codes_still_fail_closed(tmp_path, monkeypatch):
    """exit=1 without a result file remains a retryable evaluator failure."""
    seen = {}
    monkeypatch.setattr(
        verify_mod, "run_script", _fake_run_script(seen, exit_code=1)
    )
    monkeypatch.setattr(verify_mod, "read_json", lambda path, **_kwargs: None)

    evaluator = VerifyEvaluator("unused")
    result = evaluator.run(make_request(), tmp_path, 0, _deadline())

    assert result.status == "failed"
    assert result.error.code == "EVALUATOR_EXECUTION_FAILED"
    assert result.error.retryable is True


def test_verify_budget_cap_comes_from_settings(tmp_path, monkeypatch):
    """The whole-run verify budget cap is deployment-tunable, not a constant."""
    from triton_evaluation.config import Settings
    from triton_evaluation.evaluators.evaluation import EvaluationEvaluator

    seen = {}
    monkeypatch.setattr(
        verify_mod, "run_script", _fake_run_script(seen, exit_code=0)
    )
    monkeypatch.setattr(
        verify_mod, "read_json",
        lambda path, **_kwargs: {"decoy": True, "total_cases": 1},  # any dict short-circuits
    )

    settings = Settings(engine_dir="unused", verify_budget_cap_seconds=1500)
    evaluator = EvaluationEvaluator("unused", settings=settings)
    verify_evaluator = evaluator._verify_evaluator()
    deadline = _deadline(seconds=3600)
    verify_evaluator._run_verify(
        make_request(), tmp_path, 0, deadline, lambda: True
    )

    budget = int(seen["cmd"][seen["cmd"].index("--timeout") + 1])
    assert budget == 1500  # min(remaining≈3600, cap=1500)


def test_benchmark_timeout_comes_from_settings(tmp_path, monkeypatch):
    import triton_evaluation.evaluators.benchmark as benchmark_mod
    from triton_evaluation.config import Settings
    from triton_evaluation.evaluators.evaluation import EvaluationEvaluator

    seen = {}
    monkeypatch.setattr(
        benchmark_mod, "run_script", _fake_run_script(seen, exit_code=0)
    )
    monkeypatch.setattr(benchmark_mod, "read_json", lambda path, **_kwargs: {"cases": []})

    settings = Settings(engine_dir="unused", benchmark_timeout_seconds=600)
    evaluator = EvaluationEvaluator("unused", settings=settings)
    bench = evaluator._benchmark_evaluator()
    deadline = _deadline(seconds=3600)
    bench.run(make_request(), tmp_path, 0, deadline, lambda: True)

    assert seen["budget_seconds"] == 600


def test_settings_expose_the_new_budget_fields():
    from triton_evaluation.config import Settings

    settings = Settings(engine_dir="unused")
    assert settings.verify_budget_cap_seconds == 1500
    assert settings.benchmark_timeout_seconds == 600


def test_engine_watchdog_branch_exits_three():
    """engine/verify.py's outer watchdog must exit 3, distinguishable from
    both normal completion (0) and generic engine errors (1)."""
    import importlib.util
    from pathlib import Path

    engine_path = (
        Path(__file__).resolve().parents[1]
        / "triton_evaluation" / "engine" / "verify.py"
    )
    spec = importlib.util.spec_from_file_location("engine_verify", engine_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class _Proc:
        def kill(self):
            pass

        def wait(self):
            pass

    with pytest.raises(SystemExit) as exc_info:
        module._terminate_expired_subprocess(_Proc(), 960)
    assert exc_info.value.code == 3


def test_engine_crash_evidence_is_retained(tmp_path):
    """log_engine_crash appends exit/outcome/result-file presence + log tails."""
    from triton_evaluation.evaluators.base import log_engine_crash

    log_path = tmp_path / "logs" / "engine_crashes.log"
    outcome = ScriptOutcome(1, "out-line", "err-line", False, "exited")
    log_engine_crash(
        "verify", outcome, result_file_exists=False, log_path=log_path
    )
    text = log_path.read_text(encoding="utf-8")
    assert "exit=1" in text
    assert "result_file_exists=False" in text
    assert "out-line" in text
    assert "err-line" in text
