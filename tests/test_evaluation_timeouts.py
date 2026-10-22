"""One-hour evaluation defaults remain bounded by explicit request and deploy limits."""

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts import npu_feedback_comparison as feedback
from scripts import test_eval_service as smoke
from triton_evaluation.api.contract import EvaluationRequest
from triton_evaluation.config import Settings
from triton_evaluation.evaluators import verify
from triton_evaluation.evaluators.base import ScriptOutcome
from triton_evaluation.operator_input import OperatorRequest

ROOT = Path(__file__).resolve().parents[1]


def test_request_defaults_are_one_hour_and_explicit_shorter_deadlines_survive():
    public = smoke.build_request()
    assert public.pop("deadline_seconds") == 3600
    assert OperatorRequest.model_validate(public).deadline_seconds == 3600
    assert OperatorRequest.model_validate(
        public | {"deadline_seconds": 120},
    ).deadline_seconds == 120
    private = {
        "schema_version": "2.1", "op_name": "test", "reference_code": "pass",
        "kernel_code": "pass", "checks": ["correctness"],
    }
    assert EvaluationRequest.model_validate(private).deadline_seconds == 3600
    assert EvaluationRequest.model_validate(
        private | {"deadline_seconds": 120},
    ).deadline_seconds == 120


def test_engine_caps_default_to_one_hour_and_environment_overrides_remain_effective(monkeypatch):
    monkeypatch.delenv("TRITON_EVAL_VERIFY_BUDGET_CAP_SECONDS", raising=False)
    monkeypatch.delenv("TRITON_EVAL_BENCHMARK_TIMEOUT_SECONDS", raising=False)
    settings = Settings()
    assert settings.verify_budget_cap_seconds == 3600
    assert settings.benchmark_timeout_seconds == 3600
    assert verify.VerifyEvaluator("unused").budget_cap_seconds == 3600
    monkeypatch.setenv("TRITON_EVAL_VERIFY_BUDGET_CAP_SECONDS", "1200")
    monkeypatch.setenv("TRITON_EVAL_BENCHMARK_TIMEOUT_SECONDS", "300")
    overridden = Settings()
    assert overridden.verify_budget_cap_seconds == 1200
    assert overridden.benchmark_timeout_seconds == 300


@pytest.mark.parametrize("remaining, expected", [(7200, 3600), (3599, 3599), (120, 120)])
def test_verify_forwards_whole_run_cap_clamped_to_remaining_resource_time(
    tmp_path, monkeypatch, remaining, expected,
):
    observed = []
    deadline = datetime(2026, 10, 22, tzinfo=timezone.utc)
    monkeypatch.setattr(verify, "remaining_seconds", lambda _: remaining)
    monkeypatch.setattr(
        verify, "read_json",
        lambda *_a, **_kw: {"total_cases": 1, "passed_cases": 1, "failures": []},
    )

    def run(command, _cwd, _device, forwarded_deadline, _lease, *, budget_seconds):
        observed.append((command, forwarded_deadline, budget_seconds))
        return ScriptOutcome(0, "", "", False)

    request = EvaluationRequest(
        schema_version="2.1", op_name="test", reference_code="pass",
        kernel_code="pass", checks=["correctness"],
    )
    result = verify.VerifyEvaluator("unused", runner=run)._run_verify(
        request, tmp_path, 0, deadline, None,
    )
    assert result.status == "succeeded"
    command, forwarded_deadline, outer_budget = observed[0]
    assert int(command[command.index("--timeout") + 1]) == expected
    assert forwarded_deadline == deadline
    assert outer_budget == expected + 90  # Existing process-cleanup grace stays separate.


@pytest.fixture
def client_module():
    path = ROOT / "plugins/triton-eval/skills/triton-eval/scripts/triton_eval.py"
    spec = importlib.util.spec_from_file_location("evaluation_timeout_client", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "command", ["verify", "benchmark", "profile", "evaluate", "get", "progress"],
)
def test_client_wait_defaults_cover_resource_deadline_and_allow_shorter_override(
    client_module, command,
):
    args = (
        [command, "ev_" + "1" * 32] if command in {"get", "progress"} else
        [command, "--definition", "definition.json", "--workloads", "workloads.jsonl"]
    )
    assert client_module.arguments(args).timeout == 3660
    assert client_module.arguments([*args, "--timeout", "42"]).timeout == 42
    assert client_module.arguments(["version"]).timeout == 30


def test_smoke_main_forwards_one_hour_request_and_waits_through_deadline(monkeypatch):
    observed = {}
    monkeypatch.setattr(smoke.sys, "argv", ["smoke", "--token", "controlled-test-token"])
    monkeypatch.setattr(smoke, "http_json", lambda *_a, **_kw: {"status": "ok"})

    def submit(_base, body, _headers, timeout):
        observed["request_deadline"] = body["deadline_seconds"]
        observed["transport_timeout"] = timeout
        return {"state": "queued", "evaluation_id": "ev_" + "1" * 32}

    def poll(_base, _evaluation_id, _headers, *, deadline_seconds):
        observed["client_wait"] = deadline_seconds
        return {
            "state": "completed", "evaluation_id": "ev_" + "1" * 32,
            "result": {"correctness": {"status": "succeeded", "data": {"passed": True}}},
        }

    monkeypatch.setattr(smoke, "submit_and_recover", submit)
    monkeypatch.setattr(smoke, "poll_until_terminal", poll)
    assert smoke.main() == 0
    assert observed == {"request_deadline": 3600, "transport_timeout": 15, "client_wait": 3660}


def test_feedback_main_uses_one_hour_deadline_before_any_submission(tmp_path, monkeypatch):
    class RequestCaptured(Exception):
        pass

    observed = []

    def capture(body, *_):
        observed.append(body["deadline_seconds"])
        raise RequestCaptured

    monkeypatch.setattr(feedback, "requests_for", capture)
    with pytest.raises(RequestCaptured):
        feedback.main([
            "--url", "http://unused", "--capability-file", str(tmp_path / "unused"),
            "--output", str(tmp_path / "unused.json"),
        ])
    assert observed == [3600]


def test_feedback_recovery_fallback_does_not_expire_at_old_deadline(tmp_path, monkeypatch):
    service = feedback.Service("http://unused", "controlled-token", {}, tmp_path / "report", .2)
    resources = iter([
        {"evaluation_id": "ev_" + "1" * 32, "state": "queued"},
        {"evaluation_id": "ev_" + "1" * 32, "state": "completed"},
    ])
    clock = iter([0, 1800, 1801])
    monkeypatch.setattr(service, "request", lambda *_a, **_kw: next(resources))
    monkeypatch.setattr(feedback.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(feedback.time, "sleep", lambda _: None)
    monkeypatch.setattr(feedback, "save_report", lambda *_: None)
    result = service.run({"checks": ["correctness"]}, {})
    assert result["state"] == "completed"
