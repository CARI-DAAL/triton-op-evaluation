"""CPU checks for the NPU gate CLI and its HTTP recovery orchestration."""
import io
import json
import subprocess
from types import SimpleNamespace

import pytest
import test_npu_release_calibration as calibration

from scripts import npu_release_gate as gate
from triton_evaluation.api.operator import OperatorRequest


class JsonResponse(io.BytesIO):
    def __init__(self, payload):
        super().__init__(json.dumps(payload).encode())


def test_contract_identity_reads_target_schema_with_bounded_timeout(monkeypatch):
    calls = []
    schema = {"openapi": "3.1.0", "paths": {"/evaluations": {"post": {}}}}

    def respond(url, timeout):
        calls.append((url, timeout))
        return JsonResponse(schema)

    monkeypatch.setattr(gate.urllib.request, "urlopen", respond)
    first = gate._contract_identity("http://service/")
    schema["paths"]["/verify"] = {"post": {}}
    second = gate._contract_identity("http://service")
    assert calls == [("http://service/openapi.json", 10)] * 2
    assert first["source"] == "target-openapi"
    assert first["openapi_sha256"] != second["openapi_sha256"]


def test_gate_uses_capability_without_catalog_and_reports_limited_scope(tmp_path, monkeypatch):
    secret = tmp_path / "token"
    secret.write_text("test-capability-token\n")
    output = tmp_path / "report.json"
    received = []
    monkeypatch.setattr(gate, "_contract_identity", lambda url: {"source": "target-openapi"})

    def calibration_run(environment):
        received.append(environment)
        return True, ["2 passed"]

    monkeypatch.setattr(gate, "_run_calibration_with", calibration_run)
    assert gate.main([
        "--service-url", "http://service", "--capability-file", str(secret),
        "--output", str(output),
    ]) == 0
    assert received[0]["TRITON_EVAL_CAPABILITY_TOKEN"] == "test-capability-token"
    report = json.loads(output.read_text())
    assert report["passed"] is True
    assert report["scope"] == "readiness-and-add-correctness-smoke"
    assert "production_ready" not in report
    assert "test-capability-token" not in output.read_text()


def test_gate_writes_failed_report_when_target_schema_is_unavailable(tmp_path, monkeypatch, capsys):
    secret = tmp_path / "token"
    secret.write_text("private-token")
    output = tmp_path / "report.json"

    def fail(url):
        raise OSError("private-token remote diagnostic")

    monkeypatch.setattr(gate, "_contract_identity", fail)
    assert gate.main(["--capability-file", str(secret), "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["passed"] is False
    assert report["contract"] is None
    assert "private-token" not in capsys.readouterr().err + output.read_text()


@pytest.mark.parametrize("summary, exit_code, passed", [
    ("2 passed in 1.0s", 0, True),
    ("1 passed, 1 skipped in 1.0s", 0, False),
    ("2 skipped in 1.0s", 0, False),
    ("1 passed, 1 failed in 1.0s", 1, False),
    ("no tests ran in 1.0s", 5, False),
])
def test_calibration_requires_both_checks_without_skips(monkeypatch, summary, exit_code, passed):
    def run(argv, **kwargs):
        assert kwargs["timeout"] == 720
        assert "tests/test_npu_release_calibration.py" in argv
        return SimpleNamespace(returncode=exit_code, stdout=summary)

    monkeypatch.setattr(gate.subprocess, "run", run)
    assert gate._run_calibration_with({})[0] is passed


def test_calibration_timeout_fails_the_gate(monkeypatch):
    def timeout(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(gate.subprocess, "run", timeout)
    assert gate._run_calibration_with({})[0] is False


def test_target_check_submits_real_operator_input_and_polls_202(monkeypatch):
    calls = []
    terminal = {
        "evaluation_id": "ev_test", "state": "completed",
        "result": {
            "correctness": {"status": "succeeded", "data": {
                "passed": True, "passed_cases": 2, "total_cases": 2,
            }},
            "provenance": {"execution_environment": {"device": "test-double"}},
        },
    }

    def request(method, path, body=None):
        calls.append((method, path))
        if method == "POST":
            parsed = OperatorRequest.model_validate(body)
            assert parsed.checks == ["correctness"]
            assert len(parsed.workloads) == 2
            assert "def run(" in parsed.definition.reference
            assert "def run(" in parsed.triton_code
            assert "..." not in parsed.definition.reference + parsed.triton_code
            return 202, {"evaluation_id": "ev_test", "state": "queued"}
        return 200, terminal

    monkeypatch.setattr(calibration, "CAPABILITY", "cpu-test-token")
    monkeypatch.setattr(calibration, "_request", request)
    monkeypatch.setattr(calibration.time, "sleep", lambda _: None)
    calibration.test_correct_case_reaches_completed_with_environment()
    assert calls == [("POST", "/evaluations?wait_seconds=0"), ("GET", "/evaluations/ev_test")]


@pytest.mark.parametrize("state, passed", [("failed", True), ("completed", False)])
def test_target_check_rejects_nonpassing_terminal_results(monkeypatch, state, passed):
    monkeypatch.setattr(calibration, "CAPABILITY", "cpu-test-token")
    monkeypatch.setattr(calibration, "_request", lambda *args: (200, {
        "evaluation_id": "ev_test", "state": state,
        "result": {"correctness": {"status": "succeeded", "data": {"passed": passed}}},
    }))
    with pytest.raises(AssertionError):
        calibration.test_correct_case_reaches_completed_with_environment()
