"""Exercise explicit capability controls through the standalone HTTP client."""
import json

import pytest
from test_cli_plugin import EID, check, resource, run_cli, server
from test_cli_plugin import inputs as _inputs

inputs = _inputs


def test_combined_evaluation_forwards_independent_selections(inputs):
    with server(lambda _calls: (202, resource(), {})) as (url, calls):
        result, _ = run_cli(
            inputs, url, "--no-wait", "--checks", "correctness", "performance", "profile",
            "--performance-cases", "7,2", "--profile-cases", "2", "--profile-mode", "trace",
            "--warmup", "3", "--repeats", "12", "--measurement-method", "event",
            "--deadline-seconds", "120", action="evaluate",
        )
    assert result.returncode == 0
    assert len(calls) == 1 and calls[0][1] == "/evaluations"
    body = json.loads(calls[0][3])
    assert body["checks"] == ["correctness", "performance", "profile"]
    assert body["performance"] == {
        "case_indices": [2, 7], "warmup": 3, "repeats": 12, "method": "event",
    }
    assert body["profile"] == {"case_indices": [2], "mode": "trace"}
    assert body["deadline_seconds"] == 120


def test_deep_capture_controls_reach_service(inputs):
    with server(lambda _calls: (202, resource(), {})) as (url, calls):
        result, _ = run_cli(
            inputs, url, "--no-wait", "--profile-mode", "deep", "--case-indices", "4,1",
            "--metric-groups", "Memory", "L2Cache", "--no-per-core", action="profile",
        )
    assert result.returncode == 0
    assert json.loads(calls[0][3])["profile"] == {
        "mode": "deep", "case_indices": [1, 4],
        "metric_groups": ["Memory", "L2Cache"], "include_per_core": False,
    }


def test_correctness_controls_and_evidence_reference(inputs):
    with server(lambda _calls: (202, resource(), {})) as (url, calls):
        result, _ = run_cli(
            inputs, url, "--no-wait", "--stop-on", "first_failure", "--case-indices", "3,1",
        )
    assert result.returncode == 0
    assert json.loads(calls[0][3])["correctness"] == {
        "stop_on": "first_failure", "case_indices": [1, 3],
    }
    with server(lambda _calls: (202, resource(), {})) as (url, calls):
        result, _ = run_cli(
            inputs, url, "--no-wait", "--correctness-from", EID, action="benchmark",
        )
    assert result.returncode == 0
    assert json.loads(calls[0][3])["correctness_from"] == EID


@pytest.mark.parametrize("selection", ["", "1,1", "-1", "foo", "1,"])
def test_invalid_case_selection_never_submits(inputs, selection):
    result, output = run_cli(inputs, "http://127.0.0.1:1", "--case-indices=" + selection)
    assert result.returncode == 2
    assert output["error"]["code"] == "INVALID_ARGUMENT"


@pytest.mark.parametrize("state", ["running", "completed"])
def test_progress_recovers_stage_evidence_without_waiting(inputs, state):
    progress = {
        "evaluation_id": EID, "state": state, "resource_revision": 2, "progress_revision": 4,
        "attempt_id": "at_1", "active_check": "profile",
        "completed_checks": {"correctness": check(passed=True, outcome="correct")},
    }
    with server(lambda _calls: (200, progress, {})) as (url, calls):
        result, output = run_cli(inputs, url, action="progress")
    assert result.returncode == 0 and output == progress
    assert len(calls) == 1 and calls[0][1] == f"/evaluations/{EID}/progress"


def test_wait_for_returns_timing_while_profile_is_still_running(inputs):
    snapshot = {
        "evaluation_id": EID, "state": "running", "resource_revision": 2,
        "progress_revision": 5, "attempt_id": "at_1", "active_check": "profile",
        "completed_checks": {"performance": check(measurement_method="npu_event_v1")},
    }

    def respond(calls):
        return (202, resource(), {}) if len(calls) == 1 else (200, snapshot, {})

    with server(respond) as (url, calls):
        result, output = run_cli(
            inputs, url, "--checks", "correctness", "performance", "profile",
            "--profile-mode", "trace", "--wait-for", "performance", action="evaluate",
        )
    assert result.returncode == 0 and output == snapshot
    assert len(calls) == 2


def test_wait_for_unrequested_check_is_rejected_before_submission(inputs):
    result, output = run_cli(inputs, "http://127.0.0.1:1", "--wait-for", "profile")
    assert result.returncode == 2
    assert output["error"]["code"] == "INVALID_ARGUMENT"
