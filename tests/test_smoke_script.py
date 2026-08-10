from scripts import test_eval_service as smoke


def _success_response():
    return {
        "task_id": "t1",
        "status": "SUCCEEDED",
        "error": None,
        "result": {
            "correctness": {
                "passed": True,
                "compiled": True,
                "decoy": False,
                "passed_cases": 2,
                "total_cases": 2,
                "failures": [],
            },
            "performance": {"speedup_vs_torch": 2.5, "per_shape": []},
            "profile": None,
        },
    }


def test_response_exit_code_for_success_mismatch_and_timeout():
    success = _success_response()
    mismatch = _success_response()
    mismatch["result"]["correctness"] = {
        "passed": False,
        "compiled": True,
        "decoy": False,
        "passed_cases": 1,
        "total_cases": 2,
        "failures": [{"case_idx": 2, "kind": "constant_offset", "error": None}],
    }
    timed_out = {
        "task_id": "t2",
        "status": "TIMEOUT",
        "error": "evaluation request timed out",
        "result": {"correctness": None, "performance": None, "profile": None},
    }

    assert smoke.response_exit_code(success) == 0
    assert smoke.response_exit_code(mismatch) == 1
    assert smoke.response_exit_code(timed_out) == 1
    assert smoke.response_exit_code({"status": "SUCCEEDED", "result": {}}) == 2


def test_main_makes_one_synchronous_evaluation_request(monkeypatch):
    calls = []

    def fake_http_json(method, url, body=None, timeout=15.0, headers=None):
        calls.append((method, url, body, timeout, headers))
        if url.endswith("/healthz"):
            return {"redis": "ok"}
        return _success_response()

    monkeypatch.setattr(smoke, "http_json", fake_http_json)
    monkeypatch.setattr(
        smoke.sys,
        "argv",
        [
            "test_eval_service.py",
            "--url",
            "http://127.0.0.1:11907",
            "--session-id",
            "smoke-session",
            "--job-timeout",
            "10",
        ],
    )

    assert smoke.main() == 0
    assert len(calls) == 2
    method, url, body, timeout, headers = calls[1]
    assert method == "POST"
    assert url == "http://127.0.0.1:11907/evaluations"
    assert body["checks"] == ["correctness", "performance"]
    assert "measure_speed" not in body
    assert timeout == 70
    assert headers == {"X-Session-ID": "smoke-session"}


def test_main_reports_transport_failure(monkeypatch):
    def fail_request(*args, **kwargs):
        raise OSError

    monkeypatch.setattr(smoke, "http_json", fail_request)
    monkeypatch.setattr(smoke.sys, "argv", ["test_eval_service.py"])

    assert smoke.main() == 2
