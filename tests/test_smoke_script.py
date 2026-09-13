"""Smoke client against the durable API: headers, ID persistence, recovery."""
from scripts import test_eval_service as smoke

QUEUED_RESOURCE = {
    "schema_version": "2.1",
    "evaluation_id": "ev_" + "1" * 32,
    "revision": 1,
    "state": "queued",
    "result": None,
}

TERMINAL_RESOURCE = {
    "schema_version": "2.1",
    "evaluation_id": "ev_" + "1" * 32,
    "revision": 3,
    "state": "completed",
    "result": {
        "correctness": {
            "status": "succeeded",
            "data": {
                "outcome": "correct", "passed": True, "decoy": False,
                "passed_cases": 2, "total_cases": 2, "failures": [],
                "domain_failure": None,
            },
            "error": None,
            "reason": None,
        },
        "performance": None,
        "profile": None,
        "provenance": None,
    },
}


def test_response_exit_code_maps_terminal_states():
    assert smoke.response_exit_code(TERMINAL_RESOURCE) == 0
    failed = TERMINAL_RESOURCE | {"state": "failed"}
    assert smoke.response_exit_code(failed) == 1
    assert smoke.response_exit_code(QUEUED_RESOURCE) == 2


def test_response_exit_code_requires_passed_correctness():
    wrong = TERMINAL_RESOURCE.copy()
    wrong["result"] = {
        "correctness": {
            "status": "succeeded",
            "data": {
                "outcome": "wrong", "passed": False, "decoy": False,
                "passed_cases": 1, "total_cases": 2,
                "failures": [{"case_idx": 1, "kind": "value_mismatch"}],
                "domain_failure": None,
            },
            "error": None,
            "reason": None,
        },
        "performance": None,
        "profile": None,
        "provenance": None,
    }
    assert smoke.response_exit_code(wrong) == 1


def _fake_transport(monkeypatch, responses):
    calls = []

    def fake_http_json(method, url, body=None, timeout=15.0, headers=None):
        calls.append({"method": method, "url": url, "body": body,
                      "headers": dict(headers or {})})
        for pattern, payload in responses:
            if pattern(method, url):
                if isinstance(payload, Exception):
                    raise payload
                return payload
        raise AssertionError(f"unexpected {method} {url}")

    monkeypatch.setattr(smoke, "http_json", fake_http_json)
    monkeypatch.setattr(smoke.time, "sleep", lambda _s: None)
    return calls


def test_main_sends_capability_headers_and_recovers_by_get(monkeypatch):
    calls = _fake_transport(
        monkeypatch,
        [
            (lambda m, u: u.endswith("/livez"), {"status": "ok"}),
            (lambda m, u: u.endswith("/readyz"), {"ready": True}),
            (lambda m, u: m == "POST", QUEUED_RESOURCE),
            (lambda m, u: m == "GET", TERMINAL_RESOURCE),
        ],
    )
    monkeypatch.setattr(
        smoke.sys, "argv",
        ["test_eval_service.py", "--url", "http://127.0.0.1:11907",
         "--token", "secret-token"],
    )
    assert smoke.main() == 0

    post = next(call for call in calls if call["method"] == "POST")
    assert post["headers"]["Authorization"] == "Bearer secret-token"
    assert post["headers"]["X-Session-ID"] == "triton-eval-smoke"
    assert post["headers"]["Idempotency-Key"].startswith("smoke-")
    assert post["body"]["schema_version"] == "2.1"
    assert post["body"]["purpose"] == "trial"

    gets = [call for call in calls if call["method"] == "GET"
            and "/evaluations/" in call["url"]]
    assert gets, "recovery must poll the persisted evaluation_id"


def test_main_replays_post_only_without_any_response(monkeypatch):
    import urllib.error

    posts = {"count": 0}

    def fake_http_json(method, url, body=None, timeout=15.0, headers=None):
        if url.endswith("/livez"):
            return {"status": "ok"}
        if url.endswith("/readyz"):
            return {"ready": True}
        if method == "POST":
            posts["count"] += 1
            if posts["count"] == 1:
                raise urllib.error.URLError("no response obtained")
            return TERMINAL_RESOURCE
        raise AssertionError(f"unexpected {method} {url}")

    monkeypatch.setattr(smoke, "http_json", fake_http_json)
    monkeypatch.setattr(smoke.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        smoke.sys, "argv",
        ["test_eval_service.py", "--token", "secret-token"],
    )
    assert smoke.main() == 0
    assert posts["count"] == 2  # exactly one safe replay


def test_main_reports_transport_failure(monkeypatch):
    def fail_request(*args, **kwargs):
        raise OSError

    monkeypatch.setattr(smoke, "http_json", fail_request)
    monkeypatch.setattr(
        smoke.sys, "argv", ["test_eval_service.py", "--token", "secret-token"],
    )
    assert smoke.main() == 2
