"""Exercise the dependency-free plugin CLI against real local HTTP connections."""

import json
import os
import socket
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "plugins/triton-eval/skills/triton-eval/scripts"
    / "triton_eval.py"
)
TOKEN = "test-capability-that-must-not-be-printed"
EID = "ev_" + "1" * 32


def resource(state="queued", **checks):
    return {
        "evaluation_id": EID,
        "state": state,
        "result": checks if state in {"completed", "failed", "timed_out"} else None,
    }


def check(**data):
    return {"status": "succeeded", "data": data}


@contextmanager
def server(responder):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def handle_request(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            calls.append((self.command, self.path, dict(self.headers), body))
            status, value, headers = responder(calls)
            if status is None:
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            payload = value if isinstance(value, bytes) else json.dumps(value).encode()
            self.send_response(status)
            for name, text in headers.items():
                self.send_header(name, text)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_POST = do_GET = handle_request

        def log_message(self, *args):
            pass

    instance = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{instance.server_port}", calls
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join()


@pytest.fixture
def inputs(tmp_path):
    torch = tmp_path / "torch.py"
    triton = tmp_path / "triton.py"
    cases = tmp_path / "cases.json"
    token = tmp_path / "capability"
    torch.write_text("raise RuntimeError('client must never execute this')\n", encoding="utf-8")
    triton.write_text("# triton source\ndef run(x): return x\n", encoding="utf-8")
    cases.write_text('[{"inputs": [1]}]', encoding="utf-8")
    token.write_text(TOKEN + "\n", encoding="utf-8")
    token.chmod(0o600)
    return tmp_path, torch, triton, cases, token


def run_cli(inputs, url, *extra, action="verify", script=SCRIPT, timeout=10):
    cwd, torch, triton, cases, token = inputs
    args = [sys.executable, "-I", "-S", str(script), action]
    if action == "get":
        args += [EID]
    else:
        args += ["--torch", str(torch), "--triton", str(triton), "--cases", str(cases)]
    args += ["--url", url, "--capability-file", str(token), "--poll-interval", "0.01", *extra]
    result = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={key: value for key, value in os.environ.items() if not key.startswith("EVALUATION_")},
    )
    assert TOKEN not in result.stdout + result.stderr
    output = json.loads(result.stdout)
    assert len(result.stdout.splitlines()) == 1
    return result, output


@pytest.mark.parametrize("action", ["verify", "benchmark", "profile"])
def test_submit_poll_and_recover_uses_original_instance(inputs, action):
    def respond(calls):
        if len(calls) == 1:
            return 202, resource(), {"Location": "http://evil.invalid/steal-capability"}
        return 200, resource("completed", correctness=check(passed=True)), {}

    with server(respond) as (url, calls):
        result, output = run_cli(inputs, url, action=action)
    assert result.returncode == 0
    assert output["state"] == "completed"
    assert [(call[0], call[1]) for call in calls] == [
        ("POST", f"/{action}"),
        ("GET", "/evaluations/" + EID),
    ]
    assert all(call[2]["Authorization"] == "Bearer " + TOKEN for call in calls)
    body = json.loads(calls[0][3])
    assert body == {
        "torch_code": inputs[1].read_text(),
        "triton_code": inputs[2].read_text(),
        "cases": [{"inputs": [1]}],
    }
    assert calls[0][2]["Idempotency-Key"] in result.stderr
    assert body["torch_code"] not in result.stderr


@pytest.mark.parametrize("failure", [None, 429, 502, 503, 504])
def test_ambiguous_submission_retries_exact_bytes_and_key(inputs, failure):
    def respond(calls):
        if len(calls) < 3:
            return failure, b"sensitive server details " + TOKEN.encode(), {}
        return 202, resource(), {}

    with server(respond) as (url, calls):
        result, output = run_cli(
            inputs, url, "--no-wait", "--idempotency-key", "stable-key", "--session-id", "session"
        )
    assert result.returncode == 0 and output["state"] == "queued"
    assert len(calls) == 3
    assert len({call[3] for call in calls}) == 1
    assert {call[2]["Idempotency-Key"] for call in calls} == {"stable-key"}
    assert {call[2]["X-Session-Id"] for call in calls} == {"session"}


@pytest.mark.parametrize(
    "state,checks,exit_code",
    [
        ("completed", {"correctness": check(passed=True)}, 0),
        ("completed", {"correctness": check(passed=False)}, 1),
        ("completed", {"compile": check(outcome="compile_error")}, 1),
        ("completed", {"profile": check(partial=True)}, 0),
        ("completed", {"profile": {"status": "failed"}}, 2),
        ("completed", {"correctness": {"status": "timeout"}}, 2),
        ("failed", {"correctness": {"status": "failed"}}, 2),
        ("timed_out", {"correctness": {"status": "timeout"}}, 2),
    ],
)
def test_terminal_exit_codes_preserve_raw_resource(inputs, state, checks, exit_code):
    body = resource(state, **checks)
    with server(lambda _: (200, body, {})) as (url, _):
        result, output = run_cli(inputs, url)
    assert result.returncode == exit_code
    assert output == body


def test_timeout_saves_private_handle_and_get_recovers(inputs):
    handle = inputs[0] / "handle.json"
    with server(lambda _: (202, resource(), {})) as (url, calls):
        result, output = run_cli(inputs, url, "--timeout", "0.2", "--save-handle", str(handle))
        assert result.returncode == 2
        assert output["error"]["code"] == "CLIENT_TIMEOUT"
        assert output["evaluation_id"] == EID
        assert output["idempotency_key"] == calls[0][2]["Idempotency-Key"]
        assert [call[0] for call in calls].count("POST") == 1
        saved = json.loads(handle.read_text())
        assert saved == {
            "url": url,
            "evaluation_id": EID,
            "idempotency_key": output["idempotency_key"],
            "session_id": None,
        }
        assert handle.stat().st_mode & 0o777 == 0o600
        result, output = run_cli(inputs, url, "--no-wait", action="get")
        assert result.returncode == 0 and output["evaluation_id"] == EID
        assert calls[-1][0] == "GET"


def test_redirect_never_forwards_credentials(inputs):
    with server(lambda _: (200, resource(), {})) as (destination, stolen):
        with server(lambda _: (307, b"", {"Location": destination + "/steal"})) as (url, calls):
            result, output = run_cli(inputs, url)
    assert result.returncode == 2 and output["error"]["code"] == "REDIRECT_REFUSED"
    assert len(calls) == 1 and not stolen


@pytest.mark.parametrize("body", [b"{broken", b"\xff", b"[]", b'{"state":"queued"}'])
def test_invalid_response_is_safe_json_error(inputs, body):
    with server(lambda _: (200, body, {})) as (url, _):
        result, output = run_cli(inputs, url)
    assert result.returncode == 2
    assert output["error"]["code"] in {"INVALID_JSON", "INVALID_RESPONSE"}
    assert output["idempotency_key"]


def test_http_error_never_prints_server_body(inputs):
    with server(lambda _: (401, TOKEN.encode() + b" private-code", {})) as (url, _):
        result, output = run_cli(inputs, url)
    assert result.returncode == 2
    assert output["error"] == {"code": "HTTP_ERROR", "message": "service returned HTTP 401"}
    assert "private-code" not in result.stdout + result.stderr


@pytest.mark.parametrize(
    "url",
    [
        "ftp://localhost",
        "http://user:secret@localhost",
        "http://localhost?x=1",
        "http://localhost/#fragment",
    ],
)
def test_invalid_instance_url_never_sends_a_request(inputs, url):
    result, output = run_cli(inputs, url)
    assert result.returncode == 2 and output["error"]["code"] == "INVALID_URL"


def test_nonempty_case_array_required_and_source_is_utf8(inputs):
    inputs[3].write_text("{}")
    result, output = run_cli(inputs, "http://127.0.0.1:1")
    assert result.returncode == 2 and output["error"]["code"] == "INVALID_INPUT"
    inputs[3].write_text("[{}]")
    inputs[1].write_bytes(b"\xff")
    result, output = run_cli(inputs, "http://127.0.0.1:1")
    assert result.returncode == 2 and output["error"]["code"] == "INVALID_INPUT"


def test_poll_transport_failure_never_resubmits(inputs):
    def respond(calls):
        if len(calls) == 1:
            return 202, resource(), {}
        if len(calls) == 2:
            return None, b"", {}
        return 200, resource("completed", correctness=check(passed=True)), {}

    with server(respond) as (url, calls):
        result, _ = run_cli(inputs, url)
    assert result.returncode == 0
    assert [call[0] for call in calls] == ["POST", "GET", "GET"]


def test_interrupted_wait_reports_known_recovery_handle(inputs):
    import signal
    import time

    handle = inputs[0] / "interrupted.json"
    with server(lambda _: (202, resource(), {})) as (url, calls):
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-S",
                str(SCRIPT),
                "verify",
                "--url",
                url,
                "--capability-file",
                str(inputs[4]),
                "--torch",
                str(inputs[1]),
                "--triton",
                str(inputs[2]),
                "--cases",
                str(inputs[3]),
                "--save-handle",
                str(handle),
            ],
            cwd=inputs[0],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if handle.exists() and json.loads(handle.read_text()).get("evaluation_id") == EID:
                    break
                time.sleep(0.01)
            else:
                pytest.fail("CLI never saved the accepted evaluation handle")
            process.send_signal(signal.SIGINT)
            stdout, stderr = process.communicate(timeout=3)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
    assert process.returncode == 130
    output = json.loads(stdout)
    assert output["error"]["code"] == "INTERRUPTED"
    assert output["evaluation_id"] == EID
    assert output["idempotency_key"] == calls[0][2]["Idempotency-Key"]
    assert TOKEN not in stdout + stderr


def test_environment_configuration_works_without_service_dependencies(inputs):
    with server(lambda _: (202, resource(), {})) as (url, calls):
        result = subprocess.run(
            [sys.executable, "-I", "-S", str(SCRIPT), "get", EID, "--no-wait"],
            cwd=inputs[0],
            capture_output=True,
            text=True,
            timeout=5,
            env=os.environ | {"EVALUATION_URL": url, "EVALUATION_CAPABILITY_FILE": str(inputs[4])},
        )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["evaluation_id"] == EID
    assert calls[0][2]["Authorization"] == "Bearer " + TOKEN
    assert TOKEN not in result.stdout + result.stderr


def test_unresponsive_exchange_obeys_total_timeout(inputs):
    import time

    def respond(calls):
        time.sleep(0.7)
        return 200, b"", {}

    with server(respond) as (url, _):
        started = time.monotonic()
        result, output = run_cli(inputs, url, "--timeout", "0.1")
        elapsed = time.monotonic() - started
    assert result.returncode == 2 and output["error"]["code"] == "CLIENT_TIMEOUT"
    assert output["idempotency_key"]
    assert elapsed < 0.6


@pytest.mark.parametrize(
    "checks",
    [
        {"correctness": {"status": "skipped"}},
        {"correctness": {"status": "succeeded", "data": {}}},
        {"correctness": {"status": "succeeded", "data": {"passed": "false"}}},
        {"compile": {"status": "succeeded", "data": {}}},
        {"compile": {"status": "succeeded", "data": {"outcome": "unknown"}}},
        {"profile": {"status": "succeeded"}},
        {"profile": {"status": "succeeded", "data": None}},
        {"profile": {"status": ["succeeded"], "data": {}}},
    ],
)
def test_malformed_terminal_evidence_cannot_claim_success(inputs, checks):
    with server(lambda _: (200, resource("completed", **checks), {})) as (url, _):
        result, output = run_cli(inputs, url)
    assert result.returncode == 2
    assert output["error"]["code"] == "INVALID_RESPONSE"
    assert output["evaluation_id"] == EID


def test_json_escaped_credentials_are_redacted_in_output_and_progress(inputs):
    secret = 'secret"with\\json-escapes'
    inputs[4].write_text(secret)
    handle = inputs[0] / "redacted.json"
    body = resource("completed", profile=check(note=secret))
    with server(lambda _: (200, body, {})) as (url, _):
        result, output = run_cli(
            inputs, url, "--idempotency-key", secret, "--save-handle", str(handle)
        )
    assert result.returncode == 0
    assert output["result"]["profile"]["data"]["note"] == "[REDACTED]"
    assert secret not in result.stderr
    assert json.loads(handle.read_text())["idempotency_key"] == "[REDACTED]"


def test_lone_surrogate_in_cases_is_structured_input_error(inputs):
    inputs[3].write_bytes(b'[{"inputs":["\\ud800"]}]')
    result, output = run_cli(inputs, "http://127.0.0.1:1")
    assert result.returncode == 2
    assert output["error"]["code"] == "INVALID_INPUT"
    assert "Traceback" not in result.stderr


def test_exhausted_retries_keep_original_session_in_recovery_handle(inputs):
    handle = inputs[0] / "retry.json"
    with server(lambda _: (503, b"internal data", {})) as (url, calls):
        result, output = run_cli(
            inputs,
            url,
            "--session-id",
            "my-session",
            "--idempotency-key",
            "stable-key",
            "--save-handle",
            str(handle),
        )
    assert result.returncode == 2 and len(calls) == 3
    assert output["session_id"] == "my-session"
    assert output["idempotency_key"] == "stable-key"
    assert json.loads(handle.read_text())["session_id"] == "my-session"


@pytest.mark.parametrize("flag,expected", [
    ("--use-baseline-cache", None), ("--no-baseline-cache", {"use_baseline_cache": False}),
])
def test_benchmark_baseline_cache_option_reaches_service(inputs, flag, expected):
    with server(lambda calls: (202, resource(), {})) as (url, calls):
        result, output = run_cli(inputs, url, flag, "--no-wait", action="benchmark")
    assert result.returncode == 0
    assert output["state"] == "queued"
    assert json.loads(calls[0][3]).get("performance") == expected


@pytest.mark.parametrize("action", ["verify", "profile", "get"])
def test_baseline_cache_option_only_applies_to_benchmark(inputs, action):
    with server(lambda calls: (202, resource(), {})) as (url, calls):
        result, output = run_cli(inputs, url, "--no-baseline-cache", action=action)
    assert result.returncode == 2
    assert output["error"]["code"] == "INVALID_ARGUMENT"
    assert calls == []
