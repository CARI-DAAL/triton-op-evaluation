"""The portable schema probe runs isolated against a local http.server."""
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROBE = PROJECT_ROOT / "tests" / "contract_consumers" / "portable_schema_probe.py"
CONTRACT_ROOT = PROJECT_ROOT / "contracts" / "evaluation-api" / "2.0"

REQUEST_BODY = json.dumps({
    "schema_version": "2.0",
    "purpose": "trial",
    "op_name": "addmm",
    "reference_code": "REF",
    "kernel_code": "IMPL",
    "checks": ["correctness"],
})


class _Handler(BaseHTTPRequestHandler):
    behavior = {"mode": "queued"}

    def do_POST(self):  # noqa: N802 — http.server API
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        mode = _Handler.behavior["mode"]
        if mode == "queued":
            body = json.loads(
                (CONTRACT_ROOT / "fixtures" / "resource-queued.json").read_bytes()
            )
            evaluation_id = body["evaluation_id"]
            self.send_response(202)
            self.send_header("Location", f"/evaluations/{evaluation_id}")
        elif mode == "error":
            body = json.loads(
                (CONTRACT_ROOT / "fixtures" / "error-rate-limited.json").read_bytes()
            )
            self.send_response(429)
        else:
            body = json.loads(
                (CONTRACT_ROOT / "fixtures" / "resource-completed-correct.json"
                 ).read_bytes()
            )
            self.send_response(200)
        payload = json.dumps(body).encode()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802 — http.server API
        body = json.loads(
            (CONTRACT_ROOT / "fixtures" / "resource-completed-correct.json").read_bytes()
        )
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def _run_probe(*args, timeout=30):
    return subprocess.run(
        [sys.executable, str(PROBE), *args],
        capture_output=True, text=True, timeout=timeout, cwd=PROJECT_ROOT,
    )


def test_import_audit_rejects_service_frameworks():
    result = _run_probe("--token", "t", "--audit-imports")
    assert result.returncode == 0, result.stderr
    assert "disallowed" not in result.stderr


def test_probe_source_never_imports_service_frameworks():
    source = PROBE.read_text(encoding="utf-8")
    for banned in (
        "import fastapi", "import pydantic", "import redis",
        "import triton_evaluation", "from triton_evaluation",
        "from fastapi", "from pydantic", "from redis",
    ):
        assert banned not in source
    # No handwritten Evaluation field/type table: validation is $defs-driven.
    assert "EvaluationRequest\"] = {" not in source


def test_probe_accepts_queued_then_terminal_flow(tmp_path, server):
    body_file = tmp_path / "request.json"
    body_file.write_text(REQUEST_BODY)
    result = _run_probe(
        "--url", server, "--token", "t",
        "--contract-root", str(CONTRACT_ROOT),
        "--body", str(body_file),
    )
    assert result.returncode == 0, result.stderr
    assert '"evaluation_id": "ev_' in result.stdout


def test_probe_rejects_mutated_contract(tmp_path, server):
    import shutil

    mutated = tmp_path / "contract"
    shutil.copytree(CONTRACT_ROOT, mutated)
    schema = bytearray((mutated / "schema.json").read_bytes())
    schema[50] = schema[50] ^ 0x01
    (mutated / "schema.json").write_bytes(bytes(schema))
    body_file = tmp_path / "request.json"
    body_file.write_text(REQUEST_BODY)
    result = _run_probe(
        "--url", server, "--token", "t",
        "--contract-root", str(mutated),
        "--body", str(body_file),
    )
    assert result.returncode != 0
    assert "digest mismatch" in result.stderr


def test_probe_handles_error_envelope_without_supplying_id(tmp_path, server):
    _Handler.behavior["mode"] = "error"
    try:
        body_file = tmp_path / "request.json"
        body_file.write_text(REQUEST_BODY)
        result = _run_probe(
            "--url", server, "--token", "t",
            "--contract-root", str(CONTRACT_ROOT),
            "--body", str(body_file),
        )
    finally:
        _Handler.behavior["mode"] = "queued"
    assert result.returncode == 0
    # An ErrorEnvelope never supplies an ID.
    assert '"evaluation_id": null' in result.stdout
