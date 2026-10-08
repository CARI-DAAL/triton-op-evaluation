"""Target-NPU readiness and add correctness checks, run by npu_release_gate.py.

Two float32 add workloads exercise the real worker and execution provenance.
This is not a full domain-failure matrix, timing/profile calibration or crash soak.
"""
import json
import os
import time
import urllib.error
import urllib.request
from uuid import uuid4

import pytest

from scripts.test_eval_service import build_request

pytestmark = [pytest.mark.npu, pytest.mark.release_gate]

SERVICE_URL = os.environ.get("TRITON_EVAL_SERVICE_URL", "http://172.18.0.1:11907")
CAPABILITY = os.environ.get("TRITON_EVAL_CAPABILITY_TOKEN", "")


def _skip_without_service():
    if not CAPABILITY:
        pytest.skip("set TRITON_EVAL_CAPABILITY_TOKEN on the target host")


def _request(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        f"{SERVICE_URL}{path}", data=data, method=method,
        headers={
            "Authorization": f"Bearer {CAPABILITY}",
            "X-Session-ID": "npu-release-gate",
            "Idempotency-Key": f"gate-{uuid4().hex}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def _wait_for_terminal(body, timeout=600):
    evaluation_id = body["evaluation_id"]
    deadline = time.monotonic() + timeout
    while body.get("state") not in {"completed", "failed", "timed_out"}:
        assert time.monotonic() < deadline, "NPU smoke recovery window expired"
        time.sleep(0.5)
        status, body = _request("GET", f"/evaluations/{evaluation_id}")
        assert status == 200
        assert body.get("evaluation_id") == evaluation_id
    return body


def test_service_reports_readiness():
    _skip_without_service()
    status, _ = _request("GET", "/livez")
    assert status == 200
    status, body = _request("GET", "/readyz")
    assert status == 200
    assert body["ready"] is True


def test_correct_case_reaches_completed_with_environment():
    _skip_without_service()
    status, body = _request(
        "POST", "/evaluations?wait_seconds=0", build_request(measure_speed=False),
    )
    assert status in (200, 202)
    body = _wait_for_terminal(body)
    assert body["state"] == "completed"
    correctness = body["result"]["correctness"]
    assert correctness["status"] == "succeeded"
    assert correctness["data"]["passed"] is True
    assert correctness["data"]["passed_cases"] == correctness["data"]["total_cases"] == 2
    assert body["result"]["provenance"]["execution_environment"] is not None
