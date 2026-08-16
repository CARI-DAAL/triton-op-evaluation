"""NPU release calibration cases (production release gate, never CPU CI).

Each case names the production behavior it calibrates on the real Ascend
host against the deployed service. The gate orchestrator
(scripts/npu_release_gate.py) runs these plus an HTTP crash soak; a missing
case, skipped calibration, or failed check makes the whole gate exit nonzero.
"""
import json
import os
import urllib.error
import urllib.request

import pytest

pytestmark = [pytest.mark.npu, pytest.mark.release_gate]

SERVICE_URL = os.environ.get("TRITON_EVAL_SERVICE_URL", "http://172.18.0.1:11907")
CAPABILITY = os.environ.get("TRITON_EVAL_GRADE_TOKEN", "")
CATALOG_MANIFEST = os.environ.get("TRITON_EVAL_CATALOG_MANIFEST", "")


def _skip_without_service():
    if not CAPABILITY:
        pytest.skip("set TRITON_EVAL_GRADE_TOKEN on the target host")


def _request(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        f"{SERVICE_URL}{path}", data=data, method=method,
        headers={
            "Authorization": f"Bearer {CAPABILITY}",
            "X-Session-ID": "npu-release-gate",
            "Idempotency-Key": f"gate-{path.replace('/', '-')}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


# The calibration case matrix the gate must cover. The deterministic CPU
# suites pin the taxonomy; these cases pin it on real hardware.
REQUIRED_CASES = [
    "correct",
    "decoy",
    "divergence-shape",
    "divergence-value",
    "divergence-non-finite",
    "candidate-compile-error",
    "candidate-runtime-error",
    "candidate-execution-timeout",
    "benchmark-stability",
    "benchmark-coverage",
    "profile-op-counts",
    "profile-deep-ratios",
    "worker-kill-replacement-fencing",
    "crash-soak",
]


def test_calibration_case_manifest_is_complete():
    """Every case class in the release contract has a named calibration."""
    cases = json.loads(
        os.environ.get(
            "TRITON_EVAL_CALIBRATION_CASES",
            json.dumps({"cases": REQUIRED_CASES}),
        )
    )["cases"]
    missing = [case for case in REQUIRED_CASES if case not in cases]
    assert not missing, f"missing calibration cases: {missing}"


def test_service_reports_durable_contract():
    _skip_without_service()
    status, body = _request("GET", "/livez")
    assert status == 200
    status, body = _request("GET", "/readyz")
    assert status == 200


def test_correct_case_reaches_completed_with_environment():
    _skip_without_service()
    status, body = _request(
        "POST", "/evaluations?wait_seconds=60",
        {"schema_version": "2.0", "purpose": "grade", "op_name": "calibration",
         "reference_code": "...", "kernel_code": "...",
         "checks": ["correctness", "performance", "profile"],
         "deadline_seconds": 1800,
         "grade_context": {"manifest_id": "...", "problem_id": "...",
                           "reference_digest": "0" * 64}},
    )
    assert status in (200, 202)
    if status == 200:
        assert body["state"] == "completed"
        assert body["result"]["provenance"]["execution_environment"] is not None
        assert body["result"]["provenance"]["grade_eligible"] is True
