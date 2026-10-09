"""HTTP protocol tests with typed real contract fixture responses; no hardware."""

import copy
import json

import pytest

from scripts import npu_batchmatmul_gate as gate
from triton_evaluation.engine.validate_triton_impl import validate
from triton_evaluation.evaluators.shaping import shape_profile

FIXTURES = gate.ROOT / "tests/fixtures"


def terminal(body, stage):
    raw = json.loads(
        (
            FIXTURES / "evaluation_api/2.1/fixtures/resource-completed-performance-profile-op.json"
        ).read_text()
    )
    raw["reference_digest"] = gate.reference_digest(body)
    result = raw["result"]
    result["provenance"]["correctness_algorithm"] += ":npu-strict-v1"
    assert result["provenance"]["execution_environment"] is not None
    result["correctness"]["data"].update(total_cases=51, passed_cases=51)
    if stage.startswith("performance_"):
        data = result["performance"]["data"]
        data.update(
            selected_cases=51,
            measured_cases=51,
            coverage=1,
            unmeasured_cases=[],
            overall_speedup_vs_reference=2,
        )
        case = data["per_case"][0]
        tag = "npu_event_v1" if stage.endswith("event") else "profiler_kernel_sum_v1"
        data["measurement_method"] = tag
        data["per_case"] = []
        for i in range(51):
            item = copy.deepcopy(case)
            item.update(case_idx=i, measurement_method=tag)
            data["per_case"].append(item)
    else:
        result["performance"] = None
    if stage.startswith("profile_"):
        result["correctness"] = None
        mode = stage.removeprefix("profile_")
        if mode == "deep":
            engine = json.loads((FIXTURES / "engine/profile-deep-partial.json").read_text())
            data = shape_profile(engine).model_dump(mode="json")
        else:
            data = result["profile"]["data"]
        case = data["cases"][0]
        data.update(counts={"selected": 51, "complete": 51, "partial": 0, "failed": 0})
        data["cases"] = []
        for i in range(51):
            item = copy.deepcopy(case)
            item["case_idx"] = i
            data["cases"].append(item)
        result["profile"]["data"] = data
    else:
        result["profile"] = None
    return raw


def test_http_chain_exact_official_inputs(tmp_path):
    requests = []

    def transport(method, url, body=None, **kwargs):
        assert method == "POST"
        requests.append((url, body))
        stage = (
            "correctness"
            if url.endswith("/verify")
            else "performance_" + body["performance"]["method"]
            if url.endswith("/benchmark")
            else "profile_" + body["profile"]["mode"]
        )
        return terminal(body, stage)

    output = tmp_path / "report.json"
    report = gate.run_gate("http://service", "SECRET", output, transport=transport)
    assert report["passed"], report.get("error", report["checks"])
    assert len(requests) == 5
    manifest, definition, workloads = gate.fixture_check()
    for _, body in requests:
        assert body["definition"] == definition
        assert body["workloads"] == workloads
        assert body["seed"] == 200
        assert body["custom_inputs_code"] is None
    assert all(
        not body["performance"]["use_baseline_cache"]
        for _, body in requests
        if "performance" in body
    )
    serialized = output.read_text()
    assert "SECRET" not in serialized and "_bmm" not in serialized
    assert report["declared_reference_policy"] == "npu-strict-v1"


def test_adapted_candidate_passes_real_decoy_checker():
    _, definition, workloads = gate.fixture_check()
    body = gate.build_body(definition, workloads, "correctness", 3600)
    assert validate(body["triton_code"])["valid"] is True
    generated = gate.to_evaluation_request(gate.OperatorRequest.model_validate(body))
    assert validate(generated.kernel_code)["valid"] is True


def test_check_only_never_sends_http(tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("HTTP forbidden")

    report = gate.run_gate("", "", tmp_path / "report.json", check_only=True, transport=forbidden)
    assert report["passed"] and not report["hardware_verified"]
    assert report["scope"] == "fixture-integrity-only"


@pytest.mark.parametrize("fault", ["digest", "partial", "performance", "policy"])
def test_protocol_failures_cannot_pass(tmp_path, fault):
    def transport(method, url, body=None, **kwargs):
        stage = (
            "correctness"
            if url.endswith("/verify")
            else "performance_" + body["performance"]["method"]
            if url.endswith("/benchmark")
            else "profile_" + body["profile"]["mode"]
        )
        raw = terminal(body, stage)
        if fault == "digest":
            raw["reference_digest"] = "0" * 64
        if fault == "policy":
            raw["result"]["provenance"]["correctness_algorithm"] = "cpu-first"
        if fault == "performance" and stage.startswith("performance_"):
            raw["result"]["performance"]["data"]["per_case"][0]["implementation_latency_ms"] = 0
        if fault == "partial" and stage == "profile_op":
            case = raw["result"]["profile"]["data"]["cases"][0]
            case["status"] = "partial"
            case["error"] = {
                "code": "PROFILE_PARTIAL",
                "phase": "execution",
                "check": "profile",
                "retryable": False,
                "message": "gap",
            }
            raw["result"]["profile"]["data"]["counts"].update(complete=50, partial=1)
        return raw

    report = gate.run_gate(
        "http://service", "secret", tmp_path / "report.json", transport=transport
    )
    assert not report["passed"]
    assert not report["hardware_verified"]


def test_handle_persisted_before_poll_failure(tmp_path):
    output = tmp_path / "report.json"

    def transport(method, url, body=None, **kwargs):
        if method == "GET":
            saved = json.loads(output.read_text())
            assert saved["checks"]["correctness"]["evaluation_id"] == "accepted-id"
            raise OSError("connection lost")
        raw = terminal(body, "correctness")
        raw.update(
            evaluation_id="accepted-id",
            state="queued",
            result=None,
            finished_at=None,
            started_at=None,
        )
        return raw

    report = gate.run_gate(
        "http://service", "secret", output, transport=transport, sleep=lambda _: None
    )
    assert not report["passed"]
    assert report["checks"]["correctness"]["evaluation_id"] == "accepted-id"


def test_fixture_tamper_fails(tmp_path):
    import shutil

    fixture = tmp_path / "fixture"
    shutil.copytree(gate.FIXTURE, fixture)
    (fixture / "workload.jsonl").write_text("tampered")
    report = gate.run_gate("", "", tmp_path / "report.json", fixture=fixture, check_only=True)
    assert not report["passed"]


def test_lost_post_response_replays_identical_request(tmp_path):
    output = tmp_path / "report.json"
    first = None
    calls = 0

    def transport(method, url, body=None, headers=None, **kwargs):
        nonlocal calls, first
        calls += 1
        if calls == 1:
            first = (url, copy.deepcopy(body), dict(headers))
            saved = json.loads(output.read_text())["checks"]["correctness"]
            assert saved["session_id"] == headers["X-Session-ID"]
            assert saved["idempotency_key"] == headers["Idempotency-Key"]
            assert saved["route"] == "/verify" and saved["deadline_seconds"] == 3600
            raise OSError("response lost")
        if calls == 2:
            assert (url, body, headers) == first
        stage = (
            "correctness"
            if url.endswith("/verify")
            else "performance_" + body["performance"]["method"]
            if url.endswith("/benchmark")
            else "profile_" + body["profile"]["mode"]
        )
        return terminal(body, stage)

    report = gate.run_gate(
        "http://service", "TOPSECRET", output, transport=transport, sleep=lambda _: None
    )
    assert report["passed"] and calls == 6
    assert report["checks"]["correctness"]["post_retries"] == 1


def test_failed_post_keeps_recovery_metadata_and_redacts_token(tmp_path):
    output = tmp_path / "report.json"
    calls = []

    def transport(*args, **kwargs):
        calls.append(kwargs["headers"])
        raise OSError("TOPSECRET appears in transport exception")

    report = gate.run_gate(
        "http://service", "TOPSECRET", output, transport=transport, sleep=lambda _: None
    )
    assert not report["passed"]
    entry = report["checks"]["correctness"]
    assert entry["route"] == "/verify" and entry["session_id"] and entry["idempotency_key"]
    assert calls[0] == calls[1]
    assert "TOPSECRET" not in output.read_text()
    assert "TOPSECRET" not in json.dumps(report)


def test_invalid_server_text_cannot_leak_token(tmp_path):
    output = tmp_path / "report.json"
    report = gate.run_gate(
        "http://service",
        "TOPSECRET",
        output,
        transport=lambda *a, **k: {"evaluation_id": "id", "junk": "TOPSECRET"},
    )
    assert not report["passed"]
    assert "TOPSECRET" not in output.read_text()
    assert "TOPSECRET" not in json.dumps(report)
