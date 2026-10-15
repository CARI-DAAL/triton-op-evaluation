"""Single-source evidence crosses the real admission, worker and recovery seams."""

import json
from datetime import datetime, timedelta, timezone

import pytest
from test_api import build_app, seed_worker, trial_headers
from test_compile_check import app_with_precheck, cpu_worker
from test_evaluation_evaluator import (
    ENVIRONMENT,
    Harness,
    RecordingSink,
    StubCheck,
    succeeded,
    trial_pinned,
)
from test_worker import build_worker

from triton_evaluation.api.contract import CheckTimedOut, ErrorInfo
from triton_evaluation.evaluators.base import EngineCheckResult, LeaseLostError, materialize
from triton_evaluation.operator_input import OperatorRequest, to_evaluation_request
from triton_evaluation.service.terminal_results import complete_evaluation


def payload(implementation, **options):
    source = "def run(x): return x + 1\n"
    return {
        "definition": {
            "inputs": {"x": {}},
            **({"reference": source} if implementation == "torch" else {}),
        },
        **({"triton_code": source} if implementation == "triton" else {}),
        "workloads": [{"inputs": {"x": {"type": "random", "shape": [2], "dtype": "float32"}}}],
        **options,
    }


def measurements(implementation):
    return {
        "implementation": implementation,
        "correctness_verified": False,
        "measurement_method": "npu_event_v1",
        "selected_cases": 1,
        "measured_cases": 1,
        "coverage": 1.0,
        "per_case": [{
            "case_idx": 0, "input_shapes": [], "latency_ms": 0.2, "peak_memory_mb": 1.0,
        }],
        "unmeasured_cases": [],
    }


class NoVerification:
    def run(self, *args, **kwargs):
        raise AssertionError("single-source requests must not manufacture correctness")


def harness_for(implementation, raw=None):
    return Harness(
        verify=NoVerification(),
        bench=StubCheck(EngineCheckResult(
            status="succeeded", raw=raw or measurements(implementation),
        )),
        profile=succeeded("profile-op-complete.json"),
    )


def run(request, harness, tmp_path, sink=None):
    return harness.run(
        request, tmp_path, 0, datetime.now(timezone.utc) + timedelta(minutes=1),
        lambda: True, ENVIRONMENT, trial_pinned(), sink or RecordingSink(),
    )


@pytest.mark.parametrize("implementation", ["torch", "triton"])
def test_single_measurement_runs_without_a_correctness_gate(tmp_path, implementation):
    request = to_evaluation_request(OperatorRequest.model_validate(payload(implementation)))
    harness = harness_for(implementation)
    sink = RecordingSink()
    result = run(request, harness, tmp_path, sink)
    assert sink.calls == [("begin", "performance"), ("save", "performance", "succeeded")]
    assert result.correctness is None
    assert result.performance.data.implementation == implementation
    assert result.performance.data.correctness_verified is False
    assert "speedup" not in json.dumps(result.performance.model_dump(mode="json"))
    assert result.provenance.grade_eligible is False


@pytest.mark.parametrize("mutation", ["subject", "method", "selection", "verified", "speedup"])
def test_single_measurements_cannot_be_misattributed(tmp_path, mutation):
    request = to_evaluation_request(OperatorRequest.model_validate(payload(
        "torch", performance={"case_indices": [0]},
    )))
    raw = measurements("torch")
    if mutation == "subject":
        raw["implementation"] = "triton"
    elif mutation == "method":
        raw["measurement_method"] = "profiler_kernel_sum_v1"
    elif mutation == "selection":
        raw["per_case"][0]["case_idx"] = 1
    elif mutation == "verified":
        raw["correctness_verified"] = True
    else:
        raw["overall_speedup_vs_reference"] = 1.0
    result = run(request, harness_for("torch", raw), tmp_path)
    assert result.correctness is None
    assert result.performance.status == "failed"
    assert result.performance.error.code == "INVALID_ENGINE_OUTPUT"


@pytest.mark.parametrize("implementation", ["torch", "triton"])
def test_single_profile_is_explicitly_unverified(tmp_path, implementation):
    request = to_evaluation_request(OperatorRequest.model_validate(payload(
        implementation, checks=["profile"],
    )))
    result = run(request, harness_for(implementation), tmp_path)
    assert result.correctness is None
    assert result.performance is None
    assert result.profile.status == "succeeded"
    assert result.profile.data.implementation == implementation
    assert result.profile.data.correctness_verified is False


def test_single_measurement_obeys_fenced_progress(tmp_path):
    request = to_evaluation_request(OperatorRequest.model_validate(payload("triton")))
    harness = harness_for("triton")
    with pytest.raises(LeaseLostError):
        run(request, harness, tmp_path, RecordingSink(begin_ok=False))
    assert harness.benchmark.calls == 0


def test_missing_single_evidence_does_not_create_a_correctness_dependency():
    result = complete_evaluation(
        requested_checks=["compile", "performance"], outcomes={}, pinned=trial_pinned(),
        failure=CheckTimedOut(status="timeout", error=ErrorInfo(
            code="DEADLINE_EXCEEDED", phase="deadline", check=None,
            retryable=False, message="deadline expired",
        )),
    )
    assert result.correctness is None
    assert result.performance.status == "skipped"
    assert result.performance.reason.depends_on is None


def test_materialize_torch_only_does_not_create_a_missing_candidate(tmp_path):
    request = to_evaluation_request(OperatorRequest.model_validate(payload("torch")))
    stem = materialize(request, tmp_path)
    assert (tmp_path / f"{stem}_torch.py").is_file()
    assert not (tmp_path / f"{stem}_{request.triton_impl_name}.py").exists()


@pytest.mark.anyio
@pytest.mark.parametrize("implementation", ["torch", "triton"])
@pytest.mark.parametrize("endpoint", ["evaluations", "benchmark", "profile"])
async def test_single_http_worker_and_recovery(tmp_path, implementation, endpoint):
    app = build_app(tmp_path, None)
    seed_worker(app)
    worker = build_worker(
        app.store, app.settings, harness_for(implementation), app.clock,
    )
    body = payload(implementation)
    async with app.http as client:
        submitted = await client.post(f"/{endpoint}", json=body, headers=trial_headers())
        assert submitted.status_code == 202, submitted.text
        assert worker.run_once()
        replay = await client.post(f"/{endpoint}", json=body, headers=trial_headers())
        recovered = await client.get(submitted.headers["Location"], headers=trial_headers())
    assert replay.status_code == recovered.status_code == 200
    assert replay.json() == recovered.json()
    resource = recovered.json()
    assert resource["state"] == "completed"
    result = resource["result"]
    assert result["correctness"] is None
    check = "profile" if endpoint == "profile" else "performance"
    assert result[check]["data"]["implementation"] == implementation
    assert result[check]["data"]["correctness_verified"] is False
    assert "speedup" not in json.dumps(result[check])


@pytest.mark.anyio
@pytest.mark.parametrize("implementation", ["torch", "triton"])
async def test_single_verify_rejected_before_any_admission(tmp_path, implementation):
    app = build_app(tmp_path, None)
    async with app.http as client:
        response = await client.post(
            "/verify", json=payload(implementation), headers=trial_headers(),
        )
    assert response.status_code == 422
    assert not app.store._r.keys(f"{app.settings.redis_prefix}:resource:*")


@pytest.mark.anyio
@pytest.mark.parametrize("implementation", ["torch", "triton"])
async def test_single_compile_checks_the_supplied_source(tmp_path, implementation):
    app = app_with_precheck(tmp_path, None)
    body = payload(implementation, checks=["compile"])
    if implementation == "torch":
        body["definition"]["reference"] = "def broken(: pass\n"
    else:
        body["triton_code"] = "def broken(: pass\n"
    worker = cpu_worker(app)
    async with app.http as client:
        response = await client.post("/evaluations", json=body, headers=trial_headers())
        assert response.status_code == 202
        assert worker.run_once()
        resource = (await client.get(response.headers["Location"], headers=trial_headers())).json()
    assert resource["result"]["compile"]["data"]["outcome"] == "compile_error"
    assert resource["result"]["correctness"] is None
