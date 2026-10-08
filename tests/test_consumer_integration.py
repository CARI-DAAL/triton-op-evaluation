"""Cross-repository HTTP gate, enabled with EVALUATION_CONSUMER_ROOT.

Load the actual consumer module without its agent runtime registration. Its
HTTP transport reaches the real API, durable store, and Worker with a CPU
evaluator. No schema or wire response is mocked at the consumer seam.
"""
import asyncio
import importlib.util
import json
import os
from pathlib import Path

import httpx
import pytest
from conftest import make_provenance
from contract_fixtures import CONTRACT_ROOT
from test_api import TRIAL_TOKEN, build_app, seed_worker
from test_worker import StubEvaluator, build_worker

from triton_evaluation.api.contract import EvaluationResult


@pytest.fixture
def consumer():
    root = os.environ.get("EVALUATION_CONSUMER_ROOT")
    if not root:
        pytest.skip("set EVALUATION_CONSUMER_ROOT for the two-repository HTTP gate")
    path = Path(root) / "kernel_agent/rewards/triton_eval_client.py"
    spec = importlib.util.spec_from_file_location("evaluation_consumer_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("checks, fixture", [
    (["correctness"], "resource-completed-correct.json"),
    (["correctness", "performance"], "resource-completed-profiler-performance.json"),
    (["profile"], "resource-completed-profile-only.json"),
])
def test_consumer_recovers_ambiguous_submit_and_polls_real_api(
    tmp_path, release_catalog_files, consumer, checks, fixture,
):
    app = build_app(tmp_path, release_catalog_files)
    seed_worker(app)
    result = json.loads((CONTRACT_ROOT / "fixtures" / fixture).read_text())["result"]
    result["provenance"] = make_provenance(app.settings).model_dump(mode="json")
    for check in ("compile", "correctness", "performance", "profile"):
        if check not in checks:
            result[check] = None
    evaluator = StubEvaluator(app.settings, result=EvaluationResult.model_validate(result))
    worker = build_worker(app.store, app.settings, evaluator, app.clock)
    calls = []
    ids = []

    def dispatch(request):
        calls.append((request.method, request.url.path, dict(request.headers)))
        gets = sum(method == "GET" for method, _, _ in calls)
        if request.method == "GET" and gets == 2:
            assert worker.run_once()

        async def send():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app.app), base_url="http://instance-a"
            ) as client:
                return await client.request(
                    request.method, request.url.path,
                    content=request.content, headers=request.headers,
                )

        response = asyncio.run(send())
        if request.method == "POST":
            ids.append(response.json()["evaluation_id"])
            if len(ids) == 1:
                raise httpx.ReadTimeout("response lost after durable admission", request=request)
        return httpx.Response(
            response.status_code, headers=response.headers,
            content=response.content, request=request,
        )

    capability = tmp_path / "consumer-capability"
    capability.write_text(TRIAL_TOKEN)
    client = consumer.TritonEvalClient(
        "http://instance-a", timeout=30, poll_interval=0,
        capability_file=str(capability), session_id="integration-session",
    )
    client._transport = httpx.MockTransport(dispatch)
    outcome = client.evaluate(
        op_name="addmm", reference_code="REF", kernel_code="IMPL",
        non_compute=False, checks=checks,
        **({"profile": {"mode": "op"}} if checks == ["profile"] else {}),
    )
    assert outcome["infra_error"] is False, outcome
    assert len(ids) == 2 and ids[0] == ids[1]
    assert len(evaluator.calls) == 1
    assert [method for method, _, _ in calls] == ["POST", "POST", "GET", "GET"]
    posts = [headers for method, _, headers in calls if method == "POST"]
    assert posts[0]["idempotency-key"] == posts[1]["idempotency-key"]
    assert all(headers["authorization"] == f"Bearer {TRIAL_TOKEN}" for _, _, headers in calls)
    assert all(path.startswith("/evaluations") for _, path, _ in calls)
    if checks == ["profile"]:
        assert outcome["profile_state"] in {"complete", "partial"}
    else:
        assert outcome["correct"] is True
        if "performance" in checks:
            assert outcome["benchmark_available"] is True
            expected = result["performance"]["data"]["overall_speedup_vs_reference"]
            assert outcome["speedup"] == expected
            assert outcome["benchmark"]["measurement_method"] == "profiler_kernel_sum_v1"
