"""Compile evidence through admission, the durable worker, and recovery."""
import json
from datetime import datetime, timedelta, timezone
from unittest import mock

import pytest
from test_api import build_app, make_test_settings, post, seed_worker, trial_headers
from test_evaluation_evaluator import ENVIRONMENT, Harness, RecordingSink, succeeded, trial_pinned
from test_worker import build_worker

from triton_evaluation.api.contract import EvaluationRequest
from triton_evaluation.evaluators.base import engine_failure
from triton_evaluation.models import AlgorithmVersions, WorkerIdentity, new_worker_id


def payload(checks, source="class ModelNew: pass"):
    return {
        "schema_version": "2.1", "op_name": "addmm",
        "reference_code": "REF", "kernel_code": source, "checks": checks,
    }


def app_with_precheck(tmp_path, release_catalog_files, **overrides):
    settings = make_test_settings(tmp_path).model_copy(update={
        "compile_precheck_enabled": True, "verdict_cache_enabled": False, **overrides,
    })
    return build_app(tmp_path, release_catalog_files, settings=settings)


@pytest.mark.anyio
@pytest.mark.parametrize("source,status,outcome", [
    ("class ModelNew: pass", "succeeded", "compiled"),
    ("def broken(: pass", "succeeded", "compile_error"),
    ("raise OSError('temporarily unavailable')", "failed", None),
])
async def test_compile_only_is_durable_terminal_without_worker(
    tmp_path, release_catalog_files, source, status, outcome,
):
    app = app_with_precheck(tmp_path, release_catalog_files)
    async with app.http as client:
        response = await post(client, payload(["compile"], source))
        assert response.status_code == 200, response.text
        resource = response.json()
        check = resource["result"]["compile"]
        assert check["status"] == status
        assert resource["state"] == ("failed" if status == "failed" else "completed")
        if outcome is None:
            assert check["data"] is None
            assert check["error"]["retryable"] is True
        else:
            assert check["data"]["outcome"] == outcome
            assert check["data"]["duration_ms"] >= 0
        got = await client.get(f"/evaluations/{resource['evaluation_id']}", headers=trial_headers())
        assert got.json() == resource
        with mock.patch("triton_evaluation.api.routes.run_precheck", side_effect=AssertionError):
            replay = await post(client, payload(["compile"], source))
        assert replay.json() == resource
    for queue in ("interactive", "bulk"):
        assert app.store._r.zcard(f"{app.settings.redis_prefix}:ready:exec:{queue}") == 0
    assert app.store._r.zcard(f"{app.settings.redis_prefix}:deadlines") == 0


@pytest.mark.anyio
async def test_explicit_compile_executes_when_optional_precheck_is_disabled(
    tmp_path, release_catalog_files,
):
    app = app_with_precheck(tmp_path, release_catalog_files, compile_precheck_enabled=False)
    async with app.http as client:
        response = await post(client, payload(["compile"], "def broken(: pass"))
    assert response.status_code == 200, response.text
    assert response.json()["result"]["compile"]["data"]["outcome"] == "compile_error"


@pytest.mark.anyio
@pytest.mark.parametrize("source,compile_status,compile_verdict", [
    ("class ModelNew: pass", "succeeded", "compiled"),
    ("def broken(: pass", "succeeded", "compile_error"),
    ("raise OSError('temporarily unavailable')", "failed", None),
])
async def test_compile_profile_uses_admission_evidence_and_runs_profile(
    tmp_path, release_catalog_files, source, compile_status, compile_verdict,
):
    app = app_with_precheck(tmp_path, release_catalog_files)
    seed_worker(app)
    profile = succeeded("profile-op-complete.json")
    evaluator = Harness(verify=None, profile=profile)
    worker = build_worker(app.store, app.settings, evaluator, app.clock)
    async with app.http as client:
        response = await post(client, payload(["compile", "profile"], source))
        assert response.status_code == 202, response.text
        assert worker.run_once()
        got = await client.get(
            f"/evaluations/{response.json()['evaluation_id']}", headers=trial_headers()
        )
    result = got.json()["result"]
    assert result["compile"]["status"] == compile_status
    if compile_status == "failed":
        assert result["compile"]["data"] is None
        assert result["compile"]["error"]["retryable"] is True
    else:
        assert result["compile"]["data"]["outcome"] == compile_verdict
    assert profile.calls == 1
    assert result["profile"]["status"] == "succeeded"
    assert result["correctness"] is None


@pytest.mark.anyio
async def test_inconclusive_compile_is_not_inferred_from_failed_correctness(
    tmp_path, release_catalog_files,
):
    from test_evaluation_evaluator import StubCheck

    app = app_with_precheck(tmp_path, release_catalog_files)
    seed_worker(app)
    evaluator = Harness(verify=StubCheck(engine_failure("correctness", "runtime unavailable")))
    worker = build_worker(app.store, app.settings, evaluator, app.clock)
    async with app.http as client:
        response = await post(client, payload(
            ["compile", "correctness"], "raise OSError('temporarily unavailable')"
        ))
        assert worker.run_once()
        got = await client.get(
            f"/evaluations/{response.json()['evaluation_id']}", headers=trial_headers()
        )
    result = got.json()["result"]
    assert result["compile"]["status"] == "failed"
    assert result["compile"]["data"] is None
    assert result["correctness"]["status"] == "failed"


@pytest.mark.anyio
async def test_precheck_pass_does_not_claim_backend_jit_success(
    tmp_path, release_catalog_files,
):
    app = app_with_precheck(tmp_path, release_catalog_files)
    seed_worker(app)
    evaluator = Harness(verify=succeeded("verify-candidate-compile-error.json"))
    worker = build_worker(app.store, app.settings, evaluator, app.clock)
    async with app.http as client:
        response = await post(client, payload(["compile", "correctness"]))
        assert worker.run_once()
        got = await client.get(
            f"/evaluations/{response.json()['evaluation_id']}", headers=trial_headers()
        )
    result = got.json()["result"]
    assert result["compile"]["data"]["outcome"] == "compiled"
    assert result["correctness"]["data"]["outcome"] == "compile_error"


def test_missing_evidence_is_failed_and_fenced_before_other_checks(tmp_path):
    evaluator = Harness(verify=succeeded("verify-correct.json"))
    sink = RecordingSink()
    result = evaluator.run(
        request=EvaluationRequest.model_validate(payload(["compile", "correctness"])),
        work_dir=tmp_path, device_id=0,
        deadline_at=datetime.now(timezone.utc) + timedelta(minutes=10),
        lease_valid=lambda: True, execution_environment=ENVIRONMENT,
        pinned=trial_pinned(), progress=sink,
    )
    assert result.compile.status == "failed"
    assert result.compile.error.retryable is True
    assert result.correctness.data.outcome == "correct"
    assert sink.calls[:2] == [("begin", "compile"), ("save", "compile", "failed")]


@pytest.mark.anyio
async def test_replacement_and_deadline_preserve_admitted_compile_evidence(
    tmp_path, release_catalog_files,
):
    app = app_with_precheck(tmp_path, release_catalog_files)
    seed_worker(app)
    async with app.http as client:
        response = await post(client, payload(["compile", "correctness"]))
        evaluation_id = response.json()["evaluation_id"]
        identity = WorkerIdentity(new_worker_id(), app.settings.evaluator_build, AlgorithmVersions(
            app.settings.correctness_algorithm_version, app.settings.decoy_algorithm_version,
            app.settings.benchmark_algorithm_version, app.settings.profiler_algorithm_version,
        ))
        lease = app.store.claim_next(identity)
        app.clock.advance_ms(app.settings.lease_seconds * 1000 + 1)
        assert app.store.recover_expired_leases().requeued == 1
        replacement = app.store.claim_next(identity)
        assert replacement.attempt_id != lease.attempt_id
        app.clock.advance_ms(1000 * 3600)
        assert app.store.expire_due_resources() == 1
        got = await client.get(f"/evaluations/{evaluation_id}", headers=trial_headers())
    result = got.json()["result"]
    assert result["compile"]["status"] == "succeeded"
    assert result["compile"]["data"]["outcome"] == "compiled"
    assert result["correctness"]["status"] == "timeout"
    assert json.loads(app.store._r.hget(
        f"{app.settings.redis_prefix}:resource:{evaluation_id}", "resource"
    )) == got.json()


@pytest.mark.anyio
async def test_cached_compile_pass_is_evidence_without_another_subprocess(
    tmp_path, release_catalog_files,
):
    from triton_evaluation import verdict_cache

    app = app_with_precheck(tmp_path, release_catalog_files, verdict_cache_enabled=True)
    source = "class ModelNew: pass"
    digest = verdict_cache.digest_for(source, "triton_ascend_impl", app.settings.cann_fingerprint)
    verdict_cache.put(app.store, digest, "precheck_passed", 60)
    with mock.patch("triton_evaluation.precheck.subprocess.run", side_effect=AssertionError):
        async with app.http as client:
            response = await post(client, payload(["compile"], source))
    assert response.status_code == 200, response.text
    assert response.json()["result"]["compile"]["data"]["outcome"] == "compiled"
