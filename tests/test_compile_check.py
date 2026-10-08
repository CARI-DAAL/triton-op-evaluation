"""Compile evidence is acquired only by fenced CPU or NPU attempts."""
import json
import time
from datetime import datetime, timedelta, timezone
from unittest import mock

import pytest
from conftest import FakeClock
from test_api import build_app, make_test_settings, post, seed_worker, trial_headers
from test_evaluation_evaluator import ENVIRONMENT, Harness, RecordingSink, succeeded, trial_pinned
from test_worker import build_worker

from triton_evaluation.api.contract import EvaluationRequest
from triton_evaluation.evaluators.base import engine_failure
from triton_evaluation.evaluators.evaluation import EvaluationEvaluator
from triton_evaluation.runtime.worker import FencedProgress
from triton_evaluation.service.models import AlgorithmVersions, WorkerIdentity, new_worker_id


def payload(checks, source="class ModelNew: pass"):
    return {
        "schema_version": "2.1", "op_name": "addmm",
        "reference_code": "REF", "kernel_code": source, "checks": checks,
    }


def app_with_precheck(tmp_path, release_catalog_files, **overrides):
    settings = make_test_settings(tmp_path).model_copy(update={
        "verdict_cache_enabled": False, **overrides,
    })
    return build_app(
        tmp_path, release_catalog_files, settings=settings,
        clock=FakeClock(int(time.time() * 1000)),
    )


def cpu_worker(app):
    seed_worker(app, execution_class="cpu")
    return build_worker(
        app.store, app.settings, EvaluationEvaluator(app.settings.engine_dir, app.settings),
        app.clock, device_id=None,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("source,status,outcome", [
    ("class ModelNew: pass", "succeeded", "compiled"),
    ("def broken(: pass", "succeeded", "compile_error"),
    ("raise OSError('temporarily unavailable')", "failed", None),
])
async def test_compile_only_is_admitted_then_executed_by_cpu_attempt(
    tmp_path, release_catalog_files, source, status, outcome,
):
    app = app_with_precheck(tmp_path, release_catalog_files)
    with (
        mock.patch(
            "triton_evaluation.runtime.worker.health.CardHealth", side_effect=AssertionError,
        ),
        mock.patch(
            "triton_evaluation.runtime.worker.build_execution_environment",
            side_effect=AssertionError,
        ),
    ):
        worker = cpu_worker(app)
        async with app.http as client:
            response = await post(client, payload(["compile"], source))
            assert response.status_code == 202, response.text
            admitted = response.json()
            assert admitted["result"] is None
            assert worker.run_once()
            got = await client.get(
                f"/evaluations/{admitted['evaluation_id']}", headers=trial_headers()
            )
            resource = got.json()
            check = resource["result"]["compile"]
            assert check["status"] == status
            if outcome is None:
                assert check["data"] is None
                assert check["error"]["retryable"] is True
            else:
                assert check["data"]["outcome"] == outcome
                assert check["data"]["duration_ms"] >= 0
            assert resource["result"]["provenance"]["execution_environment"] is None
            assert resource["result"]["provenance"]["grade_eligible"] is False
            replay = await post(client, payload(["compile"], source))
            assert replay.json() == resource
            assert not worker.run_once()


@pytest.mark.anyio
async def test_candidate_import_is_not_executed_before_worker_claims(
    tmp_path, release_catalog_files,
):
    app = app_with_precheck(tmp_path, release_catalog_files)
    worker = cpu_worker(app)
    marker = tmp_path / "import-executed"
    source = f"open({str(marker)!r}, 'w').write('imported')"
    async with app.http as client:
        response = await post(client, payload(["compile"], source))
        assert response.status_code == 202
        assert not marker.exists()
        assert worker.run_once()
        assert marker.read_text() == "imported"


@pytest.mark.anyio
@pytest.mark.parametrize("source,compile_status,compile_verdict", [
    ("class ModelNew: pass", "succeeded", "compiled"),
    ("def broken(: pass", "succeeded", "compile_error"),
    ("raise OSError('temporarily unavailable')", "failed", None),
])
async def test_compile_profile_executes_precheck_in_npu_attempt_then_profiles(
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

    app = app_with_precheck(tmp_path, release_catalog_files, engine_failure_requeue=True)
    seed_worker(app)
    evaluator = Harness(verify=StubCheck(engine_failure("correctness", "runtime unavailable")))
    worker = build_worker(
        app.store, app.settings, evaluator, app.clock, canary=lambda **kwargs: True,
    )
    other_card = build_worker(
        app.store, app.settings, evaluator, app.clock, device_id=1,
        canary=lambda **kwargs: True,
    )
    async with app.http as client:
        response = await post(client, payload(
            ["compile", "correctness"], "raise OSError('temporarily unavailable')"
        ))
        eid = response.json()["evaluation_id"]
        # Opt-in engine failures without check evidence requeue under the shared
        # attempt budget instead of terminalizing the first attempt; the
        # failed card's own worker skips the requeued member (affinity).
        assert worker.run_once()
        requeued = await client.get(f"/evaluations/{eid}", headers=trial_headers())
        assert requeued.json()["state"] == "running"
        assert not worker.run_once()  # card 0 is avoided by last_failed_card
        assert other_card.run_once()  # attempt 2 on another card, requeued again
        assert worker.run_once()      # attempt 3: budget exhausted, terminal
        got = await client.get(f"/evaluations/{eid}", headers=trial_headers())
    assert got.json()["state"] == "completed"
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
async def test_replacement_does_not_retain_previous_attempt_compile_evidence(
    tmp_path, release_catalog_files,
):
    from triton_evaluation.service.terminal_results import compile_outcome_from_precheck

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
        progress = FencedProgress(app.store, lease)
        assert progress.begin("compile")
        assert progress.save("compile", compile_outcome_from_precheck("precheck_passed"))
        app.clock.advance_ms(app.settings.lease_seconds * 1000 + 1)
        assert app.store.recover_expired_leases().requeued == 1
        replacement = app.store.claim_next(identity)
        assert replacement.attempt_id != lease.attempt_id
        app.clock.advance_ms(1000 * 3600)
        assert app.store.expire_due_resources() == 1
        got = await client.get(f"/evaluations/{evaluation_id}", headers=trial_headers())
    result = got.json()["result"]
    assert result["compile"]["status"] == "timeout"
    assert result["compile"]["data"] is None
    assert result["correctness"]["status"] == "skipped"
    assert json.loads(app.store._r.hget(
        f"{app.settings.redis_prefix}:resource:{evaluation_id}", "resource"
    )) == got.json()


@pytest.mark.anyio
async def test_cached_compile_pass_is_used_only_after_cpu_claim(
    tmp_path, release_catalog_files,
):
    from triton_evaluation.runtime import verdict_cache

    app = app_with_precheck(tmp_path, release_catalog_files, verdict_cache_enabled=True)
    worker = cpu_worker(app)
    source = "class ModelNew: pass"
    digest = verdict_cache.digest_for(source, "triton_ascend_impl", app.settings.cann_fingerprint)
    verdict_cache.put(app.store, digest, "precheck_passed", 60)
    with mock.patch("triton_evaluation.runtime.precheck.run_script", side_effect=AssertionError):
        async with app.http as client:
            response = await post(client, payload(["compile"], source))
            assert response.status_code == 202, response.text
            assert response.json()["result"] is None
            assert worker.run_once()
            got = await client.get(
                f"/evaluations/{response.json()['evaluation_id']}", headers=trial_headers()
            )
    assert got.json()["result"]["compile"]["data"]["outcome"] == "compiled"


@pytest.mark.anyio
@pytest.mark.parametrize("include_compile", [True, False])
async def test_definitive_precheck_failure_never_replaces_correctness_or_profile(
    tmp_path, release_catalog_files, include_compile,
):
    app = app_with_precheck(tmp_path, release_catalog_files)
    seed_worker(app)
    profile = succeeded("profile-op-complete.json")
    verify = succeeded("verify-candidate-compile-error.json")
    evaluator = Harness(verify=verify, bench=None, profile=profile)
    worker = build_worker(app.store, app.settings, evaluator, app.clock)
    checks = ["correctness", "performance", "profile"]
    if include_compile:
        checks.insert(0, "compile")
    async with app.http as client:
        response = await post(client, payload(checks, "def broken(: pass"))
        assert response.status_code == 202, response.text
        assert worker.run_once()
        got = await client.get(
            f"/evaluations/{response.json()['evaluation_id']}", headers=trial_headers()
        )
    result = got.json()["result"]
    assert result["correctness"]["data"]["outcome"] == "compile_error"
    assert result["performance"]["reason"]["code"] == "DEPENDENCY_NOT_PASSED"
    assert result["profile"]["status"] == "succeeded"
    assert profile.calls == 1
    assert verify.calls == 1
