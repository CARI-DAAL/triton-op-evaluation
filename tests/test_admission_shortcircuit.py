"""Admission-time terminal synthesis for cached compile_error verdicts."""
from types import SimpleNamespace

import pytest
from conftest import make_caller, make_request

from triton_evaluation import terminal_results, verdict_cache
from triton_evaluation.api import routes
from triton_evaluation.api.contract import EvaluationResult

pytestmark = pytest.mark.anyio


def _command_for(settings, kernel_code="import torch\n"):
    caller = make_caller()
    request = make_request(kernel_code=kernel_code)
    from triton_evaluation.admission import admit
    from triton_evaluation.catalog import ReleaseCatalog

    admission = admit(request, caller, ReleaseCatalog({}))
    return routes._build_command(
        SimpleNamespace(settings=settings), caller, "s1", "k1", admission, "trial"
    )


def test_synthesize_matches_frozen_compile_error_shape(settings):
    result = terminal_results.synthesize_compile_error_result(
        requested_checks=["correctness", "performance"],
        failure_message="candidate compilation failed",
        total_cases=8,
        evaluator_build=settings.evaluator_build,
        algorithms=SimpleNamespace(
            correctness=settings.correctness_algorithm_version,
            decoy=settings.decoy_algorithm_version,
            benchmark=settings.benchmark_algorithm_version,
            profiler=settings.profiler_algorithm_version,
        ),
        manifest_id=None,
        problem_id=None,
    )
    validated = EvaluationResult.model_validate(
        result.model_dump(mode="json", exclude_none=False),
        context={"requested_checks": ["correctness", "performance"]},
    )
    correctness = validated.correctness
    assert correctness.status == "succeeded"
    assert correctness.data.outcome == "compile_error"
    assert correctness.data.passed is False
    assert correctness.data.passed_cases == 0
    assert correctness.data.total_cases == 8
    assert correctness.data.domain_failure.code == "SUBJECT_COMPILE_ERROR"
    assert correctness.data.domain_failure.phase == "compile"
    assert correctness.data.domain_failure.case_idx is None
    assert validated.performance.status == "skipped"
    assert validated.performance.reason.code == "DEPENDENCY_NOT_PASSED"
    assert validated.provenance.grade_eligible is False


def test_create_terminal_at_admission(store, settings, redis_client, clock,
                                      make_command, make_worker):
    cmd = make_command()
    result = terminal_results.synthesize_compile_error_result(
        requested_checks=list(cmd.admission.requested_checks),
        failure_message="candidate compilation failed",
        total_cases=1,
        evaluator_build=cmd.evaluator_build,
        algorithms=cmd.algorithms,
        manifest_id=None,
        problem_id=None,
    )
    outcome = store.create_terminal_at_admission(cmd, result)
    assert outcome.decision == "created"
    resource = outcome.resource
    assert resource.state == "completed"
    assert resource.revision == 1
    assert resource.started_at is not None and resource.finished_at is not None
    assert resource.result.correctness.data.outcome == "compile_error"
    # never queued: no ready membership, no inflight held
    assert redis_client.zscore("triton-eval:v2:ready:exec:interactive",
                               cmd.evaluation_id.encode()) is None
    # idempotent replay returns the same terminal resource
    replay = store.create_terminal_at_admission(cmd, result)
    assert replay.decision == "replayed"
    assert replay.resource.evaluation_id == resource.evaluation_id
    # normal claiming sees nothing
    assert store.claim_next(make_worker()) is None


def test_shortcircuit_wiring_skips_queue(store, settings, clock, make_command):
    """End-to-end at the broker+precheck level: cached verdict -> terminal."""
    cmd = make_command()
    request = __import__("json").loads(cmd.admission.canonical_request)
    digest = verdict_cache.digest_for(
        request["kernel_code"],
        request.get("triton_impl_name", "triton_ascend_impl"),
        settings.cann_fingerprint,
    )
    verdict_cache.put(store, digest, "compile_error", settings.verdict_cache_ttl_seconds)
    # admission path consults the cache before create_or_replay
    from triton_evaluation import precheck

    assert (
        precheck.run_precheck(store, settings, request["kernel_code"],
                              request.get("triton_impl_name", "triton_ascend_impl"))
        == "compile_error"
    )


# ---------------------------------------------------------------------------
# Routes composition: flag gate + profile guard on the live HTTP path
# ---------------------------------------------------------------------------


async def _post_with_cached_verdict(tmp_path, release_catalog_files, payload):
    from test_api import build_app, make_test_settings, post, seed_worker, trial_headers

    settings = make_test_settings(tmp_path)
    app = build_app(tmp_path, release_catalog_files, settings=settings)
    seed_worker(app)  # readiness
    digest = verdict_cache.digest_for(
        payload["kernel_code"],
        payload.get("triton_impl_name", "triton_ascend_impl"),
        settings.cann_fingerprint,
    )
    verdict_cache.put(app.store, digest, "compile_error",
                      settings.verdict_cache_ttl_seconds)
    response = await post(app.http, payload, trial_headers())
    return app, response


async def test_routes_shortcircuit_cached_verdict(tmp_path, release_catalog_files):
    app, response = await _post_with_cached_verdict(
        tmp_path, release_catalog_files,
        {"schema_version": "2.0", "op_name": "addmm", "reference_code": "REF",
         "kernel_code": "def broken(:\n", "checks": ["correctness"]},
    )
    assert response.status_code == 200, response.text
    resource = response.json()
    assert resource["state"] == "completed"
    assert resource["result"]["correctness"]["data"]["outcome"] == "compile_error"
    # never queued: no ready membership anywhere
    assert app.store._r.zcard("triton-eval:v2:ready:exec:interactive") == 0
    assert app.store._r.zcard("triton-eval:v2:ready:exec:bulk") == 0


async def test_routes_profile_guard_queues_normally(tmp_path, release_catalog_files):
    app, response = await _post_with_cached_verdict(
        tmp_path, release_catalog_files,
        {"schema_version": "2.0", "op_name": "addmm", "reference_code": "REF",
         "kernel_code": "def broken(:\n", "checks": ["correctness", "profile"]},
    )
    # profile guard: a cached compile verdict must not short-circuit a
    # profile request — it queues normally.
    assert response.status_code == 202, response.text
    assert response.json()["state"] == "queued"
    assert app.store._r.zcard("triton-eval:v2:state:completed") == 0


# ---------------------------------------------------------------------------
# v2.1 compile check: admission precheck execution, compile slot assembly
# ---------------------------------------------------------------------------


async def test_v21_compile_only_cached_verdict_returns_terminal(tmp_path,
                                                                release_catalog_files):
    app, response = await _post_with_cached_verdict(
        tmp_path, release_catalog_files,
        {"schema_version": "2.1", "op_name": "addmm", "reference_code": "REF",
         "kernel_code": "def broken(:\n", "checks": ["compile"]},
    )
    assert response.status_code == 200, response.text
    resource = response.json()
    assert resource["schema_version"] == "2.1"
    assert resource["state"] == "completed"
    compile_slot = resource["result"]["compile"]
    assert compile_slot["status"] == "succeeded"
    assert compile_slot["data"]["outcome"] == "compile_error"
    assert compile_slot["data"]["compiler_stage"] == "frontend"
    assert resource["result"]["correctness"] is None
    # never queued: no ready membership anywhere
    assert app.store._r.zcard("triton-eval:v2:ready:exec:interactive") == 0
    assert app.store._r.zcard("triton-eval:v2:ready:exec:bulk") == 0


async def test_v21_compile_verdict_shortcircuit_fills_compile_slot(
    tmp_path, release_catalog_files
):
    app, response = await _post_with_cached_verdict(
        tmp_path, release_catalog_files,
        {"schema_version": "2.1", "op_name": "addmm", "reference_code": "REF",
         "kernel_code": "def broken(:\n", "checks": ["compile", "correctness"]},
    )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["compile"]["data"]["outcome"] == "compile_error"
    assert result["compile"]["data"]["compiler_stage"] == "frontend"
    # the correctness carrier keeps the engine-identical compile_error shape
    assert result["correctness"]["data"]["outcome"] == "compile_error"
    assert result["correctness"]["data"]["domain_failure"]["code"] == (
        "SUBJECT_COMPILE_ERROR"
    )


async def test_v21_admission_records_precheck_verdict_on_resource(
    tmp_path, release_catalog_files
):
    app, response = await _post_with_cached_verdict(
        tmp_path, release_catalog_files,
        {"schema_version": "2.1", "op_name": "addmm", "reference_code": "REF",
         "kernel_code": "def broken(:\n", "checks": ["compile"]},
    )
    evaluation_id = response.json()["evaluation_id"]
    key = f"triton-eval:v2:resource:{evaluation_id}"
    assert app.store._r.hget(key, "precheck_verdict") == b"compile_error"
    assert int(app.store._r.hget(key, "precheck_duration_ms")) >= 0


async def test_v21_compiled_verdict_queues_normally(tmp_path, release_catalog_files):
    from test_api import build_app, make_test_settings, post, seed_worker, trial_headers

    app = build_app(tmp_path, release_catalog_files, settings=make_test_settings(tmp_path))
    seed_worker(app)
    response = await post(
        app.http,
        {"schema_version": "2.1", "op_name": "addmm", "reference_code": "REF",
         "kernel_code": "IMPL", "checks": ["compile", "correctness"]},
        trial_headers(),
    )
    assert response.status_code == 202, response.text
    resource = response.json()
    assert resource["schema_version"] == "2.1"
    assert resource["state"] == "queued"
    # the advisory precheck fields exist on the created resource hash
    key = f"triton-eval:v2:resource:{resource['evaluation_id']}"
    assert app.store._r.hget(key, "precheck_verdict") == b""
    assert app.store._r.hget(key, "precheck_excerpt") == b""


# ---------------------------------------------------------------------------
# Poison quarantine is version-aware (SUSPECTED_POISON_CANDIDATE for 2.1)
# ---------------------------------------------------------------------------


def _poison_terminal(store, settings, clock, make_command, make_worker, request,
                     idempotency_key):
    cmd = make_command(request=request, idempotency_key=idempotency_key)
    store.create_or_replay(cmd)
    for _ in range(2):
        lease = store.claim_next(make_worker())
        store.note_claimed_card(lease.evaluation_id, card=7)
        clock.advance_ms(settings.lease_seconds * 1000 + 1)
        counts = store.recover_expired_leases()
        if counts.failed:
            break
        clock.advance_ms(1)  # re-arm the replacement claim for the next death
    return store.get_visible(cmd.evaluation_id, make_caller())


def test_poison_quarantine_code_is_version_aware(store, settings, clock,
                                                 make_command, make_worker):
    from conftest import make_request

    v21 = _poison_terminal(store, settings, clock, make_command, make_worker,
                           make_request(schema_version="2.1"), "key-21")
    assert v21.schema_version == "2.1"
    assert v21.state == "failed"
    assert v21.result.correctness.error.code == "SUSPECTED_POISON_CANDIDATE"
    assert v21.result.correctness.error.retryable is False

    # the identical 2.0 resource keeps the frozen WORKER_LOST shape
    v20 = _poison_terminal(store, settings, clock, make_command, make_worker,
                           make_request(), "key-20")
    assert v20.schema_version == "2.0"
    assert v20.result.correctness.error.code == "WORKER_LOST"
    assert v20.result.correctness.error.retryable is True


# ---------------------------------------------------------------------------
# Erratum E-1: 2.0 HTTP response bodies validate against the published schema
# ---------------------------------------------------------------------------


async def test_v20_terminal_http_bodies_validate_against_published_schema(
    tmp_path, release_catalog_files, make_command
):
    import jsonschema
    from conftest import make_completed_result
    from test_api import (
        WorkerIdentity,
        algorithms,
        build_app,
        make_test_settings,
        new_worker_id,
        post,
        seed_worker,
        trial_headers,
    )
    from test_contract_v21 import published_v20_resource_validator

    app = build_app(tmp_path, release_catalog_files, settings=make_test_settings(tmp_path))
    seed_worker(app)
    command = make_command(
        caller=make_caller("trial-caller", "trial", ("trial:submit", "trial:read"))
    )
    app.store.create_or_replay(command)
    lease = app.store.claim_next(
        WorkerIdentity(new_worker_id(), app.settings.evaluator_build,
                       algorithms(app.settings))
    )
    app.store.commit_terminal(lease, "completed",
                              make_completed_result(app.settings))

    validator = published_v20_resource_validator()
    async with app.http as client:
        # an idempotent replay of the terminal answers 200 immediately
        replayed = await post(client)
        assert replayed.status_code == 200, replayed.text
        validator.validate(replayed.json())
        evaluation_id = replayed.json()["evaluation_id"]
        # and GET returns the identical wire shape
        got = await client.get(f"/evaluations/{evaluation_id}",
                               headers=trial_headers())
        assert got.status_code == 200
        validator.validate(got.json())
    with pytest.raises(jsonschema.ValidationError):
        validator.validate({"schema_version": "2.0", "result": {"compile": None}})
