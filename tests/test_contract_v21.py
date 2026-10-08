"""Contract v2.1: complete v2.1-only resource interface."""

import json
from pathlib import Path

import pytest
from conftest import make_caller, make_completed_result, make_request

from triton_evaluation.api.contract import EvaluationRequest, EvaluationResult
from triton_evaluation.service.admission import ForbiddenError, admit
from triton_evaluation.service.instance import EvaluationInstance


def published_v21_resource_validator():
    """jsonschema validator for the published 2.1 EvaluationResource $def."""
    return _published_resource_validator("2.1")


def _published_resource_validator(version):
    import jsonschema
    from referencing import Registry, Resource

    root = Path(__file__).resolve().parent.parent / "contracts" / "evaluation-api" / version
    schema = json.loads((root / "schema.json").read_text(encoding="utf-8"))
    # Register the full URN document so nested fragment refs retain their root.
    registry = Registry().with_resource(schema["$id"], Resource.from_contents(schema))
    return jsonschema.Draft202012Validator(
        {"$ref": f"{schema['$id']}#/$defs/EvaluationResource"}, registry=registry
    )


@pytest.mark.parametrize("version, validator_factory, fixture", [
    ("2.1", published_v21_resource_validator, "resource-completed-compile-check.json"),
])
def test_published_resource_validators_resolve_nested_outcomes(
    version, validator_factory, fixture,
):
    import jsonschema

    root = Path(__file__).resolve().parent.parent / "contracts" / "evaluation-api" / version
    resource = json.loads((root / "fixtures" / fixture).read_text(encoding="utf-8"))
    validator = validator_factory()
    # Reuse the validator as HTTP tests do for POST, GET, and replay bodies.
    validator.validate(resource)
    validator.validate(resource)
    resource["result"]["correctness"]["status"] = "invalid"
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(resource)


def _request(**overrides):
    payload = {
        "schema_version": "2.1",
        "op_name": "addmm",
        "reference_code": "import torch",
        "kernel_code": "import torch",
        "checks": ["correctness"],
    }
    payload.update(overrides)
    return payload


def test_v21_accepts_compile_check_and_rollout():
    request = EvaluationRequest.model_validate(
        _request(purpose="rollout", checks=["compile", "correctness"])
    )
    assert request.checks == ["compile", "correctness"]
    assert request.purpose == "rollout"


def test_v21_allows_four_checks():
    request = EvaluationRequest.model_validate(
        _request(checks=["compile", "correctness", "performance", "profile"])
    )
    assert request.checks == ["compile", "correctness", "performance", "profile"]


def test_v21_defaults_method_and_accepts_profiler():
    request = EvaluationRequest.model_validate(_request(checks=["correctness", "performance"]))
    assert request.performance.method == "event"
    request2 = EvaluationRequest.model_validate(
        _request(checks=["correctness", "performance"], performance={"method": "profiler"})
    )
    assert request2.performance.method == "profiler"


def _result_with_compile(version="2.1"):
    return {
        "compile": {
            "status": "succeeded",
            "data": {"outcome": "compiled"},
            "error": None,
            "reason": None,
        },
        "correctness": {
            "status": "succeeded",
            "data": {
                "decoy": False,
                "domain_failure": None,
                "failures": [],
                "outcome": "correct",
                "passed": True,
                "passed_cases": 1,
                "total_cases": 1,
            },
            "error": None,
            "reason": None,
        },
        "performance": None,
        "profile": None,
        "provenance": {
            "schema_version": version,
            "evaluator_build": "git:" + "a" * 40,
            "correctness_algorithm": "ca",
            "decoy_algorithm": "da",
            "benchmark_algorithm": "ba",
            "profiler_algorithm": "pa",
            "execution_environment": None,
            "grade_eligible": False,
            "manifest_id": None,
            "problem_id": None,
        },
    }


def test_performance_measurement_method_consistency():
    from triton_evaluation.api.contract import PerformanceResult

    base = {
        "aggregation": "geometric_mean_v1",
        "overall_speedup_vs_reference": 1.5,
        "selected_cases": 1,
        "measured_cases": 1,
        "coverage": 1.0,
        "measurement_method": "npu_event_v1",
        "per_case": [
            {
                "case_idx": 0,
                "input_shapes": [{"input_index": 0, "dimensions": [8]}],
                "implementation_latency_ms": 1.0,
                "reference_latency_ms": 1.5,
                "speedup_vs_reference": 1.5,
                "measurement_method": "npu_event_v1",
            }
        ],
        "unmeasured_cases": [],
    }
    assert PerformanceResult.model_validate(base).measurement_method == "npu_event_v1"
    bad = {
        **base,
        "per_case": [{**base["per_case"][0], "measurement_method": "profiler_kernel_sum_v1"}],
    }
    with pytest.raises(ValueError):
        PerformanceResult.model_validate(bad)


def test_dump_contract_does_not_mutate_model():
    from triton_evaluation.api.contract import dump_contract

    validated = EvaluationResult.model_validate(
        _result_with_compile(),
        context={"requested_checks": ["compile", "correctness"]},
    )
    dump_contract(validated)
    assert validated.compile is not None
    assert validated.model_dump(mode="json", exclude_none=False)["compile"] is not None


# ---------------------------------------------------------------------------
# Stored wire bytes validate against the published schema
# ---------------------------------------------------------------------------


def test_terminal_resource_bytes_validate_against_published_schema(
    store, settings, make_command, make_worker,
):
    """Stored terminal resources validate through the consumer's schema."""
    cmd = make_command()
    store.create_or_replay(cmd)
    lease = store.claim_next(make_worker())
    assert store.commit_terminal(lease, "completed", make_completed_result(settings))
    raw = store._r.hget(
        f"triton-eval:v2.1:resource:{cmd.evaluation_id}", b"resource"
    )
    published_v21_resource_validator().validate(json.loads(raw))


# ---------------------------------------------------------------------------
# Task 3 erratum E-2: the queued resource echoes the canonical request version
# ---------------------------------------------------------------------------


def test_queued_resource_echoes_canonical_schema_version(store, make_command):
    cmd = make_command(
        request=make_request(schema_version="2.1", checks=["compile", "correctness"])
    )
    outcome = store.create_or_replay(cmd)
    assert outcome.decision == "created"
    assert outcome.resource.schema_version == "2.1"

# ---------------------------------------------------------------------------
# Task 3 erratum E-3: rollout is admitted like a trial (scope, no catalog bind)
# ---------------------------------------------------------------------------


def test_rollout_admission_treated_as_trial(release_catalog):
    caller = make_caller(scopes=("trial:submit", "trial:read"))
    request = make_request(schema_version="2.1", purpose="rollout")
    admission = admit(request, caller, release_catalog)
    assert admission.catalog_binding is None
    assert admission.requested_checks == ("correctness",)


def test_rollout_admission_forbidden_without_trial_submit(release_catalog):
    caller = make_caller("grader-1", "trusted", ("grade:submit", "grade:read"))
    request = make_request(schema_version="2.1", purpose="rollout")
    with pytest.raises(ForbiddenError):
        admit(request, caller, release_catalog)


def test_rollout_purpose_resolves_rollout_traffic_class(
    settings, store, release_catalog, make_worker,
):
    instance = EvaluationInstance(
        settings=settings, store=store, catalog=release_catalog, artifacts_verified=True,
    )
    store.heartbeat(make_worker())
    outcome = instance.submit(
        make_request(purpose="rollout"), make_caller(), session_id="s1", idempotency_key="k1",
    )
    assert store._r.zscore(
        f"{settings.redis_prefix}:ready:npu:bulk", outcome.resource.evaluation_id,
    ) is not None


def test_rollout_owner_reads_with_trial_scope(store, make_command):
    cmd = make_command(
        request=make_request(schema_version="2.1", purpose="rollout"),
        traffic_class="rollout",
    )
    store.create_or_replay(cmd)
    visible = store.get_visible(cmd.evaluation_id, make_caller())
    assert visible is not None
    assert visible.evaluation_id == cmd.evaluation_id


@pytest.mark.anyio
async def test_leased_compile_http_body_validates_against_published_schema(
    tmp_path, release_catalog_files,
):
    import time

    import jsonschema
    from conftest import FakeClock
    from test_api import build_app, make_test_settings, post, seed_worker, trial_headers

    from triton_evaluation.evaluators.evaluation import EvaluationEvaluator
    from triton_evaluation.runtime import verdict_cache
    from triton_evaluation.runtime.worker import Worker

    settings = make_test_settings(tmp_path)
    app = build_app(tmp_path, release_catalog_files, settings=settings,
                    clock=FakeClock(int(time.time() * 1000)))
    seed_worker(app, execution_class="cpu")
    payload = _request(checks=["compile"], kernel_code="def broken(:\n")
    digest = verdict_cache.digest_for(
        payload["kernel_code"], "triton_ascend_impl", settings.cann_fingerprint,
    )
    verdict_cache.put(app.store, digest, "compile_error", settings.verdict_cache_ttl_seconds)
    worker = Worker(
        app.store, EvaluationEvaluator(settings.engine_dir, settings=settings),
        device_id=None, settings=settings, clock=app.clock,
    )
    validator = published_v21_resource_validator()
    async with app.http as client:
        response = await post(client, payload)
        assert response.status_code == 202, response.text
        validator.validate(response.json())
        assert worker.run_once()
        terminal = await client.get(
            f"/evaluations/{response.json()['evaluation_id']}", headers=trial_headers(),
        )
    assert terminal.status_code == 200, terminal.text
    body = terminal.json()
    assert body["state"] == "completed"
    assert body["result"]["compile"]["data"]["outcome"] == "compile_error"
    validator.validate(body)
    with pytest.raises(jsonschema.ValidationError):
        validator.validate({"schema_version": "2.1", "result": {"compile": None}})


@pytest.mark.parametrize("version", ["1.0", "2.0", "2.2"])
def test_only_current_schema_is_accepted(version):
    with pytest.raises(ValueError, match="2.1"):
        EvaluationRequest.model_validate(_request(schema_version=version))


@pytest.mark.anyio
async def test_removed_version_is_rejected_before_admission(tmp_path, release_catalog_files):
    from test_api import build_app, seed_worker, trial_headers

    app = build_app(tmp_path, release_catalog_files)
    seed_worker(app)
    async with app.http as client:
        rejected = await client.post(
            "/evaluations", json=_request(schema_version="2.0"), headers=trial_headers()
        )
        accepted = await client.post("/evaluations", json=_request(), headers=trial_headers())
    assert rejected.status_code == 422
    assert rejected.json()["schema_version"] == "2.1"
    assert rejected.json()["error"]["code"] == "INVALID_REQUEST"
    assert accepted.status_code == 202
