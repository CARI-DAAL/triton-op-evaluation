"""Contract v2.1: additive extensions, per-version purity, serialization stripping."""

import json
from types import SimpleNamespace

import pytest
from conftest import make_caller, make_completed_result, make_request

from triton_evaluation.admission import ForbiddenError, admit
from triton_evaluation.api import _frozen_v20, routes
from triton_evaluation.api.contract import EvaluationRequest, EvaluationResult
from triton_evaluation.catalog import ReleaseCatalog


def published_v20_resource_validator():
    """jsonschema validator for the published 2.0 EvaluationResource $def.

    Uses the pre-``referencing`` jsonschema API (``RefResolver``): the portable
    probe's ``referencing``-based approach needs a dependency that torchless
    CI hosts do not carry.
    """
    from pathlib import Path

    import jsonschema

    root = Path(__file__).resolve().parent.parent / "contracts" / "evaluation-api" / "2.0"
    schema = json.loads((root / "schema.json").read_text(encoding="utf-8"))
    resolver = jsonschema.RefResolver.from_schema(schema)
    return jsonschema.Draft202012Validator(
        schema["$defs"]["EvaluationResource"], resolver=resolver
    )


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


def test_v20_rejects_compile_check():
    with pytest.raises(ValueError, match="2.0"):
        EvaluationRequest.model_validate(_request(schema_version="2.0", checks=["compile"]))


def test_v20_rejects_rollout_purpose():
    with pytest.raises(ValueError, match="2.0"):
        EvaluationRequest.model_validate(_request(schema_version="2.0", purpose="rollout"))


def test_v20_rejects_performance_method():
    with pytest.raises(ValueError, match="2.0"):
        EvaluationRequest.model_validate(
            _request(
                schema_version="2.0",
                checks=["correctness", "performance"],
                performance={"method": "event"},
            )
        )


def test_v21_allows_four_checks():
    request = EvaluationRequest.model_validate(
        _request(checks=["compile", "correctness", "performance", "profile"])
    )
    assert request.checks == ["compile", "correctness", "performance", "profile"]


def test_v20_rejects_four_checks():
    # A 2.0 request with four checks always includes "compile", so either the
    # length rule or the compile rule fires; both are 2.0 rejections.
    with pytest.raises(ValueError, match="2.0"):
        EvaluationRequest.model_validate(
            _request(
                schema_version="2.0",
                checks=["correctness", "performance", "profile", "compile"],
            )
        )


def test_v21_defaults_method_and_accepts_profiler():
    request = EvaluationRequest.model_validate(_request(checks=["correctness", "performance"]))
    assert request.performance.method is None  # effective default resolved later
    request2 = EvaluationRequest.model_validate(
        _request(checks=["correctness", "performance"], performance={"method": "profiler"})
    )
    assert request2.performance.method == "profiler"


def _result_with_compile(version="2.0"):
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


def test_result_accepts_compile_slot_and_strips_for_v20():
    from triton_evaluation.api.contract import dump_for_version

    validated = EvaluationResult.model_validate(
        _result_with_compile(),
        context={"requested_checks": ["compile", "correctness"]},
    )
    raw = validated.model_dump(mode="json", exclude_none=False)
    assert raw["compile"] is not None
    stripped = dump_for_version(validated, "2.0")
    import json

    assert json.loads(stripped).get("compile") is None
    # the stripped 2.0 dump must still validate against the FROZEN 2.0 models
    _frozen_v20.EvaluationResult.model_validate(
        json.loads(stripped), context={"requested_checks": ["correctness"]}
    )


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


def test_dump_for_version_does_not_mutate_model():
    from triton_evaluation.api.contract import dump_for_version

    validated = EvaluationResult.model_validate(
        _result_with_compile(),
        context={"requested_checks": ["compile", "correctness"]},
    )
    dump_for_version(validated, "2.0")
    assert validated.compile is not None
    assert validated.model_dump(mode="json", exclude_none=False)["compile"] is not None


def test_strip_v21_fields_handles_resource_dump():
    from triton_evaluation.api.contract import strip_v21_fields

    dumped = {
        "schema_version": "2.1",
        "result": {
            "compile": {
                "status": "succeeded",
                "data": {"outcome": "compiled"},
                "error": None,
                "reason": None,
            },
            "performance": {
                "measurement_method": "npu_event_v1",
                "per_case": [{"measurement_method": "npu_event_v1"}],
            },
            "provenance": {"schema_version": "2.0"},
        },
    }
    strip_v21_fields(dumped)
    assert "compile" not in dumped["result"]
    assert "measurement_method" not in dumped["result"]["performance"]
    assert "measurement_method" not in dumped["result"]["performance"]["per_case"][0]


# ---------------------------------------------------------------------------
# Task 3 erratum E-1: 2.0 wire bytes must validate against the published schema
# ---------------------------------------------------------------------------


def test_v20_terminal_resource_bytes_validate_against_published_schema(
    store, settings, make_command, make_worker,
):
    """A 2.0 terminal resource's stored bytes are the same JSON a consumer
    sees on GET/POST-replay; they must carry no v2.1-only keys."""
    cmd = make_command()
    store.create_or_replay(cmd)
    lease = store.claim_next(make_worker())
    assert store.commit_terminal(lease, "completed", make_completed_result(settings))
    raw = store._r.hget(
        f"triton-eval:v2:resource:{cmd.evaluation_id}", b"resource"
    )
    published_v20_resource_validator().validate(json.loads(raw))


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

    cmd20 = make_command(idempotency_key="key-20")
    outcome20 = store.create_or_replay(cmd20)
    assert outcome20.resource.schema_version == "2.0"


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


def test_rollout_purpose_resolves_rollout_traffic_class(settings):
    context = SimpleNamespace(
        settings=settings.model_copy(update={"rollout_caller_ids": []})
    )
    caller = make_caller()
    request = make_request(schema_version="2.1", purpose="rollout")
    admission = admit(request, caller, ReleaseCatalog({}))
    command = routes._build_command(context, caller, "s1", "k1", admission, "rollout")
    assert command.traffic_class == "rollout"


def test_rollout_owner_reads_with_trial_scope(store, make_command):
    cmd = make_command(
        request=make_request(schema_version="2.1", purpose="rollout"),
        traffic_class="rollout",
    )
    store.create_or_replay(cmd)
    visible = store.get_visible(cmd.evaluation_id, make_caller())
    assert visible is not None
    assert visible.evaluation_id == cmd.evaluation_id
