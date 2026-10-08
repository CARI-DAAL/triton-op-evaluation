"""Deterministic schema and golden examples for wire compatibility tests.

The service imports only its models and serialization. These builders and
recorded bytes belong exclusively to the test suite.
"""
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

from pydantic.json_schema import models_json_schema

from triton_evaluation.api import contract as m
from triton_evaluation.serialization import canonical_json_bytes, sha256_hex

CONTRACT_ROOT = Path(__file__).parent / "fixtures/evaluation_api/2.1"
SCHEMA_ID = "urn:triton-evaluation:evaluation-api:2.1:schema"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def _build_schema_bytes(schema_id: str) -> bytes:
    _, top = models_json_schema(
        [
            (m.EvaluationRequest, "validation"),
            (m.EvaluationResource, "validation"),
            (m.ErrorEnvelope, "validation"),
        ],
        ref_template="#/$defs/{model}",
    )
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": schema_id,
        "oneOf": [
            {"$ref": "#/$defs/EvaluationRequest"},
            {"$ref": "#/$defs/EvaluationResource"},
            {"$ref": "#/$defs/ErrorEnvelope"},
        ],
        "$defs": {name: top["$defs"][name] for name in sorted(top["$defs"])},
    }
    return canonical_json_bytes(schema)


# ---------------------------------------------------------------------------
# Golden fixtures
# ---------------------------------------------------------------------------

_T0 = datetime(2026, 8, 15, 0, 0, 0, tzinfo=timezone.utc)

_REFERENCE_CODE = "import torch\n\n\nclass Model(torch.nn.Module):\n    pass\n"
_KERNEL_CODE = "import torch\n\n\nclass ModelNew(torch.nn.Module):\n    pass\n"


def _request(checks, **overrides):
    payload = {
        "schema_version": "2.1",
        "op_name": "addmm",
        "reference_code": _REFERENCE_CODE,
        "kernel_code": _KERNEL_CODE,
        "checks": list(checks),
    }
    payload.update(overrides)
    return m.EvaluationRequest.model_validate(payload)


def _grade_request():
    return _request(
        ["profile", "performance", "correctness"],
        purpose="grade",
        deadline_seconds=1200,
        performance={"warmup": 3, "repeats": 20},
        profile={"mode": "deep", "case_indices": [2, 0], "max_kernels_per_case": 100},
        grade_context={
            "manifest_id": "release-2026.08",
            "problem_id": "addmm-1",
            "reference_digest": sha256_hex(_REFERENCE_CODE.encode("utf-8")),
        },
    )


def _request_digest(request) -> str:
    canonical = request.model_dump(mode="json", exclude_none=False)
    return sha256_hex(canonical_json_bytes(canonical))


def _error(code, phase, retryable, message, check=None):
    return m.ErrorInfo(
        code=code, phase=phase, check=check, retryable=retryable, message=message
    )


def _ok(data):
    return {"status": "succeeded", "data": data, "error": None, "reason": None}


def _failed(error):
    return {"status": "failed", "data": None, "error": error, "reason": None}


def _timeout(error):
    return {"status": "timeout", "data": None, "error": error, "reason": None}


def _skipped(code, depends_on, message):
    return {
        "status": "skipped",
        "data": None,
        "error": None,
        "reason": {"code": code, "depends_on": depends_on, "message": message},
    }


def _correct_data():
    return m.CorrectnessResult(
        outcome="correct", passed=True, passed_cases=3, total_cases=3, failures=[]
    )


def _wrong_data():
    return m.CorrectnessResult(
        outcome="wrong",
        passed=False,
        passed_cases=2,
        total_cases=4,
        failures=[
            m.CaseFailure(
                case_idx=1,
                kind="value_mismatch",
                metrics=m.CaseMetrics(
                    max_abs_diff=2.5, matched_ratio=0.75, atol=1e-3, rtol=1e-4
                ),
                sample=[
                    m.SampleElement(idx=[0, 1], candidate=3.5, reference=1.0, diff=2.5),
                    m.SampleElement(idx=[0, 0], candidate=1.25, reference=1.0, diff=0.25),
                ],
            ),
            m.CaseFailure(
                case_idx=3,
                kind="non_finite_mismatch",
                metrics=m.CaseMetrics(matched_ratio=0.5),
                sample=[
                    m.SampleElement(idx=[2], candidate="inf", reference=1.0, diff="inf")
                ],
            ),
        ],
    )


def _domain_data(outcome, code, phase, case_idx, message, passed_cases=0,
                 total_cases=4, failures=()) -> "m.CorrectnessResult":
    return m.CorrectnessResult(
        outcome=outcome,
        passed=False,
        passed_cases=passed_cases,
        total_cases=total_cases,
        failures=list(failures),
        domain_failure=m.DomainFailure(
            code=code, phase=phase, case_idx=case_idx, message=message
        ),
    )


def _performance_data(measurement_method="npu_event_v1") -> "m.PerformanceResult":
    per_case = [
        m.PerformanceCase(
            case_idx=0,
            input_shapes=[
                m.InputShape(input_index=0, dimensions=[1024, 512], dtype="float16")
            ],
            implementation_latency_ms=0.5,
            reference_latency_ms=1.0,
            speedup_vs_reference=2.0,
            measurement_method=measurement_method,
            reference_latency_samples_ms=[0.96, 0.98, 0.99, 1.0, 1.0, 1.01, 1.02, 1.04],
            reference_latency_samples_source="fresh_measurements",
        ),
        m.PerformanceCase(
            case_idx=1,
            input_shapes=[
                m.InputShape(input_index=0, dimensions=[2048, 512], dtype="float16")
            ],
            implementation_latency_ms=2.0,
            reference_latency_ms=1.0,
            speedup_vs_reference=0.5,
            measurement_method=measurement_method,
        ),
    ]
    unmeasured = [
        m.UnmeasuredPerformanceCase(
            case_idx=2, input_shapes=[], reason_code="NO_STABLE_MEASUREMENT"
        )
    ]
    measured = len(per_case)
    selected = measured + len(unmeasured)
    overall = math.exp(
        sum(math.log(case.speedup_vs_reference) for case in per_case) / measured
    )
    return m.PerformanceResult(
        aggregation="geometric_mean_v1",
        overall_speedup_vs_reference=overall,
        selected_cases=selected,
        measured_cases=measured,
        coverage=measured / selected,
        per_case=per_case,
        unmeasured_cases=unmeasured,
        measurement_method=measurement_method,
    )


def _op_profile_data() -> "m.OpProfileResult":
    return m.OpProfileResult(
        mode="op",
        counts=m.ProfileCounts(selected=2, complete=2, partial=0, failed=0),
        cases=[
            m.OpProfileCase(
                case_idx=0,
                status="complete",
                kernels=[
                    m.OpKernel(
                        op_name="addmm_kernel",
                        time_ms=0.4,
                        percentage=80.0,
                        metrics=m.OpKernelMetrics(
                            aic_cube_ratio=0.72,
                            aic_mte2_ratio=0.31,
                            aic_fixpipe_ratio=0.12,
                            aiv_vec_ratio=0.02,
                            aiv_mte2_ratio=0.04,
                            aiv_mte3_ratio=0.01,
                            gm_to_l1_usage=0.61,
                            l0c_to_gm_usage=0.22,
                            gm_to_ub_usage=0.55,
                            ub_to_gm_usage=0.18,
                            aic_l2_hit=0.83,
                            aiv_l2_hit=0.41,
                        ),
                    ),
                    m.OpKernel(
                        op_name="bias_add",
                        time_ms=0.1,
                        percentage=20.0,
                        metrics=m.OpKernelMetrics(
                            aiv_vec_ratio=0.66,
                            aiv_mte2_ratio=0.21,
                            gm_to_ub_usage=0.48,
                            ub_to_gm_usage=0.44,
                            aiv_l2_hit=0.57,
                        ),
                    ),
                ],
            ),
            m.OpProfileCase(
                case_idx=1,
                status="complete",
                kernels=[m.OpKernel(op_name="addmm_kernel", time_ms=1.6, percentage=100.0)],
            ),
        ],
    )


def _deep_profile_partial_data() -> "m.DeepProfileResult":
    return m.DeepProfileResult(
        mode="deep",
        counts=m.ProfileCounts(selected=2, complete=1, partial=1, failed=0),
        cases=[
            m.DeepProfileCase(
                case_idx=0,
                status="complete",
                target_kernel="addmm_kernel",
                bound="CUBE",
                ratios=m.DeepRatios(
                    mte2=0.1, cube=0.75, vec=0.05, fixp=0.0, mte3=0.07, scalar=0.03
                ),
                per_core_spread_ratio=1.1,
                recommendations=["increase tile size"],
                missing_metrics=[],
                warnings=[],
            ),
            m.DeepProfileCase(
                case_idx=2,
                status="partial",
                target_kernel="addmm_kernel",
                bound=None,
                ratios=m.DeepRatios(mte2=0.2),
                per_core_spread_ratio=None,
                recommendations=[],
                missing_metrics=["cube_ratio"],
                warnings=["msprof exited before all metrics were emitted"],
            ),
        ],
    )


def _environment() -> "m.ExecutionEnvironment":
    return m.ExecutionEnvironment(
        hardware_model="Ascend910B3",
        hardware_count=1,
        driver_version="24.1.0",
        runtime_version="8.0.RC3",
        torch_version="2.1.0",
        torch_npu_version="2.1.0.post3",
        triton_version="3.0.0",
        toolchain_version="CANN 8.0.RC3",
    )


def _provenance(environment) -> "m.Provenance":
    return m.Provenance(
        schema_version="2.1",
        evaluator_build="sha256:" + "a" * 64,
        correctness_algorithm="correctness-v1",
        decoy_algorithm="decoy-v1",
        benchmark_algorithm="geometric-mean-v1",
        profiler_algorithm="profiler-v1",
        grade_eligible=False,
        manifest_id=None,
        problem_id=None,
        execution_environment=environment,
    )


_ENVIRONMENT_UNSET = object()


def _result(correctness=None, performance=None, profile=None, compile=None,
            environment=_ENVIRONMENT_UNSET) -> "m.EvaluationResult":
    if environment is _ENVIRONMENT_UNSET:
        environment = _environment()
    return m.EvaluationResult(
        correctness=correctness,
        performance=performance,
        profile=profile,
        provenance=_provenance(environment),
        compile=compile,
    )


def _resource(state, request, *, result=None, revision=1,
              started_at=None, finished_at=None,
              id_hex="0" * 32) -> "m.EvaluationResource":
    if result is not None:
        result = m.EvaluationResult.model_validate(
            result.model_dump(), context={"requested_checks": request.checks}
        )
        if request.purpose == "grade":
            result.provenance.manifest_id = request.grade_context.manifest_id
            result.provenance.problem_id = request.grade_context.problem_id
            result.provenance.grade_eligible = True
    return m.EvaluationResource(
        schema_version="2.1",
        evaluation_id="ev_" + id_hex,
        revision=revision,
        state=state,
        created_at=_T0,
        updated_at=finished_at or started_at or _T0,
        started_at=started_at,
        finished_at=finished_at,
        deadline_at=_T0 + timedelta(seconds=request.deadline_seconds),
        request_digest=_request_digest(request),
        reference_digest=sha256_hex(request.reference_code.encode("utf-8")),
        result=result,
    )


def _envelope(code, phase, retryable, message, id_hex) -> "m.ErrorEnvelope":
    return m.ErrorEnvelope(
        schema_version="2.1",
        request_id="rq_" + id_hex,
        error=_error(code, phase, retryable, message),
    )


def _error_fixtures() -> dict[str, "m.ErrorEnvelope"]:
    error_fixtures = [
        ("error-invalid-request.json", "INVALID_REQUEST", "request", False,
         "request failed schema validation"),
        ("error-unauthenticated.json", "UNAUTHENTICATED", "auth", False,
         "missing or invalid bearer credential"),
        ("error-forbidden.json", "FORBIDDEN", "auth", False,
         "caller lacks the required scope"),
        ("error-manifest-binding-failed.json", "MANIFEST_BINDING_FAILED", "admission",
         False, "reference does not match the bound manifest entry"),
        ("error-not-found.json", "NOT_FOUND", "request", False,
         "evaluation not found"),
        ("error-idempotency-conflict.json", "IDEMPOTENCY_CONFLICT", "admission", False,
         "idempotency key was used with a different request"),
        ("error-limit-exceeded.json", "LIMIT_EXCEEDED", "request", False,
         "request exceeds the configured size limit"),
        ("error-rate-limited.json", "RATE_LIMITED", "admission", True,
         "caller rate limit exceeded"),
        ("error-not-ready.json", "NOT_READY", "admission", True,
         "service is not ready to accept evaluations"),
        ("error-queue-unavailable.json", "QUEUE_UNAVAILABLE", "queue", True,
         "durable queue is temporarily unavailable"),
        ("error-worker-lost.json", "WORKER_LOST", "execution", True,
         "worker lease expired before a subject outcome was committed"),
        ("error-evaluator-execution-failed.json", "EVALUATOR_EXECUTION_FAILED",
         "execution", True, "evaluator failed before a subject outcome"),
        ("error-invalid-engine-output.json", "INVALID_ENGINE_OUTPUT", "shaping", False,
         "engine output did not match the private protocol"),
        ("error-deadline-exceeded.json", "DEADLINE_EXCEEDED", "deadline", False,
         "resource deadline expired"),
        ("error-internal.json", "INTERNAL", "storage", False,
         "unexpected internal error"),
    ]
    return {
        name: _envelope(code, phase, retryable, message, f"{index + 1:032x}")
        for index, (name, code, phase, retryable, message) in enumerate(error_fixtures)
    }


def _fixtures() -> dict[str, "m.StrictModel"]:
    correctness_only = _request(["correctness"])
    with_performance = _request(["correctness", "performance"])
    with_profile = _request(["correctness", "profile"])
    all_checks = _request(["correctness", "performance", "profile"])
    started = _T0 + timedelta(seconds=5)
    finished = _T0 + timedelta(seconds=95)
    dependency_skip = _skipped(
        "DEPENDENCY_NOT_PASSED", "correctness", "correctness did not pass"
    )

    def deadline_skip(depends_on):
        return _skipped(
            "RESOURCE_DEADLINE", depends_on,
            "resource deadline expired before execution",
        )
    fixtures: dict[str, m.StrictModel] = {
        "request-trial-minimal.json": correctness_only,
        "request-grade-full.json": _grade_request(),
        "resource-queued.json": _resource(
            "queued", correctness_only, id_hex="0" * 31 + "1"
        ),
        "resource-running.json": _resource(
            "running", correctness_only,
            revision=2, started_at=started, id_hex="0" * 31 + "2",
        ),
        "resource-completed-correct.json": _resource(
            "completed", with_performance,
            result=_result(
                correctness=_ok(_correct_data()),
                performance=_ok(_performance_data()),
            ),
            revision=3,
            started_at=started,
            finished_at=finished,
            id_hex="0" * 31 + "3",
        ),
        "resource-completed-wrong.json": _resource(
            "completed", with_performance,
            result=_result(
                correctness=_ok(_wrong_data()), performance=dependency_skip
            ),
            revision=3,
            started_at=started,
            finished_at=finished,
            id_hex="0" * 31 + "4",
        ),
        "resource-completed-decoy.json": _resource(
            "completed", correctness_only,
            result=_result(
                correctness=_ok(
                    m.CorrectnessResult(
                        outcome="wrong", passed=False, decoy=True,
                        passed_cases=4, total_cases=4, failures=[],
                    )
                )
            ),
            revision=3,
            started_at=started,
            finished_at=finished,
            id_hex="0" * 31 + "5",
        ),
        "resource-completed-compile-error.json": _resource(
            "completed", with_performance,
            result=_result(
                correctness=_ok(
                    _domain_data(
                        "compile_error", "SUBJECT_COMPILE_ERROR", "compile", None,
                        "candidate compilation failed", total_cases=8,
                    )
                ),
                performance=dependency_skip,
            ),
            revision=3,
            started_at=started,
            finished_at=finished,
            id_hex="0" * 31 + "6",
        ),
        "resource-completed-runtime-error.json": _resource(
            "completed", with_performance,
            result=_result(
                correctness=_ok(
                    _domain_data(
                        "runtime_error", "SUBJECT_RUNTIME_ERROR", "execute", 1,
                        "candidate raised during execute",
                        passed_cases=1,
                        failures=[
                            m.CaseFailure(
                                case_idx=0,
                                kind="value_mismatch",
                                metrics=m.CaseMetrics(max_abs_diff=1.5),
                                sample=[
                                    m.SampleElement(
                                        idx=[0], candidate=2.5, reference=1.0, diff=1.5
                                    )
                                ],
                            )
                        ],
                    )
                ),
                performance=dependency_skip,
            ),
            revision=3,
            started_at=started,
            finished_at=finished,
            id_hex="0" * 31 + "7",
        ),
        "resource-completed-execution-timeout.json": _resource(
            "completed", with_performance,
            result=_result(
                correctness=_ok(
                    _domain_data(
                        "execution_timeout", "SUBJECT_EXECUTION_TIMEOUT", "execute",
                        0, "candidate exceeded per-subject execution budget",
                    )
                ),
                performance=dependency_skip,
            ),
            revision=3,
            started_at=started,
            finished_at=finished,
            id_hex="0" * 31 + "8",
        ),
        "resource-completed-performance-profile-op.json": _resource(
            "completed", all_checks,
            result=_result(
                correctness=_ok(_correct_data()),
                performance=_ok(_performance_data()),
                profile=_ok(_op_profile_data()),
            ),
            revision=3,
            started_at=started,
            finished_at=finished,
            id_hex="0" * 31 + "9",
        ),
        "resource-completed-profile-deep-partial.json": _resource(
            "completed", with_profile,
            result=_result(
                correctness=_ok(_correct_data()),
                profile=_ok(_deep_profile_partial_data()),
            ),
            revision=3,
            started_at=started,
            finished_at=finished,
            id_hex="0" * 31 + "a",
        ),
        "resource-failed.json": _resource(
            "failed", correctness_only,
            result=_result(
                correctness=_failed(
                    _error(
                        "WORKER_LOST", "execution", True,
                        "worker lease expired before a subject outcome was committed",
                        check="correctness",
                    )
                )
            ),
            revision=3,
            started_at=started,
            finished_at=finished,
            id_hex="0" * 31 + "b",
        ),
        "resource-timed-out.json": _resource(
            "timed_out", all_checks,
            result=_result(
                correctness=_timeout(
                    _error(
                        "DEADLINE_EXCEEDED", "deadline", False,
                        "resource deadline expired before any attempt started",
                        check="correctness",
                    )
                ),
                performance=deadline_skip("correctness"),
                profile=deadline_skip(None),
                environment=None,
            ),
            revision=2,
            finished_at=_T0 + timedelta(seconds=900),
            id_hex="0" * 31 + "c",
        ),
    }
    fixtures.update(_error_fixtures())
    fixtures.update(_compile_rollout_fixtures())
    return fixtures


def _compile_rollout_fixtures() -> dict[str, "m.StrictModel"]:
    started = _T0 + timedelta(seconds=5)
    finished = _T0 + timedelta(seconds=95)
    fixtures: dict[str, m.StrictModel] = {
        "request-trial-rollout.json": _request(
            ["correctness", "performance"],
            purpose="rollout",
            performance={"warmup": 3, "repeats": 20, "method": "event"},
        ),
        "request-trial-compile-only.json": _request(["compile"]),
        "request-profile-only.json": _request(["profile"]),
        "request-profiler-performance.json": _request(
            ["correctness", "performance"], performance={"method": "profiler"}
        ),
        "request-all-checks.json": _request(
            ["compile", "correctness", "performance", "profile"]
        ),
        "resource-completed-profile-only.json": _resource(
            "completed", _request(["profile"]),
            result=_result(profile=_ok(_op_profile_data())),
            revision=3, started_at=started, finished_at=finished,
            id_hex="1" * 32,
        ),
        "resource-completed-profiler-performance.json": _resource(
            "completed",
            _request(["correctness", "performance"], performance={"method": "profiler"}),
            result=_result(
                correctness=_ok(_correct_data()),
                performance=_ok(_performance_data("profiler_kernel_sum_v1")),
            ),
            revision=3, started_at=started, finished_at=finished,
            id_hex="2" * 32,
        ),
        "resource-completed-compile-only.json": _resource(
            "completed", _request(["compile"]),
            result=_result(compile=_ok({"outcome": "compiled", "duration_ms": 12.5})),
            revision=3, started_at=started, finished_at=finished,
            id_hex="3" * 32,
        ),
        "resource-completed-compile-error-check.json": _resource(
            "completed", _request(["compile"]),
            result=_result(compile=_ok({
                "outcome": "compile_error", "compiler_stage": "frontend",
                "error_excerpt": "candidate compilation failed", "duration_ms": 3.5,
            })),
            revision=3, started_at=started, finished_at=finished,
            id_hex="4" * 32,
        ),
        "resource-completed-grade.json": _resource(
            "completed", _grade_request(),
            result=_result(
                correctness=_ok(_correct_data()),
                performance=_ok(_performance_data()),
                profile=_ok(_deep_profile_partial_data()),
            ),
            revision=3, started_at=started, finished_at=finished,
            id_hex="5" * 32,
        ),
        "resource-completed-compile-check.json": _resource(
            "completed",
            _request(["compile", "correctness", "performance"]),
            result=_result(
                compile=_ok({"outcome": "compiled", "duration_ms": 12.5}),
                correctness=_ok(_correct_data()),
                performance=_ok(_performance_data(measurement_method="npu_event_v1")),
            ),
            revision=3,
            started_at=started,
            finished_at=finished,
            id_hex="0" * 31 + "d",
        ),
        "resource-failed-poison.json": _resource(
            "failed",
            _request(["correctness"]),
            result=_result(
                correctness=_failed(
                    _error(
                        "SUSPECTED_POISON_CANDIDATE", "execution", False,
                        "worker lease expired before a subject outcome was committed; "
                        "repeated suspicious deaths — candidate quarantined as "
                        "suspected poison",
                        check="correctness",
                    )
                ),
            ),
            revision=3,
            started_at=started,
            finished_at=finished,
            id_hex="0" * 31 + "e",
        ),
    }
    return fixtures


# ---------------------------------------------------------------------------
# Build expected golden bytes
# ---------------------------------------------------------------------------


def build_artifacts() -> dict[PurePosixPath, bytes]:
    """Return generated schema and examples for comparison with golden bytes."""
    fixture_models = _fixtures()
    schema_version = "2.1"
    schema_id = SCHEMA_ID
    artifacts: dict[PurePosixPath, bytes] = {}
    schema_bytes = _build_schema_bytes(schema_id)
    artifacts[PurePosixPath("schema.json")] = schema_bytes
    fixture_paths: dict[str, PurePosixPath] = {}
    for name, model in fixture_models.items():
        path = PurePosixPath("fixtures") / name
        artifacts[path] = canonical_json_bytes(
            model.model_dump(mode="json", exclude_none=False)
        )
        fixture_paths[name] = path
    index_object = {
        "schema_version": schema_version,
        "fixtures": [
            {
                "name": fixture_name,
                "path": fixture_path.as_posix(),
                "sha256": sha256_hex(artifacts[fixture_path]),
            }
            for fixture_name, fixture_path in sorted(fixture_paths.items())
        ],
    }
    index_bytes = canonical_json_bytes(index_object)
    artifacts[PurePosixPath("fixtures/index.json")] = index_bytes
    artifacts[PurePosixPath("contract.json")] = canonical_json_bytes(
        {
            "evaluation_api_schema_version": schema_version,
            "evaluation_api_schema_digest": sha256_hex(schema_bytes),
            "evaluation_api_golden_fixture_digest": sha256_hex(index_bytes),
            "json_schema_path": "schema.json",
            "golden_fixture_index_path": "fixtures/index.json",
        }
    )
    return artifacts
