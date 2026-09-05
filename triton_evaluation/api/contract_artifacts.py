"""Deterministic evaluation-api 2.0 contract artifact builder.

The published tree under ``contracts/evaluation-api/2.0`` is the only
cross-repository contract: one Draft 2020-12 JSON schema, 29 golden fixtures,
a fixture index, and a contract digest file. Pydantic owns the generated JSON
Schema bytes, ``rfc8785.dumps()`` is the only JSON canonicalizer, and every
artifact is RFC 8785 canonical bytes with no trailing newline.
``json.dumps(sort_keys=True)`` is not RFC 8785 and is never used here.
"""
import hashlib
import math
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

import rfc8785
from pydantic.json_schema import models_json_schema

# The published 2.0 tree is byte-frozen: its bytes are generated from the
# frozen snapshot, never from the live models, so v2.1 runtime extensions
# cannot move the digest-pinned artifacts.
from ._frozen_v20 import (
    CaseFailure,
    CaseMetrics,
    CorrectnessResult,
    DeepProfileCase,
    DeepProfileResult,
    DeepRatios,
    DomainFailure,
    ErrorEnvelope,
    ErrorInfo,
    EvaluationRequest,
    EvaluationResource,
    EvaluationResult,
    ExecutionEnvironment,
    InputShape,
    OpKernel,
    OpProfileCase,
    OpProfileResult,
    PerformanceCase,
    PerformanceResult,
    ProfileCounts,
    Provenance,
    SampleElement,
    StrictModel,
    UnmeasuredPerformanceCase,
)

CONTRACT_ROOT = Path("contracts/evaluation-api/2.0")
SCHEMA_ID = "urn:triton-evaluation:evaluation-api:2.0:schema"


def canonical_json_bytes(value: object) -> bytes:
    return rfc8785.dumps(value)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def _build_schema_bytes() -> bytes:
    _, top = models_json_schema(
        [
            (EvaluationRequest, "validation"),
            (EvaluationResource, "validation"),
            (ErrorEnvelope, "validation"),
        ],
        ref_template="#/$defs/{model}",
    )
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
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

_ENVIRONMENT = ExecutionEnvironment(
    hardware_model="Ascend910B3",
    hardware_count=1,
    driver_version="24.1.0",
    runtime_version="8.0.RC3",
    torch_version="2.1.0",
    torch_npu_version="2.1.0.post3",
    triton_version="3.0.0",
    toolchain_version="CANN 8.0.RC3",
)


def _request(checks, **overrides):
    payload = {
        "schema_version": "2.0",
        "op_name": "addmm",
        "reference_code": _REFERENCE_CODE,
        "kernel_code": _KERNEL_CODE,
        "checks": list(checks),
    }
    payload.update(overrides)
    return EvaluationRequest.model_validate(payload)


def _grade_request() -> EvaluationRequest:
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


def _request_digest(request: EvaluationRequest) -> str:
    canonical = request.model_dump(mode="json", exclude_none=False)
    return sha256_hex(canonical_json_bytes(canonical))


def _error(code, phase, retryable, message, check=None) -> ErrorInfo:
    return ErrorInfo(
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


def _correct_data() -> CorrectnessResult:
    return CorrectnessResult(
        outcome="correct", passed=True, passed_cases=3, total_cases=3, failures=[]
    )


def _wrong_data() -> CorrectnessResult:
    return CorrectnessResult(
        outcome="wrong",
        passed=False,
        passed_cases=2,
        total_cases=4,
        failures=[
            CaseFailure(
                case_idx=1,
                kind="value_mismatch",
                metrics=CaseMetrics(
                    max_abs_diff=2.5, matched_ratio=0.75, atol=1e-3, rtol=1e-4
                ),
                sample=[
                    SampleElement(idx=[0, 1], candidate=3.5, reference=1.0, diff=2.5),
                    SampleElement(idx=[0, 0], candidate=1.25, reference=1.0, diff=0.25),
                ],
            ),
            CaseFailure(
                case_idx=3,
                kind="non_finite_mismatch",
                metrics=CaseMetrics(matched_ratio=0.5),
                sample=[
                    SampleElement(idx=[2], candidate="inf", reference=1.0, diff="inf")
                ],
            ),
        ],
    )


def _domain_data(outcome, code, phase, case_idx, message, passed_cases=0, total_cases=4,
                 failures=()) -> CorrectnessResult:
    return CorrectnessResult(
        outcome=outcome,
        passed=False,
        passed_cases=passed_cases,
        total_cases=total_cases,
        failures=list(failures),
        domain_failure=DomainFailure(
            code=code, phase=phase, case_idx=case_idx, message=message
        ),
    )


def _performance_data() -> PerformanceResult:
    per_case = [
        PerformanceCase(
            case_idx=0,
            input_shapes=[
                InputShape(input_index=0, dimensions=[1024, 512], dtype="float16")
            ],
            implementation_latency_ms=0.5,
            reference_latency_ms=1.0,
            speedup_vs_reference=2.0,
        ),
        PerformanceCase(
            case_idx=1,
            input_shapes=[
                InputShape(input_index=0, dimensions=[2048, 512], dtype="float16")
            ],
            implementation_latency_ms=2.0,
            reference_latency_ms=1.0,
            speedup_vs_reference=0.5,
        ),
    ]
    unmeasured = [
        UnmeasuredPerformanceCase(
            case_idx=2, input_shapes=[], reason_code="NO_STABLE_MEASUREMENT"
        )
    ]
    measured = len(per_case)
    selected = measured + len(unmeasured)
    overall = math.exp(
        sum(math.log(case.speedup_vs_reference) for case in per_case) / measured
    )
    return PerformanceResult(
        aggregation="geometric_mean_v1",
        overall_speedup_vs_reference=overall,
        selected_cases=selected,
        measured_cases=measured,
        coverage=measured / selected,
        per_case=per_case,
        unmeasured_cases=unmeasured,
    )


def _op_profile_data() -> OpProfileResult:
    return OpProfileResult(
        mode="op",
        counts=ProfileCounts(selected=2, complete=2, partial=0, failed=0),
        cases=[
            OpProfileCase(
                case_idx=0,
                status="complete",
                kernels=[
                    OpKernel(op_name="addmm_kernel", time_ms=0.4, percentage=80.0),
                    OpKernel(op_name="bias_add", time_ms=0.1, percentage=20.0),
                ],
            ),
            OpProfileCase(
                case_idx=1,
                status="complete",
                kernels=[OpKernel(op_name="addmm_kernel", time_ms=1.6, percentage=100.0)],
            ),
        ],
    )


def _deep_profile_partial_data() -> DeepProfileResult:
    return DeepProfileResult(
        mode="deep",
        counts=ProfileCounts(selected=2, complete=1, partial=1, failed=0),
        cases=[
            DeepProfileCase(
                case_idx=0,
                status="complete",
                target_kernel="addmm_kernel",
                bound="CUBE",
                ratios=DeepRatios(
                    mte2=0.1, cube=0.75, vec=0.05, fixp=0.0, mte3=0.07, scalar=0.03
                ),
                per_core_spread_ratio=1.1,
                recommendations=["increase tile size"],
                missing_metrics=[],
                warnings=[],
            ),
            DeepProfileCase(
                case_idx=2,
                status="partial",
                target_kernel="addmm_kernel",
                bound=None,
                ratios=DeepRatios(mte2=0.2),
                per_core_spread_ratio=None,
                recommendations=[],
                missing_metrics=["cube_ratio"],
                warnings=["msprof exited before all metrics were emitted"],
            ),
        ],
    )


def _provenance(environment=_ENVIRONMENT) -> Provenance:
    return Provenance(
        schema_version="2.0",
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


def _result(correctness=None, performance=None, profile=None, environment=_ENVIRONMENT):
    return EvaluationResult(
        correctness=correctness,
        performance=performance,
        profile=profile,
        provenance=_provenance(environment),
    )


def _resource(state, request, *, result=None, revision=1, started_at=None,
              finished_at=None, id_hex="0" * 32) -> EvaluationResource:
    return EvaluationResource(
        schema_version="2.0",
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


def _envelope(code, phase, retryable, message, id_hex) -> ErrorEnvelope:
    return ErrorEnvelope(
        schema_version="2.0",
        request_id="rq_" + id_hex,
        error=_error(code, phase, retryable, message),
    )


def _fixtures() -> dict[str, StrictModel]:
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
    fixtures: dict[str, StrictModel] = {
        "request-trial-minimal.json": correctness_only,
        "request-grade-full.json": _grade_request(),
        "resource-queued.json": _resource(
            "queued", correctness_only, id_hex="0" * 31 + "1"
        ),
        "resource-running.json": _resource(
            "running", correctness_only, revision=2, started_at=started,
            id_hex="0" * 31 + "2",
        ),
        "resource-completed-correct.json": _resource(
            "completed",
            with_performance,
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
            "completed",
            with_performance,
            result=_result(
                correctness=_ok(_wrong_data()), performance=dependency_skip
            ),
            revision=3,
            started_at=started,
            finished_at=finished,
            id_hex="0" * 31 + "4",
        ),
        "resource-completed-decoy.json": _resource(
            "completed",
            correctness_only,
            result=_result(
                correctness=_ok(
                    CorrectnessResult(
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
            "completed",
            with_performance,
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
            "completed",
            with_performance,
            result=_result(
                correctness=_ok(
                    _domain_data(
                        "runtime_error", "SUBJECT_RUNTIME_ERROR", "execute", 1,
                        "candidate raised during execute",
                        passed_cases=1,
                        failures=[
                            CaseFailure(
                                case_idx=0,
                                kind="value_mismatch",
                                metrics=CaseMetrics(max_abs_diff=1.5),
                                sample=[
                                    SampleElement(
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
            "completed",
            with_performance,
            result=_result(
                correctness=_ok(
                    _domain_data(
                        "execution_timeout", "SUBJECT_EXECUTION_TIMEOUT", "execute", 0,
                        "candidate exceeded per-subject execution budget",
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
            "completed",
            all_checks,
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
            "completed",
            with_profile,
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
            "failed",
            correctness_only,
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
            "timed_out",
            all_checks,
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
    for index, (name, code, phase, retryable, message) in enumerate(error_fixtures):
        fixtures[name] = _envelope(code, phase, retryable, message, f"{index + 1:032x}")
    return fixtures


# ---------------------------------------------------------------------------
# Build, write, verify
# ---------------------------------------------------------------------------


def build_artifacts() -> dict[PurePosixPath, bytes]:
    """Return every publication path and its final bytes without writing."""
    artifacts: dict[PurePosixPath, bytes] = {}
    schema_bytes = _build_schema_bytes()
    artifacts[PurePosixPath("schema.json")] = schema_bytes
    fixture_paths: dict[str, PurePosixPath] = {}
    for name, model in _fixtures().items():
        path = PurePosixPath("fixtures") / name
        artifacts[path] = canonical_json_bytes(
            model.model_dump(mode="json", exclude_none=False)
        )
        fixture_paths[name] = path
    index_object = {
        "schema_version": "2.0",
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
            "evaluation_api_schema_version": "2.0",
            "evaluation_api_schema_digest": sha256_hex(schema_bytes),
            "evaluation_api_golden_fixture_digest": sha256_hex(index_bytes),
            "json_schema_path": "schema.json",
            "golden_fixture_index_path": "fixtures/index.json",
        }
    )
    return artifacts


def write_artifacts(root: Path) -> None:
    """Atomically replace only files returned by build_artifacts()."""
    for path, data in sorted(build_artifacts().items()):
        target = root / Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            dir=target.parent, prefix=target.name + ".", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
        except BaseException:
            os.unlink(temporary)
            raise
        os.replace(temporary, target)


def verify_artifacts(root: Path) -> list[str]:
    """Return sorted missing, unexpected, or byte-different relative paths."""
    artifacts = build_artifacts()
    problems = []
    for path, data in artifacts.items():
        target = root / Path(path)
        if not target.is_file() or target.read_bytes() != data:
            problems.append(path.as_posix())
    expected = {path.as_posix() for path in artifacts}
    if root.is_dir():
        for existing in root.rglob("*"):
            if existing.is_file():
                relative = existing.relative_to(root).as_posix()
                if relative not in expected:
                    problems.append(relative)
    return sorted(problems)
