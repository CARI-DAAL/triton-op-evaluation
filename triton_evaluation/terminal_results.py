"""Synthesized terminal results for orchestration failures (design section 6).

When a resource dies without a worker-committed result — the deadline
watchdog firing, or the reaper giving up at the attempt limit — the store
assembles a complete terminal result from whatever the current attempt
durably committed: valid persisted earlier check outcomes are retained, the
first requested check without an outcome (the active check in every reachable
state) becomes ``timeout``/``failed``, later requested checks become
``skipped`` with ``RESOURCE_DEADLINE``/``PRIOR_CHECK_FAILED``, and unrequested
checks stay null. Provenance is pinned from admission and carries no
execution environment, because no attempt environment was ever recorded.
"""
import json
from typing import Literal

from pydantic import TypeAdapter

from triton_evaluation.api.contract import (
    CheckFailed,
    CheckSucceeded,
    CompileCheckOutcome,
    CompileOutcome,
    CorrectnessOutcome,
    ErrorInfo,
    EvaluationResult,
    PerformanceOutcome,
    ProfileOutcome,
)
from triton_evaluation.models import AlgorithmVersions

_CHECK_ORDER = ("compile", "correctness", "performance", "profile")
_OUTCOME_ADAPTERS = {
    "compile": TypeAdapter(CompileCheckOutcome),
    "correctness": TypeAdapter(CorrectnessOutcome),
    "performance": TypeAdapter(PerformanceOutcome),
    "profile": TypeAdapter(ProfileOutcome),
}
# Section 5 dependency rules: performance depends on correctness; profile,
# correctness, and compile depend on nothing.
_SKIP_DEPENDS_ON = {
    "compile": None,
    "correctness": None,
    "performance": "correctness",
    "profile": None,
}

_FAILURE_SHAPES = {
    "deadline": {
        "status": "timeout",
        "error": {"code": "DEADLINE_EXCEEDED", "phase": "deadline", "retryable": False},
        "skip_code": "RESOURCE_DEADLINE",
        "skip_message": "resource deadline expired before execution",
    },
    "worker_lost": {
        "status": "failed",
        "error": {"code": "WORKER_LOST", "phase": "execution", "retryable": True},
        "skip_code": "PRIOR_CHECK_FAILED",
        "skip_message": "a prior requested check did not complete",
    },
}


def persisted_check_outcomes(raw: bytes) -> dict[str, dict]:
    """Return only well-typed persisted partial outcomes, keyed by check."""
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    if not isinstance(data, dict):
        return {}
    valid: dict[str, dict] = {}
    for check in _CHECK_ORDER:
        outcome = data.get(check)
        if outcome is None:
            continue
        try:
            _OUTCOME_ADAPTERS[check].validate_python(outcome)
        except ValueError:
            continue  # a corrupt private record is treated as never committed
        valid[check] = outcome
    return valid


def synthesize_orchestration_result(
    *,
    requested_checks,
    persisted_outcomes: dict[str, dict],
    failure: Literal["deadline", "worker_lost"],
    failure_message: str,
    evaluator_build: str,
    algorithms: AlgorithmVersions,
    manifest_id: str | None,
    problem_id: str | None,
) -> EvaluationResult:
    """Assemble a complete terminal result around persisted partial outcomes."""
    return _synthesized_terminal(
        shape=_FAILURE_SHAPES[failure],
        requested_checks=requested_checks,
        persisted_outcomes=persisted_outcomes,
        failure_message=failure_message,
        evaluator_build=evaluator_build,
        algorithms=algorithms,
        manifest_id=manifest_id,
        problem_id=problem_id,
    )


def synthesize_poison_result(
    *,
    requested_checks,
    persisted_outcomes: dict[str, dict],
    code: str,
    retryable: bool,
    message: str,
    evaluator_build: str,
    algorithms: AlgorithmVersions,
    manifest_id: str | None,
    problem_id: str | None,
) -> EvaluationResult:
    """Quarantine terminal for repeated suspicious deaths (spec §10, v2.1).

    The same walk as ``synthesize_orchestration_result`` but the failure code
    and retryable flag are explicit. Poison quarantine surfaces
    ``SUSPECTED_POISON_CANDIDATE`` with ``retryable=False``.
    """
    return _synthesized_terminal(
        shape={
            "status": "failed",
            "error": {"code": code, "phase": "execution", "retryable": retryable},
            "skip_code": "PRIOR_CHECK_FAILED",
            "skip_message": "a prior requested check did not complete",
        },
        requested_checks=requested_checks,
        persisted_outcomes=persisted_outcomes,
        failure_message=message,
        evaluator_build=evaluator_build,
        algorithms=algorithms,
        manifest_id=manifest_id,
        problem_id=problem_id,
    )


def _synthesized_terminal(
    *,
    shape,
    requested_checks,
    persisted_outcomes: dict[str, dict],
    failure_message: str,
    evaluator_build: str,
    algorithms: AlgorithmVersions,
    manifest_id: str | None,
    problem_id: str | None,
) -> EvaluationResult:
    """Walk requested checks in execution order, retaining committed evidence."""
    requested = [check for check in _CHECK_ORDER if check in requested_checks]
    outcomes: dict[str, dict] = {}

    target = next(
        (check for check in requested if check not in persisted_outcomes), None
    )
    for check in requested:
        if check in persisted_outcomes:
            outcomes[check] = persisted_outcomes[check]
        elif check == target:
            outcomes[check] = {
                "status": shape["status"],
                "data": None,
                "error": {
                    **shape["error"],
                    "check": check,
                    "message": failure_message,
                },
                "reason": None,
            }
        else:
            outcomes[check] = {
                "status": "skipped",
                "data": None,
                "error": None,
                "reason": {
                    "code": shape["skip_code"],
                    "depends_on": _SKIP_DEPENDS_ON[check],
                    "message": shape["skip_message"],
                },
            }

    return EvaluationResult.model_validate(
        {
            "compile": outcomes.get("compile"),
            "correctness": outcomes.get("correctness"),
            "performance": outcomes.get("performance"),
            "profile": outcomes.get("profile"),
            "provenance": {
                "schema_version": "2.1",
                "evaluator_build": evaluator_build,
                "correctness_algorithm": algorithms.correctness,
                "decoy_algorithm": algorithms.decoy,
                "benchmark_algorithm": algorithms.benchmark,
                "profiler_algorithm": algorithms.profiler,
                "grade_eligible": False,
                "manifest_id": manifest_id,
                "problem_id": problem_id,
                "execution_environment": None,
            },
        },
        context={"requested_checks": requested},
    )


def synthesize_compile_error_result(
    *, requested_checks, failure_message, total_cases,
    evaluator_build, algorithms, manifest_id, problem_id,
    precheck_duration_ms: float | None = None,
) -> EvaluationResult:
    """Admission-time compile failure, byte-shaped like the engine's own
    compile_error terminal (contracts .../resource-completed-compile-error.json):
    correctness succeeds as a carrier with outcome compile_error; dependent
    checks skip DEPENDENCY_NOT_PASSED.

    A 2.1 request that asked for the compile check also gets the typed
    ``result.compile`` slot (spec §6): outcome compile_error, frontend stage,
    excerpt, and the measured admission precheck duration."""
    requested = [check for check in _CHECK_ORDER if check in requested_checks]
    correctness = {
        "status": "succeeded",
        "data": {
            "decoy": False,
            "domain_failure": {
                "case_idx": None,
                "code": "SUBJECT_COMPILE_ERROR",
                "message": failure_message,
                "phase": "compile",
            },
            "failures": [],
            "outcome": "compile_error",
            "passed": False,
            "passed_cases": 0,
            "total_cases": int(total_cases),
        },
        "error": None,
        "reason": None,
    }
    checks = {}
    for check in requested:
        if check == "correctness":
            checks[check] = correctness
        else:
            checks[check] = {
                "status": "skipped",
                "data": None,
                "error": None,
                "reason": {
                    "code": "DEPENDENCY_NOT_PASSED",
                    "depends_on": _SKIP_DEPENDS_ON[check],
                    "message": "correctness did not pass",
                },
            }
    if "compile" in requested:
        checks["compile"] = {
            "status": "succeeded",
            "data": {
                "outcome": "compile_error",
                "compiler_stage": "frontend",
                "error_excerpt": failure_message[:400],
                "duration_ms": precheck_duration_ms,
            },
            "error": None,
            "reason": None,
        }
    return EvaluationResult.model_validate(
        {
            "compile": checks.get("compile"),
            "correctness": checks.get("correctness"),
            "performance": checks.get("performance"),
            "profile": checks.get("profile"),
            "provenance": {
                "schema_version": "2.1",
                "evaluator_build": evaluator_build,
                "correctness_algorithm": algorithms.correctness,
                "decoy_algorithm": algorithms.decoy,
                "benchmark_algorithm": algorithms.benchmark,
                "profiler_algorithm": algorithms.profiler,
                "execution_environment": None,
                "grade_eligible": False,
                "manifest_id": manifest_id,
                "problem_id": problem_id,
            },
        },
        context={"requested_checks": requested},
    )


def compile_outcome_from_precheck(
    verdict: str | None, duration_ms: float | None = None,
) -> CompileCheckOutcome:
    """Only explicit syntax/import evidence may produce a positive compile check."""
    if verdict in ("precheck_passed", "compile_error"):
        return CheckSucceeded[CompileOutcome](
            status="succeeded",
            data=CompileOutcome(
                outcome="compiled" if verdict == "precheck_passed" else "compile_error",
                compiler_stage="frontend" if verdict == "compile_error" else None,
                error_excerpt=(
                    "candidate compilation failed (admission precheck)"
                    if verdict == "compile_error" else None
                ),
                duration_ms=duration_ms,
            ),
        )
    return CheckFailed(
        status="failed",
        error=ErrorInfo(
            code="COMPILE_PRECHECK_UNAVAILABLE", phase="execution", check="compile",
            retryable=True, message="syntax and import precheck produced no definitive verdict",
        ),
    )


def synthesize_compile_only_result(
    *, compile_outcome: CompileCheckOutcome, evaluator_build: str,
    algorithms: AlgorithmVersions, manifest_id: str | None, problem_id: str | None,
) -> EvaluationResult:
    """Compile-only admission has no execution environment and never needs a card."""
    return EvaluationResult.model_validate({
        "compile": compile_outcome,
        "correctness": None, "performance": None, "profile": None,
        "provenance": {
            "schema_version": "2.1", "evaluator_build": evaluator_build,
            "correctness_algorithm": algorithms.correctness,
            "decoy_algorithm": algorithms.decoy,
            "benchmark_algorithm": algorithms.benchmark,
            "profiler_algorithm": algorithms.profiler,
            "grade_eligible": False, "manifest_id": manifest_id, "problem_id": problem_id,
            "execution_environment": None,
        },
    }, context={"requested_checks": ["compile"]})
