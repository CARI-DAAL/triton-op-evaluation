"""Complete evaluations from check evidence and admission-pinned identity.

The worker, evaluator, and recovery paths cross the same interface. Completion
never invents successful execution: valid evidence survives, the first missing
requested check records the ending failure, and later missing checks skip.
Grade eligibility is derived here from complete evidence and a real execution
environment; failures retain the request's grade binding without granting trust.
"""
import json
from collections.abc import Iterable, Mapping

from pydantic import TypeAdapter

from triton_evaluation.api.contract import (
    Check,
    CheckFailed,
    CheckSkipped,
    CheckSucceeded,
    CheckTimedOut,
    CompileCheckOutcome,
    CompileOutcome,
    CorrectnessOutcome,
    ErrorInfo,
    EvaluationResult,
    ExecutionEnvironment,
    PerformanceOutcome,
    ProfileOutcome,
    Provenance,
    SkipInfo,
)
from triton_evaluation.models import PinnedProvenance

_CHECK_ORDER = ("compile", "correctness", "performance", "profile")
_OUTCOME_ADAPTERS = {
    "compile": TypeAdapter(CompileCheckOutcome),
    "correctness": TypeAdapter(CorrectnessOutcome),
    "performance": TypeAdapter(PerformanceOutcome),
    "profile": TypeAdapter(ProfileOutcome),
}


def persisted_check_outcomes(raw: bytes) -> dict[str, dict]:
    """Decode private progress records; malformed data cannot establish evidence."""
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        check: outcome.model_dump(mode="json", exclude_none=False)
        for check, outcome in _valid_outcomes(data).items()
    }


def _valid_outcomes(outcomes: Mapping[str, object]) -> dict[str, object]:
    valid = {}
    for check, adapter in _OUTCOME_ADAPTERS.items():
        value = outcomes.get(check)
        if value is None:
            continue
        try:
            outcome = adapter.validate_python(value)
        except ValueError:
            continue
        if outcome.error is not None and outcome.error.check != check:
            continue
        valid[check] = outcome
    return valid


def complete_evaluation(
    *,
    requested_checks: Iterable[Check],
    outcomes: Mapping[str, object],
    pinned: PinnedProvenance,
    execution_environment: ExecutionEnvironment | None = None,
    failure: CheckFailed | CheckTimedOut | None = None,
) -> EvaluationResult:
    """Produce one complete result, retaining only valid requested evidence.

    ``failure`` describes why execution stopped, with ``error.check`` rebound to
    the first missing check. An omitted failure means normal completion, but any
    missing evidence still fails closed. Even when all outcomes were saved before
    an orchestration failure, that ending cannot establish grade eligibility.
    """
    requested_set = set(requested_checks)
    if not requested_set or requested_set.difference(_CHECK_ORDER):
        raise ValueError("completion requires known, nonempty requested checks")
    requested = [check for check in _CHECK_ORDER if check in requested_set]
    valid = _valid_outcomes(outcomes)
    missing = [check for check in requested if check not in valid]
    complete = failure is None and not missing
    if missing and failure is None:
        failure = CheckFailed(
            status="failed",
            error=ErrorInfo(
                code="INVALID_ENGINE_OUTPUT", phase="shaping", check=None,
                retryable=False,
                message="evaluation finished without valid evidence for every requested check",
            ),
        )
    final = {check: valid[check] for check in requested if check in valid}
    for index, check in enumerate(missing):
        if index == 0:
            final[check] = failure.model_copy(update={
                "error": failure.error.model_copy(update={"check": check}),
            })
        else:
            deadline = failure.status == "timeout"
            final[check] = CheckSkipped(
                status="skipped",
                reason=SkipInfo(
                    code="RESOURCE_DEADLINE" if deadline else "PRIOR_CHECK_FAILED",
                    depends_on="correctness" if check == "performance" else None,
                    message=(
                        "resource deadline expired before execution" if deadline
                        else "a prior requested check did not complete"
                    ),
                ),
            )
    provenance = Provenance(
        schema_version="2.1",
        evaluator_build=pinned.evaluator_build,
        correctness_algorithm=pinned.algorithms.correctness,
        decoy_algorithm=pinned.algorithms.decoy,
        benchmark_algorithm=pinned.algorithms.benchmark,
        profiler_algorithm=pinned.algorithms.profiler,
        manifest_id=pinned.manifest_id,
        problem_id=pinned.problem_id,
        execution_environment=execution_environment,
        grade_eligible=(
            complete
            and pinned.purpose == "grade"
            and pinned.manifest_id is not None
            and pinned.problem_id is not None
            and _complete_environment(execution_environment)
            and _grade_evidence(final)
        ),
    )
    return EvaluationResult.model_validate(
        {**{check: final.get(check) for check in _CHECK_ORDER}, "provenance": provenance},
        context={"requested_checks": requested},
    )


def _complete_environment(environment: ExecutionEnvironment | None) -> bool:
    if environment is None or environment.hardware_count <= 0:
        return False
    return all(
        isinstance(value, str) and bool(value.strip())
        for field, value in environment.model_dump().items()
        if field != "hardware_count"
    )


def _grade_evidence(outcomes: Mapping[str, object]) -> bool:
    for check, outcome in outcomes.items():
        if outcome.status == "succeeded":
            if check == "performance":
                correctness = outcomes.get("correctness")
                if (
                    correctness is None or correctness.status != "succeeded"
                    or correctness.data.outcome != "correct"
                ):
                    return False
            continue
        # A measured subject failure is a valid grade input; performance cannot
        # run on it. Every other skip or infrastructure failure lacks evidence.
        correctness = outcomes.get("correctness")
        if not (
            check == "performance" and outcome.status == "skipped"
            and outcome.reason.code == "DEPENDENCY_NOT_PASSED"
            and outcome.reason.depends_on == "correctness"
            and correctness is not None and correctness.status == "succeeded"
            and correctness.data.outcome != "correct"
        ):
            return False
    return True


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
                    "candidate compilation failed (syntax and import precheck)"
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
