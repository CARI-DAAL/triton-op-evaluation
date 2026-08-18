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
    CorrectnessOutcome,
    EvaluationResult,
    PerformanceOutcome,
    ProfileOutcome,
)
from triton_evaluation.models import AlgorithmVersions

_CHECK_ORDER = ("correctness", "performance", "profile")
_OUTCOME_ADAPTERS = {
    "correctness": TypeAdapter(CorrectnessOutcome),
    "performance": TypeAdapter(PerformanceOutcome),
    "profile": TypeAdapter(ProfileOutcome),
}
# Section 5 dependency rules: performance depends on correctness; profile and
# correctness depend on nothing.
_SKIP_DEPENDS_ON = {"correctness": None, "performance": "correctness", "profile": None}

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
    shape = _FAILURE_SHAPES[failure]
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
            "correctness": outcomes.get("correctness"),
            "performance": outcomes.get("performance"),
            "profile": outcomes.get("profile"),
            "provenance": {
                "schema_version": "2.0",
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
