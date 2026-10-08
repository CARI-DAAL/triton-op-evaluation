"""Evidence completion shared by the worker, evaluator, and recovery paths."""
import json
from pathlib import Path

import pytest

from triton_evaluation.api.contract import (
    CheckFailed,
    CheckSkipped,
    CheckTimedOut,
    ErrorInfo,
    EvaluationResult,
    SkipInfo,
)
from triton_evaluation.service.models import AlgorithmVersions, PinnedProvenance
from triton_evaluation.service.terminal_results import (
    complete_evaluation,
    persisted_check_outcomes,
)

FIXTURES = Path(__file__).parents[1] / "contracts/evaluation-api/2.1/fixtures"


def result_fixture(name="correct"):
    resource = json.loads((FIXTURES / f"resource-completed-{name}.json").read_text())
    return EvaluationResult.model_validate(resource["result"])


@pytest.fixture
def pinned():
    provenance = result_fixture().provenance
    return PinnedProvenance(
        evaluator_build=provenance.evaluator_build,
        algorithms=AlgorithmVersions(
            correctness=provenance.correctness_algorithm,
            decoy=provenance.decoy_algorithm,
            benchmark=provenance.benchmark_algorithm,
            profiler=provenance.profiler_algorithm,
        ),
        purpose="grade",
        manifest_id="release-1",
        problem_id="addmm-1",
    )


@pytest.mark.parametrize("deadline", [False, True])
def test_lost_attempt_keeps_evidence_and_grade_identity_without_inventing_progress(
    pinned, deadline,
):
    evidence = result_fixture().correctness
    failed = CheckTimedOut if deadline else CheckFailed
    failure = failed(
        status="timeout" if deadline else "failed",
        error=ErrorInfo(
            code="DEADLINE_EXCEEDED" if deadline else "WORKER_LOST",
            phase="deadline" if deadline else "execution",
            check=None,
            retryable=not deadline,
            message="evaluation attempt stopped",
        ),
    )
    result = complete_evaluation(
        requested_checks=["correctness", "performance", "profile"],
        outcomes={"correctness": evidence}, pinned=pinned, failure=failure,
    )
    assert result.correctness == evidence
    assert result.performance.status == failure.status
    assert result.performance.error.check == "performance"
    assert result.profile.reason.code == (
        "RESOURCE_DEADLINE" if deadline else "PRIOR_CHECK_FAILED"
    )
    assert result.compile is None
    assert result.provenance.manifest_id == pinned.manifest_id
    assert result.provenance.problem_id == pinned.problem_id
    assert result.provenance.grade_eligible is False


def test_corrupt_evidence_and_missing_normal_completion_fail_closed(pinned):
    data = {"correctness": {"status": "succeeded", "data": {"passed": True}}}
    assert persisted_check_outcomes(json.dumps(data).encode()) == {}
    result = complete_evaluation(
        requested_checks=["correctness", "performance"], outcomes=data,
        pinned=pinned, execution_environment=result_fixture().provenance.execution_environment,
    )
    assert result.correctness.status == "failed"
    assert result.correctness.error.code == "INVALID_ENGINE_OUTPUT"
    assert result.performance.status == "skipped"
    assert result.provenance.grade_eligible is False


def test_unrequested_evidence_cannot_leak_into_result(pinned):
    result = complete_evaluation(
        requested_checks=["compile"],
        outcomes={"correctness": result_fixture().correctness}, pinned=pinned,
    )
    assert result.correctness is None
    assert result.compile.status == "failed"


@pytest.mark.parametrize("subject", ["correct", "wrong", "compile-error"])
def test_complete_subject_verdicts_are_grade_inputs_even_when_subject_did_not_pass(
    pinned, subject,
):
    source = result_fixture(subject)
    result = complete_evaluation(
        requested_checks=["correctness", "performance"],
        outcomes={"correctness": source.correctness, "performance": source.performance},
        pinned=pinned, execution_environment=source.provenance.execution_environment,
    )
    assert result.provenance.grade_eligible is True
    assert result.correctness == source.correctness


@pytest.mark.parametrize("reason", ["PRIOR_CHECK_FAILED", "RESOURCE_DEADLINE"])
def test_skipped_checks_do_not_establish_grade_evidence(pinned, reason):
    skipped = CheckSkipped(
        status="skipped", reason=SkipInfo(code=reason, message="execution did not finish"),
    )
    result = complete_evaluation(
        requested_checks=["correctness"], outcomes={"correctness": skipped},
        pinned=pinned, execution_environment=result_fixture().provenance.execution_environment,
    )
    assert result.correctness == skipped
    assert result.provenance.grade_eligible is False


@pytest.mark.parametrize("patch", [None, {"torch_npu_version": None}, {"hardware_count": 0}])
def test_grade_requires_complete_actual_execution_environment(pinned, patch):
    source = result_fixture()
    environment = (
        source.provenance.execution_environment.model_copy(update=patch)
        if patch is not None else None
    )
    result = complete_evaluation(
        requested_checks=["correctness"], outcomes={"correctness": source.correctness},
        pinned=pinned, execution_environment=environment,
    )
    assert result.correctness == source.correctness
    assert result.provenance.execution_environment == environment
    assert result.provenance.grade_eligible is False


def test_protocol_failure_after_all_check_saves_keeps_evidence_without_granting_grade(pinned):
    source = result_fixture()
    result = complete_evaluation(
        requested_checks=["correctness"], outcomes={"correctness": source.correctness},
        pinned=pinned, execution_environment=source.provenance.execution_environment,
        failure=CheckFailed(
            status="failed", error=ErrorInfo(
                code="INVALID_ENGINE_OUTPUT", phase="shaping", check=None,
                retryable=False, message="evaluator result did not match the private protocol",
            ),
        ),
    )
    assert result.correctness == source.correctness
    assert result.provenance.manifest_id == pinned.manifest_id
    assert result.provenance.grade_eligible is False


def test_error_for_different_check_cannot_be_reused_as_evidence(pinned):
    misattributed = CheckFailed(
        status="failed", error=ErrorInfo(
            code="WORKER_LOST", phase="execution", check="profile",
            retryable=True, message="worker did not complete",
        ),
    )
    result = complete_evaluation(
        requested_checks=["correctness"], outcomes={"correctness": misattributed},
        pinned=pinned,
    )
    assert result.correctness.error.code == "INVALID_ENGINE_OUTPUT"
    assert result.correctness.error.check == "correctness"
