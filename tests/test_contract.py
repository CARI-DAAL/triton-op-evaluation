import pytest
from pydantic import ValidationError

from triton_evaluation.api.contract import (
    CONTRACT_REVISION,
    CaseFailure,
    CaseMetrics,
    CorrectnessResult,
    EvaluationPayload,
    EvaluationRequest,
    EvaluationResult,
    PerformanceResult,
    ProfileResult,
    SampleElement,
)


def test_evaluation_request_correctness_only_has_clean_schema():
    r = EvaluationRequest(
        op_name="addmm", reference_code="R", kernel_code="K", checks=["correctness"]
    )
    assert r.checks == ["correctness"]
    assert r.performance is None
    assert r.profile is None
    assert r.timeout == 900
    # correctness-only request must not carry performance/profile params
    dumped = r.model_dump(exclude_none=True)
    assert "warmup" not in dumped and "repeats" not in dumped


def test_evaluation_request_rejects_empty_checks():
    with pytest.raises(ValidationError):
        EvaluationRequest(op_name="addmm", reference_code="R", kernel_code="K", checks=[])


def test_evaluation_request_rejects_unknown_check():
    with pytest.raises(ValidationError):
        EvaluationRequest(
            op_name="addmm", reference_code="R", kernel_code="K", checks=["speed"]
        )


def test_performance_params_attached_only_when_requested():
    r = EvaluationRequest(
        op_name="addmm",
        reference_code="R",
        kernel_code="K",
        checks=["correctness", "performance"],
        performance={"warmup": 2, "repeats": 20},
    )
    assert r.performance.warmup == 2
    assert r.performance.repeats == 20


def test_evaluation_result_roundtrip_success_with_payload():
    res = EvaluationResult(
        task_id="t1",
        status="SUCCEEDED",
        revision=CONTRACT_REVISION,
        error=None,
        result=EvaluationPayload(
            correctness=CorrectnessResult(
                passed=True,
                compiled=True,
                decoy=False,
                passed_cases=3,
                total_cases=3,
                failures=[],
            ),
            performance=PerformanceResult(speedup_vs_torch=2.5, per_shape=[{"case_idx": 1}]),
            profile=None,
        ),
    )
    back = EvaluationResult.model_validate(res.model_dump())
    assert back.status == "SUCCEEDED"
    assert back.result.correctness.passed is True
    assert back.result.performance.speedup_vs_torch == 2.5
    assert back.result.profile is None


def test_case_failure_with_non_finite_sample_sentinels():
    f = CaseFailure(
        case_idx=2,
        kind="sign_flip",
        metrics=CaseMetrics(
            max_abs_diff=1.0, mean_abs_err=0.5, matched_ratio=0.1, mere=0.4,
            small_count=0, normal_count=10, atol=1e-3, rtol=1.2e-4,
        ),
        sample=[
            SampleElement(idx=[0, 3], candidate="inf", reference=1.0, diff="inf"),
            SampleElement(idx=[1, 1], candidate=2.0, reference=-2.0, diff=4.0),
            SampleElement(idx=[4, 2], candidate="nan", reference="nan", diff="nan"),
        ],
        error="sign flipped",
    )
    dumped = f.model_dump_json()
    back = CaseFailure.model_validate_json(dumped)
    assert back.sample[0].candidate == "inf"
    assert back.sample[2].diff == "nan"
    assert back.sample[1].candidate == 2.0


def test_status_accepts_in_progress_states():
    for s in ("QUEUED", "RUNNING", "SUCCEEDED", "FAILED", "TIMEOUT", "REJECTED"):
        EvaluationResult(
            task_id="t", status=s, revision=CONTRACT_REVISION, result=EvaluationPayload()
        )
