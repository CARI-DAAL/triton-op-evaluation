"""Schema 2.1 invariant matrices for resources, outcomes, and domain models.

Every rule here traces to design section 4: resource lifecycle shape, the four
check-outcome discriminants, the five correctness domain outcomes, domain
failure code/phase/case rules, sample ordering and sentinel encoding,
performance aggregation arithmetic, and profile count/ratio rules.
"""
import math
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from triton_evaluation.api.contract import (
    CaseFailure,
    CaseMetrics,
    CorrectnessResult,
    DeepProfileCase,
    DeepProfileResult,
    DeepRatios,
    EvaluationResource,
    EvaluationResult,
    ExecutionEnvironment,
    InputShape,
    OpKernel,
    OpProfileCase,
    OpProfileResult,
    PerformanceResult,
    ProfileCounts,
    Provenance,
    SampleElement,
    SkipInfo,
    UnmeasuredPerformanceCase,
)

T0 = datetime(2026, 8, 15, 0, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 8, 15, 0, 0, 1, tzinfo=timezone.utc)
T2 = datetime(2026, 8, 15, 0, 0, 2, tzinfo=timezone.utc)
T3 = datetime(2026, 8, 15, 0, 15, 0, tzinfo=timezone.utc)
T_EARLIER = datetime(2026, 8, 14, 0, 0, 0, tzinfo=timezone.utc)
DIGEST = "b" * 64


def error_info(**overrides):
    base = {
        "code": "WORKER_LOST",
        "phase": "execution",
        "check": "correctness",
        "retryable": True,
        "message": "worker lease expired before a subject outcome was committed",
    }
    base.update(overrides)
    return base


def sample(idx, candidate, reference, diff):
    return {"idx": idx, "candidate": candidate, "reference": reference, "diff": diff}


def value_failure(case_idx=0, diffs=(4.0, 1.0)):
    return {
        "case_idx": case_idx,
        "kind": "value_mismatch",
        "metrics": {"max_abs_diff": max(diffs)},
        "sample": [
            sample([i], d, 0.0, d) for i, d in enumerate(diffs)
        ],
    }


def correctness_correct(total=3):
    return {
        "outcome": "correct",
        "passed": True,
        "decoy": False,
        "passed_cases": total,
        "total_cases": total,
        "failures": [],
        "domain_failure": None,
    }


def correctness_wrong(total=3):
    return {
        "outcome": "wrong",
        "passed": False,
        "decoy": False,
        "passed_cases": total - 1,
        "total_cases": total,
        "failures": [value_failure(case_idx=total - 1)],
        "domain_failure": None,
    }


def domain_failure(code="SUBJECT_COMPILE_ERROR", phase="compile", case_idx=None):
    return {
        "code": code,
        "phase": phase,
        "case_idx": case_idx,
        "message": "candidate compilation failed",
    }


def performance_payload(speedups=(2.0, 0.5), unmeasured=1):
    per_case = [
        {
            "case_idx": i,
            "input_shapes": [
                {"input_index": 0, "dimensions": [16, 16], "dtype": "float16"}
            ],
            "implementation_latency_ms": 1.0 / s,
            "reference_latency_ms": 1.0,
            "speedup_vs_reference": s,
        }
        for i, s in enumerate(speedups)
    ]
    unmeasured_cases = [
        {
            "case_idx": len(speedups) + j,
            "input_shapes": [],
            "reason_code": "NO_STABLE_MEASUREMENT",
        }
        for j in range(unmeasured)
    ]
    measured = len(per_case)
    selected = measured + len(unmeasured_cases)
    overall = math.exp(sum(math.log(s) for s in speedups) / measured)
    return {
        "aggregation": "geometric_mean_v1",
        "overall_speedup_vs_reference": overall,
        "selected_cases": selected,
        "measured_cases": measured,
        "coverage": measured / selected,
        "per_case": per_case,
        "unmeasured_cases": unmeasured_cases,
    }


def op_profile_payload():
    return {
        "mode": "op",
        "counts": {"selected": 2, "complete": 1, "partial": 0, "failed": 1},
        "cases": [
            {
                "case_idx": 0,
                "status": "complete",
                "kernels": [
                    {"op_name": "add_kernel", "time_ms": 0.3, "percentage": 60.0},
                    {"op_name": "mul_kernel", "time_ms": 0.2, "percentage": 40.0},
                ],
                "error": None,
            },
            {
                "case_idx": 1,
                "status": "failed",
                "kernels": [],
                "error": error_info(code="EVALUATOR_EXECUTION_FAILED", message="profiling failed"),
            },
        ],
    }


def deep_profile_payload():
    return {
        "mode": "deep",
        "counts": {"selected": 2, "complete": 1, "partial": 1, "failed": 0},
        "cases": [
            {
                "case_idx": 0,
                "status": "complete",
                "target_kernel": "add_kernel",
                "bound": "CUBE",
                "ratios": {"mte2": 0.1, "cube": 0.8, "vec": 0.05,
                           "fixp": 0.0, "mte3": 0.03, "scalar": 0.02},
                "per_core_spread_ratio": 1.2,
                "recommendations": ["increase tile size"],
                "missing_metrics": [],
                "warnings": [],
                "error": None,
            },
            {
                "case_idx": 1,
                "status": "partial",
                "target_kernel": "add_kernel",
                "bound": None,
                "ratios": {"mte2": None, "cube": None, "vec": None,
                           "fixp": None, "mte3": None, "scalar": None},
                "per_core_spread_ratio": None,
                "recommendations": [],
                "missing_metrics": ["cube_ratio"],
                "warnings": ["msprof exited early"],
                "error": None,
            },
        ],
    }


def provenance(**overrides):
    base = {
        "schema_version": "2.1",
        "evaluator_build": "sha256:" + "c" * 64,
        "correctness_algorithm": "correctness-v1",
        "decoy_algorithm": "decoy-v1",
        "benchmark_algorithm": "benchmark-v1",
        "profiler_algorithm": "profiler-v1",
        "grade_eligible": False,
        "manifest_id": None,
        "problem_id": None,
        "execution_environment": {
            "hardware_model": "Ascend910B",
            "hardware_count": 1,
            "driver_version": "24.1.0",
            "runtime_version": "8.0.0",
            "torch_version": "2.1.0",
            "torch_npu_version": "2.1.0.post3",
            "triton_version": "3.0.0",
            "toolchain_version": "CANN-8.0",
        },
    }
    base.update(overrides)
    return base


_UNSET = object()


def result_payload(correctness=_UNSET, performance=None, profile=None):
    if correctness is _UNSET:
        correctness = {"status": "succeeded", "data": correctness_correct(),
                       "error": None, "reason": None}
    return {
        "correctness": correctness,
        "performance": performance,
        "profile": profile,
        "provenance": provenance(),
    }


def resource_payload(state="completed", **overrides):
    base = {
        "schema_version": "2.1",
        "evaluation_id": "ev_" + "0" * 32,
        "revision": 1,
        "state": state,
        "created_at": T0,
        "updated_at": T1,
        "started_at": None,
        "finished_at": None,
        "deadline_at": T3,
        "request_digest": DIGEST,
        "reference_digest": DIGEST,
        "result": None,
    }
    if state == "running":
        base["started_at"] = T1
    elif state in ("completed", "failed", "timed_out"):
        base.update(
            revision=2,
            started_at=T1,
            finished_at=T2,
            result=result_payload(),
        )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Resource lifecycle shape across all five states
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state", ["queued", "running", "completed", "failed", "timed_out"])
def test_resource_accepts_each_well_formed_state(state):
    resource = EvaluationResource.model_validate(resource_payload(state))
    assert resource.state == state


def test_queued_deadline_timeout_may_lack_started_at():
    payload = resource_payload("timed_out", started_at=None)
    resource = EvaluationResource.model_validate(payload)
    assert resource.started_at is None
    assert resource.result is not None


@pytest.mark.parametrize(
    "state,patch",
    [
        ("queued", {"result": result_payload()}),
        ("queued", {"finished_at": T2}),
        ("queued", {"started_at": T1}),
        ("running", {"result": result_payload()}),
        ("running", {"finished_at": T2}),
        ("running", {"started_at": None}),
        ("completed", {"result": None}),
        ("completed", {"finished_at": None}),
        ("completed", {"started_at": None}),
        ("failed", {"result": None}),
        ("failed", {"started_at": None}),
        ("timed_out", {"result": None}),
        ("timed_out", {"finished_at": None}),
    ],
)
def test_resource_rejects_inconsistent_lifecycle_shape(state, patch):
    with pytest.raises(ValidationError):
        EvaluationResource.model_validate(resource_payload(state, **patch))


@pytest.mark.parametrize(
    "patch",
    [
        {"state": "SUCCEEDED"},
        {"state": "pending"},
        {"revision": 0},
        {"request_digest": "not-a-digest"},
        {"request_digest": "B" * 64},
        {"updated_at": T_EARLIER},  # before created_at
        {"deadline_at": T0},  # deadline must be after creation
        {"finished_at": T0},  # before started_at
        {"extra_field": 1},
    ],
)
def test_resource_rejects_invalid_scalars(patch):
    patch = dict(patch)
    state = patch.pop("state", "completed")
    with pytest.raises(ValidationError):
        EvaluationResource.model_validate(resource_payload(state, **patch))


# ---------------------------------------------------------------------------
# Requested versus unrequested check outcomes (validation context)
# ---------------------------------------------------------------------------


def test_result_requires_outcomes_exactly_for_requested_checks():
    payload = result_payload(
        performance={"status": "succeeded", "data": performance_payload(),
                     "error": None, "reason": None},
    )
    with pytest.raises(ValidationError):
        EvaluationResult.model_validate(
            payload, context={"requested_checks": ["correctness"]}
        )
    result = EvaluationResult.model_validate(
        payload, context={"requested_checks": ["correctness", "performance"]}
    )
    assert result.performance is not None
    with pytest.raises(ValidationError):
        EvaluationResult.model_validate(
            result_payload(correctness=None),
            context={"requested_checks": ["correctness"]},
        )


def test_result_allows_any_fill_without_requested_check_context():
    result = EvaluationResult.model_validate(result_payload())
    assert result.correctness is not None
    assert result.performance is None


# ---------------------------------------------------------------------------
# Four check outcome discriminants
# ---------------------------------------------------------------------------


def outcome(status, **overrides):
    base = {"status": status, "data": None, "error": None, "reason": None}
    base.update(overrides)
    return base


def test_succeeded_requires_data_and_no_error_or_reason():
    result = EvaluationResult.model_validate(
        result_payload(correctness=outcome("succeeded", data=correctness_correct()))
    )
    assert result.correctness.status == "succeeded"
    for bad in (
        outcome("succeeded"),
        outcome("succeeded", data=correctness_correct(), error=error_info()),
        outcome("succeeded", data=correctness_correct(),
                reason={"code": "RESOURCE_DEADLINE", "message": "deadline"}),
    ):
        with pytest.raises(ValidationError):
            EvaluationResult.model_validate(result_payload(correctness=bad))


def test_failed_and_timeout_require_error_and_no_data_or_reason():
    for status in ("failed", "timeout"):
        result = EvaluationResult.model_validate(
            result_payload(correctness=outcome(status, error=error_info()))
        )
        assert result.correctness.status == status
        for bad in (
            outcome(status),
            outcome(status, error=error_info(), data=correctness_correct()),
            outcome(status, error=error_info(),
                    reason={"code": "PRIOR_CHECK_FAILED", "message": "prior"}),
        ):
            with pytest.raises(ValidationError):
                EvaluationResult.model_validate(result_payload(correctness=bad))


def test_skipped_requires_reason_and_no_data_or_error():
    skip = {"code": "DEPENDENCY_NOT_PASSED", "depends_on": "correctness",
            "message": "correctness did not pass"}
    result = EvaluationResult.model_validate(
        result_payload(performance=outcome("skipped", reason=skip))
    )
    assert result.performance.reason.code == "DEPENDENCY_NOT_PASSED"
    for bad in (
        outcome("skipped"),
        outcome("skipped", reason=skip, data=performance_payload()),
        outcome("skipped", reason=skip, error=error_info()),
    ):
        with pytest.raises(ValidationError):
            EvaluationResult.model_validate(result_payload(performance=bad))


def test_unknown_outcome_status_is_rejected():
    with pytest.raises(ValidationError):
        EvaluationResult.model_validate(
            result_payload(correctness=outcome("cancelled"))
        )


@pytest.mark.parametrize(
    "patch",
    [
        {"code": "UNKNOWN_SKIP_CODE"},
        {"message": "x" * 257},
        {"message": "two\nlines"},
        {"depends_on": "speed"},
    ],
)
def test_skip_info_rejects_unknown_codes_and_long_messages(patch):
    skip = {"code": "RESOURCE_DEADLINE", "depends_on": None, "message": "deadline"}
    with pytest.raises(ValidationError):
        SkipInfo.model_validate(skip | patch)


# ---------------------------------------------------------------------------
# Five correctness domain outcomes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        correctness_correct(),
        correctness_wrong(),
        # decoy: all cases pass, outcome is wrong, no mismatch failures
        correctness_wrong() | {"decoy": True, "passed_cases": 3, "failures": []},
        {
            "outcome": "compile_error",
            "passed": False,
            "decoy": False,
            "passed_cases": 0,
            "total_cases": 8,
            "failures": [],
            "domain_failure": domain_failure(),
        },
        {
            "outcome": "runtime_error",
            "passed": False,
            "decoy": False,
            "passed_cases": 2,
            "total_cases": 8,
            "failures": [],
            "domain_failure": domain_failure("SUBJECT_RUNTIME_ERROR", "execute", 2),
        },
        {
            "outcome": "execution_timeout",
            "passed": False,
            "decoy": False,
            "passed_cases": 0,
            "total_cases": 8,
            "failures": [],
            "domain_failure": domain_failure(
                "SUBJECT_EXECUTION_TIMEOUT", "execute", 0
            ),
        },
    ],
)
def test_correctness_accepts_each_valid_domain_outcome(payload):
    result = CorrectnessResult.model_validate(payload)
    assert result.passed is (payload["outcome"] == "correct")


@pytest.mark.parametrize(
    "patch",
    [
        {"passed": False},  # passed iff outcome == correct
        {"passed_cases": 2},
        {"failures": [value_failure()]},
        {"decoy": True},
        {"domain_failure": domain_failure()},
        {"total_cases": 4},
    ],
)
def test_correct_outcome_rule_violations(patch):
    with pytest.raises(ValidationError):
        CorrectnessResult.model_validate(correctness_correct() | patch)


@pytest.mark.parametrize(
    "patch",
    [
        {"passed": True},
        # wrong requires every case accounted for
        {"passed_cases": 1},
        # wrong requires at least one mismatch failure or decoy
        {"passed_cases": 3, "failures": []},
        {"domain_failure": domain_failure()},
        {"failures": [value_failure(0), value_failure(0)]},  # duplicate case_idx
        {"failures": [value_failure(3)]},  # case_idx out of range
    ],
)
def test_wrong_outcome_rule_violations(patch):
    with pytest.raises(ValidationError):
        CorrectnessResult.model_validate(correctness_wrong() | patch)


def test_decoy_requires_wrong_with_all_cases_passing():
    decoy = correctness_wrong() | {"decoy": True, "passed_cases": 3, "failures": []}
    assert CorrectnessResult.model_validate(decoy).decoy is True
    with pytest.raises(ValidationError):
        CorrectnessResult.model_validate(decoy | {"failures": [value_failure(2)]})


@pytest.mark.parametrize(
    "outcome_name,code,phase,case_idx,valid",
    [
        ("compile_error", "SUBJECT_COMPILE_ERROR", "compile", None, True),
        ("compile_error", "SUBJECT_COMPILE_ERROR", "compile", 0, False),
        ("compile_error", "SUBJECT_COMPILE_ERROR", "execute", None, False),
        ("compile_error", "SUBJECT_RUNTIME_ERROR", "compile", None, False),
        ("runtime_error", "SUBJECT_RUNTIME_ERROR", "initialize", None, True),
        ("runtime_error", "SUBJECT_RUNTIME_ERROR", "execute", 4, True),
        ("runtime_error", "SUBJECT_RUNTIME_ERROR", "execute", None, False),
        ("runtime_error", "SUBJECT_RUNTIME_ERROR", "execute", 8, False),
        ("runtime_error", "SUBJECT_RUNTIME_ERROR", "compile", None, False),
        ("execution_timeout", "SUBJECT_EXECUTION_TIMEOUT", "compile", None, True),
        ("execution_timeout", "SUBJECT_EXECUTION_TIMEOUT", "initialize", None, True),
        ("execution_timeout", "SUBJECT_EXECUTION_TIMEOUT", "execute", 7, True),
        ("execution_timeout", "SUBJECT_EXECUTION_TIMEOUT", "execute", None, False),
    ],
)
def test_domain_failure_code_phase_case_rules(outcome_name, code, phase, case_idx, valid):
    payload = {
        "outcome": outcome_name,
        "passed": False,
        "decoy": False,
        "passed_cases": 0,
        "total_cases": 8,
        "failures": [],
        "domain_failure": domain_failure(code, phase, case_idx),
    }
    if valid:
        assert CorrectnessResult.model_validate(payload).outcome == outcome_name
    else:
        with pytest.raises(ValidationError):
            CorrectnessResult.model_validate(payload)


@pytest.mark.parametrize(
    "patch",
    [
        {"domain_failure": None},
        {"passed": True},
        {"decoy": True},
        {"total_cases": 0},
        {"passed_cases": 9},
    ],
)
def test_failure_outcome_rule_violations(patch):
    payload = {
        "outcome": "compile_error",
        "passed": False,
        "decoy": False,
        "passed_cases": 0,
        "total_cases": 8,
        "failures": [],
        "domain_failure": domain_failure(),
    }
    with pytest.raises(ValidationError):
        CorrectnessResult.model_validate(payload | patch)


# ---------------------------------------------------------------------------
# Samples: at most eight, stable order, sentinel encoding
# ---------------------------------------------------------------------------


def test_sample_capped_at_eight_and_sorted_by_abs_diff():
    points = [sample([i], float(8 - i), 0.0, float(8 - i)) for i in range(8)]
    failure = CaseFailure.model_validate(
        {"case_idx": 0, "kind": "value_mismatch", "metrics": {}, "sample": points}
    )
    assert len(failure.sample) == 8
    with pytest.raises(ValidationError):
        CaseFailure.model_validate(
            {"case_idx": 0, "kind": "value_mismatch", "metrics": {},
             "sample": points + [sample([8], 0.5, 0.0, 0.5)]}
        )
    # ascending order is not the stable publication order
    with pytest.raises(ValidationError):
        CaseFailure.model_validate(
            {"case_idx": 0, "kind": "value_mismatch", "metrics": {},
             "sample": list(reversed(points))}
        )


def test_sample_orders_finite_then_sentinels_with_index_tiebreak():
    ordered = [
        sample([0, 0], 2.0, 0.0, 2.0),
        sample([0, 1], 2.0, 0.0, 2.0),  # equal diff ties break by row-major index
        sample([1, 0], 1.0, 0.0, 1.0),
        sample([2], "inf", 0.0, "-inf"),
        sample([3], "nan", 1.0, "inf"),
        sample([4], 0.0, "nan", "nan"),
    ]
    failure = CaseFailure.model_validate(
        {"case_idx": 0, "kind": "value_mismatch", "metrics": {}, "sample": ordered}
    )
    assert [p.diff for p in failure.sample] == [2.0, 2.0, 1.0, "-inf", "inf", "nan"]
    for swap in ((0, 1), (2, 3), (4, 5)):
        shuffled = list(ordered)
        shuffled[swap[0]], shuffled[swap[1]] = shuffled[swap[1]], shuffled[swap[0]]
        with pytest.raises(ValidationError):
            CaseFailure.model_validate(
                {"case_idx": 0, "kind": "value_mismatch", "metrics": {}, "sample": shuffled}
            )


def test_shape_mismatch_has_empty_sample():
    failure = CaseFailure.model_validate(
        {"case_idx": 0, "kind": "shape_mismatch", "metrics": {"matched_ratio": 0.0}}
    )
    assert failure.sample == []
    with pytest.raises(ValidationError):
        CaseFailure.model_validate(
            {"case_idx": 0, "kind": "shape_mismatch", "metrics": {},
             "sample": [sample([0], 1.0, 0.0, 1.0)]}
        )


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_sample_rejects_raw_non_finite_floats(value):
    with pytest.raises(ValidationError):
        SampleElement.model_validate(sample([0], value, 0.0, 0.0))


def test_sample_sentinels_survive_json_round_trip():
    element = SampleElement.model_validate(sample([1, 2], "inf", "nan", "-inf"))
    back = SampleElement.model_validate_json(element.model_dump_json())
    assert (back.candidate, back.reference, back.diff) == ("inf", "nan", "-inf")


@pytest.mark.parametrize(
    "field,value",
    [("max_abs_diff", math.inf), ("matched_ratio", math.nan), ("rtol", -math.inf)],
)
def test_ordinary_float_fields_reject_non_finite(field, value):
    with pytest.raises(ValidationError):
        CaseMetrics.model_validate({field: value})


def test_ordinary_float_fields_accept_finite_values():
    metrics = CaseMetrics.model_validate({"max_abs_diff": 0.5, "matched_ratio": 0.99})
    assert metrics.max_abs_diff == 0.5


# ---------------------------------------------------------------------------
# Performance aggregation and coverage arithmetic
# ---------------------------------------------------------------------------


def test_performance_result_recomputes_geometric_mean_and_coverage():
    result = PerformanceResult.model_validate(performance_payload())
    assert result.measured_cases == 2
    assert result.selected_cases == 3
    assert result.coverage == pytest.approx(2 / 3)
    assert result.overall_speedup_vs_reference == pytest.approx(1.0)  # geomean(2, 0.5)


@pytest.mark.parametrize(
    "patch",
    [
        {"overall_speedup_vs_reference": 1.25},  # arithmetic mean is not geomean
        {"coverage": 1.0},
        {"selected_cases": 2},
        {"measured_cases": 3},
        {"aggregation": "arithmetic_mean_v1"},
    ],
)
def test_performance_rejects_inconsistent_arithmetic(patch):
    with pytest.raises(ValidationError):
        PerformanceResult.model_validate(performance_payload() | patch)


def test_performance_rejects_duplicate_or_zero_measurements():
    payload = performance_payload()
    payload["unmeasured_cases"][0]["case_idx"] = 0  # collides with per_case
    with pytest.raises(ValidationError):
        PerformanceResult.model_validate(payload)

    empty = performance_payload()
    empty.update(per_case=[], selected_cases=1, measured_cases=0, coverage=0.0,
                 overall_speedup_vs_reference=1.0)
    with pytest.raises(ValidationError):
        PerformanceResult.model_validate(empty)


@pytest.mark.parametrize(
    "patch",
    [
        {"implementation_latency_ms": 0.0},
        {"reference_latency_ms": -1.0},
        {"speedup_vs_reference": math.inf},
        {"input_shapes": [{"input_index": 0, "dimensions": [-1]}]},
        {"input_shapes": "20120x512"},  # compressed string shapes are forbidden
    ],
)
def test_performance_case_rejects_invalid_measurements(patch):
    payload = performance_payload()
    payload["per_case"][0] |= patch
    with pytest.raises(ValidationError):
        PerformanceResult.model_validate(payload)


def test_unmeasured_case_requires_stable_reason_code():
    with pytest.raises(ValidationError):
        UnmeasuredPerformanceCase.model_validate(
            {"case_idx": 0, "input_shapes": [], "reason_code": "SLOW"}
        )


def test_input_shape_dimensions_may_be_empty():
    shape = InputShape.model_validate({"input_index": 0, "dimensions": []})
    assert shape.dimensions == []


# ---------------------------------------------------------------------------
# Profile count and ratio rules for both discriminants
# ---------------------------------------------------------------------------


def test_op_profile_counts_must_tally_case_statuses():
    assert OpProfileResult.model_validate(op_profile_payload()).counts.selected == 2
    for counts in (
        {"selected": 1, "complete": 1, "partial": 0, "failed": 0},
        {"selected": 2, "complete": 2, "partial": 0, "failed": 0},
        {"selected": 3, "complete": 1, "partial": 0, "failed": 1},
        {"selected": 2, "complete": 1, "partial": 1, "failed": 1},
    ):
        with pytest.raises(ValidationError):
            OpProfileResult.model_validate(op_profile_payload() | {"counts": counts})


def test_profile_rejects_duplicate_case_indices():
    payload = op_profile_payload()
    payload["cases"][1]["case_idx"] = 0
    with pytest.raises(ValidationError):
        OpProfileResult.model_validate(payload)


@pytest.mark.parametrize(
    "status,error,valid",
    [
        ("complete", None, True),
        ("complete", error_info(), False),
        ("failed", None, False),
        ("failed", error_info(), True),
        ("partial", None, False),
        ("partial", error_info(), True),
    ],
)
def test_op_case_status_error_rules(status, error, valid):
    case = {
        "case_idx": 0,
        "status": status,
        "kernels": [{"op_name": "k", "time_ms": 1.0, "percentage": 100.0}],
        "error": error,
    }
    if valid:
        assert OpProfileCase.model_validate(case).status == status
    else:
        with pytest.raises(ValidationError):
            OpProfileCase.model_validate(case)


def test_op_kernel_percentage_sum_allows_only_rounding_error():
    base = {"case_idx": 0, "status": "complete", "error": None}
    OpProfileCase.model_validate(
        base | {"kernels": [{"op_name": "a", "time_ms": 1.0, "percentage": 50.5},
                            {"op_name": "b", "time_ms": 1.0, "percentage": 50.0}]}
    )
    with pytest.raises(ValidationError):
        OpProfileCase.model_validate(
            base | {"kernels": [{"op_name": "a", "time_ms": 1.0, "percentage": 50.0},
                                {"op_name": "b", "time_ms": 1.0, "percentage": 40.0}]}
        )


@pytest.mark.parametrize(
    "kernel",
    [
        {"op_name": "k", "time_ms": -1.0, "percentage": 100.0},
        {"op_name": "k", "time_ms": 1.0, "percentage": math.nan},
        {"op_name": "k", "time_ms": 1.0, "percentage": -0.1},
    ],
)
def test_op_kernel_rejects_negative_or_non_finite_values(kernel):
    with pytest.raises(ValidationError):
        OpKernel.model_validate(kernel)


def test_deep_profile_counts_and_ratios():
    assert DeepProfileResult.model_validate(deep_profile_payload()).counts.partial == 1
    payload = deep_profile_payload()
    payload["counts"] = {"selected": 2, "complete": 2, "partial": 0, "failed": 0}
    with pytest.raises(ValidationError):
        DeepProfileResult.model_validate(payload)


@pytest.mark.parametrize("value", [-0.1, 1.1, math.inf, math.nan])
def test_deep_ratios_are_finite_and_within_unit_interval(value):
    with pytest.raises(ValidationError):
        DeepRatios.model_validate({"mte2": value})


def test_deep_ratios_reject_unknown_metric_names():
    with pytest.raises(ValidationError):
        DeepRatios.model_validate({"mte4": 0.5})


@pytest.mark.parametrize(
    "patch,valid",
    [
        ({"status": "complete", "error": error_info()}, False),
        ({"status": "failed", "error": None}, False),
        ({"status": "failed", "error": error_info()}, True),
        ({"status": "partial", "missing_metrics": [], "warnings": [], "error": None}, False),
        ({"status": "partial", "missing_metrics": [], "warnings": ["degraded"],
          "error": None}, True),
        ({"bound": "MTE2 BOUND"}, False),  # v1 bound strings are not schema 2.1
        ({"bound": "NONE"}, True),
    ],
)
def test_deep_case_status_and_bound_rules(patch, valid):
    case = deep_profile_payload()["cases"][1] | patch
    if valid:
        assert DeepProfileCase.model_validate(case).case_idx == 1
    else:
        with pytest.raises(ValidationError):
            DeepProfileCase.model_validate(case)


def test_profile_mode_discriminator_rejects_cross_mode_cases():
    # deep case payload is not valid under op mode
    op_payload = op_profile_payload() | {"cases": deep_profile_payload()["cases"]}
    with pytest.raises(ValidationError):
        OpProfileResult.model_validate(op_payload)


def test_profile_counts_selected_must_equal_component_sum():
    with pytest.raises(ValidationError):
        ProfileCounts.model_validate(
            {"selected": 3, "complete": 1, "partial": 0, "failed": 1}
        )


# ---------------------------------------------------------------------------
# Provenance and serialization hygiene
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "patch,valid",
    [
        ({"grade_eligible": True}, False),  # requires manifest identity
        ({"grade_eligible": True, "manifest_id": "release-1", "problem_id": "addmm-1"},
         True),
        ({"manifest_id": "release-1"}, False),  # manifest and problem move together
        ({"problem_id": "addmm-1"}, False),
        ({"execution_environment": None}, True),  # no attempt ever started
    ],
)
def test_provenance_grade_eligibility_requires_manifest_identity(patch, valid):
    if valid:
        assert Provenance.model_validate(provenance(**patch)).grade_eligible == patch.get(
            "grade_eligible", False
        )
    else:
        with pytest.raises(ValidationError):
            Provenance.model_validate(provenance(**patch))


def test_serialized_models_never_contain_removed_v1_fields():
    resource = EvaluationResource.model_validate(
        resource_payload(
            "completed",
            result=result_payload(
                performance={"status": "succeeded", "data": performance_payload(),
                             "error": None, "reason": None},
                profile={"status": "succeeded", "data": deep_profile_payload(),
                         "error": None, "reason": None},
            ),
        )
    )
    text = resource.model_dump_json()
    assert "mean_abs_err" not in text
    assert "throughput" not in text


def test_removed_v1_fields_are_rejected_on_input():
    with pytest.raises(ValidationError):
        CaseMetrics.model_validate({"mean_abs_err": 0.1})
    payload = performance_payload()
    payload["per_case"][0]["throughput"] = 12.0
    with pytest.raises(ValidationError):
        PerformanceResult.model_validate(payload)


def test_execution_environment_carries_no_host_or_serial():
    with pytest.raises(ValidationError):
        ExecutionEnvironment.model_validate(
            provenance()["execution_environment"] | {"hostname": "npu-node-7"}
        )
