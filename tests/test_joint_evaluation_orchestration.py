"""Combined execution preserves independently recoverable, typed check evidence."""
from datetime import datetime, timedelta, timezone

import pytest
from test_evaluation_evaluator import RecordingSink, full_harness, load, trial_pinned

from triton_evaluation.api.contract import EvaluationRequest
from triton_evaluation.evaluators.base import EngineCheckResult, LeaseLostError, engine_failure
from triton_evaluation.evaluators.evaluation import EvaluationEvaluator
from triton_evaluation.evaluators.shaping import ShapingError, shape_profile_runs


def trace_raw():
    return {
        "mode": "trace", "measurement_method": "profiler_kernel_sum_v1", "total_cases": 3,
        "counts": {"selected": 1, "complete": 1, "partial": 0, "failed": 0},
        "cases": [{
            "case_idx": 0, "status": "complete", "error": None,
            "kernels": [{"op_name": "add", "time_ms": 0.2, "percentage": 100.0}],
        }],
    }


class UnexpectedProfile:
    def run(self, *args, **kwargs):
        pytest.fail("a joint trace must not launch a separate profile evaluator")


def run(evaluator, sink, tmp_path, *, lease=lambda: True):
    request = EvaluationRequest(
        schema_version="2.1", op_name="add", reference_code="REF", kernel_code="CANDIDATE",
        checks=["correctness", "performance", "profile"],
        profile={"mode": "trace", "case_indices": [0]},
    )
    return evaluator.run(
        request, tmp_path, 0, datetime.now(timezone.utc) + timedelta(minutes=5),
        lease, None, trial_pinned(), sink,
    )


@pytest.mark.parametrize("trace_fails", [False, True])
def test_timing_is_saved_before_trace_finishes(tmp_path, trace_fails):
    sink = RecordingSink()

    class Joint:
        def run_with_trace(self, *args, on_performance, **kwargs):
            timing = EngineCheckResult(
                status="succeeded", raw=load("benchmark-event-timing.json"), error=None,
            )
            assert on_performance(timing) is True
            assert sink.calls[-2:] == [
                ("save", "performance", "succeeded"), ("begin", "profile"),
            ]
            trace = (
                engine_failure("profile", "capture failed") if trace_fails else EngineCheckResult(
                    status="succeeded", raw=trace_raw(), error=None,
                )
            )
            return timing, trace

    result = run(EvaluationEvaluator(
        verify=full_harness().verify, benchmark=Joint(), profile=UnexpectedProfile(),
    ), sink, tmp_path)
    assert result.performance.status == "succeeded"
    assert result.profile.status == ("failed" if trace_fails else "succeeded")
    assert sum(call[:2] == ("save", "performance") for call in sink.calls) == 1


def test_invalid_timing_stops_shared_trace(tmp_path):
    class Joint:
        def run_with_trace(self, *args, on_performance, **kwargs):
            timing = EngineCheckResult(status="succeeded", raw={}, error=None)
            assert on_performance(timing) is False
            return timing, engine_failure("profile", "not run")

    result = run(EvaluationEvaluator(
        verify=full_harness().verify, benchmark=Joint(), profile=UnexpectedProfile(),
    ), RecordingSink(), tmp_path)
    assert result.performance.status == "failed"
    assert result.profile.status == "skipped"


def test_joint_checkpoint_cannot_publish_after_lease_loss(tmp_path):
    valid = [True]

    class Joint:
        def run_with_trace(self, *args, on_performance, **kwargs):
            valid[0] = False
            on_performance(EngineCheckResult(
                status="succeeded", raw=load("benchmark-event-timing.json"), error=None,
            ))
            pytest.fail("fenced checkpoint returned")

    with pytest.raises(LeaseLostError):
        run(EvaluationEvaluator(
            verify=full_harness().verify, benchmark=Joint(), profile=UnexpectedProfile(),
        ), RecordingSink(), tmp_path, lease=lambda: valid[0])


@pytest.mark.parametrize("invalid", ["method", "inventory"])
def test_joint_timing_must_match_requested_protocol_and_verified_inventory(tmp_path, invalid):
    verifier = full_harness().verify

    class Joint:
        def run_with_trace(self, *args, on_performance, **kwargs):
            raw = load("benchmark-event-timing.json")
            if invalid == "method":
                raw["measurement_method"] = "profiler_kernel_sum_v1"
                for case in raw["per_shape_results"]:
                    case["measurement_method"] = "profiler_kernel_sum_v1"
            timing = EngineCheckResult(status="succeeded", raw=raw, error=None)
            assert on_performance(timing) is False
            return timing, engine_failure("profile", "not run")

    if invalid == "inventory":
        verifier.result.raw.update(total_cases=2, passed_cases=2)
    result = run(EvaluationEvaluator(
        verify=verifier, benchmark=Joint(), profile=UnexpectedProfile(),
    ), RecordingSink(), tmp_path)
    assert result.performance.error.code == "INVALID_ENGINE_OUTPUT"
    assert result.profile.status == "skipped"


@pytest.mark.parametrize("invalid", ["zero_duration", "duplicate_names", "wrong_percentages"])
def test_trace_shaping_rejects_invalid_duration_breakdown(invalid):
    from triton_evaluation.api.contract import ProfileParams

    raw = trace_raw()
    kernels = raw["cases"][0]["kernels"]
    if invalid == "zero_duration":
        kernels[0]["time_ms"] = 0
    else:
        kernels[0]["percentage"] = 50.0
        kernels.append({
            "op_name": "add" if invalid == "duplicate_names" else "other",
            "time_ms": 0.6, "percentage": 50.0,
        })
    with pytest.raises(ShapingError):
        shape_profile_runs(raw, ProfileParams(mode="trace", case_indices=[0]))
