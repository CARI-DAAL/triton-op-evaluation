import pytest

from triton_evaluation.evaluators.shaping import (
    ShapingError,
    shape_payload,
    shape_performance,
    shape_profile,
)


def _verify_json(passed=1, total=2, failures=None):
    return {
        "op_name": "addmm",
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": total - passed,
        "failures": failures or [],
    }


def _failure(kind="constant_offset"):
    return {
        "case_idx": 2,
        "error_type": "AccuracyError",
        "error_msg": "boom",
        "kind": kind,
        "sample": [{"idx": [0], "candidate": 6.0, "reference": 1.0, "diff": 5.0}],
        "metrics": {
            "max_abs_diff": 5.0,
            "matched_ratio": 0.0,
            "MERE": 1.0,
            "small_count": 0,
            "normal_count": 4,
            "atol": 1e-3,
            "rtol": 1.2e-4,
        },
    }


def _complete_perf_json():
    return {
        "speedup_vs_torch": 2.5,
        "failed_cases": 0,
        "per_shape_results": [
            {
                "case_idx": 1,
                "input_desc": [
                    {"type": "tensor", "shape": [20120, 512], "dtype": "torch.float32"}
                ],
                "status": "pass",
                "framework": {"avg_latency_ms": 1.0, "peak_memory_mb": 2.0},
                "implementation": {"avg_latency_ms": 0.4, "peak_memory_mb": 1.0},
                "speedup_vs_torch": 2.5,
                "error_type": None,
                "error_msg": None,
            }
        ],
    }


def test_correctness_maps_classified_failure_without_error_text():
    payload = shape_payload(
        _verify_json(passed=1, failures=[_failure()]),
        perf_json=None,
        profile_json=None,
        checks=["correctness"],
    )

    failure = payload["correctness"]["failures"][0]
    assert failure["kind"] == "constant_offset"
    assert failure["sample"][0]["diff"] == 5.0
    assert failure["metrics"]["mean_abs_err"] is None
    assert failure["metrics"]["mere"] == 1.0
    assert failure["error"] is None


def test_correctness_keeps_compile_error_and_sets_compiled_false():
    raw = _failure(kind=None)
    raw["error_type"] = "CompilationError"
    raw["sample"] = []
    payload = shape_payload(
        _verify_json(passed=0, total=1, failures=[raw]),
        None,
        None,
        checks=["correctness"],
    )

    assert payload["correctness"]["compiled"] is False
    assert payload["correctness"]["failures"][0]["error"] == "boom"


def test_performance_projects_only_stable_per_shape_fields():
    assert shape_performance(_complete_perf_json()) == {
        "speedup_vs_torch": 2.5,
        "per_shape": [
            {
                "shape": "[20120,512]",
                "latency_ms": 0.4,
                "throughput": None,
                "speedup_vs_torch": 2.5,
            }
        ],
    }


@pytest.mark.parametrize(
    "mutation",
    [
        {"failed_cases": 1},
        {"speedup_vs_torch": None},
        {"per_shape_results": []},
    ],
)
def test_performance_rejects_incomplete_engine_result(mutation):
    raw = _complete_perf_json()
    raw.update(mutation)

    with pytest.raises(ShapingError):
        shape_performance(raw)


def test_performance_present_only_when_correctness_passed_and_run():
    payload = shape_payload(
        _verify_json(passed=2, total=2),
        perf_json=_complete_perf_json(),
        profile_json=None,
        checks=["correctness", "performance"],
    )
    assert payload["performance"]["speedup_vs_torch"] == 2.5

    payload_fail = shape_payload(
        _verify_json(passed=1, failures=[_failure()]),
        perf_json=_complete_perf_json(),
        profile_json=None,
        checks=["correctness", "performance"],
    )
    assert payload_fail["performance"] is None


def test_op_profile_maps_kernel_durations_and_percentages():
    raw = {
        "profile_mode": "op",
        "selected_cases": 1,
        "completed_cases": 1,
        "partial_cases": 0,
        "failed_cases": 0,
        "timed_out": False,
        "per_case_results": [
            {
                "case_idx": 0,
                "status": "complete",
                "kernels": [
                    {"name": "slow", "duration_us": 400.0, "metrics": {"private": 1}},
                    {"name": "fast", "duration_us": 100.0, "metrics": {"private": 2}},
                ],
                "raw_metrics": {"must_not_leak": 1},
            }
        ],
    }

    profile = shape_profile(raw)

    assert profile == {
        "profile_state": "complete",
        "data": {
            "mode": "op",
            "selected_cases": 1,
            "completed_cases": 1,
            "partial_cases": 0,
            "failed_cases": 0,
            "timed_out": False,
            "ops": [
                {"case_idx": 0, "op_name": "slow", "time_ms": 0.4, "percentage": 80.0},
                {"case_idx": 0, "op_name": "fast", "time_ms": 0.1, "percentage": 20.0},
            ],
            "summaries": None,
        },
    }


def test_deep_profile_maps_fixed_diagnostics_without_raw_evidence():
    raw = {
        "profile_mode": "deep",
        "selected_cases": 1,
        "completed_cases": 0,
        "partial_cases": 1,
        "failed_cases": 0,
        "timed_out": False,
        "per_case_results": [
            {
                "case_idx": 3,
                "status": "partial",
                "target_kernel": "kernel_a",
                "bound": {
                    "name": "MTE2 BOUND",
                    "rule": "MTE2 > 0.80",
                    "ratios": {
                        "MTE2": 0.9,
                        "CUBE": 0.1,
                        "PUSHQ": 0.2,
                        "FIXP": 0.3,
                        "MTE3": 0.4,
                        "SCALAR": 0.05,
                        "SCALARLDST": 0.07,
                    },
                    "evidence": {"must_not_leak": "raw"},
                },
                "per_core": {"spread_ratio": 0.25, "verdict": "warning"},
                "recommendations": [
                    {"recommendation": "reduce loads", "evidence": {"private": 1}}
                ],
                "missing_metrics": ["Memory"],
                "warnings": ["partial capture"],
                "error": None,
                "raw_metrics": {"must_not_leak": 1},
            }
        ],
    }

    summary = shape_profile(raw)["data"]["summaries"][0]

    assert summary == {
        "case_idx": 3,
        "status": "partial",
        "target_kernel": "kernel_a",
        "bound": "MTE2 BOUND",
        "rule": "MTE2 > 0.80",
        "mte2_ratio": 0.9,
        "cube_ratio": 0.1,
        "vec_ratio": 0.2,
        "fixp_ratio": 0.3,
        "mte3_ratio": 0.4,
        "scalar_ratio": 0.07,
        "per_core_spread_ratio": 0.25,
        "per_core_verdict": "warning",
        "recommendations": ["reduce loads"],
        "missing_metrics": ["Memory"],
        "warnings": ["partial capture"],
        "error": None,
    }


def test_profile_partial_when_failures_remain():
    raw = {
        "profile_mode": "op",
        "selected_cases": 2,
        "completed_cases": 1,
        "failed_cases": 1,
        "partial_cases": 0,
        "timed_out": False,
        "per_case_results": [],
    }
    assert shape_profile(raw)["profile_state"] == "partial"


def test_decoy_correctness_result():
    payload = shape_payload(None, None, None, checks=["correctness"], decoy=True)
    correctness = payload["correctness"]
    assert correctness["decoy"] is True
    assert correctness["passed"] is False
    assert correctness["compiled"] is False
    assert correctness["failures"] == []
