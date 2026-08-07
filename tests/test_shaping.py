from triton_evaluation.evaluators.shaping import shape_payload


def _verify_json(passed=1, total=2, failures=None):
    return {
        "op_name": "addmm",
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": total - passed,
        "failures": failures or [],
    }


def _failure():
    return {
        "case_idx": 2,
        "error_type": "AccuracyError",
        "error_msg": "boom",
        "kind": "constant_offset",
        "sample": [{"idx": [0], "candidate": 6.0, "reference": 1.0, "diff": 5.0}],
        "metrics": {"max_abs_diff": 5.0, "matched_ratio": 0.0, "MERE": 1.0,
                    "small_count": 0, "normal_count": 4, "atol": 1e-3, "rtol": 1.2e-4},
    }


def _perf_json():
    return {
        "speedup_vs_torch": 2.5, "per_shape_results": [{"case_idx": 1, "status": "pass"}],
    }


def test_correctness_only_maps_failures_with_kind_and_sample():
    payload = shape_payload(_verify_json(passed=1, failures=[_failure()]),
                            perf_json=None, profile_json=None, checks=["correctness"])
    corr = payload["correctness"]
    assert corr["passed"] is False
    assert corr["compiled"] is True
    assert corr["passed_cases"] == 1 and corr["total_cases"] == 2
    assert corr["failures"][0]["kind"] == "constant_offset"
    assert corr["failures"][0]["sample"][0]["diff"] == 5.0
    assert corr["failures"][0]["metrics"]["mean_abs_err"] is None  # engine has no mean abs err
    assert corr["failures"][0]["metrics"]["mere"] == 1.0  # MERE (mean rel err) passes through
    assert payload["performance"] is None
    assert payload["profile"] is None


def test_performance_present_only_when_correctness_passed_and_run():
    # correctness passed AND perf run -> performance populated
    payload = shape_payload(_verify_json(passed=2, total=2), perf_json=_perf_json(),
                            profile_json=None, checks=["correctness", "performance"])
    assert payload["performance"]["speedup_vs_torch"] == 2.5
    # correctness failed -> performance is null even if perf_json somehow present
    payload_fail = shape_payload(_verify_json(passed=1, failures=[_failure()]),
                                 perf_json=_perf_json(), profile_json=None,
                                 checks=["correctness", "performance"])
    assert payload_fail["performance"] is None


def test_profile_passthrough_with_state_derived():
    profile_json = {
        "profile_mode": "op", "selected_cases": 2, "completed_cases": 2,
        "failed_cases": 0, "partial_cases": 0, "per_case_results": [],
    }
    payload = shape_payload(None, perf_json=None, profile_json=profile_json, checks=["profile"])
    assert payload["profile"]["profile_state"] == "complete"
    assert payload["profile"]["data"]["profile_mode"] == "op"
    assert payload["correctness"] is None  # correctness not requested


def test_profile_partial_when_failures_remain():
    profile_json = {"selected_cases": 2, "completed_cases": 1, "failed_cases": 1, "partial_cases": 0}
    payload = shape_payload(None, None, profile_json, checks=["profile"])
    assert payload["profile"]["profile_state"] == "partial"


def test_decoy_correctness_result():
    payload = shape_payload(None, None, None, checks=["correctness"], decoy=True)
    corr = payload["correctness"]
    assert corr["decoy"] is True
    assert corr["passed"] is False
    assert corr["compiled"] is False
    assert corr["failures"] == []
