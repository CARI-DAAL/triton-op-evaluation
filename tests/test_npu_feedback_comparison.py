"""The manual NPU harness compares equal protocols and fails on missing evidence."""
import copy

import pytest

from scripts.npu_feedback_comparison import (
    compare_measurements,
    requests_for,
    summarize,
    validate_result,
)
from scripts.test_eval_service import build_request


def test_separate_and_joint_requests_share_sources_and_protocols():
    original = build_request()
    original["non_compute"] = True
    before = copy.deepcopy(original)
    requests = requests_for(original, "event", [1, 0])
    assert original == before
    assert requests["combined"]["performance"] == requests["timing"]["performance"]
    assert requests["combined"]["profile"] == requests["trace"]["profile"]
    assert requests["combined"]["performance"] == {
        "method": "event", "warmup": 5, "repeats": 20,
        "use_baseline_cache": False, "case_indices": [0, 1],
    }
    assert "correctness" in requests["timing"]["checks"]
    assert "correctness" in requests["combined"]["checks"]
    assert requests["trace"]["checks"] == ["profile"]
    for field in ("definition", "workloads", "triton_code", "seed", "non_compute"):
        assert requests["combined"][field] == requests["trace"][field]


@pytest.mark.parametrize("selection", [[], [2], [-1], [0, 0]])
def test_invalid_selections_are_rejected(selection):
    with pytest.raises(ValueError):
        requests_for(build_request(), "event", selection)


def terminal():
    return {
        "state": "completed", "result": {
            "provenance": {"evaluator_build": "build-a", "execution_environment": {"card": "test"}},
            "correctness": {"status": "succeeded", "data": {
                "outcome": "correct", "passed": True, "passed_cases": 2, "total_cases": 2,
            }},
            "performance": {"status": "succeeded", "data": {
                "measurement_method": "npu_event_v1", "unmeasured_cases": [],
                "per_case": [{"case_idx": 1, "baseline_cache_hit": False,
                              "implementation_latency_ms": 0.1, "reference_latency_ms": 0.2}],
            }},
            "profile": {"status": "succeeded", "data": {
                "mode": "trace", "measurement_method": "profiler_kernel_sum_v1", "total_cases": 2,
                "counts": {"selected": 1, "complete": 1, "partial": 0, "failed": 0},
                "cases": [{"case_idx": 1, "status": "complete", "kernels": [
                    {"op_name": "kernel", "time_ms": 0.1, "percentage": 100.0},
                ]}],
            }},
        },
    }


def test_acceptance_requires_successful_matching_evidence():
    resource = terminal()
    checks = ["correctness", "performance", "profile"]
    validate_result(resource, checks, [1], "event", 2)
    for check in checks:
        bad = copy.deepcopy(resource)
        bad["result"][check]["status"] = "skipped"
        with pytest.raises(ValueError):
            validate_result(bad, checks, [1], "event", 2)
    resource["result"]["performance"]["data"]["per_case"][0]["baseline_cache_hit"] = True
    with pytest.raises(ValueError, match="fresh Torch"):
        validate_result(resource, checks, [1], "event", 2)


def test_summary_excludes_warmup_and_failed_trials():
    def measurement(seconds, accepted=True):
        return {"wall_seconds": seconds, "time_to_performance_seconds": seconds / 2,
                "accepted": accepted}

    summary = summarize([
        {"warmup": True, "separate": measurement(99), "combined": measurement(88)},
        {"warmup": False, "separate": measurement(10), "combined": measurement(5)},
        {"warmup": False, "separate": measurement(999, False), "combined": {}},
        {"warmup": False, "separate": measurement(100), "combined": {}},
    ])
    assert summary["separate"]["n"] == 1
    assert summary["combined"]["median_wall_seconds"] == 5
    assert summary["separate_over_combined_wall_ratio"] == 2


def test_measurement_drift_is_reported_separately_from_wall_time():
    old, new = terminal(), terminal()
    new["result"]["performance"]["data"]["per_case"][0]["implementation_latency_ms"] *= 1000
    pair = {
        "separate": {"requests": [{"resource": old}, {"resource": old}]},
        "combined": {"requests": [{"resource": new}]},
    }
    report = compare_measurements(pair)
    assert report[0]["implementation_latency_ms_combined_over_separate"] == 1000
    assert report[0]["within_requested_drift"] is None
    assert compare_measurements(pair, 0.2)[0]["within_requested_drift"] is False
