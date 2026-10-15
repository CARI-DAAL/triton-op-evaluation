"""Shaping boundary tests: private engine JSON -> stable v2 domain models.

Every fixture under tests/fixtures/engine mirrors the engine's private output
protocol byte-shape (1-based case indices for verify/benchmark, 0-based for
profile, engine divergence names, sentinel strings for non-finite samples).
The shaping layer is the only boundary allowed to read that protocol, and it
must fail closed on anything outside it.
"""

import copy
import json
import math
from pathlib import Path

import pytest

from triton_evaluation.api.contract import (
    DeepProfileResult,
    OpProfileResult,
)
from triton_evaluation.engine._divergence import classify_divergence
from triton_evaluation.evaluators.shaping import (
    ShapingError,
    UnattributedFailure,
    shape_correctness,
    shape_performance,
    shape_profile,
)

FIXTURES = Path(__file__).parent / "fixtures" / "engine"


def load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def mutated(name, *path, value):
    """Return a fixture copy with ``value`` written at ``path`` (keys/indices)."""
    raw = load(name)
    node = raw
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return raw


def without(name, *path):
    raw = load(name)
    node = raw
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]
    return raw


def _walk_strings(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, str):
        yield value


# ---------------------------------------------------------------------------
# Correctness: the private verify protocol shapes into CorrectnessResult
# ---------------------------------------------------------------------------


def test_correct_fixture_is_correct():
    result = shape_correctness(load("verify-correct.json"), decoy=False)
    assert result.outcome == "correct"
    assert result.passed is True
    assert result.decoy is False
    assert result.passed_cases == 2
    assert result.total_cases == 2
    assert result.failures == []
    assert result.domain_failure is None


def test_wrong_value_fixture_maps_experimental_kind_to_coarse_value_mismatch():
    result = shape_correctness(load("verify-wrong-value.json"), decoy=False)
    assert result.outcome == "wrong"
    assert result.passed is False
    assert result.passed_cases == 1
    (failure,) = result.failures
    # engine case_idx is 1-based; the stable contract is 0-based.
    assert failure.case_idx == 1
    assert failure.kind == "value_mismatch"
    assert failure.metrics.max_abs_diff == 2.5
    assert failure.metrics.matched_ratio == 0.75
    assert failure.metrics.mere == 0.5
    assert failure.metrics.small_count == 0
    assert failure.metrics.normal_count == 4
    assert failure.metrics.atol == pytest.approx(0.001)
    assert failure.metrics.rtol == pytest.approx(0.0001220703125)
    assert [point.diff for point in failure.sample] == [2.5, 0.25]
    dumped = result.model_dump()
    assert "mean_abs_err" not in dumped["failures"][0]["metrics"]


def test_engine_sample_with_matched_non_finite_pairs_shapes_as_wrong():
    # Producer to reader: the engine's own sample for a wrong candidate whose
    # NaN/inf positions match the reference must not become INVALID_ENGINE_OUTPUT.
    sample = classify_divergence(
        [math.nan, 1.0, math.inf, 3.0], [math.nan, 1.0, math.inf, 7.0], (2, 2),
        atol=1e-3, rtol=2**-13, small_value_threshold=2**-14,
        small_value_error=2**-30, rel_threshold=2**-13, small_count=0, normal_count=4,
    )["sample"]
    raw = mutated("verify-wrong-value.json", "failures", 0, "sample", value=sample)

    result = shape_correctness(raw, decoy=False)

    assert result.outcome == "wrong"
    assert [point.diff for point in result.failures[0].sample] == [4.0, 0.0, 0.0, 0.0]


def test_shape_mismatch_fixture_has_empty_sample():
    result = shape_correctness(load("verify-shape-mismatch.json"), decoy=False)
    assert result.outcome == "wrong"
    (failure,) = result.failures
    assert failure.kind == "shape_mismatch"
    assert failure.case_idx == 0
    assert failure.sample == []


def test_non_finite_fixture_maps_to_sentinels_and_contract_order():
    result = shape_correctness(load("verify-non-finite-mismatch.json"), decoy=False)
    assert result.outcome == "wrong"
    (failure,) = result.failures
    assert failure.kind == "non_finite_mismatch"
    # The engine ranks non-finite diffs first; the stable contract ranks finite
    # abs(diff) descending first, then sentinels lexically.
    assert [point.diff for point in failure.sample] == [1.5, "inf"]
    assert failure.sample[0].candidate == 2.5
    assert failure.sample[1].candidate == "inf"


def test_decoy_fixture_shapes_wrong_with_full_credit_counts():
    result = shape_correctness(load("verify-decoy.json"), decoy=True)
    assert result.outcome == "wrong"
    assert result.passed is False
    assert result.decoy is True
    assert result.passed_cases == 4
    assert result.total_cases == 4
    assert result.failures == []
    assert result.domain_failure is None


def test_candidate_compile_error_fixture():
    result = shape_correctness(load("verify-candidate-compile-error.json"), decoy=False)
    assert result.outcome == "compile_error"
    assert result.passed is False
    assert result.failures == []
    assert result.domain_failure is not None
    assert result.domain_failure.code == "SUBJECT_COMPILE_ERROR"
    assert result.domain_failure.phase == "compile"
    assert result.domain_failure.case_idx is None
    assert result.domain_failure.message == "candidate module import failed: SyntaxError"


def test_candidate_initialize_error_fixture():
    result = shape_correctness(load("verify-candidate-initialize-error.json"), decoy=False)
    assert result.outcome == "runtime_error"
    assert result.passed_cases == 1
    assert result.failures == []
    failure = result.domain_failure
    assert failure is not None
    assert failure.code == "SUBJECT_RUNTIME_ERROR"
    assert failure.phase == "initialize"
    assert failure.case_idx is None


def test_candidate_execute_error_fixture_keeps_earlier_mismatch():
    result = shape_correctness(load("verify-candidate-execute-error.json"), decoy=False)
    assert result.outcome == "runtime_error"
    assert result.passed_cases == 1
    assert result.total_cases == 4
    failure = result.domain_failure
    assert failure is not None
    assert failure.code == "SUBJECT_RUNTIME_ERROR"
    assert failure.phase == "execute"
    assert failure.case_idx == 1
    (mismatch,) = result.failures
    assert mismatch.case_idx == 0
    assert mismatch.kind == "value_mismatch"


def test_candidate_execution_timeout_fixture():
    result = shape_correctness(load("verify-candidate-execution-timeout.json"), decoy=False)
    assert result.outcome == "execution_timeout"
    assert result.passed is False
    failure = result.domain_failure
    assert failure is not None
    assert failure.code == "SUBJECT_EXECUTION_TIMEOUT"
    assert failure.phase == "execute"
    assert failure.case_idx == 0


def test_ambiguous_process_timeout_is_never_a_subject_timeout():
    with pytest.raises(UnattributedFailure) as raised:
        shape_correctness(load("verify-ambiguous-timeout.json"), decoy=False)
    # The detail is carried redacted: single line, no raw paths, bounded.
    assert isinstance(raised.value, ShapingError)
    assert "\n" not in raised.value.detail
    assert "/home" not in raised.value.detail
    assert len(raised.value.detail.encode("utf-8")) <= 512


def test_non_bool_decoy_flag_is_rejected():
    with pytest.raises(ShapingError):
        shape_correctness(load("verify-correct.json"), decoy="false")


def test_decoy_raw_requires_trusted_count():
    with pytest.raises(ShapingError):
        shape_correctness({"decoy": True}, decoy=True)
    with pytest.raises(ShapingError):
        shape_correctness({"decoy": True, "total_cases": 0}, decoy=True)


# ---------------------------------------------------------------------------
# Correctness rejection tests: the private parser fails closed
# ---------------------------------------------------------------------------


def test_correctness_rejects_extra_top_level_key():
    raw = mutated("verify-correct.json", "unexpected", value=True)
    with pytest.raises(ShapingError):
        shape_correctness(raw, decoy=False)


def test_correctness_rejects_extra_failure_key():
    raw = mutated("verify-wrong-value.json", "failures", 0, "extra", value=1)
    with pytest.raises(ShapingError):
        shape_correctness(raw, decoy=False)


def test_correctness_rejects_missing_case_identity():
    raw = without("verify-wrong-value.json", "failures", 0, "case_idx")
    with pytest.raises(ShapingError):
        shape_correctness(raw, decoy=False)


def test_correctness_rejects_duplicate_case():
    raw = load("verify-wrong-value.json")
    raw["failures"].append(copy.deepcopy(raw["failures"][0]))
    raw["failed_cases"] = 2
    raw["total_cases"] = 3
    with pytest.raises(ShapingError):
        shape_correctness(raw, decoy=False)


def test_correctness_rejects_contradictory_counts():
    raw = mutated("verify-correct.json", "failed_cases", value=1)
    with pytest.raises(ShapingError):
        shape_correctness(raw, decoy=False)


def test_correctness_rejects_ordinary_nan_and_inf_in_samples():
    raw = mutated("verify-wrong-value.json", "failures", 0, "sample", 0, "diff", value=float("nan"))
    with pytest.raises(ShapingError):
        shape_correctness(raw, decoy=False)
    raw = mutated(
        "verify-wrong-value.json",
        "failures",
        0,
        "sample",
        0,
        "reference",
        value=float("inf"),
    )
    with pytest.raises(ShapingError):
        shape_correctness(raw, decoy=False)


def test_correctness_rejects_more_than_eight_samples():
    raw = load("verify-wrong-value.json")
    base = raw["failures"][0]["sample"][0]
    raw["failures"][0]["sample"] = [{**base, "idx": [0, 8 - i], "diff": 10.0 - i} for i in range(9)]
    with pytest.raises(ShapingError):
        shape_correctness(raw, decoy=False)


def test_correctness_rejects_unsorted_samples():
    raw = load("verify-wrong-value.json")
    sample = raw["failures"][0]["sample"]
    sample[0], sample[1] = sample[1], sample[0]
    with pytest.raises(ShapingError):
        shape_correctness(raw, decoy=False)


def test_correctness_rejects_unknown_divergence_kind():
    raw = mutated("verify-wrong-value.json", "failures", 0, "kind", value="per_axis_v9")
    with pytest.raises(ShapingError):
        shape_correctness(raw, decoy=False)


def test_correctness_rejects_traceback_in_attribution_message():
    raw = mutated(
        "verify-candidate-execute-error.json",
        "failures",
        1,
        "candidate_error",
        "message",
        value='Traceback (most recent call last):\n  File "/tmp/x.py", line 1',
    )
    with pytest.raises(ShapingError):
        shape_correctness(raw, decoy=False)


def test_correctness_rejects_absolute_path_in_attribution_message():
    raw = mutated(
        "verify-candidate-execute-error.json",
        "failures",
        1,
        "candidate_error",
        "message",
        value="candidate failed at /tmp/eval/work/addmm_triton_ascend_impl.py",
    )
    with pytest.raises(ShapingError):
        shape_correctness(raw, decoy=False)


def test_correctness_rejects_unknown_attribution_phase():
    raw = mutated(
        "verify-candidate-execute-error.json",
        "failures",
        1,
        "candidate_error",
        "phase",
        value="reference",
    )
    with pytest.raises(ShapingError):
        shape_correctness(raw, decoy=False)


# ---------------------------------------------------------------------------
# Performance: per-case measurements -> geometric_mean_v1 aggregate
# ---------------------------------------------------------------------------


def test_benchmark_mixed_coverage_computes_per_case_geometric_mean():
    result = shape_performance(load("benchmark-mixed-coverage.json"))
    assert result.aggregation == "geometric_mean_v1"
    assert result.selected_cases == 3
    assert result.measured_cases == 2
    assert result.coverage == pytest.approx(2 / 3)
    # speedups 2.0 and 0.5 -> geometric mean exactly 1.0; the engine aggregate
    # (0.8) must never be trusted.
    assert result.overall_speedup_vs_reference == pytest.approx(1.0)
    first, second = result.per_case
    assert first.case_idx == 0
    assert first.reference_latency_ms == 1.0
    assert first.implementation_latency_ms == 0.5
    assert first.speedup_vs_reference == pytest.approx(2.0)
    assert first.input_shapes[0].dimensions == [1024, 512]
    assert first.input_shapes[0].dtype == "torch.float16"
    assert first.input_shapes[0].input_index == 0
    assert second.speedup_vs_reference == pytest.approx(0.5)
    (unmeasured,) = result.unmeasured_cases
    assert unmeasured.case_idx == 2
    assert unmeasured.reason_code == "NO_STABLE_MEASUREMENT"
    assert "throughput" not in result.model_dump()


def test_benchmark_rejects_invalid_measurement():
    with pytest.raises(ShapingError):
        shape_performance(load("benchmark-invalid-measurement.json"))


def test_benchmark_rejects_extra_key():
    raw = mutated("benchmark-mixed-coverage.json", "surprise", value=1)
    with pytest.raises(ShapingError):
        shape_performance(raw)


def test_benchmark_rejects_compressed_string_input_shape():
    raw = mutated(
        "benchmark-mixed-coverage.json",
        "per_shape_results",
        0,
        "input_desc",
        0,
        "shape",
        value="[1024,512]",
    )
    with pytest.raises(ShapingError):
        shape_performance(raw)


def test_benchmark_rejects_missing_case_identity():
    raw = without("benchmark-mixed-coverage.json", "per_shape_results", 0, "case_idx")
    with pytest.raises(ShapingError):
        shape_performance(raw)


def test_benchmark_rejects_duplicate_case():
    raw = load("benchmark-mixed-coverage.json")
    raw["per_shape_results"][1]["case_idx"] = 1
    with pytest.raises(ShapingError):
        shape_performance(raw)


def test_benchmark_rejects_contradictory_counts():
    raw = mutated("benchmark-mixed-coverage.json", "passed_cases", value=3)
    with pytest.raises(ShapingError):
        shape_performance(raw)


def test_benchmark_rejects_non_finite_latency():
    raw = mutated(
        "benchmark-mixed-coverage.json",
        "per_shape_results",
        0,
        "framework",
        "avg_latency_ms",
        value=float("inf"),
    )
    with pytest.raises(ShapingError):
        shape_performance(raw)


def test_benchmark_rejects_no_valid_measurement():
    raw = load("benchmark-mixed-coverage.json")
    for case in raw["per_shape_results"]:
        case["status"] = "fail"
        case["framework"] = None
        case["implementation"] = None
        case["speedup_vs_torch"] = None
        case["error_type"] = "RuntimeError"
        case["error_msg"] = "crashed"
    raw["passed_cases"] = 0
    raw["failed_cases"] = 3
    with pytest.raises(ShapingError):
        shape_performance(raw)


# ---------------------------------------------------------------------------
# Profile: mode-discriminated union with count invariants
# ---------------------------------------------------------------------------


def test_profile_op_complete_shapes_kernels_and_counts():
    result = shape_profile(load("profile-op-complete.json"))
    assert isinstance(result, OpProfileResult)
    assert result.counts.selected == 1
    assert result.counts.complete == 1
    (case,) = result.cases
    assert case.status == "complete"
    assert case.error is None
    assert [kernel.op_name for kernel in case.kernels] == ["addmm_kernel", "bias_add"]
    assert case.kernels[0].time_ms == pytest.approx(0.4)
    assert case.kernels[0].percentage == pytest.approx(80.0)
    assert case.kernels[1].percentage == pytest.approx(20.0)


def test_profile_op_kernel_metrics_normalize_scales_and_drop_invalid():
    result = shape_profile(load("profile-op-complete.json"))
    (case,) = result.cases
    dominant, secondary = case.kernels
    # Pipe ratios are captured as fractions and pass through unchanged.
    assert dominant.metrics.aic_cube_ratio == pytest.approx(0.72)
    # Memory/L2 columns are percentages and are normalized to fractions.
    assert dominant.metrics.gm_to_ub_usage == pytest.approx(0.55)
    assert dominant.metrics.ub_to_gm_usage == pytest.approx(0.18)
    assert dominant.metrics.aiv_l2_hit == pytest.approx(0.41)
    # Unparsable and out-of-range values stay None instead of failing the case.
    assert dominant.metrics.aic_l2_hit is None
    assert secondary.metrics.aiv_vec_ratio is None
    assert secondary.metrics.aiv_mte2_ratio is None


def test_profile_op_partial_and_failed_cases_carry_typed_errors():
    result = shape_profile(load("profile-op-partial.json"))
    assert isinstance(result, OpProfileResult)
    assert result.counts.partial == 1
    assert result.counts.failed == 1
    partial, failed = result.cases
    assert partial.status == "partial"
    assert partial.error is not None
    assert partial.error.check == "profile"
    assert partial.kernels[0].op_name == "addmm_kernel"
    assert failed.status == "failed"
    assert failed.error is not None
    assert failed.error.code == "PROFILE_CASE_FAILED"


def test_profile_deep_partial_normalizes_ratios_and_bound():
    result = shape_profile(load("profile-deep-partial.json"))
    assert isinstance(result, DeepProfileResult)
    complete, partial = result.cases
    assert complete.status == "complete"
    assert complete.bound == "CUBE"
    assert complete.ratios.cube == 0.75
    assert complete.ratios.vec == 0.05  # PUSHQ is the vector-issue ratio
    assert complete.ratios.scalar == 0.03  # max(SCALAR, SCALARLDST)
    assert complete.per_core_spread_ratio == pytest.approx(1 / 3)
    assert complete.recommendations == [
        "Increase useful Cube work per launch and validate tiling with measured latency."
    ]
    assert partial.status == "partial"
    assert partial.bound is None
    assert partial.ratios.mte2 is None
    assert partial.missing_metrics == ["PipeUtilization.CUBE"]


def test_profile_rejects_invalid_counts():
    with pytest.raises(ShapingError):
        shape_profile(load("profile-invalid-counts.json"))


def test_profile_rejects_extra_key():
    raw = mutated("profile-op-complete.json", "debug", value={})
    with pytest.raises(ShapingError):
        shape_profile(raw)


def test_profile_rejects_duplicate_case():
    raw = load("profile-op-partial.json")
    raw["per_case_results"][1]["case_idx"] = 0
    with pytest.raises(ShapingError):
        shape_profile(raw)


def test_profile_rejects_invalid_ratio():
    raw = mutated(
        "profile-deep-partial.json",
        "per_case_results",
        0,
        "bound",
        "ratios",
        "CUBE",
        value=1.5,
    )
    with pytest.raises(ShapingError):
        shape_profile(raw)
    raw = mutated(
        "profile-deep-partial.json",
        "per_case_results",
        0,
        "bound",
        "ratios",
        "CUBE",
        value=float("nan"),
    )
    with pytest.raises(ShapingError):
        shape_profile(raw)


def test_profile_rejects_unknown_bound_name():
    raw = mutated(
        "profile-deep-partial.json",
        "per_case_results",
        0,
        "bound",
        "name",
        value="HBM BOUND",
    )
    with pytest.raises(ShapingError):
        shape_profile(raw)


def test_profile_rejects_invalid_kernel_percentage_basis():
    raw = mutated(
        "profile-op-complete.json",
        "per_case_results",
        0,
        "kernels",
        0,
        "duration_us",
        value=0.0,
    )
    raw["per_case_results"][0]["kernels"][1]["duration_us"] = 0.0
    with pytest.raises(ShapingError):
        shape_profile(raw)


def test_profile_rejects_negative_kernel_duration():
    raw = mutated(
        "profile-op-complete.json",
        "per_case_results",
        0,
        "kernels",
        0,
        "duration_us",
        value=-1.0,
    )
    with pytest.raises(ShapingError):
        shape_profile(raw)


def test_profile_rejects_engine_timeout_envelope():
    raw = mutated("profile-op-complete.json", "timed_out", value=True)
    with pytest.raises(ShapingError):
        shape_profile(raw)


def test_profile_rejects_unknown_mode():
    raw = mutated("profile-op-complete.json", "profile_mode", value="flamegraph")
    with pytest.raises(ShapingError):
        shape_profile(raw)


# ---------------------------------------------------------------------------
# Stable outputs never carry engine-private content
# ---------------------------------------------------------------------------


def test_stable_outputs_never_leak_engine_private_content():
    results = [
        shape_correctness(load("verify-wrong-value.json"), decoy=False),
        shape_correctness(load("verify-candidate-execute-error.json"), decoy=False),
        shape_correctness(load("verify-non-finite-mismatch.json"), decoy=False),
        shape_performance(load("benchmark-mixed-coverage.json")),
        shape_profile(load("profile-op-partial.json")),
        shape_profile(load("profile-deep-partial.json")),
    ]
    forbidden_keys = {
        "mean_abs_err",
        "throughput",
        "error_msg",
        "error_type",
        "input_desc",
        "raw_metrics",
        "evidence",
        "divergence",
    }
    forbidden_kinds = {
        "sign_flip",
        "value_clamp",
        "constant_offset",
        "per_axis",
        "precision_loss",
        "scattered",
    }
    for result in results:
        dumped = result.model_dump(mode="json")
        strings = set(_walk_strings(dumped))
        assert not (strings & forbidden_keys)
        assert not (strings & forbidden_kinds)
        for text in strings:
            assert "Traceback" not in text


def test_finite_metric_values_stay_finite():
    # A non-finite MERE/max_abs_diff from the engine becomes null, never an
    # ordinary non-finite float in the stable contract.
    raw = mutated("verify-wrong-value.json", "failures", 0, "metrics", "MERE", value=float("nan"))
    result = shape_correctness(raw, decoy=False)
    assert result.failures[0].metrics.mere is None
    for value in result.failures[0].metrics.model_dump().values():
        assert not (isinstance(value, float) and not math.isfinite(value))


# ---------------------------------------------------------------------------
# Performance: measurement_method projection (event timing, v2.1)
# ---------------------------------------------------------------------------


def test_benchmark_event_measurement_method_projects_per_case_and_overall():
    result = shape_performance(load("benchmark-event-timing.json"))
    assert result.measurement_method == "npu_event_v1"
    first, second = result.per_case
    assert first.measurement_method == "npu_event_v1"
    assert second.measurement_method == "npu_event_v1"
    (unmeasured,) = result.unmeasured_cases
    assert unmeasured.reason_code == "NO_STABLE_MEASUREMENT"


def test_benchmark_without_measurement_method_projects_none():
    # Older engine JSON lacks the field entirely; every projection stays None.
    result = shape_performance(load("benchmark-mixed-coverage.json"))
    assert result.measurement_method is None
    assert all(case.measurement_method is None for case in result.per_case)


def test_benchmark_rejects_unknown_measurement_method():
    raw = mutated("benchmark-event-timing.json", "measurement_method", value="wall_clock_v1")
    with pytest.raises(ShapingError):
        shape_performance(raw)


def test_benchmark_mismatched_measurement_method_fails_closed():
    # Top-level says profiler while measured cases say event: the contract
    # consistency rule must surface as a shaping failure, not a silent pass.
    raw = mutated(
        "benchmark-event-timing.json", "measurement_method", value="profiler_kernel_sum_v1"
    )
    with pytest.raises(ShapingError):
        shape_performance(raw)


def test_benchmark_measurement_method_round_trips_through_evaluation_result():
    from triton_evaluation.api.contract import EvaluationResult, dump_contract

    performance = shape_performance(load("benchmark-event-timing.json"))
    result = EvaluationResult.model_validate(
        {
            "correctness": None,
            "performance": {
                "status": "succeeded",
                "data": performance.model_dump(mode="json", exclude_none=False),
                "error": None,
                "reason": None,
            },
            "profile": None,
            "provenance": {
                "schema_version": "2.1",
                "evaluator_build": "git:" + "a" * 40,
                "correctness_algorithm": "ca",
                "decoy_algorithm": "da",
                "benchmark_algorithm": "ba",
                "profiler_algorithm": "pa",
                "execution_environment": None,
                "grade_eligible": False,
                "manifest_id": None,
                "problem_id": None,
            },
        },
        context={"requested_checks": ["performance"]},
    )
    assert result.performance.data.measurement_method == "npu_event_v1"
    wire = json.loads(dump_contract(result))
    serialized = wire["performance"]["data"]
    assert serialized["measurement_method"] == "npu_event_v1"
    assert serialized["per_case"][0]["measurement_method"] == "npu_event_v1"
