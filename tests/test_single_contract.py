"""Single implementations carry measurements without comparison claims."""

import copy
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from triton_evaluation.api.contract import (
    DeepProfileResult,
    EvaluationRequest,
    OpProfileResult,
    PerformanceOutcome,
    SinglePerformanceResult,
)
from triton_evaluation.api.operator import OperatorRequest
from triton_evaluation.evaluators.operator import to_evaluation_request
from triton_evaluation.serialization import canonical_json_bytes, sha256_hex

SOURCE = "def run(x):\n    return x + 1\n"
CASES = [{"inputs": [{"shape": [4], "dtype": "float32"}]}]


def operator(**values):
    return OperatorRequest.model_validate({"cases": CASES, **values})


@pytest.mark.parametrize("implementation", ["torch", "triton"])
def test_single_implementation_defaults_to_performance(implementation):
    body = operator(**{implementation + "_code": SOURCE})
    assert body.checks == ["performance"]
    durable = to_evaluation_request(body)
    assert durable.implementation == implementation
    assert durable.checks == ["performance"]
    assert durable.performance is not None
    assert durable.profile is None
    assert durable.grade_context is None
    serialized = durable.model_dump(mode="json")
    assert serialized["implementation"] == implementation
    assert EvaluationRequest.model_validate_json(durable.model_dump_json()) == durable


@pytest.mark.parametrize("implementation", ["torch", "triton"])
@pytest.mark.parametrize("checks", [["compile"], ["profile"], ["compile", "performance"]])
def test_single_checks_do_not_require_comparison(implementation, checks):
    body = operator(**{implementation + "_code": SOURCE}, checks=checks)
    durable = to_evaluation_request(body)
    assert durable.checks == checks
    assert (durable.profile is not None) == ("profile" in checks)


@pytest.mark.parametrize("implementation", ["torch", "triton"])
def test_single_implementation_cannot_request_correctness(implementation):
    with pytest.raises(ValidationError, match="both implementations"):
        operator(**{implementation + "_code": SOURCE}, checks=["correctness"])
    body = operator(**{implementation + "_code": SOURCE})
    with pytest.raises(ValidationError, match="both implementations"):
        to_evaluation_request(body, checks=["correctness"])


@pytest.mark.parametrize("values", [
    {}, {"torch_code": None, "triton_code": None}, {"torch_code": ""},
    {"triton_code": " \n\t"}, {"torch_code": SOURCE, "triton_code": " "},
])
def test_at_least_one_real_source_is_required_and_supplied_blanks_are_rejected(values):
    with pytest.raises(ValidationError):
        operator(**values)


@pytest.mark.parametrize("implementation", ["torch", "triton"])
def test_cases_still_required_for_single_code(implementation):
    with pytest.raises(ValidationError):
        OperatorRequest.model_validate({implementation + "_code": SOURCE})


def test_torch_only_generates_only_a_real_reference_model():
    pytest.importorskip("torch")
    source = "# preserve leading/trailing source whitespace \n" + SOURCE + "\n"
    durable = to_evaluation_request(operator(torch_code=source))
    assert durable.reference_code.startswith(source + "\n")
    assert durable.kernel_code == ""
    namespace = {}
    exec(durable.reference_code, namespace)
    inputs = namespace["get_inputs"]()
    assert namespace["Model"]()(*inputs).equal(inputs[0] + 1)


def test_triton_only_generates_inputs_without_an_invented_reference_model():
    pytest.importorskip("torch")
    durable = to_evaluation_request(operator(triton_code=SOURCE))
    reference, candidate = {}, {}
    exec(durable.reference_code, reference)
    exec(durable.kernel_code, candidate)
    assert "run" not in reference and "Model" not in reference
    assert reference["get_init_inputs"]() == []
    inputs = reference["get_input_groups"]()[0]
    assert candidate["ModelNew"]()(*inputs).equal(inputs[0] + 1)
    assert durable.kernel_code.startswith(SOURCE + "\n")


def test_comparison_generation_and_identity_preserve_previous_bytes():
    body = operator(torch_code=SOURCE, triton_code=SOURCE)
    assert body.checks == ["correctness", "performance"]
    durable = to_evaluation_request(body)
    assert durable.implementation == "comparison"
    assert "implementation" not in durable.model_dump(mode="json")
    # Computed with the converter at main 352d6d4, before single-mode support.
    assert sha256_hex(canonical_json_bytes(durable.model_dump(mode="json"))) == (
        "5392e3d2e29743b1d16120a06a60fe4befe38862e256229dc11796c8a09975e6"
    )


def test_historical_grade_request_decodes_without_new_identity_fields():
    path = Path(__file__).parent / "fixtures/evaluation_api/2.1/fixtures/request-grade-full.json"
    raw = path.read_bytes()
    request = EvaluationRequest.model_validate_json(raw)
    assert request.implementation == "comparison"
    assert canonical_json_bytes(request.model_dump(mode="json")) == raw
    with pytest.raises(ValidationError, match="grade records require"):
        EvaluationRequest.model_validate({
            **request.model_dump(mode="json"), "checks": ["performance"],
            "profile": None, "implementation": "torch",
        })


def performance(**overrides):
    return {
        "implementation": "torch", "correctness_verified": False,
        "measurement_method": "npu_event_v1", "selected_cases": 2,
        "measured_cases": 1, "coverage": 0.5,
        "per_case": [{
            "case_idx": 2, "input_shapes": [], "latency_ms": 0.25, "peak_memory_mb": 0.0,
        }],
        "unmeasured_cases": [{
            "case_idx": 5, "input_shapes": [], "reason_code": "UNSUPPORTED_CASE",
        }],
        **overrides,
    }


@pytest.mark.parametrize("implementation", ["torch", "triton"])
def test_single_measurement_success_is_a_performance_outcome(implementation):
    payload = performance(implementation=implementation)
    result = SinglePerformanceResult.model_validate(payload)
    parsed = TypeAdapter(PerformanceOutcome).validate_python({
        "status": "succeeded", "data": payload, "error": None, "reason": None,
    })
    assert parsed.data == result
    assert parsed.data.correctness_verified is False
    assert "speedup" not in parsed.model_dump_json()
    assert "reference_latency" not in parsed.model_dump_json()


@pytest.mark.parametrize("patch", [
    {"correctness_verified": True}, {"correctness_verified": 0},
    {"correctness_verified": None}, {"implementation": "comparison"},
    {"measurement_method": None}, {"measured_cases": 0}, {"measured_cases": 2},
    {"selected_cases": 3}, {"coverage": 1.0}, {"coverage": float("nan")},
    {"per_case": []}, {"overall_speedup_vs_reference": 1.0},
])
def test_single_measurement_rejects_unverified_or_inconsistent_claims(patch):
    with pytest.raises(ValidationError):
        SinglePerformanceResult.model_validate(performance(**patch))


@pytest.mark.parametrize("field", ["implementation", "correctness_verified", "measurement_method"])
def test_single_measurement_evidence_identity_is_required(field):
    payload = performance()
    payload.pop(field)
    with pytest.raises(ValidationError):
        SinglePerformanceResult.model_validate(payload)


@pytest.mark.parametrize("patch", [
    {"case_idx": 5}, {"latency_ms": 0}, {"latency_ms": float("inf")},
    {"peak_memory_mb": -1}, {"speedup_vs_reference": 1.0}, {"reference_latency_ms": 1.0},
])
def test_single_measurement_case_validates_actual_measurements(patch):
    payload = copy.deepcopy(performance())
    payload["per_case"][0].update(patch)
    with pytest.raises(ValidationError):
        SinglePerformanceResult.model_validate(payload)


@pytest.mark.parametrize("model,mode", [(OpProfileResult, "op"), (DeepProfileResult, "deep")])
def test_profile_single_identity_is_paired_and_comparison_fields_stay_absent(model, mode):
    payload = {
        "mode": mode, "counts": {"selected": 0, "complete": 0, "partial": 0, "failed": 0},
        "cases": [],
    }
    assert model.model_validate(payload).model_dump(mode="json") == payload
    single = model.model_validate({
        **payload, "implementation": "triton", "correctness_verified": False,
    })
    assert single.model_dump(mode="json")["correctness_verified"] is False
    for incomplete in [
        {"implementation": "torch"}, {"correctness_verified": False},
        {"implementation": "torch", "correctness_verified": True},
        {"implementation": "torch", "correctness_verified": 0},
    ]:
        with pytest.raises(ValidationError):
            model.model_validate({**payload, **incomplete})
