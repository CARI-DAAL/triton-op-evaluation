"""Schema 2.0 strict request and error-envelope contract tests.

These tests pin the frozen Evaluation API 2.0 request boundary: strict scalar
types, no unknown fields, check/parameter/purpose coupling, UTF-8 byte limits,
and rejection of nonstandard JSON constants.
"""
import json

import pytest
from pydantic import ValidationError

from triton_evaluation.api.contract import (
    ErrorEnvelope,
    ErrorInfo,
    EvaluationRequest,
)

GRADE_CONTEXT = {
    "manifest_id": "release-1",
    "problem_id": "addmm-1",
    "reference_digest": "a" * 64,
}


def valid_request(**overrides):
    base = {
        "schema_version": "2.0",
        "op_name": "addmm",
        "reference_code": "import torch\n\n\nclass Model:\n    pass\n",
        "kernel_code": "import torch\n\n\nclass ModelNew:\n    pass\n",
        "checks": ["correctness"],
    }
    base.update(overrides)
    return base


def test_minimal_trial_request_uses_documented_defaults():
    request = EvaluationRequest.model_validate(valid_request())
    assert request.purpose == "trial"
    assert request.triton_impl_name == "triton_ascend_impl"
    assert request.deadline_seconds == 900
    assert request.non_compute is False
    assert request.performance is None
    assert request.profile is None
    assert request.grade_context is None


def test_requested_params_default_to_explicit_objects():
    request = EvaluationRequest.model_validate(
        valid_request(checks=["profile", "performance", "correctness"])
    )
    # check order normalizes to the fixed execution order
    assert request.checks == ["correctness", "performance", "profile"]
    assert request.performance is not None
    assert request.performance.warmup == 5
    assert request.performance.repeats == 50
    assert request.profile is not None
    assert request.profile.mode == "op"
    assert request.profile.case_indices is None
    assert request.profile.max_kernels_per_case == 200
    assert request.profile.include_recommendations is True


def test_grade_request_requires_and_accepts_grade_context():
    request = EvaluationRequest.model_validate(
        valid_request(purpose="grade", grade_context=GRADE_CONTEXT)
    )
    assert request.grade_context.manifest_id == "release-1"


@pytest.mark.parametrize(
    "patch",
    [
        # strict scalar types: no string/int coercion
        {"deadline_seconds": "900"},
        {"non_compute": 1},
        # performance without correctness is not executable
        {"checks": ["performance"]},
        # duplicate checks are rejected before normalization
        {"checks": ["correctness", "correctness"]},
        # parameters without the matching check are never silently discarded
        {"checks": ["correctness"], "performance": {"warmup": 1, "repeats": 2}},
        {"checks": ["correctness"], "profile": {"mode": "op"}},
        # purpose/context coupling
        {"purpose": "trial", "grade_context": GRADE_CONTEXT},
        {"purpose": "grade", "grade_context": None},
    ],
)
def test_request_rejects_invalid_contract_combinations(patch):
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(valid_request(**patch))


@pytest.mark.parametrize(
    "patch",
    [
        {"checks": []},
        {"checks": ["correctness", "speed"]},
        {"checks": "correctness"},
        {"checks": ["correctness"] * 2 + ["profile"], "profile": None},
        {"schema_version": "1.0"},
        {"op_name": ""},
        {"op_name": "x" * 129},
        {"op_name": 42},
        {"deadline_seconds": 9},
        {"deadline_seconds": 3601},
        {"triton_impl_name": ""},
        {"unknown_field": True},
        {"grade_context": GRADE_CONTEXT | {"extra": 1}, "purpose": "grade"},
        {"grade_context": {**GRADE_CONTEXT, "reference_digest": "A" * 64},
         "purpose": "grade"},
    ],
)
def test_request_rejects_malformed_shapes(patch):
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(valid_request(**patch))


def test_request_rejects_missing_required_fields():
    for field in ("schema_version", "op_name", "reference_code", "kernel_code", "checks"):
        payload = valid_request()
        del payload[field]
        with pytest.raises(ValidationError):
            EvaluationRequest.model_validate(payload)


def test_source_limits_are_utf8_bytes_not_code_points():
    two_mib = 2 * 1024 * 1024
    # exactly 2 MiB of 4-byte characters is accepted
    request = EvaluationRequest.model_validate(
        valid_request(reference_code="😀" * (two_mib // 4))
    )
    assert len(request.reference_code.encode("utf-8")) == two_mib
    # one code point over the byte limit is rejected
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(
            valid_request(reference_code="😀" * (two_mib // 4) + "x")
        )
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(valid_request(kernel_code="k" * (two_mib + 1)))


def test_profile_case_indices_are_unique_bounded_and_sorted():
    params = {"mode": "deep", "case_indices": [3, 1, 2]}
    request = EvaluationRequest.model_validate(
        valid_request(checks=["profile"], profile=params)
    )
    assert request.profile.case_indices == [1, 2, 3]

    request = EvaluationRequest.model_validate(
        valid_request(checks=["profile"], profile={"case_indices": list(range(128))})
    )
    assert len(request.profile.case_indices) == 128

    for bad in (list(range(129)), [0, 0], [-1], ["0"]):
        with pytest.raises(ValidationError):
            EvaluationRequest.model_validate(
                valid_request(checks=["profile"], profile={"case_indices": bad})
            )


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_request_json_rejects_nonstandard_constants(constant):
    text = json.dumps(valid_request()).replace(
        '"checks":', f'"padding": {constant}, "checks":', 1
    )
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate_json(text)


def test_finite_float_rejected_for_int_fields_even_when_integral():
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(valid_request(deadline_seconds=900.0))


def make_error_info(**overrides):
    base = {
        "code": "INVALID_REQUEST",
        "phase": "request",
        "check": None,
        "retryable": False,
        "message": "request failed validation",
    }
    base.update(overrides)
    return base


def test_error_envelope_round_trip():
    envelope = ErrorEnvelope.model_validate(
        {
            "schema_version": "2.0",
            "request_id": "rq_" + "0" * 32,
            "error": make_error_info(check="correctness"),
        }
    )
    back = ErrorEnvelope.model_validate_json(envelope.model_dump_json())
    assert back.error.code == "INVALID_REQUEST"
    assert back.error.check == "correctness"


@pytest.mark.parametrize(
    "patch",
    [
        {"code": "invalid_request"},  # lowercase is not a stable code
        {"code": ""},
        {"code": "X" * 65},
        {"phase": "domain"},  # domain outcomes are not ErrorInfo phases
        {"message": "two\nlines"},
        {"message": "carriage\rreturn"},
        {"message": ""},
        {"message": "😀" * 129},  # 516 UTF-8 bytes > 512
        {"retryable": "true"},
        {"unexpected": "field"},
    ],
)
def test_error_info_rejects_unstable_or_unredacted_shapes(patch):
    with pytest.raises(ValidationError):
        ErrorInfo.model_validate(make_error_info(**patch))


def test_error_info_message_allows_512_utf8_bytes():
    info = ErrorInfo.model_validate(make_error_info(message="😀" * 128))
    assert len(info.message.encode("utf-8")) == 512
