"""Typed operator inputs and pure, reproducible engine module generation."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from triton_evaluation.api.contract import MAX_SOURCE_BYTES
from triton_evaluation.api.operator import Case, OperatorRequest, TensorInput
from triton_evaluation.evaluators.operator import to_evaluation_request
from triton_evaluation.serialization import canonical_json_bytes, sha256_hex


def request(**overrides):
    return OperatorRequest.model_validate({
        "torch_code": "def run(x):\n    return x + 1\n",
        "triton_code": "def run(x):\n    return x + 1\n",
        "cases": [{"inputs": [{"shape": [4], "dtype": "float32"}]}],
        **overrides,
    })


def generated_modules(body):
    pytest.importorskip("torch")
    durable = to_evaluation_request(body)
    reference, candidate = {}, {}
    exec(durable.reference_code, reference)
    exec(durable.kernel_code, candidate)
    return reference, candidate


def test_minimal_request_and_durable_defaults():
    body = request()
    assert body.checks == ["correctness", "performance"]
    assert body.deadline_seconds == 900
    durable = to_evaluation_request(body)
    assert durable.schema_version == "2.1"
    assert durable.purpose == "trial"
    assert durable.op_name == "operator"
    assert durable.triton_impl_name == "triton_ascend_impl"
    assert durable.performance is not None
    assert durable.grade_context is None
    assert "triton_evaluation" not in durable.reference_code
    assert "official" not in durable.reference_code


@pytest.mark.parametrize("field", ["torch_code", "triton_code", "cases"])
def test_three_inputs_are_required(field):
    payload = request().model_dump()
    payload.pop(field)
    with pytest.raises(ValidationError):
        OperatorRequest.model_validate(payload)


@pytest.mark.parametrize("cases", [[], [{"inputs": []}] * 129])
def test_case_count_is_bounded(cases):
    with pytest.raises(ValidationError):
        request(cases=cases)


@pytest.mark.parametrize("overrides", [
    {"shape": [-1]}, {"shape": [True]}, {"shape": [1.0]}, {"shape": ["2"]},
    {"shape": [1] * 33}, {"dtype": "complex64"}, {"device": "npu"},
    {"range": [1]}, {"range": [0, 1, 2]}, {"range": [True, 2]},
    {"range": [0, "1"]}, {"range": [0, float("inf")]}, {"range": [0, float("nan")]},
    {"range": [2, 2]}, {"range": [2, 1]}, {"dtype": "bool", "range": [0, 1]},
    {"dtype": "int32", "range": [0.5, 2]},
    {"dtype": "int8", "range": [-129, 2]}, {"dtype": "uint8", "range": [0, 257]},
])
def test_tensor_specs_reject_invalid_or_coerced_values(overrides):
    with pytest.raises(ValidationError):
        TensorInput.model_validate({"shape": [2], **overrides})


@pytest.mark.parametrize("value", [[1, 2], {"arbitrary": "object"}, float("inf")])
def test_case_inputs_reject_nested_objects_and_nonfinite_scalars(value):
    with pytest.raises(ValidationError):
        Case.model_validate({"inputs": [value]})


@pytest.mark.parametrize("seed", [True, 1.5, "1", -(2**63) - 1, 2**64])
def test_seed_is_a_strict_supported_integer(seed):
    with pytest.raises(ValidationError):
        Case.model_validate({"inputs": [], "seed": seed})


def test_scalar_types_survive_validation():
    values = [1, 1.5, True, "text", None]
    case = Case.model_validate({"inputs": values})
    assert case.inputs == values
    assert [type(value) for value in case.inputs] == [int, float, bool, str, type(None)]
    assert case.seed == 0
    assert TensorInput(shape=[0, 2]).shape == [0, 2]
    assert TensorInput(shape=[]).dtype == "float32"
    assert TensorInput(shape=[1], dtype="uint8", range=[0.0, 256.0]).range == [0, 256]


def test_check_coupling_uses_final_endpoint_checks():
    profile = request(profile={"mode": "deep"})
    durable = to_evaluation_request(profile, checks=["profile"])
    assert durable.checks == ["profile"]
    assert durable.profile.mode == "deep"
    assert durable.performance is None
    with pytest.raises(ValidationError, match="profile params"):
        to_evaluation_request(profile)
    with pytest.raises(ValidationError, match="requires the correctness"):
        to_evaluation_request(request(checks=["performance"]))
    with pytest.raises(ValidationError, match="duplicates"):
        request(checks=["correctness", "correctness"])
    with pytest.raises(ValidationError):
        request(checks=["unknown"])
    durable = to_evaluation_request(
        request(checks=["profile", "correctness"], non_compute=True, deadline_seconds=42)
    )
    assert durable.checks == ["correctness", "profile"]
    assert durable.non_compute is True
    assert durable.deadline_seconds == 42


def test_generation_never_executes_or_compiles_caller_source(tmp_path):
    marker = tmp_path / "executed"
    source = f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise SystemExit(1)\n"
    to_evaluation_request(request(torch_code=source, triton_code=source))
    assert not marker.exists()
    # Invalid Python is a worker outcome, never a parsing step in admission.
    converted = to_evaluation_request(request(torch_code="def (", triton_code="def ("))
    assert converted.reference_code.startswith("def (")
    assert converted.kernel_code.startswith("def (")


def test_generated_source_limit_is_an_explicit_input_error():
    with pytest.raises(ValueError, match="generated operator source"):
        to_evaluation_request(request(torch_code="#" * MAX_SOURCE_BYTES))


def test_case_order_values_and_seeds_are_part_of_durable_identity():
    base = request(cases=[{"inputs": [1]}, {"inputs": [2]}])

    def identity(body):
        return sha256_hex(canonical_json_bytes(to_evaluation_request(body).model_dump(mode="json")))

    original = identity(base)
    assert original == identity(OperatorRequest.model_validate_json(base.model_dump_json()))
    for cases in [
        [{"inputs": [2]}, {"inputs": [1]}],
        [{"inputs": [1], "seed": 7}, {"inputs": [2]}],
        [{"inputs": [3]}, {"inputs": [2]}],
    ]:
        assert identity(request(cases=cases)) != original


def test_wrappers_call_run_with_the_same_positional_inputs():
    torch = pytest.importorskip("torch")
    source = (
        "def run(x, scale, label, enabled, absent):\n"
        "    return x * scale, label, enabled, absent\n"
    )
    reference, candidate = generated_modules(request(
        torch_code=source, triton_code=source,
        cases=[{"inputs": [{"shape": [3]}, 2, "tag", True, None]}],
    ))
    inputs = reference["get_inputs"]()
    expected = reference["Model"]()(*inputs)
    actual = candidate["ModelNew"]()(*inputs)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    assert actual[1:] == expected[1:] == ("tag", True, None)
    assert reference["get_init_inputs"]() == []


def test_inputs_are_lazy_repeatable_and_do_not_change_global_rng(monkeypatch):
    torch = pytest.importorskip("torch")
    reference, _ = generated_modules(request(cases=[
        {"inputs": [{"shape": [8]}], "seed": 41},
        {"inputs": [{"shape": [8]}], "seed": 42},
    ]))
    original_randn = torch.randn
    calls = []

    def tracked(*args, **kwargs):
        calls.append(args)
        return original_randn(*args, **kwargs)

    monkeypatch.setattr(torch, "randn", tracked)
    groups = reference["get_input_groups"]()
    assert len(groups) == 2
    assert calls == []
    before = torch.random.get_rng_state().clone()
    second = groups[1][0]
    first = groups[0][0]
    torch.testing.assert_close(groups[1][0], second, rtol=0, atol=0)
    assert not torch.equal(first, second)
    assert torch.equal(before, torch.random.get_rng_state())
    sequential = list(groups)
    torch.testing.assert_close(sequential[0][0], first, rtol=0, atol=0)
    torch.testing.assert_close(sequential[1][0], second, rtol=0, atol=0)
    with pytest.raises(IndexError):
        groups[2]
    assert len(groups[:1]) == 1


@pytest.mark.parametrize("dtype", [
    "float16", "bfloat16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "bool",
])
def test_default_tensor_generation_for_each_dtype(dtype):
    torch = pytest.importorskip("torch")
    reference, _ = generated_modules(request(cases=[{"inputs": [{"shape": [64], "dtype": dtype}]}]))
    tensor = reference["get_inputs"]()[0]
    assert tensor.dtype == getattr(torch, dtype)
    assert list(tensor.shape) == [64]
    assert tensor.device.type == "cpu"
    if dtype.startswith("int"):
        assert ((tensor >= -5) & (tensor < 5)).all()
    elif dtype == "uint8":
        assert ((tensor >= 0) & (tensor < 5)).all()
    elif dtype == "bool":
        assert tensor.any() and not tensor.all()
    else:
        assert torch.isfinite(tensor).all()


@pytest.mark.parametrize("dtype,bounds", [
    ("float16", [-0.3, 0.3]), ("bfloat16", [0.1, 0.1006]),
    ("float32", [-3, 2]), ("float64", [-3, 2]),
    ("int8", [-128, 128]), ("int16", [-10, 20]), ("int32", [5, 9]),
    ("uint8", [0, 256]), ("int64", [2**63 - 2, 2**63]), ("int64", [-(2**63), 2**63]),
])
def test_explicit_range_is_half_open_in_final_dtype(dtype, bounds):
    torch = pytest.importorskip("torch")
    reference, _ = generated_modules(request(cases=[{
        "inputs": [{"shape": [512], "dtype": dtype, "range": bounds}], "seed": 13,
    }]))
    values = reference["get_inputs"]()[0].tolist()
    assert all(bounds[0] <= value < bounds[1] for value in values)
    assert isinstance(values[0], float if dtype.startswith(("float", "bfloat")) else int)
    assert torch.isfinite(reference["get_inputs"]()[0]).all()


def test_empty_and_scalar_tensor_shapes():
    reference, _ = generated_modules(request(
        cases=[{"inputs": [{"shape": []}, {"shape": [0, 2]}]}]
    ))
    scalar, empty = reference["get_inputs"]()
    assert scalar.shape == ()
    assert empty.shape == (0, 2)


def test_candidate_wrapper_remains_visible_to_engine_decoy_check(tmp_path):
    source = '''import torch
import triton
import triton.language as tl

@triton.jit
def add(X, Y, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(Y + i, tl.load(X + i) + 1)

def run(x):
    result = torch.empty_like(x)
    add[(1,)](x, result, x.numel())
    return result
'''
    generated = to_evaluation_request(request(triton_code=source))
    candidate = tmp_path / "candidate.py"
    candidate.write_text(generated.kernel_code)
    validator = (
        Path(__file__).resolve().parents[1] / "triton_evaluation/engine/validate_triton_impl.py"
    )
    result = subprocess.run(
        [sys.executable, str(validator), str(candidate), "--json"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout[result.stdout.index("{"):])
    assert report["valid"] is True


def test_baseline_cache_defaults_preserve_durable_request_bytes():
    from triton_evaluation.api.contract import EvaluationRequest, dump_contract

    legacy = to_evaluation_request(request())
    original = dump_contract(legacy)
    assert b"use_baseline_cache" not in original
    recovered = EvaluationRequest.model_validate_json(original)
    assert recovered.performance.use_baseline_cache is True
    assert dump_contract(recovered) == original
    explicit = to_evaluation_request(request(performance={"use_baseline_cache": True}))
    assert dump_contract(explicit) == original
    fresh = to_evaluation_request(request(performance={"use_baseline_cache": False}))
    assert fresh.performance.use_baseline_cache is False
    assert dump_contract(fresh) != original
    assert json.loads(dump_contract(fresh))["performance"]["use_baseline_cache"] is False


@pytest.mark.parametrize("value", [0, 1, "true", "false", None])
def test_baseline_cache_requires_boolean(value):
    with pytest.raises(ValidationError):
        request(performance={"use_baseline_cache": value})


def test_baseline_cache_evidence_preserves_legacy_result_bytes():
    from triton_evaluation.api.contract import PerformanceCase, dump_contract

    case = PerformanceCase(
        case_idx=0, input_shapes=[], implementation_latency_ms=1.0,
        reference_latency_ms=2.0, speedup_vs_reference=2.0,
    )
    original = dump_contract(case)
    assert b"baseline_cache_hit" not in original
    assert dump_contract(PerformanceCase.model_validate_json(original)) == original
    for hit in (True, False):
        evidence = case.model_copy(update={"baseline_cache_hit": hit})
        assert json.loads(dump_contract(evidence))["baseline_cache_hit"] is hit
