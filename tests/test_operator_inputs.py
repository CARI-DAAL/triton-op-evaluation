"""Official input requests retain worker-only execution and durable identities."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from triton_evaluation.api.contract import MAX_SOURCE_BYTES
from triton_evaluation.operator_input import OperatorRequest, to_evaluation_request
from triton_evaluation.serialization import canonical_json_bytes, sha256_hex


def request(*, reference="def run(x): return x + 1", **overrides):
    return OperatorRequest.model_validate({
        "definition": {"inputs": {"x": {}}, "reference": reference},
        "triton_code": "def run(x): return x + 1",
        "workloads": [{"inputs": {"x": {"type": "random", "shape": [4], "dtype": "float32"}}}],
        **overrides,
    })


def test_minimal_request_and_durable_defaults():
    body = request()
    assert body.seed == 200
    assert body.checks == ["correctness", "performance"]
    durable = to_evaluation_request(body)
    assert durable.schema_version == "2.1"
    assert durable.purpose == "trial"
    assert durable.performance is not None
    assert durable.grade_context is None
    assert "triton_evaluation" not in durable.reference_code


def test_check_coupling_uses_final_endpoint_checks():
    profile = request(profile={"mode": "deep"})
    durable = to_evaluation_request(profile, checks=["profile"])
    assert durable.checks == ["profile"]
    assert durable.profile.mode == "deep"
    assert durable.performance is None
    with pytest.raises(ValueError, match="profile params"):
        to_evaluation_request(profile)
    with pytest.raises(ValueError, match="requires the correctness"):
        to_evaluation_request(request(checks=["performance"]))
    with pytest.raises(ValueError, match="duplicates"):
        to_evaluation_request(request(checks=["correctness", "correctness"]))


def test_generation_never_executes_or_compiles_caller_source(tmp_path):
    marker = tmp_path / "executed"
    source = f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise SystemExit(1)\n"
    to_evaluation_request(request(reference=source, triton_code=source))
    assert not marker.exists()
    converted = to_evaluation_request(request(reference="def (", triton_code="def ("))
    assert converted.reference_code.startswith("def (")
    assert converted.kernel_code.startswith("def (")


def test_generated_source_limit_is_an_explicit_input_error():
    with pytest.raises(ValueError, match="generated operator source"):
        to_evaluation_request(request(reference="#" * MAX_SOURCE_BYTES))


def test_workload_order_values_and_seed_are_part_of_durable_identity():
    cases = [{"inputs": {"x": {"type": "scalar", "value": n}}} for n in (1, 2)]
    def identity(body):
        return sha256_hex(canonical_json_bytes(to_evaluation_request(body).model_dump(mode="json")))
    body = request(workloads=cases)
    assert identity(body) == identity(OperatorRequest.model_validate_json(body.model_dump_json()))
    assert identity(body) != identity(request(workloads=list(reversed(cases))))
    assert identity(body) != identity(request(workloads=cases, seed=201))
    changed = request(workloads=[{"inputs": {"x": {"type": "scalar", "value": 3}}}])
    assert identity(body) != identity(changed)


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
