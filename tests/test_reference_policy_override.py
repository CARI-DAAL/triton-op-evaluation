"""Per-request reference-device-policy override (trial correctness only).

A trial request may pin cpu-first correctness under an npu-strict deployment
for references that cannot execute on NPU verbatim (fp64 conv1d, CPU-mixing
internals). The override must flow to the verify engine, be refused for grade
records, and be pinned honestly into the result provenance so cross-policy
correctness reuse keeps failing the evidence validator.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from triton_evaluation.api.contract import EvaluationRequest, Provenance
from triton_evaluation.config import Settings
from triton_evaluation.operator_input import OperatorRequest, to_evaluation_request
from triton_evaluation.service.terminal_results import PinnedProvenance, complete_evaluation


def _definition() -> dict:
    return {
        "name": "op",
        "axes": {},
        "inputs": {"x": {"shape": ["8"], "dtype": "float32"}},
        "outputs": {},
        "reference": "def run(x):\n    return x\n",
    }


def _workloads() -> list[dict]:
    return [
        {
            "uuid": "case-1",
            "inputs": {
                "x": {
                    "type": "random", "shape": [8], "dtype": "float32",
                    "distribution": "uniform", "low": -1.0, "high": 1.0,
                }
            },
        }
    ]


def test_operator_request_accepts_policy_and_translates():
    request = OperatorRequest(
        definition=_definition(), workloads=_workloads(),
        triton_code="def run(x):\n    return x\n",
        reference_device_policy="cpu-first",
    )
    assert request.reference_device_policy == "cpu-first"
    translated = to_evaluation_request(request, checks=["correctness"])
    assert translated.reference_device_policy == "cpu-first"


def test_operator_request_default_policy_is_absent():
    request = OperatorRequest(
        definition=_definition(), workloads=_workloads(),
        triton_code="def run(x):\n    return x\n",
    )
    assert request.reference_device_policy is None
    translated = to_evaluation_request(request, checks=["correctness"])
    assert translated.reference_device_policy is None


def test_evaluation_request_refuses_policy_for_grade():
    kwargs = dict(
        schema_version="2.1", purpose="grade", op_name="operator",
        implementation="comparison",
        reference_code="def run(x):\n    return x\n",
        kernel_code="def run(x):\n    return x\n",
        checks=["correctness"], reference_device_policy="cpu-first",
    )
    with pytest.raises(ValidationError):
        EvaluationRequest(**kwargs)


def test_settings_correctness_algorithm_for_recomputes_suffix():
    settings = Settings(reference_device_policy="npu-strict")
    assert settings.effective_correctness_algorithm_version.endswith(":npu-strict-v1")
    cpu = settings.correctness_algorithm_for("cpu-first")
    assert not cpu.endswith(":npu-strict-v1")
    assert cpu.startswith(settings.correctness_algorithm_version)
    with pytest.raises(ValueError):
        settings.correctness_algorithm_for("nonsense")


def _pinned() -> PinnedProvenance:
    from triton_evaluation.service.models import AlgorithmVersions

    return PinnedProvenance(
        evaluator_build="git:" + "0" * 40,
        algorithms=AlgorithmVersions(
            correctness="correctness-v1:npu-strict-v1",
            decoy="decoy-v1",
            benchmark="geometric-mean-baseline-v2",
            profiler="profiler-v1",
        ),
        manifest_id=None,
        problem_id=None,
        purpose="trial",
    )


def test_complete_evaluation_pins_policy_provenance(tmp_path: Path):
    from triton_evaluation.api.contract import CheckSucceeded, CorrectnessResult

    correctness = CheckSucceeded(
        status="succeeded",
        data=CorrectnessResult(
            outcome="correct", passed=True, passed_cases=1, total_cases=1, failures=[]
        ),
    )
    result = complete_evaluation(
        requested_checks=["correctness"],
        outcomes={"correctness": correctness},
        pinned=_pinned(),
        correctness_algorithm_override="correctness-v1",
        reference_device_policy="cpu-first",
    )
    assert result.provenance.correctness_algorithm == "correctness-v1"
    assert result.provenance.reference_device_policy == "cpu-first"


def test_complete_evaluation_defaults_keep_pinned_provenance(tmp_path: Path):
    from triton_evaluation.api.contract import CheckSucceeded, CorrectnessResult

    correctness = CheckSucceeded(
        status="succeeded",
        data=CorrectnessResult(
            outcome="correct", passed=True, passed_cases=1, total_cases=1, failures=[]
        ),
    )
    result = complete_evaluation(
        requested_checks=["correctness"],
        outcomes={"correctness": correctness},
        pinned=_pinned(),
    )
    assert result.provenance.correctness_algorithm == "correctness-v1:npu-strict-v1"
    assert result.provenance.reference_device_policy is None


def test_provenance_accepts_missing_policy_for_old_records():
    provenance = Provenance(
        schema_version="2.1",
        evaluator_build="git:" + "0" * 40,
        correctness_algorithm="correctness-v1:npu-strict-v1",
        decoy_algorithm="decoy-v1",
        benchmark_algorithm="geometric-mean-baseline-v2",
        profiler_algorithm="profiler-v1",
        grade_eligible=False,
        manifest_id=None,
        problem_id=None,
        execution_environment=None,
    )
    assert provenance.reference_device_policy is None


def test_verify_engine_cmd_honours_request_policy(tmp_path, monkeypatch):
    import triton_evaluation.evaluators.verify as verify_mod

    seen = {}

    def fake_run_script(cmd, cwd, device_id, deadline_at=None, lease_valid=None, **kwargs):
        seen["cmd"] = cmd
        from triton_evaluation.evaluators.base import ScriptOutcome

        return ScriptOutcome(1, "", "", False)

    monkeypatch.setattr(verify_mod, "run_script", fake_run_script)
    monkeypatch.setattr(
        verify_mod, "read_json", lambda path, **_: None,
    )
    monkeypatch.setattr(
        verify_mod, "remaining_seconds", lambda deadline: 100.0,
    )
    evaluator = verify_mod.VerifyEvaluator("unused", reference_device_policy="npu-strict")
    request = EvaluationRequest(
        schema_version="2.1", purpose="trial", op_name="operator",
        implementation="comparison",
        reference_code="def run(x):\n    return x\n",
        kernel_code="def run(x):\n    return x\n",
        checks=["correctness"], reference_device_policy="cpu-first",
    )
    from datetime import datetime, timedelta, timezone

    evaluator._run_verify(
        request, tmp_path, 0, datetime.now(timezone.utc) + timedelta(seconds=100), None,
    )
    cmd = seen["cmd"]
    assert cmd[cmd.index("--reference-device-policy") + 1] == "cpu-first"
