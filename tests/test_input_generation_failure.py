"""A workload the pinned generator refuses is the request's error, never the card's."""

import json
from datetime import datetime, timedelta, timezone

import pytest
from conftest import make_request

from triton_evaluation.api.operator import OperatorRequest
from triton_evaluation.evaluators.base import ScriptOutcome, input_generation_failure
from triton_evaluation.evaluators.benchmark import BenchmarkEvaluator
from triton_evaluation.evaluators.operator import (
    INPUT_GENERATION_FAILURE_PREFIX,
    to_evaluation_request,
)
from triton_evaluation.evaluators.profile import ProfileEvaluator
from triton_evaluation.evaluators.verify import VerifyEvaluator
from triton_evaluation.runtime.input_generation import InputGenerationError, generate_inputs


def custom_case():
    inputs = {"x": {"type": "custom"}}
    return {"uuid": "c", "axes": {"N": 4}, "inputs": inputs, "input_order": ["x"]}


@pytest.mark.parametrize(
    "code",
    [
        "def build(axes, device):\n    return {'x': axes['missing']}",
        "def build(axes, device) return {}",
        "def build(axes, device):\n    return [1]",
        "def build(axes, device):\n    return {}",
        "build = 1",
    ],
)
def test_request_code_faults_are_request_errors(code):
    with pytest.raises(InputGenerationError):
        generate_inputs(custom_case(), 200, "cpu", code, "build")


@pytest.mark.parametrize("error", ["RuntimeError('device fault')", "MemoryError()"])
def test_runtime_and_memory_errors_keep_the_engine_failure_path(error):
    # torch reports argument errors and device faults alike as RuntimeError.
    code = f"def build(axes, device):\n    raise {error}"
    with pytest.raises((RuntimeError, MemoryError)) as raised:
        generate_inputs(custom_case(), 200, "cpu", code, "build")
    assert not isinstance(raised.value, InputGenerationError)


def test_negative_sigma_draw_is_a_request_error():
    spec = {
        "type": "random", "dtype": "float32", "shape": [4],
        "distribution": "normal", "sigma_low": -2.0, "sigma_high": -1.0,
    }
    case = {"uuid": "n", "axes": {}, "inputs": {"x": spec}, "input_order": ["x"]}
    with pytest.raises(InputGenerationError, match="sigma draw"):
        generate_inputs(case, 200, "cpu")


def reference_module(seed):
    x = {"type": "random", "dtype": "float32", "shape": [4]}
    request = OperatorRequest.model_validate(
        {
            "definition": {"inputs": {"x": {}}, "reference": "def run(x): return x"},
            # Below the default uniform low of -5: refused only by the uniform family.
            "workloads": [{"uuid": "ok", "inputs": {"x": x}},
                          {"uuid": "bad", "inputs": {"x": dict(x, high=-6.0)}}],
            "seed": seed,
        }
    )
    namespace = {}
    exec(to_evaluation_request(request).reference_code, namespace)
    generate = namespace["_operator_generate_inputs"]
    # The adapter always targets the NPU; materialize on CPU here.
    namespace["_operator_generate_inputs"] = (
        lambda case, **kw: generate(case, **kw | {"device": "cpu"})
    )
    return namespace


def test_reference_module_names_the_refused_workload(capsys):
    # Seed 0 draws the uniform case family, seed 200 the normal one.
    assert len(reference_module(200)["get_input_groups"]()[1]) == 1
    namespace = reference_module(0)
    groups = namespace["get_input_groups"]()
    assert len(groups[0]) == 1
    capsys.readouterr()
    with pytest.raises(namespace["_operator_input_error"]):
        groups[1]
    [line] = capsys.readouterr().err.splitlines()
    assert line.startswith(INPUT_GENERATION_FAILURE_PREFIX)
    failure = json.loads(line[len(INPUT_GENERATION_FAILURE_PREFIX):])
    assert failure["case_idx"] == 2 and failure["uuid"] == "bad"
    assert "uniform needs low <= high" in failure["message"]


def test_other_generation_errors_are_not_marked(capsys):
    namespace = reference_module(0)

    def device_fault(case, **_):
        raise RuntimeError("device fault")

    namespace["_operator_generate_inputs"] = device_fault
    with pytest.raises(RuntimeError, match="device fault"):
        namespace["get_input_groups"]()[0]
    assert INPUT_GENERATION_FAILURE_PREFIX not in capsys.readouterr().err


def marker(message="uniform needs low <= high, got -5.0 > -6.0"):
    failure = {"case_idx": 2, "uuid": "bad", "message": message}
    return INPUT_GENERATION_FAILURE_PREFIX + json.dumps(failure)


def failed_run(stderr):
    def run(cmd, *args, **kwargs):
        return ScriptOutcome(exit_code=1, stdout="", stderr=stderr, timed_out=False)

    return run


def test_marker_message_is_one_bounded_line():
    outcome = ScriptOutcome(1, "", marker("multi\nline " + "é" * 600) + "\nTraceback\n", False)
    error = input_generation_failure("correctness", outcome).error
    assert error.code == "INPUT_GENERATION_FAILED" and error.retryable is False
    assert "\n" not in error.message and len(error.message.encode()) <= 512
    assert error.message.startswith("workload bad (case 2) inputs cannot be generated: multi line")
    torn = ScriptOutcome(1, "", INPUT_GENERATION_FAILURE_PREFIX + "{not json\n", False)
    assert input_generation_failure("correctness", torn) is None


@pytest.mark.parametrize(
    ("evaluator", "check"),
    [
        (lambda run: VerifyEvaluator("engine", runner=run), "correctness"),
        (lambda run: BenchmarkEvaluator("engine", runner=run), "performance"),
        (lambda run: ProfileEvaluator("engine", runner=run), "profile"),
    ],
)
def test_evaluators_report_refused_workloads_as_request_errors(
    evaluator, check, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)  # engine crash evidence lands in ./logs
    # Comparison performance requires correctness; each evaluator runs only its own check.
    request = make_request(checks=sorted({"correctness", check}))
    deadline = datetime.now(timezone.utc) + timedelta(minutes=5)
    stderr = "Traceback (most recent call last):\n" + marker() + "\nInputGenerationError: x\n"
    refused = evaluator(failed_run(stderr)).run(request, tmp_path / "a", 0, deadline)
    assert refused.status == "failed" and refused.error.check == check
    assert refused.error.code == "INPUT_GENERATION_FAILED" and refused.error.retryable is False
    assert "workload bad (case 2)" in refused.error.message
    crashed = evaluator(failed_run("RuntimeError: device fault\n")).run(
        request, tmp_path / "b", 0, deadline
    )
    assert crashed.error.code == "EVALUATOR_EXECUTION_FAILED" and crashed.error.retryable
