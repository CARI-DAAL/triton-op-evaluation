"""Explicit correctness coverage must never silently become whole-task success."""

import json
import sys
from datetime import datetime, timedelta, timezone
from types import ModuleType, SimpleNamespace

import pytest
from conftest import make_request
from pydantic import ValidationError

from triton_evaluation.api.contract import CorrectnessParams, CorrectnessResult, dump_contract
from triton_evaluation.api.operator import OperatorRequest
from triton_evaluation.engine import verify
from triton_evaluation.evaluators.base import ScriptOutcome
from triton_evaluation.evaluators.operator import to_evaluation_request
from triton_evaluation.evaluators.shaping import (
    ShapingError,
    UnattributedFailure,
    shape_correctness,
)
from triton_evaluation.evaluators.verify import VerifyEvaluator


def operator_request(**overrides):
    return OperatorRequest.model_validate({
        "definition": {"inputs": {"x": {}}, "reference": "def run(x): return x"},
        "workloads": [
            {"inputs": {"x": {"type": "random", "dtype": "float32", "shape": [i + 1]}}}
            for i in range(5)
        ],
        "triton_code": "def run(x): return x",
        "checks": ["correctness"],
        **overrides,
    })


@pytest.mark.parametrize("indices", [[], [-1], [True], [1, 1], [0.5], ["1"], list(range(129))])
def test_invalid_selection_is_rejected(indices):
    with pytest.raises(ValidationError):
        CorrectnessParams(case_indices=indices)


def test_selection_is_sorted_and_bound_to_workload_inventory():
    request = operator_request(correctness={"case_indices": [4, 1]})
    assert request.correctness.case_indices == [1, 4]
    durable = to_evaluation_request(request)
    assert durable.correctness == request.correctness
    assert to_evaluation_request(request, checks=["profile"]).correctness is None
    with pytest.raises(ValidationError, match="out of range"):
        operator_request(correctness={"case_indices": [5]})
    with pytest.raises(ValidationError, match="correctness check"):
        operator_request(checks=["profile"], correctness={"case_indices": [1]})


@pytest.mark.parametrize("params", [{"case_indices": [0]}, {"stop_on": "first_failure"}])
def test_grade_cannot_request_partial_correctness(params):
    with pytest.raises(ValidationError, match="grade correctness"):
        make_request(
            purpose="grade", correctness=params,
            grade_context={
                "manifest_id": "release", "problem_id": "op", "reference_digest": "a" * 64,
            },
        )


def test_default_request_and_result_bytes_omit_new_optional_fields():
    request = make_request()
    assert dump_contract(request) == dump_contract(make_request(correctness=None))
    assert "correctness" not in json.loads(dump_contract(request))
    result = shape_correctness({
        "op_name": "op", "total_cases": 2, "passed_cases": 2,
        "failed_cases": 0, "failures": [],
    }, decoy=False)
    assert result.model_dump(mode="json", exclude_none=False) == {
        "outcome": "correct", "passed": True, "decoy": False,
        "passed_cases": 2, "total_cases": 2, "failures": [], "domain_failure": None,
    }


@pytest.fixture
def engine_run(tmp_path, monkeypatch):
    def run(*, indices=None, stop_on="complete", failures=None, compile_error=False,
            truncated_after=None):
        request = operator_request()
        torch = ModuleType("torch")
        torch.Tensor = type("FakeTensor", (), {})
        torch.nn = SimpleNamespace(Module=object)
        torch.manual_seed = lambda seed: None
        torch.npu = SimpleNamespace(manual_seed=lambda seed: None)
        torch.device = lambda name: name
        monkeypatch.setitem(sys.modules, "torch", torch)
        monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
        reference = ModuleType("operator_torch")
        exec(to_evaluation_request(request).reference_code, reference.__dict__)
        generated, executed = [], []

        def inputs(case, **kwargs):
            index = case["inputs"]["x"]["shape"][0]
            generated.append(index)
            return [index]

        reference._operator_generate_inputs = inputs
        if truncated_after is not None:
            reference._OperatorInputGroups.__iter__ = lambda self: (
                self[index] for index in range(truncated_after)
            )

        def load(*args):
            if compile_error:
                raise verify._CandidateImportError(
                    reference, "SyntaxError: bad source", "SyntaxError",
                )
            return {"torch_module": reference}

        def attempt(modules, case_ctx, inputs, device, **kwargs):
            assert inputs == [case_ctx.case_idx]
            executed.append(case_ctx.case_idx)
            failure = (failures or {}).get(case_ctx.case_idx)
            return failure is None, failure

        monkeypatch.setattr(verify, "_load_verify_modules", load)
        monkeypatch.setattr(verify, "_ensure_npu_ambient_shared", lambda *a, **kw: True)
        monkeypatch.setattr(verify, "_try_run_case", attempt)
        output = tmp_path / "verify_result.json"
        counts = verify.verify_implementations(
            "operator", str(tmp_path), output_path=str(output),
            case_indices=indices, stop_on=stop_on,
        )
        return json.loads(output.read_text()), generated, executed, counts

    return run


def mismatch(case_idx):
    return verify._shape_mismatch_failure(case_idx, (3,), (4,))


def test_selected_pass_only_materializes_selected_workloads_and_is_incomplete(engine_run):
    raw, generated, executed, counts = engine_run(indices=[4, 2])
    assert generated == executed == [2, 4]
    assert counts == (2, 5)
    assert raw["failed_cases"] == 0
    result = shape_correctness(raw, decoy=False)
    assert result.outcome == "incomplete" and not result.passed
    assert result.total_cases == 5 and result.passed_cases == 2
    assert result.coverage.model_dump() == {
        "selected_case_indices": [1, 3], "evaluated_case_indices": [1, 3],
        "stop_reason": "complete",
    }


def test_first_failure_returns_counterexample_without_generating_later_cases(engine_run):
    raw, generated, executed, counts = engine_run(
        stop_on="first_failure", failures={2: mismatch(2)},
    )
    assert generated == executed == [1, 2]
    assert counts == (1, 5) and raw["failed_cases"] == 1
    result = shape_correctness(raw, decoy=False)
    assert result.outcome == "wrong"
    assert result.coverage.selected_case_indices == [0, 1, 2, 3, 4]
    assert result.coverage.evaluated_case_indices == [0, 1]
    assert result.coverage.stop_reason == "first_failure"


def test_first_failure_on_final_selected_case_keeps_complete_selected_coverage(engine_run):
    raw, generated, _, _ = engine_run(
        indices=[2, 5], stop_on="first_failure", failures={5: mismatch(5)},
    )
    assert generated == [2, 5]
    result = shape_correctness(raw, decoy=False)
    assert result.outcome == "wrong" and result.passed_cases == 1
    assert result.coverage.evaluated_case_indices == result.coverage.selected_case_indices == [1, 4]


def test_explicit_full_inventory_passing_can_still_be_correct(engine_run):
    raw, generated, _, _ = engine_run(indices=[1, 2, 3, 4, 5], stop_on="first_failure")
    assert generated == [1, 2, 3, 4, 5]
    result = shape_correctness(raw, decoy=False)
    assert result.outcome == "correct" and result.passed
    assert result.coverage.stop_reason == "complete"


def test_default_wrong_run_still_evaluates_every_case_and_has_no_coverage(engine_run):
    raw, generated, _, _ = engine_run(failures={1: mismatch(1)})
    assert generated == [1, 2, 3, 4, 5]
    assert "coverage" not in raw
    assert shape_correctness(raw, decoy=False).outcome == "wrong"


def test_selected_compile_error_needs_no_inputs_and_only_marks_selected_cases(engine_run):
    raw, generated, executed, counts = engine_run(indices=[2, 4], compile_error=True)
    assert generated == executed == []
    assert counts == (0, 5) and raw["failed_cases"] == 2
    assert [failure["case_idx"] for failure in raw["failures"]] == [2, 4]
    result = shape_correctness(raw, decoy=False)
    assert result.outcome == "compile_error"
    assert result.coverage.evaluated_case_indices == [1, 3]
    assert result.coverage.stop_reason == "candidate_error"


def test_selected_candidate_runtime_failure_does_not_mark_unrun_cases_failed(engine_run):
    failure = {
        "case_idx": 2, "error_type": "_CandidateExecuteError", "error_msg": "boom",
        "candidate_error": {"phase": "execute", "message": "candidate execute failed"},
    }
    raw, generated, _, _ = engine_run(indices=[2, 4], failures={2: failure})
    assert generated == [2] and raw["failed_cases"] == 1
    result = shape_correctness(raw, decoy=False)
    assert result.outcome == "runtime_error"
    assert result.coverage.evaluated_case_indices == [1]
    assert result.domain_failure.case_idx == 1


def test_unattributed_failure_is_not_a_counterexample(engine_run):
    failure = {"case_idx": 2, "error_type": "RuntimeError", "error_msg": "reference failed"}
    raw, generated, _, _ = engine_run(stop_on="first_failure", failures={2: failure})
    assert generated == [1, 2, 3, 4, 5]
    with pytest.raises(UnattributedFailure):
        shape_correctness(raw, decoy=False)


def test_exhausted_input_provider_reports_incomplete_evidence(engine_run):
    raw, generated, _, _ = engine_run(stop_on="first_failure", truncated_after=2)
    assert generated == [1, 2] and raw["failed_cases"] == 0
    result = shape_correctness(raw, decoy=False)
    assert result.outcome == "incomplete" and result.passed_cases == 2
    assert result.coverage.stop_reason == "input_exhausted"
    assert result.coverage.selected_case_indices == [0, 1, 2, 3, 4]
    assert result.coverage.evaluated_case_indices == [0, 1]


@pytest.mark.parametrize("indices", [[], [0], [2, 2], [True], [6]])
def test_engine_rejects_invalid_selection(engine_run, indices):
    with pytest.raises(ValueError, match="case.ind"):
        engine_run(indices=indices)


def partial_raw():
    return {
        "op_name": "operator", "total_cases": 5, "passed_cases": 1,
        "failed_cases": 1, "failures": [mismatch(4)],
        "coverage": {
            "selected_case_indices": [2, 4, 5], "evaluated_case_indices": [2, 4],
            "stop_reason": "first_failure",
        },
    }


@pytest.mark.parametrize("mutation", [
    lambda raw: raw["coverage"].update(evaluated_case_indices=[4]),
    lambda raw: raw["coverage"].update(selected_case_indices=[2, 4, 6]),
    lambda raw: raw["coverage"].update(stop_reason="complete"),
    lambda raw: raw.update(failed_cases=4),  # unrun cases cannot be labelled failed
    lambda raw: raw["failures"][0].update(case_idx=5),
    lambda raw: raw.pop("coverage"),  # old all-case accounting remains strict
])
def test_inconsistent_partial_evidence_fails_closed(mutation):
    raw = partial_raw()
    mutation(raw)
    with pytest.raises(ShapingError):
        shape_correctness(raw, decoy=False)


def test_partial_success_cannot_be_relabelled_global_correct():
    raw = partial_raw()
    raw.update(passed_cases=2, failed_cases=0, failures=[])
    raw["coverage"].update(stop_reason="input_exhausted")
    result = shape_correctness(raw, decoy=False)
    assert result.outcome == "incomplete"
    with pytest.raises(ValidationError, match="every case"):
        CorrectnessResult.model_validate({
            **result.model_dump(), "outcome": "correct", "passed": True,
        })


def test_adapter_runner_and_shaper_preserve_selection(tmp_path):
    request = to_evaluation_request(operator_request(correctness={
        "case_indices": [4, 1, 3], "stop_on": "first_failure",
    }))
    commands = []

    def runner(cmd, work_dir, *args, **kwargs):
        commands.append(cmd)
        (tmp_path / "verify_result.json").write_text(json.dumps(partial_raw()))
        return ScriptOutcome(1, "", "", False)

    raw = VerifyEvaluator("engine", runner=runner)._run_verify(
        request, tmp_path, 0, datetime.now(timezone.utc) + timedelta(minutes=1), lambda: True,
    )
    assert raw.status == "succeeded"
    cmd = commands[0]
    assert cmd[cmd.index("--case-indices") + 1] == "2,4,5"
    assert cmd[cmd.index("--stop-on") + 1] == "first_failure"
    assert shape_correctness(raw.raw, decoy=False).coverage.evaluated_case_indices == [1, 3]


@pytest.mark.parametrize("raw", [
    {"op_name": "op", "total_cases": 5, "passed_cases": 5, "failed_cases": 0, "failures": []},
    partial_raw(),  # asks for a different selected set
])
def test_runner_rejects_evidence_which_ignores_requested_selection(tmp_path, raw):
    def runner(*args, **kwargs):
        (tmp_path / "verify_result.json").write_text(json.dumps(raw))
        return ScriptOutcome(0, "", "", False)

    result = VerifyEvaluator("engine", runner=runner)._run_verify(
        make_request(correctness={"case_indices": [1]}), tmp_path, 0,
        datetime.now(timezone.utc) + timedelta(minutes=1), lambda: True,
    )
    assert result.status == "failed" and result.error.code == "INVALID_ENGINE_OUTPUT"
    assert result.error.retryable is False


def test_selected_decoy_has_no_executed_cases(tmp_path, monkeypatch):
    evaluator = VerifyEvaluator("engine")
    monkeypatch.setattr(evaluator, "_reference_case_count", lambda *args: 5)
    raw = evaluator._decoy_result(
        make_request(correctness={"case_indices": [1, 3]}), tmp_path, 0,
        datetime.now(timezone.utc) + timedelta(minutes=1), lambda: True,
    )
    result = shape_correctness(raw.raw, decoy=True)
    assert result.outcome == "wrong" and result.decoy
    assert result.passed_cases == 0
    assert result.coverage.selected_case_indices == [1, 3]
    assert result.coverage.evaluated_case_indices == []
    assert result.coverage.stop_reason == "decoy"


@pytest.mark.anyio
async def test_http_submission_retains_controls_in_durable_request(tmp_path, release_catalog_files):
    from test_api import build_app, seed_worker, trial_headers

    app = build_app(tmp_path, release_catalog_files)
    seed_worker(app)
    request = operator_request(correctness={"case_indices": [3, 1], "stop_on": "first_failure"})
    async with app.http as client:
        response = await client.post(
            "/verify", json=request.model_dump(mode="json"), headers=trial_headers(),
        )
    assert response.status_code == 202, response.text
    canonical = app.store._r.hget(
        f"{app.settings.redis_prefix}:resource:{response.json()['evaluation_id']}",
        "canonical_request",
    )
    assert json.loads(canonical)["correctness"] == {
        "case_indices": [1, 3], "stop_on": "first_failure",
    }


def test_partial_pass_cannot_authorize_performance(tmp_path):
    from test_evaluation_evaluator import (
        Harness,
        RecordingSink,
        StubCheck,
        run_evaluation,
    )

    from triton_evaluation.evaluators.base import EngineCheckResult

    raw = partial_raw()
    raw.update(passed_cases=2, failed_cases=0, failures=[])
    raw["coverage"].update(selected_case_indices=[2, 4], stop_reason="complete")
    benchmark = StubCheck(EngineCheckResult(status="succeeded", raw={}))
    harness = Harness(
        verify=StubCheck(EngineCheckResult(status="succeeded", raw=raw)), bench=benchmark,
    )
    result = run_evaluation(
        harness, ["correctness", "performance"], RecordingSink(), tmp_path,
    )
    assert result.correctness.data.outcome == "incomplete"
    assert result.performance.status == "skipped"
    assert result.performance.reason.code == "DEPENDENCY_NOT_PASSED"
    assert benchmark.calls == 0
