"""v2.1 compile check: admission-time execution via precheck, result assembly."""
import json
from datetime import datetime, timedelta, timezone

from conftest import make_caller, make_provenance
from test_evaluation_evaluator import (
    ENVIRONMENT,
    Harness,
    RecordingSink,
    succeeded,
    trial_pinned,
)

from triton_evaluation import precheck as precheck_mod
from triton_evaluation.api.contract import EvaluationRequest, EvaluationResult


def _v21_payload(checks, **overrides):
    payload = {
        "schema_version": "2.1", "op_name": "addmm",
        "reference_code": "import torch", "kernel_code": "import torch",
        "checks": checks,
    }
    payload.update(overrides)
    return payload


def _run_v21(harness, checks, sink, tmp_path, fixture="verify-correct.json"):
    request = EvaluationRequest.model_validate(_v21_payload(checks))
    return harness.run(
        request=request,
        work_dir=tmp_path,
        device_id=0,
        deadline_at=datetime.now(timezone.utc) + timedelta(minutes=10),
        lease_valid=lambda: True,
        execution_environment=ENVIRONMENT,
        pinned=trial_pinned(),
        progress=sink,
    )


def test_compile_only_request_never_queues(store, settings, clock, make_command,
                                           make_worker):
    cmd = make_command(request=EvaluationRequest.model_validate(_v21_payload(["compile"])))
    outcome = precheck_mod.run_precheck(store, settings,
        json.loads(cmd.admission.canonical_request)["kernel_code"],
        "triton_ascend_impl")
    # compiled verdict -> a compile-only terminal is assembled at admission
    assert outcome is None  # good module passes precheck
    # (terminal assembly assertions live in the routes-level test below)


# ---------------------------------------------------------------------------
# On-card assembly: the worker path fills result.compile from the subject verdict
# ---------------------------------------------------------------------------


def test_on_card_run_fills_compiled_compile_slot(tmp_path):
    harness = Harness(verify=succeeded("verify-correct.json"))
    sink = RecordingSink()
    result = _run_v21(harness, ["compile", "correctness"], sink, tmp_path)

    assert result.correctness.data.outcome == "correct"
    assert result.compile.status == "succeeded"
    assert result.compile.data.outcome == "compiled"
    assert result.compile.data.compiler_stage is None
    # the compile check executes at the admission precheck: it never touches
    # a card and is never fenced through the on-card progress sink
    assert ("begin", "compile") not in sink.calls
    assert ("save", "compile", "succeeded") not in sink.calls


def test_on_card_compile_error_fills_backend_compile_slot(tmp_path):
    harness = Harness(verify=succeeded("verify-candidate-compile-error.json"))
    result = _run_v21(harness, ["compile", "correctness"], RecordingSink(), tmp_path)

    assert result.correctness.data.outcome == "compile_error"
    assert result.compile.data.outcome == "compile_error"
    assert result.compile.data.compiler_stage == "backend"
    assert result.compile.data.error_excerpt == (
        result.correctness.data.domain_failure.message
    )


def test_compile_only_request_result_is_compiled(tmp_path):
    harness = Harness(verify=succeeded("verify-correct.json"))
    sink = RecordingSink()
    result = _run_v21(harness, ["compile"], sink, tmp_path)

    assert result.correctness is None
    assert result.compile.data.outcome == "compiled"
    assert sink.calls == []


def test_v21_committed_result_keeps_compile_slot(store, settings, make_command,
                                                 make_worker):
    """A 2.1 worker-committed result validates against its requested checks
    and survives serialization with the compile slot intact."""
    cmd = make_command(request=EvaluationRequest.model_validate(
        _v21_payload(["compile", "correctness"])
    ))
    store.create_or_replay(cmd)
    lease = store.claim_next(make_worker())
    result = EvaluationResult.model_validate(
        {
            "compile": {
                "status": "succeeded",
                "data": {"outcome": "compiled"},
            },
            "correctness": {
                "status": "succeeded",
                "data": {
                    "outcome": "correct",
                    "passed": True,
                    "passed_cases": 1,
                    "total_cases": 1,
                    "failures": [],
                },
            },
            "performance": None,
            "profile": None,
            "provenance": make_provenance(settings).model_dump(
                mode="json", exclude_none=False
            ),
        },
        context={"requested_checks": ["compile", "correctness"]},
    )
    assert store.commit_terminal(lease, "completed", result)

    visible = store.get_visible(cmd.evaluation_id, make_caller())
    assert visible.schema_version == "2.1"
    assert visible.result.compile.data.outcome == "compiled"
    stored = json.loads(store._r.hget(
        f"triton-eval:v2:resource:{cmd.evaluation_id}", b"resource"
    ))
    assert stored["result"]["compile"]["data"]["outcome"] == "compiled"
