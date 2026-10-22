"""Exercise actual evaluator checkpoint callbacks before process termination."""

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from test_evaluation_evaluator import ENVIRONMENT, succeeded, trial_pinned
from test_shaping import load

from triton_evaluation.api.contract import EvaluationRequest
from triton_evaluation.evaluators.base import LeaseLostError, ScriptOutcome
from triton_evaluation.evaluators.benchmark import BenchmarkEvaluator
from triton_evaluation.evaluators.evaluation import EvaluationEvaluator


def checkpoint_raw(*, empty=False):
    raw = load("benchmark-event-timing.json")
    raw.update(
        total_cases=2, passed_cases=0 if empty else 1, failed_cases=0,
        case_indices=[1, 3], per_shape_results=[] if empty else raw["per_shape_results"][:1],
        progress={
            "selected_case_indices": [0, 2],
            "completed_case_indices": [] if empty else [0],
            "complete": False,
        },
    )
    return raw


class Sink:
    def __init__(self):
        self.saved = {}
        self.checkpoints = []

    def begin(self, check):
        return True

    def save(self, check, outcome):
        self.saved[check] = outcome
        if check == "performance":
            self.checkpoints.append(outcome)
        return True


def request(joint=False):
    return EvaluationRequest(
        schema_version="2.1", op_name="addmm", reference_code="reference", kernel_code="kernel",
        checks=(
            ["correctness", "performance", "profile"] if joint else ["correctness", "performance"]
        ),
        performance={"method": "event", "case_indices": [0, 2], "use_baseline_cache": False},
        profile={"mode": "trace"} if joint else None,
    )


@pytest.mark.parametrize("joint", [False, True], ids=["benchmark", "timing-trace"])
def test_checkpoint_is_published_while_running_and_returned_with_timeout_note(tmp_path, joint):
    sink = Sink()

    def execute(cmd, cwd, device, deadline, poll, **kwargs):
        path = Path(cmd[cmd.index("--output") + 1])
        path.write_text(json.dumps(checkpoint_raw(empty=True)))
        assert poll() and sink.checkpoints == []
        path.write_text(json.dumps(checkpoint_raw()))
        assert poll()
        saved = sink.saved["performance"]
        assert saved.data.complete is False
        assert saved.data.pending_case_indices == [2]
        assert saved.data.termination_reason is None
        assert saved.data.per_case[0].implementation_latency_ms == 0.5
        # A torn or malformed next snapshot cannot replace the saved numbers.
        path.write_text("{")
        assert poll()
        assert sink.saved["performance"] == saved
        return ScriptOutcome(-9, "", "", True, "budget")

    verify = succeeded("verify-correct.json")
    verify.result.raw.update(total_cases=3, passed_cases=3)
    evaluator = EvaluationEvaluator(
        verify=verify,
        benchmark=BenchmarkEvaluator("unused", runner=execute), profile=None,
    )
    result = evaluator.run(
        request(joint), tmp_path, 0, datetime.now(timezone.utc) + timedelta(hours=1),
        lambda: True, ENVIRONMENT, trial_pinned(), sink,
    )
    assert result.correctness.data.outcome == "correct"
    assert result.performance.status == "succeeded"
    data = result.performance.data
    assert data.termination_reason == "timeout" and data.complete is False
    assert data.coverage == 0.5 and data.measured_cases == 1 and data.selected_cases == 2
    assert data.pending_case_indices == [2]
    assert data.overall_speedup_vs_reference == 2.0
    assert not result.provenance.grade_eligible


def test_zero_completed_measurements_remains_unmeasured_not_a_zero_speedup(tmp_path):
    def execute(cmd, cwd, device, deadline, poll, **kwargs):
        Path(cmd[cmd.index("--output") + 1]).write_text(json.dumps(checkpoint_raw(empty=True)))
        assert poll()
        return ScriptOutcome(-9, "", "", True, "budget")

    saved = []
    result = BenchmarkEvaluator("unused", runner=execute).run(
        request(), tmp_path, 0, datetime.now(timezone.utc) + timedelta(hours=1),
        on_checkpoint=lambda value: saved.append(value),
    )
    assert saved == [] and result.status == "timeout" and result.raw is None


def test_fenced_worker_cannot_publish_a_later_snapshot(tmp_path):
    valid = True
    saved = []

    def execute(cmd, cwd, device, deadline, poll, **kwargs):
        nonlocal valid
        path = Path(cmd[cmd.index("--output") + 1])
        path.write_text(json.dumps(checkpoint_raw()))
        assert poll() and len(saved) == 1
        valid = False
        changed = copy.deepcopy(checkpoint_raw())
        changed["per_shape_results"][0]["implementation"]["avg_latency_ms"] = 0.25
        path.write_text(json.dumps(changed))
        assert poll() is False
        return ScriptOutcome(-9, "", "", False, "fenced")

    with pytest.raises(LeaseLostError):
        BenchmarkEvaluator("unused", runner=execute).run(
            request(), tmp_path, 0, datetime.now(timezone.utc) + timedelta(hours=1),
            lease_valid=lambda: valid, on_checkpoint=lambda result: saved.append(result),
        )
    assert len(saved) == 1
    assert saved[0].raw["per_shape_results"][0]["implementation"]["avg_latency_ms"] == 0.5
