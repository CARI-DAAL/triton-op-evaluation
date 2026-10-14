"""Profile evidence through the real runner, shaping and fenced orchestration."""

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from triton_evaluation.api.contract import EvaluationRequest, ProfileParams
from triton_evaluation.evaluators import profile as profile_module
from triton_evaluation.evaluators.base import ScriptOutcome
from triton_evaluation.evaluators.benchmark import BenchmarkEvaluator
from triton_evaluation.evaluators.evaluation import EvaluationEvaluator
from triton_evaluation.evaluators.profile import ProfileEvaluator
from triton_evaluation.evaluators.shaping import ShapingError, shape_profile_runs
from triton_evaluation.evaluators.verify import VerifyEvaluator
from triton_evaluation.service.models import AlgorithmVersions, PinnedProvenance

FIXTURES = Path(__file__).parent / "fixtures" / "engine"


def envelope(index=0, *, mode="op", status="complete", total=3):
    filename = "profile-op-complete.json" if mode == "op" else "profile-deep-partial.json"
    raw = json.loads((FIXTURES / filename).read_text())
    case = raw["per_case_results"][0]
    case.update(case_idx=index, status=status)
    if status != "complete":
        case["error"] = "profile capture incomplete"
    raw.update(
        requested_case_idx=index,
        total_cases=total,
        selected_cases=1,
        completed_cases=int(status == "complete"),
        partial_cases=int(status == "partial"),
        failed_cases=int(status == "failed"),
        per_case_results=[case],
    )
    return raw


def batch_envelope(indices, *, mode="op", statuses=None, total=3):
    statuses = statuses or ["complete"] * len(indices)
    cases = [
        envelope(index, mode=mode, status=status, total=total)["per_case_results"][0]
        for index, status in zip(indices, statuses, strict=True)
    ]
    raw = envelope(mode=mode, total=total)
    raw.update(
        requested_case_idx=None,
        requested_case_indices=list(indices),
        selected_cases=len(indices),
        completed_cases=statuses.count("complete"),
        partial_cases=statuses.count("partial"),
        failed_cases=statuses.count("failed"),
        per_case_results=cases,
    )
    return raw


class Progress:
    def __init__(self):
        self.saved = {}

    def begin(self, check):
        return True

    def save(self, check, outcome):
        self.saved[check] = outcome
        return True


def evaluate(monkeypatch, tmp_path, outputs, *, indices=(0,), mode="op", recommendations=True):
    pending = iter(outputs)
    calls = []

    def execute(cmd, *args, **kwargs):
        calls.append(cmd)
        Path(cmd[cmd.index("--output") + 1]).write_text(json.dumps(next(pending)))
        return ScriptOutcome(0, "", "", False)

    monkeypatch.setattr(profile_module, "run_script", execute)
    request = EvaluationRequest.model_validate({
        "schema_version": "2.1",
        "op_name": "profile_evidence",
        "reference_code": "REF",
        "kernel_code": "IMPL",
        "checks": ["profile"],
        "profile": {
            "mode": mode,
            "case_indices": list(indices) if indices is not None else None,
            "include_recommendations": recommendations,
        },
    })
    progress = Progress()
    result = EvaluationEvaluator(
        verify=VerifyEvaluator("unused"), benchmark=BenchmarkEvaluator("unused"),
        profile=ProfileEvaluator("unused"),
    ).run(
        request, tmp_path, 0, datetime.now(timezone.utc) + timedelta(minutes=10),
        lambda: True, None,
        PinnedProvenance(
            evaluator_build="git:" + "0" * 40,
            algorithms=AlgorithmVersions(
                correctness="correctness-v1", decoy="decoy-v1",
                benchmark="geometric-mean-v1", profiler="profiler-v1",
            ),
            purpose="trial", manifest_id=None, problem_id=None,
        ),
        progress,
    )
    assert progress.saved["profile"] == result.profile
    assert len(calls) == 1
    if indices is not None:
        start = calls[0].index("--case-indices") + 1
        assert calls[0][start:start + len(indices)] == [str(index) for index in indices]
    return result.profile


def assert_invalid(outcome):
    assert outcome.status == "failed"
    assert outcome.error.code == "INVALID_ENGINE_OUTPUT"
    assert outcome.error.phase == "shaping"
    assert outcome.error.retryable is False


@pytest.mark.parametrize("indices", [(0,), (0, 2)])
@pytest.mark.parametrize("field,value", [
    ("timed_out", True),
    ("remaining_case_indices", [0]),
    ("completed_cases", 99),
    ("selected_cases", 0),
])
def test_invalid_run_cannot_be_laundered_by_combining(
    monkeypatch, tmp_path, indices, field, value,
):
    outputs = [batch_envelope(indices)]
    outputs[0][field] = value
    assert_invalid(evaluate(monkeypatch, tmp_path, outputs, indices=indices))


@pytest.mark.parametrize("indices", [None, (0,), (0, 2)])
@pytest.mark.parametrize("recommendations", [True, False])
def test_malformed_json_is_protocol_failure(monkeypatch, tmp_path, indices, recommendations):
    outputs = [{}]
    assert_invalid(evaluate(
        monkeypatch, tmp_path, outputs, indices=indices, recommendations=recommendations,
    ))


@pytest.mark.parametrize("mutation", ["mode", "requested", "case", "total", "duplicate"])
def test_inconsistent_run_identity_is_protocol_failure(monkeypatch, tmp_path, mutation):
    raw = batch_envelope((0, 2))
    if mutation == "mode":
        raw["profile_mode"] = "deep"
    elif mutation == "requested":
        raw["requested_case_indices"] = [0, 1]
    elif mutation == "case":
        raw["per_case_results"][1]["case_idx"] = 1
    elif mutation == "total":
        raw["total_cases"] = 2
    else:
        raw["per_case_results"][1] = copy.deepcopy(raw["per_case_results"][0])
    assert_invalid(evaluate(monkeypatch, tmp_path, [raw], indices=(0, 2)))


def test_unselected_run_must_match_requested_mode(monkeypatch, tmp_path):
    raw = envelope(mode="deep", total=1)
    raw["requested_case_idx"] = None
    assert_invalid(evaluate(monkeypatch, tmp_path, [raw], indices=None))


@pytest.mark.parametrize("mode", ["op", "deep"])
@pytest.mark.parametrize("recommendations", [True, False])
def test_valid_run_statuses_and_recommendations_survive_combination(
    monkeypatch, tmp_path, mode, recommendations,
):
    outputs = [batch_envelope(
        (0, 1, 2), mode=mode, statuses=["complete", "partial", "failed"],
    )]
    outcome = evaluate(
        monkeypatch, tmp_path, outputs, indices=(0, 1, 2), mode=mode,
        recommendations=recommendations,
    )
    assert outcome.status == "succeeded"
    assert outcome.data.mode == mode
    assert outcome.data.counts.model_dump() == {
        "selected": 3, "complete": 1, "partial": 1, "failed": 1,
    }
    assert [case.case_idx for case in outcome.data.cases] == [0, 1, 2]
    assert [case.status for case in outcome.data.cases] == ["complete", "partial", "failed"]
    if mode == "deep":
        assert bool(outcome.data.cases[0].recommendations) is recommendations


@pytest.mark.parametrize("mode", ["op", "deep"])
def test_all_case_run_remains_valid(monkeypatch, tmp_path, mode):
    raw = envelope(mode=mode, total=1)
    raw["requested_case_idx"] = None
    outcome = evaluate(monkeypatch, tmp_path, [raw], indices=None, mode=mode)
    assert outcome.status == "succeeded"
    assert outcome.data.counts.selected == 1


@pytest.mark.parametrize("mutation", ["selection", "missing", "empty"])
def test_all_case_run_must_cover_the_reference(monkeypatch, tmp_path, mutation):
    raw = envelope(total=1)
    raw["requested_case_idx"] = None
    if mutation == "selection":
        raw["requested_case_idx"] = 0
    elif mutation == "missing":
        raw["total_cases"] = 2
    else:
        raw.update(selected_cases=0, completed_cases=0, per_case_results=[])
    assert_invalid(evaluate(monkeypatch, tmp_path, [raw], indices=None))


@pytest.mark.parametrize("indices", [None, (0,)])
def test_hidden_recommendations_must_still_be_valid(monkeypatch, tmp_path, indices):
    raw = envelope(mode="deep", total=1) if indices is None else batch_envelope(
        indices, mode="deep", total=1,
    )
    if indices is None:
        raw["requested_case_idx"] = None
    raw["per_case_results"][0]["recommendations"] = "invalid recommendations"
    assert_invalid(evaluate(
        monkeypatch, tmp_path, [raw], indices=indices, mode="deep", recommendations=False,
    ))


@pytest.mark.parametrize("field,value", [
    ("timed_out", True), ("remaining_case_indices", [2]),
    ("completed_cases", 99), ("selected_cases", 0), ("total_cases", 4),
])
def test_legacy_envelopes_are_validated_before_combining(field, value):
    raw = {"profile_runs": [envelope(0), envelope(2)]}
    raw["profile_runs"][1][field] = value
    with pytest.raises(ShapingError):
        shape_profile_runs(raw, ProfileParams(case_indices=[0, 2]))


def test_legacy_envelope_collection_remains_readable():
    result = shape_profile_runs(
        {"profile_runs": [envelope(0), envelope(2)]},
        ProfileParams(case_indices=[0, 2]),
    )
    assert [case.case_idx for case in result.cases] == [0, 2]


def test_deep_capture_controls_reach_single_engine_invocation(tmp_path):
    calls = []
    raw = batch_envelope((0, 2), mode="deep")

    def execute(cmd, *args, **kwargs):
        calls.append(cmd)
        Path(cmd[cmd.index("--output") + 1]).write_text(json.dumps(raw))
        return ScriptOutcome(0, "", "", False)

    request = EvaluationRequest.model_validate({
        "schema_version": "2.1", "op_name": "profile_evidence",
        "reference_code": "REF", "kernel_code": "IMPL", "checks": ["profile"],
        "profile": {"mode": "deep", "case_indices": [0, 2],
                    "metric_groups": ["Memory", "L2Cache"], "include_per_core": False},
    })
    outcome = ProfileEvaluator("unused", runner=execute).run(
        request, tmp_path, 0, datetime.now(timezone.utc) + timedelta(minutes=10),
    )
    assert outcome.status == "succeeded"
    assert outcome.raw == raw
    assert len(calls) == 1
    assert calls[0][-6:] == [
        "0", "2", "--metric-groups", "Memory", "L2Cache", "--no-per-core",
    ]
