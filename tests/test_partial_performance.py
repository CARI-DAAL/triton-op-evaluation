"""Measured timing survives interruption without inventing unfinished cases."""

import copy

import pytest
from conftest import make_caller, make_correctness_success, make_grade_caller, make_grade_request
from pydantic import ValidationError
from test_api import algorithms, build_app, request_payload, seed_worker, trial_headers
from test_shaping import load
from test_terminal_evidence import result_fixture

from triton_evaluation.api.contract import (
    CheckSucceeded,
    EvaluationProgress,
    EvaluationResource,
    PerformanceResult,
)
from triton_evaluation.evaluators.shaping import (
    ShapingError,
    UnattributedFailure,
    shape_performance,
    shape_single_performance,
)
from triton_evaluation.service.models import WorkerIdentity, new_worker_id
from triton_evaluation.service.terminal_results import complete_evaluation

SELECTED = [2, 7, 12, 20]


def partial_raw():
    raw = load("benchmark-mixed-coverage.json")
    raw.update(
        total_cases=4,
        case_indices=[index + 1 for index in SELECTED],
        progress={
            "selected_case_indices": SELECTED.copy(),
            "completed_case_indices": SELECTED[:3],
            "complete": False,
        },
    )
    for case, index in zip(raw["per_shape_results"], SELECTED, strict=False):
        case["case_idx"] = index + 1
    return raw


def complete_raw():
    raw = partial_raw()
    measured = raw["per_shape_results"][:2]
    raw["per_shape_results"] = [copy.deepcopy(measured[i % 2]) for i in range(4)]
    for case, index in zip(raw["per_shape_results"], SELECTED, strict=True):
        case["case_idx"] = index + 1
    raw.update(passed_cases=4, failed_cases=0)
    raw["progress"].update(completed_case_indices=SELECTED.copy(), complete=True)
    return raw


def outcome(raw=None):
    return CheckSucceeded[PerformanceResult](
        status="succeeded", data=shape_performance(partial_raw() if raw is None else raw),
    )


def test_sparse_partial_shaping_preserves_original_case_partition_and_measurements():
    raw = partial_raw()
    original = copy.deepcopy(raw)
    result = shape_performance(raw)

    assert raw == original
    assert result.complete is False and result.termination_reason is None
    assert result.selected_cases == 4 and result.measured_cases == 2
    assert result.coverage == 0.5
    assert [case.case_idx for case in result.per_case] == [2, 7]
    assert [case.case_idx for case in result.unmeasured_cases] == [12]
    assert result.pending_case_indices == [20]
    assert result.unmeasured_cases[0].reason_code == "NO_STABLE_MEASUREMENT"
    assert [case.implementation_latency_ms for case in result.per_case] == [0.5, 2.0]
    assert result.overall_speedup_vs_reference == pytest.approx(1.0)


@pytest.mark.parametrize("reason", [None, "timeout", "interrupted"])
def test_partial_completion_wire_roundtrip_preserves_ending_reason(reason):
    raw = partial_raw()
    raw["termination_reason"] = reason
    data = shape_performance(raw)
    restored = PerformanceResult.model_validate_json(data.model_dump_json())
    assert restored == data
    assert restored.termination_reason == reason
    assert restored.pending_case_indices == [20]


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_selection", "duplicate_completed", "mismatched_completed",
        "mismatched_row", "duplicate_row", "wrong_selection_count",
        "one_based_progress", "zero_based_row", "premature_complete",
        "pending_abnormal_case",
    ],
)
def test_partial_shaping_rejects_invalid_original_case_identities(mutation):
    raw = partial_raw()
    progress = raw["progress"]
    if mutation == "duplicate_selection":
        progress["selected_case_indices"][1] = SELECTED[0]
    elif mutation == "duplicate_completed":
        progress["completed_case_indices"][1] = SELECTED[0]
    elif mutation == "mismatched_completed":
        progress["completed_case_indices"][1] = 8
    elif mutation == "mismatched_row":
        raw["per_shape_results"][1]["case_idx"] = 9
    elif mutation == "duplicate_row":
        raw["per_shape_results"][1]["case_idx"] = 3
    elif mutation == "wrong_selection_count":
        raw["total_cases"] = 5
    elif mutation == "one_based_progress":
        progress["selected_case_indices"] = [index + 1 for index in SELECTED]
    elif mutation == "zero_based_row":
        raw["per_shape_results"][0]["case_idx"] = 2
    elif mutation == "premature_complete":
        progress["complete"] = True
    else:
        raw["none_indices"] = [21]
    with pytest.raises(ShapingError):
        shape_performance(raw)


@pytest.mark.parametrize(
    "patch",
    [
        {"complete": True},
        {"pending_case_indices": []},
        {"pending_case_indices": [20, 20], "selected_cases": 5, "coverage": 0.4},
        {"pending_case_indices": [7, 20], "selected_cases": 5, "coverage": 0.4},
        {"pending_case_indices": [20, 19], "selected_cases": 5, "coverage": 0.4},
        {"selected_cases": 3},
        {"measured_cases": 3},
        {"coverage": 1.0},
    ],
)
def test_public_partial_result_requires_disjoint_complete_identity_partition(patch):
    payload = shape_performance(partial_raw()).model_dump()
    payload.update(patch)
    with pytest.raises(ValidationError):
        PerformanceResult.model_validate(payload)


def test_zero_completed_cases_cannot_fabricate_a_performance_result():
    raw = partial_raw()
    raw.update(passed_cases=0, failed_cases=0, per_shape_results=[])
    raw["progress"]["completed_case_indices"] = []
    with pytest.raises(UnattributedFailure):
        shape_performance(raw)


def test_single_source_partial_preserves_sparse_ids_without_reference_measurements():
    raw = {
        "implementation": "triton", "correctness_verified": False,
        "measurement_method": "npu_event_v1", "selected_cases": 4,
        "measured_cases": 1, "coverage": 0.25,
        "per_case": [{"case_idx": 2, "input_shapes": [], "latency_ms": 0.75}],
        "unmeasured_cases": [],
        "progress": {
            "selected_case_indices": SELECTED,
            "completed_case_indices": [2], "complete": False,
        },
    }
    result = shape_single_performance(raw)
    assert result.complete is False and result.pending_case_indices == [7, 12, 20]
    assert result.per_case[0].latency_ms == 0.75
    assert "overall_speedup_vs_reference" not in result.model_dump()
    raw["progress"]["completed_case_indices"] = [7]
    with pytest.raises(ShapingError):
        shape_single_performance(raw)


@pytest.mark.anyio
@pytest.mark.parametrize("ending", ["deadline", "complete"])
async def test_http_progress_and_reaper_preserve_partial_timing(
    tmp_path, release_catalog_files, ending,
):
    app = build_app(tmp_path, release_catalog_files)
    seed_worker(app)
    async with app.http as client:
        submitted = await client.post(
            "/evaluations", headers=trial_headers(),
            json=request_payload(
                workloads=[{"inputs": {}} for _ in range(21)],
                checks=["correctness", "performance"],
                performance={"case_indices": SELECTED}, deadline_seconds=60,
            ),
        )
        assert submitted.status_code == 202, submitted.text
        evaluation_id = submitted.json()["evaluation_id"]
        lease = app.store.claim_next(WorkerIdentity(
            new_worker_id(), app.settings.evaluator_build, algorithms(app.settings),
        ))
        assert lease is not None
        correctness = make_correctness_success(passed_cases=21, total_cases=21)
        partial = outcome()
        assert app.store.save_check_outcome(lease, "correctness", correctness)
        assert app.store.set_active_check(lease, "performance")
        assert app.store.save_check_outcome(lease, "performance", partial)
        progress_response = await client.get(
            f"/evaluations/{evaluation_id}/progress", headers=trial_headers(),
        )
        assert progress_response.status_code == 200
        progress = EvaluationProgress.model_validate_json(progress_response.content)
        assert progress.state == "running" and progress.active_check == "performance"
        assert progress.attempt_id == lease.attempt_id
        assert progress.completed_checks.correctness == correctness
        assert progress.completed_checks.performance == partial
        resource = await client.get(f"/evaluations/{evaluation_id}", headers=trial_headers())
        assert resource.json()["result"] is None

        if ending == "deadline":
            app.clock.advance_ms(60_001)
            assert app.store.expire_due_resources() == 1
            expected = partial.data.model_copy(update={"termination_reason": "timeout"})
            state = "timed_out"
        else:
            completed = outcome(complete_raw())
            assert app.store.save_check_outcome(lease, "performance", completed)
            response = await client.get(
                f"/evaluations/{evaluation_id}/progress", headers=trial_headers(),
            )
            assert response.json()["active_check"] is None
            assert "complete" not in response.json()["completed_checks"]["performance"]["data"]
            result = complete_evaluation(
                requested_checks=["correctness", "performance"],
                outcomes={"correctness": correctness, "performance": completed},
                pinned=lease.pinned,
            )
            assert app.store.commit_terminal(lease, "completed", result)
            expected, state = completed.data, "completed"

        response = await client.get(f"/evaluations/{evaluation_id}", headers=trial_headers())
        assert response.status_code == 200 and "Retry-After" not in response.headers
        terminal = EvaluationResource.model_validate_json(response.content)
        assert terminal.state == state
        assert terminal.result.correctness == correctness
        assert terminal.result.performance.status == "succeeded"
        assert terminal.result.performance.data == expected
        assert terminal.result.provenance.grade_eligible is False
        final_progress = await client.get(
            f"/evaluations/{evaluation_id}/progress", headers=trial_headers(),
        )
        final = EvaluationProgress.model_validate_json(final_progress.content)
        assert final.active_check is None and final.attempt_id == lease.attempt_id
        assert final.completed_checks.performance == terminal.result.performance
        assert final.progress_revision > progress.progress_revision
        assert not app.store.save_check_outcome(lease, "performance", outcome(complete_raw()))
        assert app.store.expire_due_resources() == 0


def test_replacement_attempt_never_merges_old_partial_measurements(
    store, make_command, make_worker, clock, settings,
):
    command = make_command(request=make_grade_request(
        checks=["correctness", "performance"], deadline_seconds=120,
    ), caller=make_grade_caller())
    store.create_or_replay(command)
    first = store.claim_next(make_worker())
    caller = make_grade_caller()
    assert first is not None
    assert store.save_check_outcome(first, "correctness", make_correctness_success())
    assert store.set_active_check(first, "performance")
    assert store.save_check_outcome(first, "performance", outcome())
    clock.advance_ms(settings.lease_seconds * 1000 + 1)
    assert store.recover_expired_leases().requeued == 1
    hidden = store.get_progress_visible(command.evaluation_id, caller)
    assert hidden.completed_checks.performance is None and hidden.active_check is None

    second = store.claim_next(make_worker())
    assert second is not None and second.attempt_id != first.attempt_id
    assert not store.save_check_outcome(first, "performance", outcome(complete_raw()))
    fresh = store.get_progress_visible(command.evaluation_id, caller)
    assert fresh.completed_checks.correctness is None
    assert fresh.completed_checks.performance is None
    raw = partial_raw()
    raw["per_shape_results"] = raw["per_shape_results"][:1]
    raw["per_shape_results"][0]["implementation"]["avg_latency_ms"] = 0.25
    raw["per_shape_results"][0]["speedup_vs_torch"] = 4.0
    raw.update(passed_cases=1, failed_cases=0)
    raw["progress"]["completed_case_indices"] = [2]
    second_evidence = outcome(raw)
    assert store.set_active_check(second, "performance")
    assert store.save_check_outcome(second, "performance", second_evidence)
    current = store.get_progress_visible(command.evaluation_id, caller)
    assert current.attempt_id == second.attempt_id
    assert current.completed_checks.performance == second_evidence
    assert current.completed_checks.performance.data.pending_case_indices == [7, 12, 20]
    assert not store.save_check_outcome(first, "performance", outcome())
    clock.advance_ms(120_001)
    assert store.expire_due_resources() == 1
    terminal = store.get_visible(command.evaluation_id, caller)
    assert terminal.result.performance.data.per_case == second_evidence.data.per_case
    assert terminal.result.performance.data.termination_reason == "timeout"
    assert terminal.result.correctness.status == "timeout"
    assert terminal.result.provenance.grade_eligible is False


@pytest.mark.parametrize("complete", [False, True])
def test_grade_requires_full_measurement_even_with_valid_correctness_and_environment(
    store, make_command, make_worker, complete,
):
    command = make_command(
        request=make_grade_request(checks=["correctness", "performance"]),
        caller=make_grade_caller(),
    )
    store.create_or_replay(command)
    lease = store.claim_next(make_worker())
    assert lease is not None
    correctness = make_correctness_success(passed_cases=4, total_cases=4)
    raw = complete_raw() if complete else partial_raw()
    raw["case_indices"] = [1, 2, 3, 4]
    raw["progress"]["selected_case_indices"] = [0, 1, 2, 3]
    raw["progress"]["completed_case_indices"] = list(range(len(raw["per_shape_results"])))
    for index, case in enumerate(raw["per_shape_results"], start=1):
        case["case_idx"] = index
    performance = outcome(raw)
    result = complete_evaluation(
        requested_checks=["correctness", "performance"],
        outcomes={"correctness": correctness, "performance": performance},
        pinned=lease.pinned,
        execution_environment=result_fixture().provenance.execution_environment,
    )
    assert result.correctness == correctness and result.performance == performance
    assert result.provenance.grade_eligible is complete
    assert store.get_visible(command.evaluation_id, make_caller()) is None
