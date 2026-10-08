"""Grade-boundary tests (design sections 4.5 and 5).

The only grade decision this repository makes is ``Provenance.grade_eligible``;
it must never create a ``GradeRecord`` or a numeric reward. The scored/unscored
mapping belongs to the trusted coordinator. These tests pin the boundary from
the published terminal fixtures: which fixture states carry a scorable subject
outcome (scored numeric zero by the coordinator), which carry only
infrastructure outcomes (unscored), and why eligibility is not a score.
"""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import triton_evaluation.api.contract as contract
from triton_evaluation.api.contract import EvaluationResource, Provenance

FIXTURES = (
    Path(__file__).parent.parent
    / "tests"
    / "fixtures"
    / "evaluation_api"
    / "2.1"
    / "fixtures"
)

# correctness succeeded with a non-correct subject outcome: the coordinator
# scores these numeric zero. dependency-skipped performance does not change it.
SCORED_ZERO_FIXTURES = {
    "resource-completed-wrong.json": "wrong",
    "resource-completed-decoy.json": "wrong",
    "resource-completed-compile-error.json": "compile_error",
    "resource-completed-runtime-error.json": "runtime_error",
    "resource-completed-execution-timeout.json": "execution_timeout",
}

# evaluator/resource failure: no trustworthy subject outcome exists, so the
# coordinator must leave the run unscored (never a disguised numeric zero).
UNSCORED_FIXTURES = ["resource-failed.json", "resource-timed-out.json"]


def load(name):
    return EvaluationResource.model_validate_json(
        (FIXTURES / name).read_text(encoding="utf-8")
    )


def _walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def test_subject_failure_fixtures_carry_scorable_zero_inputs():
    for name, outcome in SCORED_ZERO_FIXTURES.items():
        resource = load(name)
        assert resource.state == "completed"
        correctness = resource.result.correctness
        assert correctness.status == "succeeded"
        assert correctness.data.outcome == outcome
        assert correctness.data.passed is False
        assert correctness.error is None
        # a dependency-skipped performance must not change the scored-zero mapping
        if correctness.data.outcome != "correct" and resource.result.performance:
            assert resource.result.performance.status in ("skipped", "succeeded")


def test_correct_fixture_is_the_only_positive_subject_outcome():
    resource = load("resource-completed-correct.json")
    assert resource.result.correctness.data.outcome == "correct"
    assert resource.result.correctness.data.passed is True


def test_infra_failure_fixtures_carry_no_subject_outcome():
    for name in UNSCORED_FIXTURES:
        resource = load(name)
        assert resource.state in ("failed", "timed_out")
        correctness = resource.result.correctness
        assert correctness.status in ("failed", "timeout")
        assert correctness.data is None
        assert correctness.error is not None
        assert resource.result.provenance.grade_eligible is False


def test_no_published_fixture_contains_reward_or_score_fields():
    for path in sorted(FIXTURES.glob("*.json")):
        if path.name == "index.json":
            continue
        keys = {key.lower() for key in _walk_keys(json.loads(path.read_text()))}
        assert not ({"reward", "score", "scoring_state", "grade_record"} & keys), path.name


def test_every_published_terminal_fixture_is_trial_shaped_not_eligible():
    for name in (*SCORED_ZERO_FIXTURES, *UNSCORED_FIXTURES):
        provenance = load(name).result.provenance
        assert provenance.grade_eligible is False
        assert provenance.manifest_id is None
        assert provenance.problem_id is None


def test_check_data_always_comes_with_a_full_execution_environment():
    for path in sorted(FIXTURES.glob("resource-*.json")):
        resource = load(path.name)
        if resource.result is None:
            continue
        outcomes = (
            resource.result.correctness,
            resource.result.performance,
            resource.result.profile,
        )
        if any(outcome is not None and outcome.data is not None for outcome in outcomes):
            environment = resource.result.provenance.execution_environment
            assert environment is not None, path.name
            assert environment.hardware_count == 1
            assert environment.hardware_model
            assert environment.driver_version
            assert environment.runtime_version
            assert environment.toolchain_version


def _provenance_payload(**overrides):
    payload = {
        "schema_version": "2.1",
        "evaluator_build": "sha256:" + "a" * 64,
        "correctness_algorithm": "correctness-v1",
        "decoy_algorithm": "decoy-v1",
        "benchmark_algorithm": "geometric-mean-v1",
        "profiler_algorithm": "profiler-v1",
        "grade_eligible": False,
        "manifest_id": None,
        "problem_id": None,
        "execution_environment": None,
    }
    payload.update(overrides)
    return payload


def test_grade_eligibility_requires_manifest_and_problem_binding():
    with pytest.raises(ValidationError):
        Provenance(**_provenance_payload(grade_eligible=True))
    with pytest.raises(ValidationError):
        Provenance(**_provenance_payload(manifest_id="release-1"))
    eligible = Provenance(
        **_provenance_payload(
            grade_eligible=True, manifest_id="release-1", problem_id="addmm-1"
        )
    )
    assert eligible.grade_eligible is True


def test_grade_eligible_is_not_a_score():
    # Eligibility is a trust precondition recorded in provenance; the scored /
    # unscored disposition (including coordinator cancellation records) lives
    # only in the consumer-owned GradeRecord, which this repo must not create.
    eligible = Provenance(
        **_provenance_payload(
            grade_eligible=True, manifest_id="release-1", problem_id="addmm-1"
        )
    )
    assert set(eligible.model_dump()) == {
        "schema_version",
        "evaluator_build",
        "correctness_algorithm",
        "decoy_algorithm",
        "benchmark_algorithm",
        "profiler_algorithm",
        "grade_eligible",
        "manifest_id",
        "problem_id",
        "execution_environment",
    }


def test_contract_defines_no_grade_record_or_reward():
    assert not hasattr(contract, "GradeRecord")
    for name in dir(contract):
        model = getattr(contract, name)
        fields = getattr(model, "model_fields", None)
        if not fields:
            continue
        lowered = {field.lower() for field in fields}
        assert not ({"reward", "score", "scoring_state"} & lowered), name
