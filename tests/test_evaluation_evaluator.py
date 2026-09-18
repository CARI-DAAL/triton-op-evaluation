"""Orchestration tests for the v2 EvaluationEvaluator.

The evaluator runs the fixed correctness -> performance -> profile sequence,
fences every check behind ProgressSink begin/save, runs performance only after
a succeeded *and* correct subject outcome, keeps profile independent of the
subject verdict, and fails closed when the lease or the engine output cannot
be trusted.
"""

import importlib.metadata
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from triton_evaluation.api.contract import (
    EvaluationRequest,
    ExecutionEnvironment,
)
from triton_evaluation.config import Settings
from triton_evaluation.evaluators.base import (
    EngineCheckResult,
    LeaseLostError,
    engine_failure,
    engine_timeout,
)
from triton_evaluation.evaluators.evaluation import EvaluationEvaluator
from triton_evaluation.execution_environment import build_execution_environment
from triton_evaluation.models import AlgorithmVersions, PinnedProvenance

FIXTURES = Path(__file__).parent / "fixtures" / "engine"


def load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class RecordingSink:
    def __init__(self, begin_ok=True, save_ok=True):
        self.begin_ok = begin_ok
        self.save_ok = save_ok
        self.calls = []

    def begin(self, check):
        self.calls.append(("begin", check))
        return self.begin_ok

    def save(self, check, outcome):
        self.calls.append(("save", check, outcome.status))
        return self.save_ok


class StubCheck:
    """Sub-evaluator stub returning a fixed EngineCheckResult."""

    def __init__(self, result):
        self.result = result
        self.calls = 0

    def run(self, request, work_dir, device_id, deadline_at, lease_valid=None):
        self.calls += 1
        return self.result


def succeeded(name):
    return StubCheck(EngineCheckResult(status="succeeded", raw=load(name), error=None))


class Harness(EvaluationEvaluator):
    def __init__(self, verify, bench=None, profile=None):
        super().__init__("unused")
        self.verify = verify
        self.bench = bench
        self.profile = profile

    def _verify_evaluator(self):
        return self.verify

    def _benchmark_evaluator(self):
        return self.bench

    def _profile_evaluator(self):
        return self.profile


ALGORITHMS = AlgorithmVersions(
    correctness="correctness-v1",
    decoy="decoy-v1",
    benchmark="geometric-mean-v1",
    profiler="profiler-v1",
)

ENVIRONMENT = ExecutionEnvironment(
    hardware_model="Ascend910B3",
    hardware_count=1,
    driver_version="24.1.0",
    runtime_version="8.0.RC3",
    torch_version="2.1.0",
    torch_npu_version="2.1.0.post3",
    triton_version="3.0.0",
    toolchain_version="CANN 8.0.RC3",
)


def trial_pinned():
    return PinnedProvenance(
        evaluator_build="git:" + "0" * 40,
        algorithms=ALGORITHMS,
        purpose="trial",
        manifest_id=None,
        problem_id=None,
    )


def grade_pinned():
    return PinnedProvenance(
        evaluator_build="git:" + "0" * 40,
        algorithms=ALGORITHMS,
        purpose="grade",
        manifest_id="release-1",
        problem_id="addmm-1",
    )


def run_evaluation(harness, checks, sink, tmp_path, *, pinned=None, lease=lambda: True):
    request = EvaluationRequest.model_validate(
        {
            "schema_version": "2.1",
            "op_name": "addmm",
            "reference_code": "REF",
            "kernel_code": "IMPL",
            "checks": checks,
        }
    )
    return harness.run(
        request=request,
        work_dir=tmp_path,
        device_id=0,
        deadline_at=datetime.now(timezone.utc) + timedelta(minutes=10),
        lease_valid=lease,
        execution_environment=ENVIRONMENT,
        pinned=pinned or trial_pinned(),
        progress=sink,
    )


def full_harness(verify_fixture="verify-correct.json"):
    return Harness(
        verify=succeeded(verify_fixture),
        bench=succeeded("benchmark-mixed-coverage.json"),
        profile=succeeded("profile-op-complete.json"),
    )


# ---------------------------------------------------------------------------
# Fixed order, fencing, and slot filling
# ---------------------------------------------------------------------------


def test_correct_kernel_runs_all_checks_in_fixed_order(tmp_path):
    harness = full_harness()
    sink = RecordingSink()
    # The request array order must not change execution semantics.
    result = run_evaluation(harness, ["profile", "performance", "correctness"], sink, tmp_path)

    assert sink.calls == [
        ("begin", "correctness"),
        ("save", "correctness", "succeeded"),
        ("begin", "performance"),
        ("save", "performance", "succeeded"),
        ("begin", "profile"),
        ("save", "profile", "succeeded"),
    ]
    assert result.correctness.data.outcome == "correct"
    assert result.performance.data.aggregation == "geometric_mean_v1"
    assert result.profile.data.mode == "op"
    assert result.provenance.execution_environment == ENVIRONMENT
    assert result.provenance.grade_eligible is False


def test_unrequested_checks_stay_null(tmp_path):
    sink = RecordingSink()
    result = run_evaluation(
        Harness(verify=succeeded("verify-correct.json")), ["correctness"], sink, tmp_path
    )
    assert result.correctness.status == "succeeded"
    assert result.performance is None
    assert result.profile is None
    assert sink.calls == [
        ("begin", "correctness"),
        ("save", "correctness", "succeeded"),
    ]


def test_wrong_subject_skips_performance_but_still_profiles(tmp_path):
    harness = full_harness("verify-wrong-value.json")
    sink = RecordingSink()
    result = run_evaluation(harness, ["correctness", "performance", "profile"], sink, tmp_path)

    assert result.correctness.data.outcome == "wrong"
    performance = result.performance
    assert performance.status == "skipped"
    assert performance.reason.code == "DEPENDENCY_NOT_PASSED"
    assert performance.reason.depends_on == "correctness"
    assert harness.bench.calls == 0
    assert result.profile.status == "succeeded"
    assert harness.profile.calls == 1
    # skipped outcomes are fenced through the sink like real ones
    assert ("save", "performance", "skipped") in sink.calls


def test_decoy_is_wrong_with_full_counts_and_still_profiles(tmp_path):
    harness = full_harness("verify-decoy.json")
    sink = RecordingSink()
    result = run_evaluation(harness, ["correctness", "performance", "profile"], sink, tmp_path)

    data = result.correctness.data
    assert data.outcome == "wrong"
    assert data.decoy is True
    assert data.passed_cases == data.total_cases
    assert data.failures == []
    assert result.performance.status == "skipped"
    assert harness.bench.calls == 0
    assert result.profile.status == "succeeded"


def test_compile_error_is_a_subject_outcome_not_an_infra_error(tmp_path):
    harness = full_harness("verify-candidate-compile-error.json")
    sink = RecordingSink()
    result = run_evaluation(harness, ["correctness", "performance", "profile"], sink, tmp_path)

    assert result.correctness.status == "succeeded"
    assert result.correctness.data.outcome == "compile_error"
    assert result.performance.reason.code == "DEPENDENCY_NOT_PASSED"
    assert result.profile.status == "succeeded"


def test_correctness_infra_failure_skips_dependents(tmp_path):
    harness = full_harness()
    harness.verify = StubCheck(engine_failure("correctness", "verify engine crashed"))
    sink = RecordingSink()
    result = run_evaluation(harness, ["correctness", "performance", "profile"], sink, tmp_path)

    assert result.correctness.status == "failed"
    assert result.correctness.error.code == "EVALUATOR_EXECUTION_FAILED"
    assert result.correctness.error.retryable is True
    assert result.performance.status == "skipped"
    assert result.performance.reason.code == "PRIOR_CHECK_FAILED"
    assert result.profile.status == "skipped"
    assert result.profile.reason.code == "PRIOR_CHECK_FAILED"
    assert harness.bench.calls == 0
    assert harness.profile.calls == 0


def test_engine_process_timeout_is_never_a_subject_execution_timeout(tmp_path):
    harness = full_harness()
    harness.verify = StubCheck(engine_timeout("correctness", "resource deadline expired"))
    sink = RecordingSink()
    result = run_evaluation(harness, ["correctness", "performance"], sink, tmp_path)

    assert result.correctness.status == "timeout"
    assert result.correctness.error.code == "DEADLINE_EXCEEDED"
    assert result.correctness.data is None
    assert result.performance.status == "skipped"


def test_invalid_engine_output_fails_closed(tmp_path):
    harness = full_harness("verify-unknown-kind.json")
    sink = RecordingSink()
    result = run_evaluation(harness, ["correctness", "performance", "profile"], sink, tmp_path)

    assert result.correctness.status == "failed"
    assert result.correctness.error.code == "INVALID_ENGINE_OUTPUT"
    assert result.correctness.error.phase == "shaping"
    assert result.correctness.error.retryable is False
    assert result.performance.status == "skipped"
    assert result.profile.status == "skipped"
    assert harness.profile.calls == 0


def test_unattributed_case_failure_is_retryable_internal(tmp_path):
    """kind=None failures without candidate_error are a measurement gap.

    The engine records framework-side crashes, device errors, and output
    flattening TypeErrors this way; they are not protocol violations, so the
    check fails retryable instead of voiding the evaluation as invalid.
    """
    harness = full_harness("verify-ambiguous-timeout.json")
    sink = RecordingSink()
    result = run_evaluation(harness, ["correctness", "performance", "profile"], sink, tmp_path)

    assert result.correctness.status == "failed"
    assert result.correctness.error.code == "INTERNAL"
    assert result.correctness.error.phase == "execution"
    assert result.correctness.error.retryable is True
    message = result.correctness.error.message
    assert "unattributed case failure" in message
    assert "\n" not in message
    assert "TimeoutError" in message
    assert result.performance.status == "skipped"
    assert result.profile.status == "skipped"
    assert harness.profile.calls == 0


def test_performance_infra_failure_keeps_correctness_and_skips_profile(tmp_path):
    harness = full_harness()
    harness.bench = succeeded("benchmark-invalid-measurement.json")
    sink = RecordingSink()
    result = run_evaluation(harness, ["correctness", "performance", "profile"], sink, tmp_path)

    assert result.correctness.status == "succeeded"
    assert result.performance.status == "failed"
    assert result.performance.error.code == "INVALID_ENGINE_OUTPUT"
    assert result.profile.status == "skipped"
    assert result.profile.reason.code == "PRIOR_CHECK_FAILED"
    assert harness.profile.calls == 0


# ---------------------------------------------------------------------------
# Fencing: begin/save rejection and lease loss discard the attempt
# ---------------------------------------------------------------------------


def test_rejected_begin_raises_lease_lost_before_any_engine_work(tmp_path):
    harness = full_harness()
    with pytest.raises(LeaseLostError):
        run_evaluation(harness, ["correctness"], RecordingSink(begin_ok=False), tmp_path)
    assert harness.verify.calls == 0


def test_rejected_save_raises_lease_lost(tmp_path):
    harness = full_harness()
    with pytest.raises(LeaseLostError):
        run_evaluation(harness, ["correctness"], RecordingSink(save_ok=False), tmp_path)


def test_lease_loss_between_checks_discards_remaining_work(tmp_path):
    harness = full_harness()
    sink = RecordingSink()
    calls = {"count": 0}

    def lease():
        calls["count"] += 1
        return calls["count"] <= 2  # validity lost after correctness completes

    with pytest.raises(LeaseLostError):
        run_evaluation(
            harness, ["correctness", "performance", "profile"], sink, tmp_path, lease=lease
        )
    assert harness.bench.calls == 0
    assert harness.profile.calls == 0
    assert sink.calls == [
        ("begin", "correctness"),
        ("save", "correctness", "succeeded"),
        ("begin", "performance"),
    ]


# ---------------------------------------------------------------------------
# Provenance and the grade boundary
# ---------------------------------------------------------------------------


def test_completed_trusted_grade_is_grade_eligible(tmp_path):
    result = run_evaluation(
        full_harness(),
        ["correctness", "performance"],
        RecordingSink(),
        tmp_path,
        pinned=grade_pinned(),
    )
    provenance = result.provenance
    assert provenance.grade_eligible is True
    assert provenance.manifest_id == "release-1"
    assert provenance.problem_id == "addmm-1"
    assert provenance.evaluator_build == "git:" + "0" * 40
    assert provenance.correctness_algorithm == "correctness-v1"
    # grade_eligible is a trust precondition, not a score: no reward fields exist.
    assert not ({"reward", "score", "scoring_state"} & set(result.model_dump()))


def test_grade_with_wrong_subject_is_eligible_but_scored_zero_by_coordinator(tmp_path):
    result = run_evaluation(
        full_harness("verify-wrong-value.json"),
        ["correctness", "performance"],
        RecordingSink(),
        tmp_path,
        pinned=grade_pinned(),
    )
    # Eligibility is a trust property of the completed record; the subject
    # outcome (wrong) is what the coordinator maps to scored numeric zero.
    assert result.provenance.grade_eligible is True
    assert result.correctness.data.outcome == "wrong"
    assert result.correctness.data.passed is False


def test_grade_with_infra_failure_is_not_eligible(tmp_path):
    harness = full_harness()
    harness.verify = StubCheck(engine_failure("correctness", "verify engine crashed"))
    result = run_evaluation(
        harness, ["correctness"], RecordingSink(), tmp_path, pinned=grade_pinned()
    )
    assert result.provenance.grade_eligible is False


def test_trial_is_never_grade_eligible(tmp_path):
    result = run_evaluation(full_harness(), ["correctness"], RecordingSink(), tmp_path)
    assert result.provenance.grade_eligible is False
    assert result.provenance.manifest_id is None
    assert result.provenance.problem_id is None


# ---------------------------------------------------------------------------
# Worker execution environment (design section 4.5)
# ---------------------------------------------------------------------------


def test_build_execution_environment_uses_attested_settings():
    settings = Settings(
        worker_hardware_model="Ascend910B4",
        worker_driver_version="25.0.0",
        worker_runtime_version="8.1.RC1",
        worker_toolchain_version="CANN 8.1.RC1",
    )
    env = build_execution_environment(settings)
    assert env.hardware_model == "Ascend910B4"
    assert env.hardware_count == 1  # each worker process exposes exactly one card
    assert env.driver_version == "25.0.0"
    assert env.runtime_version == "8.1.RC1"
    assert env.toolchain_version == "CANN 8.1.RC1"


def test_build_execution_environment_reads_optional_versions_in_process():
    env = build_execution_environment(Settings())

    def expected(distribution):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return None

    assert env.torch_version == expected("torch")
    assert env.torch_npu_version == expected("torch-npu")
    assert env.triton_version == expected("triton")


def test_execution_environment_never_carries_host_or_card_identity():
    env = build_execution_environment(Settings())
    assert set(env.model_dump()) == {
        "hardware_model",
        "hardware_count",
        "driver_version",
        "runtime_version",
        "torch_version",
        "torch_npu_version",
        "triton_version",
        "toolchain_version",
    }


# ---------------------------------------------------------------------------
# Verify evaluator outer-kill ordering (review fix round 1)
# ---------------------------------------------------------------------------


def _request(checks=("correctness",)):
    return EvaluationRequest.model_validate(
        {
            "schema_version": "2.1",
            "op_name": "addmm",
            "reference_code": "REF",
            "kernel_code": "IMPL",
            "checks": list(checks),
        }
    )


def test_evaluator_outer_kill_fires_after_the_engine_grace(tmp_path, monkeypatch):
    import triton_evaluation.evaluators.verify as verify_mod
    from triton_evaluation.evaluators.base import ScriptOutcome

    seen = {}

    def fake_run_script(
        cmd, cwd, device_id, deadline_at=None, lease_valid=None, *, budget_seconds=None
    ):
        seen["cmd"] = cmd
        seen["budget"] = budget_seconds
        # validate pre-check yields no JSON verdict -> proceed to verify
        return ScriptOutcome(0, "", "", False)

    monkeypatch.setattr(verify_mod, "run_script", fake_run_script)
    monkeypatch.setattr(verify_mod, "read_json", lambda path: load("verify-correct.json"))

    evaluator = verify_mod.VerifyEvaluator("unused")
    deadline = datetime.now(timezone.utc) + timedelta(seconds=120)
    result = evaluator.run(_request(), tmp_path, 0, deadline)

    assert result.status == "succeeded"
    budget = int(seen["cmd"][seen["cmd"].index("--timeout") + 1])
    # The evaluator kill must come strictly after the engine's own outer kill
    # (budget + engine grace), so the per-subject attribution wins the race.
    assert seen["budget"] > budget + verify_mod._ENGINE_OUTER_GRACE_SECONDS


def test_ambiguous_exit_past_deadline_is_timeout_never_attributed(tmp_path, monkeypatch):
    import triton_evaluation.evaluators.verify as verify_mod
    from triton_evaluation.evaluators.base import ScriptOutcome

    # run(), decoy pre-check, and _run_verify's start see budget; the
    # post-run check observes the expired deadline.
    remaining = iter([100.0, 100.0, 100.0, -1.0])
    monkeypatch.setattr(verify_mod, "remaining_seconds", lambda deadline: next(remaining, -1.0))
    monkeypatch.setattr(
        verify_mod,
        "run_script",
        lambda *args, **kwargs: ScriptOutcome(1, "", "", False),
    )
    monkeypatch.setattr(verify_mod, "read_json", lambda path: None)

    evaluator = verify_mod.VerifyEvaluator("unused")
    deadline = datetime.now(timezone.utc) + timedelta(seconds=100)
    result = evaluator.run(_request(), tmp_path, 0, deadline)

    assert result.status == "timeout"
    assert result.error.code == "DEADLINE_EXCEEDED"
    assert result.error.retryable is False


def test_compile_precheck_is_fenced_before_candidate_import(tmp_path):
    sink = RecordingSink()
    harness = Harness(verify=succeeded("verify-correct.json"))
    calls = []

    def precheck():
        assert sink.calls == [("begin", "compile")]
        calls.append("import")
        return "precheck_passed"

    request = EvaluationRequest.model_validate(
        {
            "schema_version": "2.1",
            "op_name": "addmm",
            "reference_code": "REF",
            "kernel_code": "IMPL",
            "checks": ["compile", "correctness"],
        }
    )
    result = harness.run(
        request,
        tmp_path,
        0,
        datetime.now(timezone.utc) + timedelta(minutes=1),
        lambda: True,
        ENVIRONMENT,
        trial_pinned(),
        sink,
        precheck=precheck,
    )
    assert calls == ["import"]
    assert result.compile.data.outcome == "compiled"
    assert result.correctness.data.outcome == "correct"


def test_correctness_does_not_execute_an_unrequested_compile_check(tmp_path):
    harness = Harness(verify=succeeded("verify-correct.json"))
    request = EvaluationRequest.model_validate(
        {
            "schema_version": "2.1",
            "op_name": "addmm",
            "reference_code": "REF",
            "kernel_code": "IMPL",
            "checks": ["correctness"],
        }
    )

    def unexpected_precheck():
        raise AssertionError("compile was not requested")

    result = harness.run(
        request,
        tmp_path,
        0,
        datetime.now(timezone.utc) + timedelta(minutes=1),
        lambda: True,
        ENVIRONMENT,
        trial_pinned(),
        RecordingSink(),
        precheck=unexpected_precheck,
    )
    assert result.compile is None
    assert result.correctness.data.outcome == "correct"
