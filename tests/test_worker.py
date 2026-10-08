"""Durable worker protocol tests: leases, fencing, maintenance, and hygiene.

The evaluator is stubbed at the EvaluationEvaluator.run boundary — these tests
pin the worker's fenced protocol, not engine behavior (covered by the Task 6
suites).
"""
import json
import time
from pathlib import Path

import pytest
import redis
from conftest import make_caller, make_completed_result, make_provenance, make_request

from triton_evaluation.api.contract import (
    CheckSucceeded,
    CorrectnessResult,
    DomainFailure,
    EvaluationResource,
    EvaluationResult,
)
from triton_evaluation.config import Settings
from triton_evaluation.evaluators.base import LeaseLostError
from triton_evaluation.runtime.worker import LeaseKeeper, Worker
from triton_evaluation.service.models import AlgorithmVersions, WorkerIdentity, new_worker_id


class StubEvaluator:
    """Scripted EvaluationEvaluator standing in for the real engine stack."""

    def __init__(self, settings, *, result=None, error=None, hook=None):
        self.settings = settings
        self.result = result
        self.error = error
        self.hook = hook
        self.calls = []

    def run(
        self, request, work_dir, device_id, deadline_at, lease_valid,
        execution_environment, pinned, progress, *, precheck=None,
    ):
        self.calls.append(
            {
                "request": request,
                "work_dir": work_dir,
                "deadline_at": deadline_at,
                "lease_valid": lease_valid,
                "pinned": pinned,
            }
        )
        if self.hook is not None:
            self.hook(self)
        if self.error is not None:
            raise self.error
        return self.result if self.result is not None else make_completed_result(
            self.settings
        )


def algorithms(settings):
    return AlgorithmVersions(
        correctness=settings.correctness_algorithm_version,
        decoy=settings.decoy_algorithm_version,
        benchmark=settings.benchmark_algorithm_version,
        profiler=settings.profiler_algorithm_version,
    )


def build_worker(store, settings, evaluator, clock, identity=None, **kwargs):
    return Worker(
        store, evaluator, device_id=kwargs.pop("device_id", 0),
        settings=settings, identity=identity,
        clock=clock, sleep=lambda _s: None, **kwargs,
    )


def terminalize_everything(store, clock):
    clock.advance_ms(3 * 24 * 3600 * 1000)
    store.expire_due_resources(limit=100)


def resource_state_key(store, state):
    members = store._r.zrange(f"{store._s.redis_prefix}:state:{state}", 0, -1)
    assert len(members) == 1, f"expected one {state} resource, got {members}"
    return f"{store._s.redis_prefix}:resource:{members[0].decode()}"


def read_resource(store, state):
    return EvaluationResource.model_validate_json(
        store._r.hget(resource_state_key(store, state), b"resource")
    )


def fenced_writes(store):
    return int(
        store._r.get(f"{store._s.redis_prefix}:metrics:fenced-write-rejected") or 0
    )


def current_resource(store):
    for state in ("completed", "failed", "timed_out", "running", "queued"):
        members = store._r.zrange(f"{store._s.redis_prefix}:state:{state}", 0, -1)
        if members:
            key = f"{store._s.redis_prefix}:resource:{members[0].decode()}"
            return EvaluationResource.model_validate_json(
                store._r.hget(key, b"resource")
            )
    raise AssertionError("no resource in any state index")


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def settings(tmp_path):
    # Fast heartbeats so lease-loss tests observe invalidation quickly.
    # Startup canary off: these protocol tests predate card health and keep
    # their exact pre-feature code path; card-health tests opt in via
    # canary_settings.
    return Settings(
        engine_dir=str(tmp_path),
        heartbeat_interval_seconds=0.05,
        queue_poll_interval_seconds=0.05,
        card_canary_on_startup=False,
    )


@pytest.fixture
def queued(store, make_command):
    resource = store.create_or_replay(make_command()).resource
    assert resource.state == "queued"
    return resource


def sabotage_attempt(store, settings):
    """Another actor invalidates the running attempt's fencing identity."""
    key = resource_state_key(store, "running")
    store._r.hset(key, "attempt_id", "at_other")


# ---------------------------------------------------------------------------
# Idle protocol: heartbeat, maintenance, no busy-spin
# ---------------------------------------------------------------------------


def test_empty_claim_heartbeats_and_runs_maintenance(store, settings, clock, queued):
    terminalize_everything(store, clock)
    worker = build_worker(store, settings, StubEvaluator(settings), clock)
    assert worker.run_once() is False

    class Compatibility:
        evaluator_build = settings.evaluator_build
        algorithms = algorithms(settings)
        execution_class = "npu"

    assert store.has_fresh_compatible_worker(Compatibility())


def test_incompatible_worker_build_is_not_fresh(store, settings, clock, queued):
    terminalize_everything(store, clock)
    worker = build_worker(store, settings, StubEvaluator(settings), clock)
    worker.run_once()

    class Other:
        evaluator_build = "sha256:" + "b" * 64
        algorithms = algorithms(settings)
        execution_class = "npu"

    assert not store.has_fresh_compatible_worker(Other())


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_normal_completion_commits_completed(store, settings, clock, queued):
    evaluator = StubEvaluator(settings)
    worker = build_worker(store, settings, evaluator, clock)
    assert worker.run_once() is True
    resource = read_resource(store, "completed")
    assert resource.result.correctness.status == "succeeded"


def test_candidate_failure_completion_still_completes(store, settings, clock, queued):
    failed = EvaluationResult(
        correctness=CheckSucceeded[CorrectnessResult](
            status="succeeded",
            data=CorrectnessResult(
                outcome="compile_error",
                passed=False,
                passed_cases=0,
                total_cases=1,
                failures=[],
                domain_failure={
                    "code": "SUBJECT_COMPILE_ERROR",
                    "phase": "compile",
                    "case_idx": None,
                    "message": "candidate compilation failed",
                },
            ),
        ),
        performance=None,
        profile=None,
        provenance=make_provenance(settings),
    )
    worker = build_worker(store, settings, StubEvaluator(settings, result=failed), clock)
    assert worker.run_once() is True
    resource = read_resource(store, "completed")
    assert resource.result.correctness.data.outcome == "compile_error"


def test_work_directory_uses_safe_stem_and_is_cleaned(store, settings, clock, queued):
    evaluator = StubEvaluator(settings)
    worker = build_worker(store, settings, evaluator, clock)
    assert worker.run_once() is True
    work_dir = Path(evaluator.calls[0]["work_dir"])
    assert not work_dir.exists()
    assert work_dir.name.startswith("eval_")
    assert "addmm" not in work_dir.name
    assert len(work_dir.name.split("_")[1]) == 24  # digest hex portion


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def test_evaluator_protocol_failure_commits_redacted_failed(store, settings, clock,
                                                            queued):
    evaluator = StubEvaluator(
        settings, error=RuntimeError("secret /etc/passwd trace <torch source>")
    )
    worker = build_worker(store, settings, evaluator, clock)
    assert worker.run_once() is True
    resource = read_resource(store, "failed")
    assert resource.result.correctness.error.code == "EVALUATOR_EXECUTION_FAILED"
    dumped = json.dumps(resource.result.model_dump(mode="json"), default=str)
    assert "secret" not in dumped
    assert "/etc/passwd" not in dumped
    assert "torch" not in dumped


def test_resource_deadline_attempts_timed_out_commit(store, settings, clock, queued):
    original_claim = store.claim_next

    def claim_and_expire(worker_identity, avoid_card=None, *, card=None):
        lease = original_claim(worker_identity, avoid_card=avoid_card, card=card)
        if lease is not None:
            clock.advance_ms(lease.deadline_at_ms - clock.now_ms + 1)
        return lease

    store.claim_next = claim_and_expire
    worker = build_worker(store, settings, StubEvaluator(settings), clock)
    # The worker's own timed-out commit is fenced out past the deadline (the
    # watchdog owns that transition); the next empty claim runs maintenance,
    # whose deadline watchdog terminalizes the resource.
    assert worker.run_once() is True
    assert worker.run_once() is False
    resource = read_resource(store, "timed_out")
    assert resource.result.correctness.error.code == "DEADLINE_EXCEEDED"


def test_late_delivery_deadline_completion_preserves_requested_checks(
    store, settings, clock, make_command
):
    """Late-delivery completion keeps the independently pinned compile slot."""
    cmd = make_command(
        request=make_request(schema_version="2.1", checks=["compile", "correctness"])
    )
    store.create_or_replay(cmd)
    lease = store.claim_next(
        WorkerIdentity(new_worker_id(), settings.evaluator_build, algorithms(settings))
    )
    worker = build_worker(store, settings, StubEvaluator(settings), clock)
    # Directly exercise the worker's own synthesis; the commit wins the CAS
    # while the lease fence still holds.
    worker._commit_deadline(lease, ["compile", "correctness"], {})
    resource = read_resource(store, "timed_out")
    # The compile slot carries the timeout carrier; correctness is skipped.
    assert resource.result.compile.status == "timeout"
    assert resource.result.compile.error.code == "DEADLINE_EXCEEDED"
    assert resource.result.correctness.status == "skipped"
    assert resource.result.correctness.reason.code == "RESOURCE_DEADLINE"
    # Schema-valid: the stored terminal bytes re-validate against the models
    # with the full requested-checks context.
    EvaluationResource.model_validate_json(
        store._r.hget(resource_state_key(store, "timed_out"), b"resource")
    )


def test_late_delivery_watchdog_terminal_carries_compile_slot(
    store, settings, clock, make_command
):
    """End-to-end: a 2.1 compile+correctness request delivered past its
    deadline terminalizes with the compile carrier on the wire."""
    cmd = make_command(
        request=make_request(schema_version="2.1", checks=["compile", "correctness"])
    )
    store.create_or_replay(cmd)
    original_claim = store.claim_next

    def claim_and_expire(worker_identity, avoid_card=None, *, card=None):
        lease = original_claim(worker_identity, avoid_card=avoid_card, card=card)
        if lease is not None:
            clock.advance_ms(lease.deadline_at_ms - clock.now_ms + 1)
        return lease

    store.claim_next = claim_and_expire
    worker = build_worker(store, settings, StubEvaluator(settings), clock)
    assert worker.run_once() is True  # late-delivery commit fenced out
    assert worker.run_once() is False  # maintenance: watchdog terminalizes
    resource = read_resource(store, "timed_out")
    assert resource.result.compile.status == "timeout"
    assert resource.result.compile.error.code == "DEADLINE_EXCEEDED"


def test_stale_terminal_commit_is_accepted(store, settings, clock, queued):
    evaluator = StubEvaluator(
        settings, hook=lambda _stub: sabotage_attempt(store, settings)
    )
    worker = build_worker(store, settings, evaluator, clock)
    before = fenced_writes(store)
    assert worker.run_once() is True  # False from commit_terminal is accepted
    assert fenced_writes(store) > before


def test_corrupted_canonical_request_fails_closed(store, settings, make_command,
                                                  clock):
    resource = store.create_or_replay(make_command()).resource
    key = f"{settings.redis_prefix}:resource:{resource.evaluation_id}"
    payload = json.loads(store._r.hget(key, b"canonical_request"))
    tampered = json.dumps(dict(reversed(list(payload.items())))).encode()
    store._r.hset(key, "canonical_request", tampered)

    evaluator = StubEvaluator(settings)
    worker = build_worker(store, settings, evaluator, clock)
    assert worker.run_once() is True
    assert not evaluator.calls  # never materialized
    resource = read_resource(store, "failed")
    assert resource.result.correctness.error.code == "INTERNAL"


# ---------------------------------------------------------------------------
# Lease loss discards work
# ---------------------------------------------------------------------------


def test_lease_loss_during_execution_discards_work(store, settings, clock, queued):
    observed = {}

    def hook(stub):
        sabotage_attempt(store, settings)
        lease_valid = stub.calls[0]["lease_valid"]
        observed["invalid"] = wait_until(lambda: not lease_valid())

    evaluator = StubEvaluator(settings, hook=hook)
    worker = build_worker(store, settings, evaluator, clock)
    before = fenced_writes(store)
    assert worker.run_once() is True
    assert observed.get("invalid") is True
    assert fenced_writes(store) > before
    # No terminal result was committed; the reaper owns recovery.
    assert current_resource(store).state == "running"


def test_renewal_outage_invalidates_and_discards(store, settings, clock, queued):
    observed = {}

    def hook(stub):
        def broken_renew(lease):
            raise redis.exceptions.ConnectionError("connection lost")

        store.renew_lease = broken_renew
        lease_valid = stub.calls[0]["lease_valid"]
        observed["invalid"] = wait_until(lambda: not lease_valid())

    evaluator = StubEvaluator(settings, hook=hook)
    worker = build_worker(store, settings, evaluator, clock)
    assert worker.run_once() is True
    assert observed.get("invalid") is True
    assert current_resource(store).state == "running"


class TestLeaseKeeperUnit:
    @pytest.fixture
    def claimed_lease(self, store, settings, make_command, make_worker, clock):
        store.create_or_replay(make_command())
        return store.claim_next(make_worker())

    def test_renewal_keeps_validity_while_stopped_on_exit(
        self, store, settings, claimed_lease
    ):
        keeper = LeaseKeeper(store, claimed_lease, settings, interval=0.05)
        with keeper:
            time.sleep(0.15)
            assert keeper.is_valid()
        assert not keeper.is_valid()

    def test_false_renewal_permanently_clears(self, store, settings, claimed_lease):
        keeper = LeaseKeeper(store, claimed_lease, settings, interval=0.05)
        with keeper:
            store.renew_lease = lambda lease: False
            assert wait_until(lambda: not keeper.is_valid())
            store.renew_lease = lambda lease: True
            time.sleep(0.15)
            assert not keeper.is_valid()  # permanently cleared

    def test_redis_error_permanently_clears(self, store, settings, claimed_lease):
        keeper = LeaseKeeper(store, claimed_lease, settings, interval=0.05)
        with keeper:
            def broken(lease):
                raise redis.exceptions.ConnectionError("boom")

            store.renew_lease = broken
            assert wait_until(lambda: not keeper.is_valid())


# ---------------------------------------------------------------------------
# Loop behavior
# ---------------------------------------------------------------------------


def test_serve_forever_stops_on_event_and_backs_off(store, settings, clock):
    import threading

    worker = build_worker(store, settings, StubEvaluator(settings), clock)
    stop = threading.Event()
    attempts = []

    def flaky_run_once(**kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise redis.exceptions.ConnectionError("redis down")
        stop.set()
        return False

    worker.run_once = flaky_run_once
    sleeps = []
    worker.serve_forever(stop_event=stop, sleep=sleeps.append)
    assert len(attempts) == 3
    assert sleeps and all(s > 0 for s in sleeps)


def test_worker_never_logs_source_or_paths(store, settings, clock, queued, caplog):
    evaluator = StubEvaluator(
        settings, error=RuntimeError("Traceback: /abs/path/secret.py import torch")
    )
    worker = build_worker(store, settings, evaluator, clock)
    with caplog.at_level("DEBUG"):
        worker.run_once()
    text = caplog.text
    assert "torch" not in text
    assert "/abs/path" not in text
    assert "Traceback" not in text


# ---------------------------------------------------------------------------
# Card health: canary gating, drain-on-unhealthy, recovery ladder
# ---------------------------------------------------------------------------


@pytest.fixture
def canary_settings(settings):
    """Startup suspicion enabled; the card-health tests opt in explicitly."""
    return settings.model_copy(update={"card_canary_on_startup": True})


class _ScriptedCanary:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, *, device_id, budget_seconds):
        self.calls += 1
        return self._outcomes.pop(0)


def _healthy_worker(store, settings, evaluator, clock, canary, **kwargs):
    return build_worker(
        store, settings, evaluator, clock, canary=canary,
        recovery_step=lambda step, *, card, reset_command: (
            "cleanup" if step == 1 else "reset"
        ),
        **kwargs,
    )


def test_canary_runs_before_first_claim_when_startup_canary_enabled(store, canary_settings,
                                                                    clock, make_command):
    canary = _ScriptedCanary([True])
    worker = _healthy_worker(
        store, canary_settings,
        StubEvaluator(canary_settings, result=make_completed_result(canary_settings)),
        clock, canary,
    )
    store.create_or_replay(make_command())
    assert worker.run_once() is True
    assert canary.calls == 1


def test_suspicious_ending_triggers_canary_before_next_claim(store, canary_settings,
                                                             clock, make_command):
    canary = _ScriptedCanary([True, True])
    evaluator = StubEvaluator(canary_settings,
                              result=make_completed_result(canary_settings))
    worker = _healthy_worker(store, canary_settings, evaluator, clock, canary)
    store.create_or_replay(make_command())
    assert worker.run_once() is True
    assert canary.calls == 1  # startup
    evaluator.error = RuntimeError("boom")  # protocol failure -> suspicious
    store.create_or_replay(make_command(session_id="s-2", idempotency_key="k-2"))
    assert worker.run_once() is True
    assert canary.calls == 2  # suspicion triggered a re-canary


def test_failed_canary_drains_worker_and_skips_claim_and_heartbeat(store, canary_settings,
                                                                   clock, make_command):
    # Outcome budget: the probe consumes 1; each ladder rung re-probes once
    # (card_recovery_max_step=2).
    canary = _ScriptedCanary([False, False, False])
    worker = _healthy_worker(store, canary_settings, StubEvaluator(canary_settings),
                             clock, canary)
    resource = store.create_or_replay(make_command()).resource
    assert worker.run_once() is False
    assert store.is_card_unhealthy(0)
    assert store.get_visible(resource.evaluation_id, make_caller()).state == "queued"
    assert not store.has_fresh_compatible_worker(worker.identity)


def test_avoid_card_passed_to_claim(store, canary_settings, clock, make_command):
    from unittest import mock

    canary = _ScriptedCanary([True])
    worker = _healthy_worker(
        store, canary_settings,
        StubEvaluator(canary_settings, result=make_completed_result(canary_settings)),
        clock, canary,
    )
    store.create_or_replay(make_command())
    with mock.patch.object(store, "claim_next", wraps=store.claim_next) as spy:
        worker.run_once()
    assert spy.call_args.kwargs.get("avoid_card") == 0


def test_drained_worker_resumes_claims_after_recovery_ladder_succeeds(
    store, canary_settings, clock, make_command,
):
    worker = Worker(
        store, StubEvaluator(canary_settings), device_id=0,
        settings=canary_settings, clock=clock,
        canary=_ScriptedCanary([False, False, False, False, True]),
        recovery_step=lambda step, **kwargs: "cleanup" if step == 1 else "reset",
    )
    resource = store.create_or_replay(make_command()).resource
    assert worker.run_once() is False
    assert store.is_card_unhealthy(0)

    clock.advance_ms(300_001)
    # The drained probe fails, then cleanup restores the card. The recovery
    # iteration does not claim; the next iteration must resume useful work.
    assert worker.run_once() is False
    assert not store.is_card_unhealthy(0)
    assert worker.run_once() is True
    assert store.get_visible(resource.evaluation_id, make_caller()).state == "completed"
    assert store.has_fresh_compatible_worker(worker.identity)


def test_worker_records_attempt_card_even_when_affinity_avoidance_is_disabled(
    store, settings, clock, make_command,
):
    settings = settings.model_copy(update={"card_affinity_avoidance": False})
    worker = Worker(
        store, StubEvaluator(settings, error=LeaseLostError()),
        device_id=0, settings=settings, clock=clock, canary=lambda **kwargs: True,
    )
    resource = store.create_or_replay(make_command()).resource
    for _ in range(2):
        assert worker.run_once() is True
        clock.advance_ms(settings.lease_seconds * 1000 + 1)
        counts = store.recover_expired_leases()
    # Both deaths have provenance even though the same card may claim again.
    assert counts.failed == 1
    terminal = store.get_visible(resource.evaluation_id, make_caller())
    assert terminal.state == "failed"
    assert "poison" in terminal.result.correctness.error.message


def test_shape_dependent_compile_error_is_not_reused_for_other_inputs(store, settings, clock,
                                                    make_command, redis_client):
    from triton_evaluation.runtime import verdict_cache

    settings = settings.model_copy(update={"cann_fingerprint": "cann-test"})
    result = make_completed_result(settings)
    result.correctness.data.outcome = "compile_error"
    result.correctness.data.passed = False
    result.correctness.data.domain_failure = DomainFailure(
        code="SUBJECT_COMPILE_ERROR", phase="compile",
        message="candidate compilation failed",
    )
    cache = verdict_cache.VerdictCache(redis_client, prefix=settings.redis_prefix)
    worker = build_worker(
        store, settings, StubEvaluator(settings, result=result), clock, verdict_cache=cache,
    )
    cmd = make_command()
    store.create_or_replay(cmd)
    assert worker.run_once() is True
    request = json.loads(cmd.admission.canonical_request)
    digest = verdict_cache.digest_for(
        request["kernel_code"], request.get("triton_impl_name", "triton_ascend_impl"),
        settings.cann_fingerprint,
    )
    assert digest is not None
    assert cache.get(digest) is None


def test_correct_terminal_does_not_feed_cache(store, settings, clock, make_command, redis_client):
    from triton_evaluation.runtime import verdict_cache

    cache = verdict_cache.VerdictCache(redis_client, prefix=settings.redis_prefix)
    worker = build_worker(
        store, settings, StubEvaluator(settings, result=make_completed_result(settings)),
        clock, verdict_cache=cache,
    )
    cmd = make_command()
    store.create_or_replay(cmd)
    assert worker.run_once() is True
    request = json.loads(cmd.admission.canonical_request)
    digest = verdict_cache.digest_for(
        request["kernel_code"], request.get("triton_impl_name", "triton_ascend_impl"),
        settings.cann_fingerprint,
    )
    assert cache.get(digest) is None


@pytest.mark.parametrize("malformed", [object(), {}, {"correctness": None}])
def test_malformed_evaluator_result_is_failed_without_waiting_for_reaper(
    store, settings, clock, queued, malformed,
):
    evaluator = StubEvaluator(settings, result=malformed)
    worker = build_worker(store, settings, evaluator, clock)
    assert worker.run_once()
    resource = read_resource(store, "failed")
    assert resource.result.correctness.error.code == "EVALUATOR_EXECUTION_FAILED"
    assert resource.result.provenance.grade_eligible is False


def test_evaluator_cannot_replace_admission_pinned_provenance(
    store, settings, clock, queued,
):
    result = make_completed_result(settings)
    forged = result.model_copy(update={
        "provenance": result.provenance.model_copy(update={
            "evaluator_build": "sha256:" + "a" * 64,
            "grade_eligible": True,
            "manifest_id": "forged-manifest", "problem_id": "forged-problem",
        }),
    })
    worker = build_worker(store, settings, StubEvaluator(settings, result=forged), clock)
    assert worker.run_once()
    provenance = read_resource(store, "completed").result.provenance
    assert provenance.evaluator_build == settings.evaluator_build
    assert provenance.manifest_id is None
    assert provenance.problem_id is None
    assert provenance.grade_eligible is False


@pytest.mark.parametrize("corruption", [b"not-json", b"{}"])
def test_corrupted_compile_request_keeps_cpu_check_identity(
    store, settings, clock, make_command, corruption,
):
    command = make_command(request=make_request(checks=["compile"]))
    admitted = store.create_or_replay(command).resource
    key = f"{settings.redis_prefix}:resource:{admitted.evaluation_id}"
    store._r.hset(key, "canonical_request", corruption)
    evaluator = StubEvaluator(settings)
    worker = build_worker(store, settings, evaluator, clock, device_id=None)
    assert worker.run_once()
    assert not evaluator.calls
    resource = read_resource(store, "failed")
    assert resource.result.compile.error.code == "INTERNAL"
    assert resource.result.correctness is None
    assert resource.result.provenance.grade_eligible is False


def test_canonical_source_replacement_cannot_change_admitted_request_identity(
    store, settings, clock, make_command,
):
    from triton_evaluation.serialization import canonical_json_bytes

    resource = store.create_or_replay(make_command()).resource
    key = f"{settings.redis_prefix}:resource:{resource.evaluation_id}"
    request = json.loads(store._r.hget(key, "canonical_request"))
    request["kernel_code"] = "class ReplacedCandidate: pass"
    store._r.hset(key, "canonical_request", canonical_json_bytes(request))
    evaluator = StubEvaluator(settings)
    worker = build_worker(store, settings, evaluator, clock)
    assert worker.run_once()
    assert not evaluator.calls
    failed = read_resource(store, "failed")
    assert failed.result.correctness.error.code == "INTERNAL"
