"""Opt-in retries of unknown engine failures preserve evidence and fencing."""
import pytest
from conftest import (
    make_caller,
    make_completed_result,
    make_request,
)
from test_worker import StubEvaluator, _healthy_worker, _ScriptedCanary, build_worker

from triton_evaluation.api.contract import (
    CheckFailed,
    CheckSkipped,
    ErrorInfo,
    EvaluationResult,
    SkipInfo,
)
from triton_evaluation.runtime.worker import _retryable_engine_failure
from triton_evaluation.service.broker import MAX_ATTEMPTS
from triton_evaluation.service.terminal_results import compile_outcome_from_precheck


def test_engine_failure_retry_requires_explicit_opt_in(settings):
    assert settings.engine_failure_requeue is False


def _profile_failure_result(settings):
    result = _engine_failure_result(settings)
    result.profile = result.correctness.model_copy(deep=True)
    result.profile.error.check = "profile"
    result.correctness = result.performance = None
    return EvaluationResult.model_validate(result.model_dump())


def test_profile_failure_without_evidence_can_retry(settings):
    assert _retryable_engine_failure(_profile_failure_result(settings))


def test_profile_only_worker_retries_on_another_card(store, make_command, settings, clock):
    settings = settings.model_copy(update={"engine_failure_requeue": True})
    command = make_command(request=make_request(checks=["profile"], profile={"mode": "op"}))
    store.create_or_replay(command)
    evaluator = StubEvaluator(settings, result=_profile_failure_result(settings))
    workers = [
        build_worker(
            store, settings, evaluator, clock, device_id=card, canary=lambda **kwargs: True,
        )
        for card in (0, 1)
    ]
    assert workers[0].run_once()
    assert not workers[0].run_once()
    assert workers[1].run_once()
    assert workers[0].run_once()
    result = store.get_visible(command.evaluation_id, make_caller())
    assert result.state == "completed"
    assert result.result.profile.error.code == "EVALUATOR_EXECUTION_FAILED"
    assert len(evaluator.calls) == MAX_ATTEMPTS


def test_failed_health_probe_blocks_same_card_fallback(store, make_command, settings, clock):
    settings = settings.model_copy(update={"engine_failure_requeue": True})
    command = make_command(request=make_request(checks=["correctness", "performance"]))
    store.create_or_replay(command)
    evaluator = StubEvaluator(settings, result=_engine_failure_result(settings))
    worker = build_worker(
        store, settings, evaluator, clock, canary=lambda **kwargs: False,
        recovery_step=lambda *args, **kwargs: "noop",
    )
    assert worker.run_once()
    assert store.is_card_unhealthy(0)
    clock.advance_ms(settings.lease_seconds * 1000)
    assert not worker.run_once()
    assert len(evaluator.calls) == 1


def test_failure_at_deadline_still_checks_card_health(store, make_command, settings, clock):
    command = make_command(request=make_request(
        checks=["correctness", "performance"], deadline_seconds=10,
    ))
    store.create_or_replay(command)
    probes = []

    def canary(**kwargs):
        probes.append(kwargs["device_id"])
        return True

    evaluator = StubEvaluator(
        settings, result=_engine_failure_result(settings), hook=lambda _: clock.advance_ms(10_000),
    )
    worker = build_worker(store, settings, evaluator, clock, canary=canary)
    assert worker.run_once()
    assert probes == [0]
    # Past-deadline writes are fenced; the deadline watchdog owns the terminal.
    assert store.expire_due_resources() == 1
    assert store.get_visible(command.evaluation_id, make_caller()).state == "timed_out"


def test_other_check_evidence_prevents_retry(settings):
    result = _engine_failure_result(settings)
    result.compile = compile_outcome_from_precheck("precheck_passed")
    result = EvaluationResult.model_validate(result.model_dump())
    assert not _retryable_engine_failure(result)


@pytest.mark.parametrize("deadline_seconds", [10, 1800])
def test_failed_card_can_retry_before_deadline_after_bounded_avoidance(
    store, make_command, make_worker, settings, clock, deadline_seconds,
):
    command = make_command(request=make_request(deadline_seconds=deadline_seconds))
    store.create_or_replay(command)
    worker = make_worker()
    lease = store.claim_next(worker, card=0, avoid_card=0)
    assert store.requeue_retryable_engine_failure(lease, failed_card=0)
    assert store.claim_next(worker, card=0, avoid_card=0) is None
    wait_ms = min(settings.lease_seconds * 1000, deadline_seconds * 1000 // 2)
    clock.advance_ms(wait_ms)
    replacement = store.claim_next(worker, card=0, avoid_card=0)
    assert replacement is not None
    assert clock() < replacement.deadline_at_ms
    assert not store.commit_terminal(lease, "completed", make_completed_result(settings))
    assert store.commit_terminal(replacement, "completed", make_completed_result(settings))


@pytest.mark.parametrize("retry_enabled", [True, False])
def test_engine_failure_probes_card_even_when_retry_is_disabled(
    store, make_command, settings, clock, retry_enabled,
):
    settings = settings.model_copy(update={"engine_failure_requeue": retry_enabled})
    command = make_command(request=make_request(checks=["correctness", "performance"]))
    store.create_or_replay(command)
    probes = []

    def canary(**kwargs):
        probes.append(kwargs["device_id"])
        return True

    worker = build_worker(
        store, settings, StubEvaluator(settings, result=_engine_failure_result(settings)),
        clock, canary=canary,
    )
    assert worker.run_once()
    assert probes == [0]
    visible = store.get_visible(command.evaluation_id, make_caller())
    assert visible.state == ("running" if retry_enabled else "completed")


def test_single_card_worker_exhausts_shared_budget_and_checks_health(
    store, make_command, settings, clock, redis_client,
):
    settings = settings.model_copy(update={"engine_failure_requeue": True})
    command = make_command(request=make_request(checks=["correctness", "performance"]))
    store.create_or_replay(command)
    probes = []

    def canary(**kwargs):
        probes.append(kwargs["device_id"])
        return True

    evaluator = StubEvaluator(settings, result=_engine_failure_result(settings))
    worker = build_worker(store, settings, evaluator, clock, canary=canary)
    for attempt in range(MAX_ATTEMPTS):
        assert worker.run_once()
        assert len(probes) == attempt + 1
        assert not worker.run_once()
        clock.advance_ms(settings.lease_seconds * 1000)
    assert len(evaluator.calls) == MAX_ATTEMPTS
    assert store.get_visible(command.evaluation_id, make_caller()).state == "completed"
    resource_key = _key(settings, f"resource:{command.evaluation_id}")
    assert redis_client.hget(resource_key, "attempt_count") == str(MAX_ATTEMPTS).encode()


def _key(settings, suffix):
    return f"{settings.redis_prefix}:{suffix}"


def _engine_failure_result(settings):
    """The exact terminal shape the evaluator produced for the uni grades."""
    from conftest import make_provenance

    return EvaluationResult(
        correctness=CheckFailed(
            status="failed",
            error=ErrorInfo(
                code="EVALUATOR_EXECUTION_FAILED",
                phase="execution",
                check="correctness",
                retryable=True,
                message="verify engine produced no usable result (engine exit=1)",
            ),
        ),
        performance=CheckSkipped(
            status="skipped",
            reason=SkipInfo(
                code="PRIOR_CHECK_FAILED",
                depends_on="correctness",
                message="a prior requested check did not complete",
            ),
        ),
        profile=None,
        provenance=make_provenance(settings),
    )


def test_decision_helper_matches_only_retryable_engine_failures(settings):
    failed = _engine_failure_result(settings)
    assert _retryable_engine_failure(failed) is True

    # A produced verdict never requeues.
    completed = make_completed_result(settings)
    assert _retryable_engine_failure(completed) is False

    # Non-retryable execution failures (watchdog/poison) stay terminal.
    watchdog = failed.model_copy(deep=True)
    fixed = CheckFailed(
        status="failed",
        error=ErrorInfo(
            code="DEADLINE_EXCEEDED",
            phase="deadline",
            check="correctness",
            retryable=False,
            message="correctness exceeded the resource deadline",
        ),
    )
    watchdog.correctness = fixed
    assert _retryable_engine_failure(watchdog) is False


def test_requeue_publishes_replacement_and_card_avoidance(
    store, make_command, make_worker, clock, settings, redis_client
):
    command = make_command(request=make_request(deadline_seconds=1800))
    store.create_or_replay(command)
    worker = make_worker()
    lease = store.claim_next(worker, card=6)
    assert lease is not None

    assert (
        store.requeue_retryable_engine_failure(lease, failed_card=6) is True
    )

    stored = redis_client.hgetall(_key(settings, f"resource:{command.evaluation_id}"))
    assert stored[b"replacement_pending"] == b"1"
    assert stored[b"attempt_id"] == b""
    assert stored[b"last_failed_card"] == b"6"
    assert stored[b"fencing_token"] == str(lease.fencing_token + 1).encode()
    assert redis_client.zscore(_key(settings, "running-leases"), command.evaluation_id) is None
    assert redis_client.zrange(_key(settings, "ready:npu:interactive"), 0, -1) == [
        command.evaluation_id.encode()
    ]

    # The failed card's worker skips the requeued member; another card claims it.
    assert store.claim_next(worker, card=6, avoid_card=6) is None
    replacement = store.claim_next(make_worker(), card=4, avoid_card=4)
    assert replacement is not None
    assert replacement.fencing_token == lease.fencing_token + 1
    # The retry can still terminalize normally.
    assert (
        store.commit_terminal(replacement, "completed", make_completed_result(settings))
        is True
    )


def test_requeue_shares_the_attempt_budget(store, make_command, make_worker, clock, settings):
    command = make_command()
    store.create_or_replay(command)
    # A lost lease consumes the first attempt; engine retries share that budget.
    assert store.claim_next(make_worker()) is not None
    clock.advance_ms(settings.lease_seconds * 1000 + 1)
    assert store.recover_expired_leases().requeued == 1
    for attempt in range(2, MAX_ATTEMPTS + 1):
        lease = store.claim_next(make_worker())
        assert lease is not None
        assert store.requeue_retryable_engine_failure(lease) is (attempt < MAX_ATTEMPTS)
    assert store.commit_terminal(lease, "completed", make_completed_result(settings))
    assert store.claim_next(make_worker()) is None


def test_engine_avoidance_does_not_weaken_later_lease_loss_policy(
    store, make_command, make_worker, settings, clock,
):
    command = make_command()
    store.create_or_replay(command)
    worker = make_worker()
    first = store.claim_next(worker, card=0)
    assert store.requeue_retryable_engine_failure(first, failed_card=0)
    assert store.claim_next(worker, card=1, avoid_card=1) is not None
    clock.advance_ms(settings.lease_seconds * 1000 + 1)
    assert store.recover_expired_leases().requeued == 1
    # The previous engine's avoidance timer must not permit card 1 to pick up
    # its own lost lease; the lease reaper retains its existing poison policy.
    assert store.claim_next(worker, card=1, avoid_card=1) is None
    assert store.claim_next(worker, card=0, avoid_card=0) is not None


def test_deadline_rejects_requeue(store, make_command, make_worker, clock):
    command = make_command(request=make_request(deadline_seconds=10))
    store.create_or_replay(command)
    lease = store.claim_next(make_worker(), card=0)
    clock.advance_ms(10_000)
    assert not store.requeue_retryable_engine_failure(lease, failed_card=0)
    assert store.expire_due_resources() == 1
    assert store.get_visible(command.evaluation_id, make_caller()).state == "timed_out"


def test_stale_lease_cannot_requeue(
    store, make_command, make_worker, clock, settings
):
    command = make_command()
    store.create_or_replay(command)
    lease = store.claim_next(make_worker())
    assert lease is not None
    assert store.commit_terminal(lease, "completed", make_completed_result(settings))
    # The terminalized resource is no longer this lease's to requeue.
    assert store.requeue_retryable_engine_failure(lease, failed_card=3) is False
    visible = store.get_visible(command.evaluation_id, make_caller())
    assert visible.state == "completed"


def _input_generation_result(settings, code="INPUT_GENERATION_FAILED", retryable=False):
    result = _engine_failure_result(settings)
    result.correctness = CheckFailed(
        status="failed",
        error=ErrorInfo(
            code=code, phase="execution", check="correctness", retryable=retryable,
            message="workload bad (case 2) inputs cannot be generated: uniform low > high",
        ),
    )
    return result


@pytest.mark.parametrize(
    ("code", "retryable", "suspected"),
    [("EVALUATOR_EXECUTION_FAILED", True, True), ("INPUT_GENERATION_FAILED", False, False)],
)
def test_refused_workload_never_suspects_the_card(
    store, make_command, settings, clock, code, retryable, suspected
):
    settings = settings.model_copy(update={"card_canary_on_startup": True})
    canary = _ScriptedCanary([True] * 8)
    evaluator = StubEvaluator(settings, result=_input_generation_result(settings, code, retryable))
    worker = _healthy_worker(store, settings, evaluator, clock, canary)
    checks = make_request(checks=["correctness", "performance"])
    store.create_or_replay(make_command(request=checks))
    assert worker.run_once() is True
    store.create_or_replay(
        make_command(request=checks, session_id="s-2", idempotency_key="k-2")
    )
    assert worker.run_once() is True
    # Only the startup canary runs unless a failure the card may own re-probes it.
    assert (canary.calls > 1) is suspected


def test_refused_workload_is_terminal_even_with_requeue(store, make_command, settings, clock):
    settings = settings.model_copy(update={"engine_failure_requeue": True})
    result = _input_generation_result(settings)
    assert _retryable_engine_failure(result) is False
    command = make_command(request=make_request(checks=["correctness", "performance"]))
    store.create_or_replay(command)
    evaluator = StubEvaluator(settings, result=result)
    worker = build_worker(store, settings, evaluator, clock, canary=lambda **kwargs: True)
    assert worker.run_once() is True
    visible = store.get_visible(command.evaluation_id, make_caller())
    assert visible.state == "completed"
    assert visible.result.correctness.error.code == "INPUT_GENERATION_FAILED"
    assert len(evaluator.calls) == 1
