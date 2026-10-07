"""Retryable pre-subject engine failures requeue instead of terminalizing.

Covers the 2026-10-07 uni-agent grade failures: EVALUATOR_EXECUTION_FAILED
(engine exit=1, retryable, phase=execution, no subject outcome) terminalized
the resource on the first attempt while other cards were healthy. The fix
requeues the same resource under the shared MAX_ATTEMPTS claim budget and
publishes last_failed_card so affinity avoidance steers the retry away.
"""
from conftest import (
    make_caller,
    make_completed_result,
    make_correctness_success,
    make_request,
)

from triton_evaluation.api.contract import (
    CheckFailed,
    CheckSkipped,
    ErrorInfo,
    EvaluationResult,
    SkipInfo,
)
from triton_evaluation.broker import MAX_ATTEMPTS
from triton_evaluation.worker import _retryable_engine_failure


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


def test_requeue_shares_the_attempt_budget(
    store, make_command, make_worker, clock, settings
):
    command = make_command()
    store.create_or_replay(command)
    for _ in range(MAX_ATTEMPTS):
        lease = store.claim_next(make_worker())
        assert lease is not None
        if store.requeue_retryable_engine_failure(lease, failed_card=0) is False:
            break  # budget exhausted exactly at the last claim
    # No further requeue is possible; the next claim never happens.
    final = None
    for _ in range(MAX_ATTEMPTS):
        lease = store.claim_next(make_worker())
        if lease is None:
            break
        final = lease
    if final is not None:
        assert store.requeue_retryable_engine_failure(final, failed_card=0) is False


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
