"""Crash recovery: an abandoned worker's token can never mutate the resource."""
from conftest import make_completed_result
from test_worker import build_worker

from triton_evaluation.api.contract import EvaluationResource


def test_abandoned_worker_token_cannot_mutate_after_replacement(
    store, settings, clock, make_command, make_worker
):
    resource = store.create_or_replay(make_command()).resource

    # Worker 1 claims the attempt, then dies without committing anything.
    first_identity = make_worker()
    lease1 = store.claim_next(first_identity)
    assert lease1 is not None

    # The lease expires; the reaper requeues the same evaluation.
    clock.advance_ms(settings.lease_seconds * 1000 + 1)
    counts = store.recover_expired_leases()
    assert counts.requeued == 1

    # Worker 2 claims the replacement attempt and commits a terminal result.
    evaluator = _RecordingEvaluator(settings)
    worker2 = build_worker(store, settings, evaluator, clock, identity=make_worker())
    assert worker2.run_once() is True

    key = f"{settings.redis_prefix}:resource:{resource.evaluation_id}"
    terminal = EvaluationResource.model_validate_json(store._r.hget(key, b"resource"))
    assert terminal.state == "completed"
    assert terminal.result.correctness.status == "succeeded"

    # The abandoned token cannot mutate or rewrite the persisted winner.
    outcome = make_completed_result(settings).correctness
    assert store.save_check_outcome(lease1, "correctness", outcome) is False
    assert store.set_active_check(lease1, "correctness") is False
    assert store.renew_lease(lease1) is False
    assert (
        store.commit_terminal(lease1, "failed", terminal.result.model_copy(deep=True))
        is False
    )
    after = EvaluationResource.model_validate_json(store._r.hget(key, b"resource"))
    assert after.state == "completed"
    assert after.revision == terminal.revision


def test_worker2_replaces_and_succeeds_after_crash_at_attempt_limit(
    store, settings, clock, make_command, make_worker
):
    """Crashes below the limit requeue; the reaper, not the worker, fails at it."""
    resource = store.create_or_replay(make_command()).resource

    for _ in range(2):  # attempts 1 and 2 crash
        lease = store.claim_next(make_worker())
        assert lease is not None
        clock.advance_ms(settings.lease_seconds * 1000 + 1)
        counts = store.recover_expired_leases()
        assert counts.requeued == 1

    # Third claim: the last permitted attempt.
    evaluator = _RecordingEvaluator(settings)
    worker = build_worker(store, settings, evaluator, clock, identity=make_worker())
    assert worker.run_once() is True

    key = f"{settings.redis_prefix}:resource:{resource.evaluation_id}"
    terminal = EvaluationResource.model_validate_json(store._r.hget(key, b"resource"))
    assert terminal.state == "completed"


class _RecordingEvaluator:
    def __init__(self, settings):
        self.settings = settings
        self.calls = []

    def run(
        self, request, work_dir, device_id, deadline_at, lease_valid,
        execution_environment, pinned, progress, *, compile_outcome=None,
    ):
        self.calls.append({"request": request})
        return make_completed_result(self.settings)
