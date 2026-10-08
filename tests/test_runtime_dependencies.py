"""Explicit metrics dependencies remain advisory during durable execution."""
from threading import Event
from unittest import mock

from conftest import make_caller, make_completed_result
from test_worker import StubEvaluator, build_worker, wait_until

from triton_evaluation.runtime.worker import LeaseKeeper
from triton_evaluation.service.broker import EvaluationStore


def test_worker_metrics_failure_still_publishes_success(
    store, settings, clock, make_command,
):
    metrics = mock.Mock()
    metrics.increment.side_effect = RuntimeError("metrics unavailable")
    command = make_command()
    store.create_or_replay(command)
    worker = build_worker(
        store, settings, StubEvaluator(settings), clock, metrics=metrics,
    )

    assert worker.run_once()
    resource = store.get_visible(command.evaluation_id, make_caller())
    assert resource.state == "completed"
    assert resource.result.correctness.data.passed
    metrics.increment.assert_any_call("worker_attempts_total", {})


def test_lease_metrics_failure_preserves_renewal_and_eventual_fencing(
    store, settings, clock, make_command, make_worker,
):
    command = make_command()
    store.create_or_replay(command)
    lease = store.claim_next(make_worker())
    renewed = Event()
    rejected = Event()
    recovered = Event()
    metrics = mock.Mock()

    def unavailable_counter(name, labels, amount=1):
        if name == "lease_renewals_total":
            if labels["outcome"] == "renewed":
                renewed.set()
                recovered.wait(2)
            else:
                rejected.set()
        raise RuntimeError("metrics unavailable")

    metrics.increment.side_effect = unavailable_counter
    with LeaseKeeper(store, lease, settings, interval=0.01, metrics=metrics) as keeper:
        assert renewed.wait(2), "lease was not renewed"
        assert keeper.is_valid()
        clock.advance_ms(settings.lease_seconds * 1000 + 1)
        assert store.recover_expired_leases().requeued == 1
        recovered.set()
        assert rejected.wait(2), "replaced lease was not rejected"
        assert wait_until(lambda: not keeper.is_valid())
        assert not store.commit_terminal(lease, "completed", make_completed_result(settings))
    assert store.get_visible(command.evaluation_id, make_caller()).result is None


def test_postcommit_duration_metrics_failure_returns_durable_success(
    redis_client, settings, clock, make_command, make_worker,
):
    metrics = mock.Mock()
    metrics.observe.side_effect = RuntimeError("metrics unavailable")
    store = EvaluationStore(redis_client, settings, clock=clock, metrics=metrics)
    command = make_command()
    store.create_or_replay(command)
    lease = store.claim_next(make_worker())
    clock.advance_ms(250)

    assert store.commit_terminal(lease, "completed", make_completed_result(settings))
    resource = store.get_visible(command.evaluation_id, make_caller())
    assert resource.state == "completed"
    assert resource.result.correctness.data.passed
    metrics.observe.assert_called_once_with(
        "resource_terminal_duration_seconds", {"state": "completed"}, 0.25,
    )
