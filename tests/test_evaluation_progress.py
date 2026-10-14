"""Progress reveals fenced evidence without weakening terminal resource contracts."""
from conftest import make_caller, make_completed_result, make_correctness_success


def claimed(store, make_command, make_worker):
    command = make_command()
    store.create_or_replay(command)
    lease = store.claim_next(make_worker())
    assert lease is not None
    return command, lease


def test_check_evidence_is_visible_before_terminal(store, make_command, make_worker):
    command, lease = claimed(store, make_command, make_worker)
    caller = make_caller()
    initial = store.get_progress_visible(command.evaluation_id, caller)
    assert initial.completed_checks.correctness is None
    store.set_active_check(lease, "correctness")
    active = store.get_progress_visible(command.evaluation_id, caller)
    assert active.active_check == "correctness"
    assert active.progress_revision > initial.progress_revision
    store.save_check_outcome(lease, "correctness", make_correctness_success())
    observed = store.get_progress_visible(command.evaluation_id, caller)
    assert observed.state == "running"
    assert observed.completed_checks.correctness.data.outcome == "correct"
    assert observed.attempt_id == lease.attempt_id
    assert observed.active_check is None
    assert observed.progress_revision > active.progress_revision
    assert store.get_visible(command.evaluation_id, caller).result is None


def test_progress_uses_resource_authorization(store, make_command, make_worker):
    command, _ = claimed(store, make_command, make_worker)
    assert store.get_progress_visible(command.evaluation_id, make_caller("someone-else")) is None
    assert store.get_progress_visible(
        command.evaluation_id, make_caller(scopes=("trial:submit",))
    ) is None
    observer = make_caller("observer", "observer", ("observer:read",))
    assert store.get_progress_visible(command.evaluation_id, observer) is not None
    assert store.get_progress_visible("ev_" + "0" * 32, observer) is None


def test_replacement_hides_previous_attempt_evidence(
    store, make_command, make_worker, settings, clock,
):
    command, first = claimed(store, make_command, make_worker)
    store.save_check_outcome(first, "correctness", make_correctness_success())
    before = store.get_progress_visible(command.evaluation_id, make_caller())
    clock.advance_ms(settings.lease_seconds * 1000 + 1)
    store.recover_expired_leases()
    pending = store.get_progress_visible(command.evaluation_id, make_caller())
    assert pending.completed_checks.correctness is None
    assert pending.progress_revision > before.progress_revision
    second = store.claim_next(make_worker())
    assert second is not None and second.attempt_id != first.attempt_id
    assert store.save_check_outcome(first, "correctness", make_correctness_success()) is False
    observed = store.get_progress_visible(command.evaluation_id, make_caller())
    assert observed.attempt_id == second.attempt_id
    assert observed.progress_revision > pending.progress_revision
    assert observed.completed_checks.correctness is None


def test_invalid_private_outcome_is_not_published(
    store, make_command, make_worker, redis_client, settings,
):
    command, _ = claimed(store, make_command, make_worker)
    redis_client.hset(
        f"{settings.redis_prefix}:resource:{command.evaluation_id}",
        "check_outcomes", '{"correctness":{"status":"succeeded","data":{}}}',
    )
    assert store.get_progress_visible(
        command.evaluation_id, make_caller()
    ).completed_checks.correctness is None


def test_terminal_progress_retains_its_evidence_attempt(store, make_command, make_worker, settings):
    command, lease = claimed(store, make_command, make_worker)
    store.save_check_outcome(lease, "correctness", make_correctness_success())
    before = store.get_progress_visible(command.evaluation_id, make_caller())
    assert store.commit_terminal(lease, "completed", make_completed_result(settings))
    terminal = store.get_progress_visible(command.evaluation_id, make_caller())
    assert terminal.attempt_id == before.attempt_id == lease.attempt_id
    assert terminal.progress_revision > before.progress_revision
    assert terminal.active_check is None
    assert terminal.completed_checks.correctness.data.outcome == "correct"
