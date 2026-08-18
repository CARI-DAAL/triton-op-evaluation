"""Resource visibility: owner scope rules, observer reads, uniform hiding."""
from types import SimpleNamespace

import pytest
from conftest import make_caller, make_grade_caller, make_grade_request


@pytest.fixture
def trial_resource(store, make_command):
    command = make_command()
    result = store.create_or_replay(command)
    assert result.decision == "created"
    return SimpleNamespace(command=command, resource=result.resource)


@pytest.fixture
def grade_resource(store, make_command):
    command = make_command(caller=make_grade_caller(), request=make_grade_request())
    result = store.create_or_replay(command)
    assert result.decision == "created"
    return SimpleNamespace(command=command, resource=result.resource)


def test_owner_with_purpose_read_scope_sees_resource(store, trial_resource):
    caller = make_caller("caller-1", "trial", ("trial:read",))
    visible = store.get_visible(trial_resource.resource.evaluation_id, caller)
    assert visible == trial_resource.resource


def test_owner_without_read_scope_gets_none(store, trial_resource):
    caller = make_caller("caller-1", "trial", ("trial:submit",))
    assert store.get_visible(trial_resource.resource.evaluation_id, caller) is None


def test_owner_with_wrong_purpose_scope_gets_none(store, trial_resource):
    caller = make_caller("caller-1", "trial", ("grade:read",))
    assert store.get_visible(trial_resource.resource.evaluation_id, caller) is None


def test_different_caller_gets_none(store, trial_resource):
    other = make_caller("caller-2", "trial", ("trial:submit", "trial:read"))
    assert store.get_visible(trial_resource.resource.evaluation_id, other) is None


def test_unknown_evaluation_id_gets_none(store, trial_resource):
    caller = make_caller("caller-1", "trial", ("trial:read",))
    assert store.get_visible("ev_" + "0" * 32, caller) is None


def test_guessed_session_grants_no_visibility(store, trial_resource):
    # The resource was created with session_id "session-1"; a different caller
    # who knows or guesses that correlation ID still cannot read the resource.
    # get_visible takes no session argument at all — session never authorizes.
    assert trial_resource.command.session_id == "session-1"
    guesser = make_caller("caller-2", "trial", ("trial:submit", "trial:read"))
    assert store.get_visible(trial_resource.resource.evaluation_id, guesser) is None


def test_observer_reads_without_owning(store, redis_client, settings, trial_resource):
    observer = make_caller("observer-1", "observer", ("observer:read",))

    visible = store.get_visible(trial_resource.resource.evaluation_id, observer)

    assert visible == trial_resource.resource
    # Observer reads never change ownership.
    resource_key = (
        f"{settings.redis_prefix}:resource:{trial_resource.resource.evaluation_id}"
    )
    assert redis_client.hget(resource_key, "owner") == b"caller-1"
    owner = make_caller("caller-1", "trial", ("trial:read",))
    assert store.get_visible(trial_resource.resource.evaluation_id, owner) is not None


def test_observer_reads_grade_resource(store, grade_resource):
    observer = make_caller("observer-1", "observer", ("observer:read",))
    assert store.get_visible(grade_resource.resource.evaluation_id, observer) is not None


def test_grade_owner_with_grade_read_sees_resource(store, grade_resource):
    caller = make_grade_caller("grader-1", ("grade:read",))
    visible = store.get_visible(grade_resource.resource.evaluation_id, caller)
    assert visible == grade_resource.resource


def test_grade_owner_without_grade_read_gets_none(store, grade_resource):
    caller = make_grade_caller("grader-1", ("grade:submit",))
    assert store.get_visible(grade_resource.resource.evaluation_id, caller) is None


def test_trusted_non_owner_with_grade_read_gets_none(store, grade_resource):
    other = make_grade_caller("grader-2", ("grade:submit", "grade:read"))
    assert store.get_visible(grade_resource.resource.evaluation_id, other) is None


def test_unknown_and_invisible_are_indistinguishable(store, trial_resource):
    caller = make_caller("caller-2", "trial", ("trial:read",))
    unknown = store.get_visible("ev_" + "0" * 32, caller)
    invisible = store.get_visible(trial_resource.resource.evaluation_id, caller)
    assert unknown is None
    assert invisible is None
    assert type(unknown) is type(invisible)
