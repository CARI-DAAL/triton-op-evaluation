import time

import pytest
import redis

from triton_evaluation.broker import Broker
from triton_evaluation.models import Job, JobSpec, JobType, TaskStatus


def _job(task_id="t1", type_="verify"):
    spec = JobSpec(op_name="softmax", reference_code="R", kernel_code="K")
    return Job(task_id=task_id, type=type_, spec=spec)


def test_create_then_get(broker):
    broker.create_task(_job())
    rec = broker.get_task("t1")
    assert rec["task_id"] == "t1"
    assert rec["type"] == "verify"
    assert rec["status"] == "QUEUED"
    assert isinstance(rec["created_at"], float)


def test_get_missing_returns_none(broker):
    assert broker.get_task("nope") is None


def test_enqueue_dequeue_roundtrip(broker):
    broker.create_task(_job("t2"))
    job = broker.dequeue()
    assert job.task_id == "t2"
    assert job.spec.op_name == "softmax"


def test_dequeue_empty_returns_none(broker):
    assert broker.dequeue() is None


def test_store_result_parses_json(broker):
    broker.create_task(_job("t3"))
    broker.store_result(
        "t3", TaskStatus.SUCCEEDED.value, result={"passed_cases": 3, "total_cases": 3}
    )
    rec = broker.get_task("t3")
    assert rec["status"] == "SUCCEEDED"
    assert rec["result"]["passed_cases"] == 3


def test_mark_running_sets_device(broker):
    broker.create_task(_job("t4"))
    broker.mark_running("t4", 5, deadline=time.time() + 100)
    rec = broker.get_task("t4")
    assert rec["status"] == "RUNNING"
    assert rec["device_id"] == 5


def test_dequeue_returns_none_on_socket_timeout(settings):
    # A blocking BRPOP that hits the client socket read timeout must be "no task",
    # not a crash — dequeue honours its Job | None contract.
    class _TimingOutClient:
        def brpop(self, *args, **kwargs):
            raise redis.exceptions.TimeoutError("read timed out")

    broker = Broker(_TimingOutClient(), settings)
    assert broker.dequeue() is None


def test_session_lease_has_single_owner(broker):
    assert broker.acquire_session("session-a", "task-1", 60) is True
    assert broker.acquire_session("session-a", "task-2", 60) is False
    assert broker.release_session("session-a", "task-2") is False
    assert broker.release_session("session-a", "task-1") is True
    assert broker.acquire_session("session-a", "task-2", 60) is True


def test_sessions_are_independent(broker):
    assert broker.acquire_session("session-a", "task-a", 60) is True
    assert broker.acquire_session("session-b", "task-b", 60) is True


def test_session_lease_rejects_non_positive_ttl(broker):
    with pytest.raises(ValueError, match="positive"):
        broker.acquire_session("session-a", "task-a", 0)


def test_submit_atomically_creates_lease_task_and_queue_item(broker):
    spec = JobSpec(
        op_name="softmax",
        reference_code="R",
        kernel_code="K",
        checks=["correctness"],
        session_id="session-a",
    )
    job = Job("task-a", JobType.EVALUATION.value, spec)

    assert broker.submit(job, lease_ttl_seconds=60) is True
    assert broker.get_task("task-a")["status"] == TaskStatus.QUEUED.value
    assert broker.get_task("task-a")["session_id"] == "session-a"
    assert broker.acquire_session("session-a", "other", 60) is False
    assert broker.dequeue().task_id == "task-a"


def test_submit_conflict_does_not_create_or_enqueue_second_task(broker):
    first = JobSpec(
        op_name="a",
        reference_code="R",
        kernel_code="K",
        checks=["correctness"],
        session_id="session-a",
    )
    second = JobSpec(
        op_name="b",
        reference_code="R",
        kernel_code="K",
        checks=["correctness"],
        session_id="session-a",
    )

    assert broker.submit(Job("task-a", "evaluation", first), 60) is True
    assert broker.submit(Job("task-b", "evaluation", second), 60) is False
    assert broker.get_task("task-b") is None
    assert broker.dequeue().task_id == "task-a"
    assert broker.dequeue() is None


def test_terminal_store_and_owned_lease_release_are_atomic(broker):
    spec = JobSpec(
        op_name="a",
        reference_code="R",
        kernel_code="K",
        checks=["correctness"],
        session_id="session-a",
    )
    assert broker.submit(Job("task-a", "evaluation", spec), 60) is True

    assert broker.store_result("task-a", "SUCCEEDED", result={"ok": True}) is True

    assert broker.get_task("task-a")["status"] == "SUCCEEDED"
    assert broker.acquire_session("session-a", "task-b", 60) is True


def test_old_terminal_store_does_not_delete_new_session_owner(broker):
    old_spec = JobSpec(
        op_name="a",
        reference_code="R",
        kernel_code="K",
        checks=["correctness"],
        session_id="session-a",
    )
    broker.create_task(Job("old-task", "evaluation", old_spec))
    assert broker.acquire_session("session-a", "new-task", 60) is True

    assert broker.store_result("old-task", "FAILED", error="late") is True

    assert broker.acquire_session("session-a", "third-task", 60) is False
