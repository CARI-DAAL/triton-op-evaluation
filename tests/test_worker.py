import time

import pytest
import redis

from triton_evaluation.evaluators.base import EvaluatorResult
from triton_evaluation.models import Job, JobSpec, JobType, TaskStatus
from triton_evaluation.worker import (
    Worker,
    compute_session_lease_ttl,
    reap_stale_tasks,
)


class FakeEvaluator:
    def __init__(self):
        self.seen = []

    def run(self, spec, verify_dir, device_id):
        self.seen.append((spec, verify_dir, device_id))
        return EvaluatorResult(TaskStatus.SUCCEEDED.value, result={"ok": True})


def _spec():
    return JobSpec(op_name="s", reference_code="R", kernel_code="K")


def test_run_once_dispatches_and_stores(broker, settings):
    ev = FakeEvaluator()
    broker.create_task(Job("t1", JobType.VERIFY.value, _spec()))
    w = Worker(broker, {"verify": ev}, device_id=4, settings=settings)
    assert w.run_once() is True
    rec = broker.get_task("t1")
    assert rec["status"] == "SUCCEEDED"
    assert rec["result"] == {"ok": True}
    assert rec["device_id"] == 4
    assert ev.seen[0][2] == 4  # device_id passed through


@pytest.mark.parametrize(
    ("measure_speed", "expected_deadline"),
    [
        (False, 2020.0),
        (True, 2320.0),
    ],
)
def test_run_once_deadline_includes_benchmark_only_when_requested(
    broker,
    settings,
    monkeypatch,
    measure_speed,
    expected_deadline,
):
    import triton_evaluation.worker as worker_mod

    evaluator = FakeEvaluator()
    spec = _spec()
    spec.measure_speed = measure_speed
    broker.create_task(Job("deadline-task", JobType.VERIFY.value, spec))
    monkeypatch.setattr(worker_mod.time, "time", lambda: 1000.0)

    worker = Worker(
        broker,
        {"verify": evaluator},
        device_id=0,
        settings=settings,
    )
    worker.run_once()

    assert broker.get_task("deadline-task")["deadline"] == expected_deadline


def test_profile_job_runs_on_whichever_worker_dequeues_it(broker, settings):
    evaluator = FakeEvaluator()
    spec = JobSpec(
        op_name="softmax",
        reference_code="R",
        kernel_code="K",
        profile_mode="deep",
    )
    broker.create_task(Job("profile-1", JobType.PROFILE.value, spec))
    worker = Worker(
        broker,
        {"profile": evaluator},
        device_id=7,
        settings=settings,
    )
    assert worker.run_once() is True
    record = broker.get_task("profile-1")
    assert record["device_id"] == 7
    assert evaluator.seen[0][2] == 7


def test_run_once_idle_returns_false(broker, settings):
    w = Worker(broker, {}, device_id=0, settings=settings)
    assert w.run_once() is False


def test_evaluator_exception_marks_failed(broker, settings):
    class Boom:
        def run(self, *a):
            raise RuntimeError("kaboom")

    broker.create_task(Job("t2", JobType.VERIFY.value, _spec()))
    w = Worker(broker, {"verify": Boom()}, device_id=0, settings=settings)
    w.run_once()
    rec = broker.get_task("t2")
    assert rec["status"] == "FAILED"
    assert "kaboom" in rec["error"]


def test_reaper_marks_stale_running_and_releases_session(broker, settings):
    spec = _spec()
    spec.session_id = "session-a"
    assert broker.acquire_session("session-a", "t3", 60) is True
    broker.create_task(Job("t3", JobType.VERIFY.value, spec))
    broker.mark_running("t3", 0, deadline=time.time() + 10000)
    # backdate the deadline into the past
    broker.r.hset(broker._task_key("t3"), "deadline", repr(time.time() - 10))
    n = reap_stale_tasks(broker, settings)
    assert n == 1
    assert broker.get_task("t3")["status"] == "FAILED"
    assert broker.acquire_session("session-a", "next-task", 60) is True


def test_reaper_does_not_reap_live_task(broker, settings):
    broker.create_task(Job("t4", JobType.VERIFY.value, _spec()))
    broker.mark_running("t4", 0, deadline=time.time() + 10000)
    n = reap_stale_tasks(broker, settings)
    assert n == 0
    assert broker.get_task("t4")["status"] == "RUNNING"


def test_serve_forever_survives_redis_connection_error(settings, monkeypatch):
    import triton_evaluation.worker as wmod

    sleeps = []
    monkeypatch.setattr(wmod.time, "sleep", lambda s: sleeps.append(s))

    class _Stop(BaseException):
        pass

    class FlakyBroker:
        def __init__(self):
            self.calls = 0

        def dequeue(self):
            self.calls += 1
            if self.calls <= 2:
                raise redis.exceptions.ConnectionError("redis down")
            raise _Stop()  # break the otherwise-infinite loop for the test

    w = Worker(FlakyBroker(), {}, device_id=0, settings=settings)
    with pytest.raises(_Stop):
        w.serve_forever()
    # survived 2 connection errors with capped exponential backoff, then continued
    assert sleeps == [2, 4]


def test_deadline_budgets_performance_and_profile_checks():
    from triton_evaluation.config import BENCHMARK_TIMEOUT_SECONDS
    from triton_evaluation.models import JobSpec
    from triton_evaluation.worker import compute_max_runtime

    base = JobSpec(op_name="addmm", reference_code="R", kernel_code="K", timeout=100)
    assert compute_max_runtime(base) == 100  # v1 verify, no checks

    perf = JobSpec(op_name="a", reference_code="R", kernel_code="K", timeout=100,
                   checks=["correctness", "performance"])
    assert compute_max_runtime(perf) == 100 + BENCHMARK_TIMEOUT_SECONDS

    both = JobSpec(op_name="a", reference_code="R", kernel_code="K", timeout=100,
                   checks=["correctness", "performance", "profile"])
    assert compute_max_runtime(both) == 100 + BENCHMARK_TIMEOUT_SECONDS + 100


def test_session_lease_ttl_covers_worker_and_request_budgets(settings):
    spec = JobSpec(
        op_name="a",
        reference_code="R",
        kernel_code="K",
        timeout=100,
        checks=["correctness", "performance", "profile"],
    )

    assert compute_session_lease_ttl(spec, settings) == 100 + 300 + 100 + 120 + 100


def test_worker_releases_session_after_storing_success(broker, settings):
    spec = JobSpec(
        op_name="a",
        reference_code="R",
        kernel_code="K",
        checks=["correctness"],
        session_id="session-a",
    )
    assert broker.acquire_session("session-a", "task-a", 60) is True
    broker.create_task(Job("task-a", JobType.EVALUATION.value, spec))

    worker = Worker(
        broker,
        {JobType.EVALUATION.value: FakeEvaluator()},
        device_id=0,
        settings=settings,
    )
    assert worker.run_once() is True
    assert broker.get_task("task-a")["status"] == TaskStatus.SUCCEEDED.value
    assert broker.acquire_session("session-a", "task-b", 60) is True


def test_worker_releases_session_after_storing_evaluator_failure(broker, settings):
    class Boom:
        def run(self, *args):
            raise RuntimeError("kaboom")

    spec = JobSpec(
        op_name="a",
        reference_code="R",
        kernel_code="K",
        checks=["correctness"],
        session_id="session-a",
    )
    assert broker.acquire_session("session-a", "task-a", 60) is True
    broker.create_task(Job("task-a", JobType.EVALUATION.value, spec))

    Worker(
        broker,
        {JobType.EVALUATION.value: Boom()},
        device_id=0,
        settings=settings,
    ).run_once()

    assert broker.get_task("task-a")["status"] == TaskStatus.FAILED.value
    assert broker.acquire_session("session-a", "task-b", 60) is True


def test_worker_retains_session_when_terminal_store_fails(broker, settings, monkeypatch):
    spec = JobSpec(
        op_name="a",
        reference_code="R",
        kernel_code="K",
        checks=["correctness"],
        session_id="session-a",
    )
    assert broker.acquire_session("session-a", "task-a", 60) is True
    broker.create_task(Job("task-a", JobType.EVALUATION.value, spec))

    def fail_store(*args, **kwargs):
        raise redis.exceptions.ConnectionError("redis unavailable")

    monkeypatch.setattr(broker, "store_result", fail_store)
    worker = Worker(
        broker,
        {JobType.EVALUATION.value: FakeEvaluator()},
        device_id=0,
        settings=settings,
    )

    with pytest.raises(redis.exceptions.ConnectionError):
        worker.run_once()
    assert broker.acquire_session("session-a", "task-b", 60) is False
