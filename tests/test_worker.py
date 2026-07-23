import time

from triton_evaluation.models import Job, JobSpec, JobType, TaskStatus
from triton_evaluation.worker import Worker, reap_stale_tasks
from triton_evaluation.evaluators.base import EvaluatorResult


class FakeEvaluator:
    name = "fake"
    needs_bound_card = True

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


def test_run_once_idle_returns_false(broker, settings):
    w = Worker(broker, {}, device_id=0, settings=settings)
    assert w.run_once() is False


def test_benchmark_gate_resolution_injects_seed(broker, settings):
    # prior verify task with a stored result
    broker.create_task(Job("v1", JobType.VERIFY.value, _spec()))
    broker.store_result("v1", TaskStatus.SUCCEEDED.value,
                        result={"total_cases": 2, "passed_cases": 2})
    ev = FakeEvaluator()
    bspec = _spec()
    bspec.verify_task_id = "v1"
    broker.create_task(Job("b1", JobType.BENCHMARK.value, bspec))
    w = Worker(broker, {"benchmark": ev}, device_id=0, settings=settings)
    w.run_once()
    seen_spec = ev.seen[0][0]
    assert seen_spec.verify_result_seed == {"total_cases": 2, "passed_cases": 2}


def test_evaluator_exception_marks_failed(broker, settings):
    class Boom:
        name = "verify"
        needs_bound_card = True
        def run(self, *a):
            raise RuntimeError("kaboom")

    broker.create_task(Job("t2", JobType.VERIFY.value, _spec()))
    w = Worker(broker, {"verify": Boom()}, device_id=0, settings=settings)
    w.run_once()
    rec = broker.get_task("t2")
    assert rec["status"] == "FAILED"
    assert "kaboom" in rec["error"]


def test_reaper_marks_stale_running(broker, settings):
    broker.create_task(Job("t3", JobType.VERIFY.value, _spec()))
    broker.mark_running("t3", 0)
    # backdate started_at beyond the grace window
    broker.r.hset(broker._task_key("t3"),
                  "started_at", repr(time.time() - settings.reaper_grace_seconds - 10))
    n = reap_stale_tasks(broker, settings)
    assert n == 1
    assert broker.get_task("t3")["status"] == "FAILED"
