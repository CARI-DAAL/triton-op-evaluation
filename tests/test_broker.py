import time

from triton_evaluation.models import Job, JobSpec, TaskStatus


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
