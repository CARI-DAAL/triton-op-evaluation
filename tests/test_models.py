import pytest
from pydantic import ValidationError
from triton_evaluation.models import (
    TaskStatus, JobType, JobSpec, Job, VerifyRequest, BenchmarkRequest,
)


def test_job_json_roundtrip():
    spec = JobSpec(op_name="softmax", reference_code="R", kernel_code="K", non_compute=True)
    job = Job(task_id="t1", type=JobType.VERIFY.value, spec=spec)
    back = Job.from_json(job.to_json())
    assert back.task_id == "t1"
    assert back.type == "verify"
    assert back.spec.op_name == "softmax"
    assert back.spec.non_compute is True
    assert back.spec.triton_impl_name == "triton_ascend_impl"


def test_benchmark_request_requires_gate():
    with pytest.raises(ValidationError):
        BenchmarkRequest(op_name="s", reference_code="R", kernel_code="K")


def test_benchmark_request_accepts_verify_task_id():
    r = BenchmarkRequest(op_name="s", reference_code="R", kernel_code="K", verify_task_id="v1")
    assert r.verify_task_id == "v1"


def test_verify_request_defaults():
    r = VerifyRequest(op_name="s", reference_code="R", kernel_code="K")
    assert r.timeout == 900
    assert r.non_compute is False
