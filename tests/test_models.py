from triton_evaluation.models import (
    Job,
    JobSpec,
    JobType,
    VerifyRequest,
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


def test_job_json_roundtrip_carries_measure_speed():
    spec = JobSpec(
        op_name="softmax", reference_code="R", kernel_code="K",
        measure_speed=True, warmup=3, repeats=7,
    )
    back = Job.from_json(Job(task_id="t1", type=JobType.VERIFY.value, spec=spec).to_json())
    assert back.spec.measure_speed is True
    assert back.spec.warmup == 3
    assert back.spec.repeats == 7


def test_verify_request_defaults():
    r = VerifyRequest(op_name="s", reference_code="R", kernel_code="K")
    assert r.timeout == 900
    assert r.non_compute is False
    assert r.measure_speed is False
    assert r.warmup == 5
    assert r.repeats == 50


def test_verify_request_accepts_measure_speed():
    r = VerifyRequest(
        op_name="s", reference_code="R", kernel_code="K",
        measure_speed=True, warmup=2, repeats=20,
    )
    assert r.measure_speed is True
    assert r.warmup == 2
    assert r.repeats == 20
