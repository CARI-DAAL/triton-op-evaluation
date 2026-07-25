import pytest
from pydantic import ValidationError

from triton_evaluation.models import (
    Job,
    JobSpec,
    JobType,
    ProfileRequest,
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


def test_profile_job_roundtrip_carries_profile_fields():
    spec = JobSpec(
        op_name="softmax",
        reference_code="R",
        kernel_code="K",
        timeout=7200,
        profile_mode="deep",
        case_idx=2,
    )
    job = Job(task_id="p1", type=JobType.PROFILE.value, spec=spec)
    back = Job.from_json(job.to_json())
    assert back.type == "profile"
    assert back.spec.timeout == 7200
    assert back.spec.profile_mode == "deep"
    assert back.spec.case_idx == 2
    assert back.spec.profile_launch_count is None


def test_profile_request_defaults_to_all_cases_in_op_mode():
    request = ProfileRequest(
        op_name="softmax",
        reference_code="R",
        kernel_code="K",
    )
    assert request.profile_mode == "op"
    assert request.case_idx is None
    assert request.profile_launch_count is None
    assert request.timeout == 3600


@pytest.mark.parametrize("mode", ["bad", "", "OP"])
def test_profile_request_rejects_unknown_mode(mode):
    with pytest.raises(ValidationError):
        ProfileRequest(
            op_name="softmax",
            reference_code="R",
            kernel_code="K",
            profile_mode=mode,
        )


def test_profile_request_rejects_launch_count_in_deep_mode():
    with pytest.raises(ValidationError, match="profile_launch_count is only valid"):
        ProfileRequest(
            op_name="softmax",
            reference_code="R",
            kernel_code="K",
            profile_mode="deep",
            profile_launch_count=2,
        )
