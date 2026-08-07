import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class TaskStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    REJECTED = "REJECTED"


class JobType(str, Enum):
    VERIFY = "verify"
    PROFILE = "profile"
    EVALUATION = "evaluation"


ProfileMode = Literal["op", "deep"]


@dataclass
class JobSpec:
    op_name: str
    reference_code: str
    kernel_code: str
    triton_impl_name: str = "triton_ascend_impl"
    # verify
    non_compute: bool = False
    timeout: int = 900
    # measure speed in the same job once verify passes (runs benchmark.py in-place)
    measure_speed: bool = False
    warmup: int = 5
    repeats: int = 50
    # standalone profile job
    profile_mode: ProfileMode = "op"
    case_idx: int | None = None
    profile_launch_count: int | None = None
    # v2 evaluation jobs: which checks to run. None for v1 verify/profile jobs.
    checks: list[str] | None = None


@dataclass
class Job:
    task_id: str
    type: str
    spec: JobSpec

    def to_json(self) -> str:
        return json.dumps({"task_id": self.task_id, "type": self.type, "spec": asdict(self.spec)})

    @staticmethod
    def from_json(payload: str) -> "Job":
        d = json.loads(payload)
        return Job(task_id=d["task_id"], type=d["type"], spec=JobSpec(**d["spec"]))


class VerifyRequest(BaseModel):
    op_name: str
    reference_code: str
    kernel_code: str
    non_compute: bool = False
    triton_impl_name: str = "triton_ascend_impl"
    timeout: int = Field(default=900, ge=10, le=3600)
    # measure speed in the same job once verify passes
    measure_speed: bool = False
    warmup: int = Field(default=5, ge=0, le=100)
    repeats: int = Field(default=50, ge=1, le=1000)


class ProfileRequest(BaseModel):
    op_name: str
    reference_code: str
    kernel_code: str
    triton_impl_name: str = "triton_ascend_impl"
    profile_mode: ProfileMode = "op"
    case_idx: int | None = Field(default=None, ge=0)
    profile_launch_count: int | None = Field(default=None, ge=1, le=100)
    timeout: int = Field(default=3600, ge=10, le=86400)

    @model_validator(mode="after")
    def validate_mode_specific_fields(self):
        if self.profile_mode == "deep" and self.profile_launch_count is not None:
            raise ValueError("profile_launch_count is only valid in op mode")
        return self


class SubmitResponse(BaseModel):
    task_id: str


class TaskResponse(BaseModel):
    task_id: str
    type: str
    status: str
    result: dict | None = None
    error: str | None = None
    device_id: int | None = None
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
