import json
from dataclasses import asdict, dataclass
from enum import Enum

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
    BENCHMARK = "benchmark"


@dataclass
class JobSpec:
    op_name: str
    reference_code: str
    kernel_code: str
    triton_impl_name: str = "triton_ascend_impl"
    # verify
    non_compute: bool = False
    timeout: int = 900
    # benchmark
    warmup: int = 5
    repeats: int = 50
    verify_task_id: str | None = None
    verify_not_required: bool = False
    verify_result_seed: dict | None = None  # injected by worker for the L1 gate


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


class BenchmarkRequest(BaseModel):
    op_name: str
    reference_code: str
    kernel_code: str
    triton_impl_name: str = "triton_ascend_impl"
    warmup: int = Field(default=5, ge=0, le=100)
    repeats: int = Field(default=50, ge=1, le=1000)
    timeout: int = Field(default=1800, ge=10, le=3600)
    verify_task_id: str | None = None
    verify_not_required: bool = False

    @model_validator(mode="after")
    def _require_gate(self):
        if not self.verify_task_id and not self.verify_not_required:
            raise ValueError("benchmark requires verify_task_id or verify_not_required=true")
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
