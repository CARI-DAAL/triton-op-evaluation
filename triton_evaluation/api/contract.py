"""Stable v2 wire contract for the evaluation service.

This module is the public API surface: the request shapes a consumer sends and
the typed response they get back. It depends only on pydantic — no app internals
— so it can be imported by clients and tests in isolation.
"""
from typing import Literal

from pydantic import BaseModel, Field, model_validator

CONTRACT_REVISION = "v2"

Check = Literal["correctness", "performance", "profile"]

DivergenceKind = Literal[
    "constant_offset",
    "per_axis",
    "precision_loss",
    "sign_flip",
    "value_clamp",
    "shape_mismatch",
    "scattered",
]

FiniteValue = float | Literal["inf", "-inf", "nan"]

TaskState = Literal["QUEUED", "RUNNING", "SUCCEEDED", "FAILED", "TIMEOUT", "REJECTED"]


class PerformanceParams(BaseModel):
    warmup: int = Field(default=5, ge=0, le=100)
    repeats: int = Field(default=50, ge=1, le=1000)


class ProfileParams(BaseModel):
    mode: Literal["op", "deep"] = "op"
    case_idx: int | None = Field(default=None, ge=0)
    launch_count: int | None = Field(default=None, ge=1, le=100)

    @model_validator(mode="after")
    def _launch_count_only_in_op_mode(self):
        if self.mode == "deep" and self.launch_count is not None:
            raise ValueError("launch_count is only valid in op mode")
        return self


class EvaluationRequest(BaseModel):
    op_name: str
    reference_code: str
    kernel_code: str
    triton_impl_name: str = "triton_ascend_impl"
    checks: list[Check] = Field(min_length=1)
    non_compute: bool = False
    timeout: int = Field(default=900, ge=10, le=3600)
    performance: PerformanceParams | None = None
    profile: ProfileParams | None = None

    @model_validator(mode="after")
    def _params_only_with_matching_check(self):
        if "performance" not in self.checks:
            self.performance = None
        if "profile" not in self.checks:
            self.profile = None
        return self


class SampleElement(BaseModel):
    idx: list[int]
    candidate: FiniteValue
    reference: FiniteValue
    diff: FiniteValue


class CaseMetrics(BaseModel):
    max_abs_diff: float | None = None
    mean_abs_err: float | None = None
    matched_ratio: float | None = None
    mere: float | None = None
    small_count: int | None = None
    normal_count: int | None = None
    atol: float | None = None
    rtol: float | None = None


class CaseFailure(BaseModel):
    case_idx: int
    kind: DivergenceKind | None = None
    metrics: CaseMetrics = Field(default_factory=CaseMetrics)
    sample: list[SampleElement] = Field(default_factory=list)
    error: str = ""


class CorrectnessResult(BaseModel):
    passed: bool
    compiled: bool
    decoy: bool = False
    passed_cases: int
    total_cases: int
    failures: list[CaseFailure] = Field(default_factory=list)


class PerformanceResult(BaseModel):
    speedup_vs_torch: float | None = None
    per_shape: list[dict] = Field(default_factory=list)


class ProfileResult(BaseModel):
    profile_state: Literal["complete", "partial"]
    data: dict = Field(default_factory=dict)


class EvaluationPayload(BaseModel):
    correctness: CorrectnessResult | None = None
    performance: PerformanceResult | None = None
    profile: ProfileResult | None = None


class EvaluationResult(BaseModel):
    task_id: str
    status: TaskState
    revision: str = CONTRACT_REVISION
    error: str | None = None
    result: EvaluationPayload = Field(default_factory=EvaluationPayload)
