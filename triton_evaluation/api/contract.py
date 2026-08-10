"""Stable v2 wire contract for the evaluation service.

This module is the public API surface: the request shapes a consumer sends and
the typed response they get back. It depends only on pydantic — no app internals
— so it can be imported by clients and tests in isolation.
"""
from typing import Literal

from pydantic import BaseModel, Field, model_validator

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

TerminalStatus = Literal["SUCCEEDED", "FAILED", "TIMEOUT", "REJECTED"]
ProfileState = Literal["complete", "partial"]
ProfileCaseState = Literal["complete", "partial", "failed"]
BoundKind = Literal[
    "MTE2 BOUND",
    "CUBE BOUND",
    "VEC BOUND",
    "FIXP BOUND",
    "MTE3 BOUND",
    "SCALAR BOUND",
    "NO BOUND",
]


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
        if len(set(self.checks)) != len(self.checks):
            raise ValueError("checks must not contain duplicates")
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
    error: str | None = None


class CorrectnessResult(BaseModel):
    passed: bool
    compiled: bool
    decoy: bool = False
    passed_cases: int
    total_cases: int
    failures: list[CaseFailure] = Field(default_factory=list)


class PerShapeResult(BaseModel):
    shape: str
    latency_ms: float
    throughput: float | None = None
    speedup_vs_torch: float


class PerformanceResult(BaseModel):
    speedup_vs_torch: float
    per_shape: list[PerShapeResult] = Field(default_factory=list)


class OpProfile(BaseModel):
    case_idx: int
    op_name: str
    time_ms: float
    percentage: float | None = None


class DeepProfileSummary(BaseModel):
    case_idx: int
    status: ProfileCaseState
    target_kernel: str | None = None
    bound: BoundKind | None = None
    rule: str | None = None
    mte2_ratio: float | None = None
    cube_ratio: float | None = None
    vec_ratio: float | None = None
    fixp_ratio: float | None = None
    mte3_ratio: float | None = None
    scalar_ratio: float | None = None
    per_core_spread_ratio: float | None = None
    per_core_verdict: str | None = None
    recommendations: list[str] = Field(default_factory=list)
    missing_metrics: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class ProfileData(BaseModel):
    mode: Literal["op", "deep"]
    selected_cases: int
    completed_cases: int
    partial_cases: int
    failed_cases: int
    timed_out: bool
    ops: list[OpProfile] | None = None
    summaries: list[DeepProfileSummary] | None = None

    @model_validator(mode="after")
    def _mode_specific_payload(self):
        if self.mode == "op":
            self.ops = self.ops or []
            self.summaries = None
        else:
            self.summaries = self.summaries or []
            self.ops = None
        return self


class ProfileResult(BaseModel):
    profile_state: ProfileState
    data: ProfileData


class EvaluationPayload(BaseModel):
    correctness: CorrectnessResult | None = None
    performance: PerformanceResult | None = None
    profile: ProfileResult | None = None


class EvaluationResult(BaseModel):
    task_id: str
    status: TerminalStatus
    error: str | None = None
    result: EvaluationPayload = Field(default_factory=EvaluationPayload)
