"""Validated HTTP requests and responses for the kernel evaluation tools.

FastAPI uses these same models for validation, serialization and OpenAPI.
Shared evaluation evidence uses the existing domain models.
"""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .contract import (
    CaseFailure,
    CorrectnessResult,
    DeepProfileCase,
    InputShape,
    OpProfileCase,
    SourceCode,
    StrictModel,
)
from .schemas import SessionId


class ToolRequest(StrictModel):
    task_id: str = Field(description="Task identifier in the configured catalog.")
    kernel_code: SourceCode = Field(
        description="Candidate source, at most 2 MiB of UTF-8; reference code is server-owned.",
    )
    session: SessionId | None = None


class ToolProfileRequest(ToolRequest):
    profile_mode: Literal["op", "deep"] = "op"
    case_idx: Annotated[int, Field(ge=0)] | None = None
    profile_launch_count: Annotated[int, Field(ge=1, le=100)] | None = Field(
        default=None,
        description="Legacy option; validated but not forwarded. Forbidden in deep mode.",
    )


    @model_validator(mode="after")
    def _deep_mode_has_no_launch_count(self):
        if self.profile_mode == "deep" and self.profile_launch_count is not None:
            raise ValueError("profile_launch_count is not supported in deep mode")
        return self


class ToolError(StrictModel):
    error: str


class _ToolEnvelope(StrictModel):
    action: Literal["verify", "benchmark", "profile"]
    artifact_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    eval_turn: int | None
    eval_budget: str
    observation: str
    diagnostics: str | None = None


class ToolBudgetExhausted(_ToolEnvelope):
    request_status: Literal["budget_exhausted"]


class ToolInfrastructureError(_ToolEnvelope):
    request_status: Literal["infra_error"]
    infra_error: Literal[True]
    error: str


class _Latency(StrictModel):
    avg_latency_ms: float


class ToolBenchmarkCase(StrictModel):
    case_idx: int
    input_desc: list[InputShape]
    status: Literal["pass", "fail"]
    framework: _Latency | None
    implementation: _Latency | None
    speedup_vs_torch: float | None
    error_type: str | None = None
    error_msg: str | None = None


class ToolBenchmarkResult(StrictModel):
    speedup_vs_torch: float | None
    total_cases: int
    passed_cases: int
    coverage: float
    per_shape_results: list[ToolBenchmarkCase]


class ToolVerificationEnvelope(_ToolEnvelope):
    request_status: Literal["completed", "infra_error"]
    correct: bool
    correctness_raw: CorrectnessResult | None
    decoy: bool
    speedup: float
    compiled: bool
    error: str | None
    error_class: str | None
    max_abs_err: float | None
    mean_abs_err: float | None
    pass_rate: float | None
    infra_error: bool
    total_cases: int
    passed_cases: int
    failures: list[CaseFailure]
    benchmark_available: bool
    benchmark: ToolBenchmarkResult | None


class ToolVerifyEnvelope(ToolVerificationEnvelope):
    action: Literal["verify"]


class ToolBenchmarkEnvelope(ToolVerificationEnvelope):
    action: Literal["benchmark"]


class ToolProfileResult(StrictModel):
    selected_cases: int
    failed_cases: int
    partial_cases: int
    remaining_case_indices: list[int]
    timed_out: bool
    cases: list[OpProfileCase | DeepProfileCase]


class ToolProfileEnvelope(_ToolEnvelope):
    action: Literal["profile"]
    request_status: Literal["completed", "infra_error"]
    task_status: Literal["SUCCEEDED", "REJECTED", "TIMEOUT", "FAILED"]
    profile_state: Literal["complete", "partial", "failed", "timeout", "infra_error"]
    result: ToolProfileResult | None
    error: str | None
    infra_error: bool


VerifyResponse = ToolVerifyEnvelope | ToolInfrastructureError | ToolBudgetExhausted
BenchmarkResponse = ToolBenchmarkEnvelope | ToolInfrastructureError | ToolBudgetExhausted
ProfileResponse = ToolProfileEnvelope | ToolInfrastructureError | ToolBudgetExhausted
