"""Strict durable evaluation models and typed public resource responses.

``EvaluationRequest`` is the internal engine/storage format retained for
persisted request recovery. Public submissions use ``api.operator.OperatorRequest``
and are translated into that format at the HTTP boundary. Resource and error
models here define the responses returned to callers.

This module depends only on pydantic, so clients and tests can import response
models in isolation. Models reject unknown fields, scalar coercion and
non-finite ordinary floats; correctness samples represent NaN/Inf as explicit
string sentinels. Recorded resource examples live in the test suite.
"""
import math
import statistics
from collections.abc import Callable
from datetime import datetime
from typing import Annotated, Generic, Literal, TypeVar

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationInfo,
    model_validator,
)

SchemaVersion = Literal["2.1"]
Purpose = Literal["trial", "grade", "rollout"]
Check = Literal["correctness", "performance", "profile", "compile"]
MeasurementMethod = Literal["npu_event_v1", "profiler_kernel_sum_v1"]
ResourceState = Literal["queued", "running", "completed", "failed", "timed_out"]

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

FiniteOrSentinel = float | Literal["inf", "-inf", "nan"]
DivergenceKind = Literal["shape_mismatch", "value_mismatch", "non_finite_mismatch"]
CorrectnessDomainOutcome = Literal[
    "correct", "wrong", "compile_error", "runtime_error", "execution_timeout"
]
ProfileCaseStatus = Literal["complete", "partial", "failed"]

MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_ERROR_MESSAGE_BYTES = 512
MAX_SKIP_MESSAGE_BYTES = 256

_CHECK_ORDER = ("compile", "correctness", "performance", "profile")

T = TypeVar("T")


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_default=True,
        allow_inf_nan=False,
    )


def utf8_len(max_bytes: int) -> Callable[[str], str]:
    """Return a validator enforcing a UTF-8 byte (not code-point) limit."""

    def _check(value: str) -> str:
        if len(value.encode("utf-8")) > max_bytes:
            raise ValueError(f"must be at most {max_bytes} UTF-8 bytes")
        return value

    return _check


SourceCode = Annotated[str, AfterValidator(utf8_len(MAX_SOURCE_BYTES))]


def redacted_text(max_bytes: int) -> Callable[[str], str]:
    """Return a validator for single-line redacted text bounded in UTF-8 bytes."""

    def _check(value: str) -> str:
        if "\r" in value or "\n" in value:
            raise ValueError("must be a single line without CR/LF")
        size = len(value.encode("utf-8"))
        if not 1 <= size <= max_bytes:
            raise ValueError(f"must be 1..{max_bytes} UTF-8 bytes")
        return value

    return _check


# ---------------------------------------------------------------------------
# Request parameters and purpose context
# ---------------------------------------------------------------------------


class PerformanceParams(StrictModel):
    use_baseline_cache: bool = Field(
        default=True, exclude_if=lambda value: value is True,
        description=(
            "Reuse cached Torch baseline timing; false measures fresh "
            "without reading or writing the cache."
        ),
    )
    warmup: int = Field(default=5, ge=0, le=100)
    repeats: int = Field(default=50, ge=1, le=1000)
    method: Literal["event", "profiler"] = "event"
    # Zero-based original case identities, mirroring ProfileParams. None keeps
    # full-task performance coverage; the engine preserves original indices.
    case_indices: list[Annotated[int, Field(ge=0)]] | None = Field(
        default=None, max_length=128
    )

    @model_validator(mode="after")
    def _case_indices_unique_then_sorted(self):
        if self.case_indices is not None:
            if len(set(self.case_indices)) != len(self.case_indices):
                raise ValueError("case_indices must not contain duplicates")
            self.case_indices = sorted(self.case_indices)
        return self


class ProfileParams(StrictModel):
    mode: Literal["op", "deep"] = "op"
    case_indices: list[Annotated[int, Field(ge=0)]] | None = Field(
        default=None, max_length=128
    )
    max_kernels_per_case: int = Field(default=200, ge=1, le=1000)
    include_recommendations: bool = True

    @model_validator(mode="after")
    def _case_indices_unique_then_sorted(self):
        if self.case_indices is not None:
            if len(set(self.case_indices)) != len(self.case_indices):
                raise ValueError("case_indices must not contain duplicates")
            self.case_indices = sorted(self.case_indices)
        return self


class GradeContext(StrictModel):
    manifest_id: str = Field(min_length=1, max_length=128)
    problem_id: str = Field(min_length=1, max_length=128)
    reference_digest: Digest


# Internal storage/engine model. Public input is api.operator.OperatorRequest.
# Existing fields and default serialization preserve historical record bytes.
class EvaluationRequest(StrictModel):
    """A durable request for evaluation evidence.

    Checks must be unique. Comparison performance requires correctness; performance and
    profile parameters may only accompany their corresponding checks. Grade
    requests require grade_context; trial and rollout requests cannot carry it.
    Omitted check parameters use their declared defaults. Each source string
    is limited to 2 MiB of UTF-8. Accepted work survives HTTP wait expiration
    and is recovered by ID or by replaying the same idempotent submission.
    """

    schema_version: SchemaVersion
    purpose: Purpose = "trial"
    implementation: Literal["comparison", "torch", "triton"] = Field(
        default="comparison", exclude_if=lambda value: value == "comparison"
    )
    op_name: str = Field(min_length=1, max_length=128)
    reference_code: SourceCode
    kernel_code: SourceCode
    triton_impl_name: str = Field(
        default="triton_ascend_impl", min_length=1, max_length=128
    )
    checks: list[Check] = Field(min_length=1, max_length=4)
    non_compute: bool = False
    deadline_seconds: int = Field(default=900, ge=10, le=3600)
    performance: PerformanceParams | None = None
    profile: ProfileParams | None = None
    grade_context: GradeContext | None = None

    @model_validator(mode="after")
    def _normalize_and_couple(self):
        if len(set(self.checks)) != len(self.checks):
            raise ValueError("checks must not contain duplicates")
        self.checks = sorted(self.checks, key=_CHECK_ORDER.index)
        if self.implementation != "comparison":
            if "correctness" in self.checks:
                raise ValueError("the correctness check requires both implementations")
            if self.purpose == "grade":
                raise ValueError("grade records require comparison implementations")
        if (
            self.implementation == "comparison"
            and "performance" in self.checks and "correctness" not in self.checks
        ):
            raise ValueError("the performance check requires the correctness check")
        if "performance" in self.checks:
            if self.performance is None:
                self.performance = PerformanceParams()
        elif self.performance is not None:
            raise ValueError("performance params require the performance check")
        if "profile" in self.checks:
            if self.profile is None:
                self.profile = ProfileParams()
        elif self.profile is not None:
            raise ValueError("profile params require the profile check")
        if self.purpose != "grade" and self.grade_context is not None:
            raise ValueError("only grade requests may carry grade_context")
        if self.purpose == "grade" and self.grade_context is None:
            raise ValueError("grade requests require grade_context")
        return self


# ---------------------------------------------------------------------------
# Error envelope (design section 3.4)
# ---------------------------------------------------------------------------


class ErrorInfo(StrictModel):
    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")
    phase: Literal[
        "request",
        "auth",
        "admission",
        "queue",
        "execution",
        "shaping",
        "storage",
        "deadline",
    ]
    check: Check | None
    retryable: bool
    message: Annotated[str, AfterValidator(redacted_text(MAX_ERROR_MESSAGE_BYTES))]


class ErrorEnvelope(StrictModel):
    schema_version: SchemaVersion
    request_id: str
    error: ErrorInfo


# ---------------------------------------------------------------------------
# Check outcomes (design section 4.1)
# ---------------------------------------------------------------------------


class SkipInfo(StrictModel):
    code: Literal["DEPENDENCY_NOT_PASSED", "PRIOR_CHECK_FAILED", "RESOURCE_DEADLINE"]
    depends_on: Check | None = None
    message: Annotated[str, AfterValidator(redacted_text(MAX_SKIP_MESSAGE_BYTES))]


class CheckSucceeded(StrictModel, Generic[T]):
    status: Literal["succeeded"]
    data: T
    error: None = None
    reason: None = None


class CheckFailed(StrictModel):
    status: Literal["failed"]
    data: None = None
    error: ErrorInfo
    reason: None = None


class CheckSkipped(StrictModel):
    status: Literal["skipped"]
    data: None = None
    error: None = None
    reason: SkipInfo


class CheckTimedOut(StrictModel):
    status: Literal["timeout"]
    data: None = None
    error: ErrorInfo
    reason: None = None


# typing.Generic (not PEP 695 syntax) keeps Python 3.10 support. Pydantic
# parametrizes CheckSucceeded[T] eagerly, so a generic alias like
# ``Annotated[CheckSucceeded[T] | ..., Field(discriminator=...)]`` cannot be
# subscripted later; the per-check concrete aliases below are the field types.
CheckOutcome = Annotated[
    CheckSucceeded[T] | CheckFailed | CheckSkipped | CheckTimedOut,
    Field(discriminator="status"),
]


# ---------------------------------------------------------------------------
# Correctness (design section 4.2)
# ---------------------------------------------------------------------------


class CaseMetrics(StrictModel):
    max_abs_diff: float | None = None
    matched_ratio: float | None = None
    mere: float | None = None
    small_count: int | None = None
    normal_count: int | None = None
    atol: float | None = None
    rtol: float | None = None


class SampleElement(StrictModel):
    idx: list[int]
    candidate: FiniteOrSentinel
    reference: FiniteOrSentinel
    diff: FiniteOrSentinel


def _sample_order_key(point: SampleElement) -> tuple:
    """Finite diffs first by descending abs value, then sentinels lexically."""
    if isinstance(point.diff, str):
        return (1, point.diff, point.idx)
    return (0, -abs(point.diff), point.idx)


class CaseFailure(StrictModel):
    case_idx: Annotated[int, Field(ge=0)]
    kind: DivergenceKind
    metrics: CaseMetrics
    sample: list[SampleElement] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def _sample_rules(self):
        if self.kind == "shape_mismatch" and self.sample:
            raise ValueError("shape_mismatch must have an empty sample")
        keys = [_sample_order_key(point) for point in self.sample]
        if keys != sorted(keys):
            raise ValueError(
                "sample must rank finite abs(diff) descending, then sentinels "
                "lexically, breaking ties by row-major index"
            )
        return self


class DomainFailure(StrictModel):
    code: Literal[
        "SUBJECT_COMPILE_ERROR", "SUBJECT_RUNTIME_ERROR", "SUBJECT_EXECUTION_TIMEOUT"
    ]
    phase: Literal["compile", "initialize", "execute"]
    case_idx: int | None = Field(default=None, ge=0)
    message: Annotated[str, AfterValidator(redacted_text(MAX_ERROR_MESSAGE_BYTES))]


_OUTCOME_TO_DOMAIN_CODE = {
    "compile_error": "SUBJECT_COMPILE_ERROR",
    "runtime_error": "SUBJECT_RUNTIME_ERROR",
    "execution_timeout": "SUBJECT_EXECUTION_TIMEOUT",
}


class CorrectnessResult(StrictModel):
    outcome: CorrectnessDomainOutcome
    passed: bool
    decoy: bool = False
    passed_cases: Annotated[int, Field(ge=0)]
    total_cases: Annotated[int, Field(ge=1)]
    failures: list[CaseFailure]
    domain_failure: DomainFailure | None = None

    @model_validator(mode="after")
    def _domain_rules(self):
        if self.passed_cases > self.total_cases:
            raise ValueError("passed_cases must not exceed total_cases")
        indices = [failure.case_idx for failure in self.failures]
        if len(set(indices)) != len(indices):
            raise ValueError("failures must have unique case_idx values")
        if any(index >= self.total_cases for index in indices):
            raise ValueError("failure case_idx out of range")
        if self.passed_cases + len(self.failures) > self.total_cases:
            raise ValueError("passed_cases plus failures must not exceed total_cases")
        if self.passed != (self.outcome == "correct"):
            raise ValueError("passed is true iff outcome is correct")
        if self.outcome == "correct":
            if self.passed_cases != self.total_cases or self.failures:
                raise ValueError("correct requires every case passing")
            if self.domain_failure is not None or self.decoy:
                raise ValueError("correct forbids domain_failure and decoy")
        elif self.outcome == "wrong":
            if self.domain_failure is not None:
                raise ValueError("wrong forbids domain_failure")
            if not self.failures and not self.decoy:
                raise ValueError("wrong requires a mismatch failure or decoy")
            if self.passed_cases + len(self.failures) != self.total_cases:
                raise ValueError("wrong requires every case accounted for")
        else:
            failure = self.domain_failure
            if failure is None or failure.code != _OUTCOME_TO_DOMAIN_CODE[self.outcome]:
                raise ValueError("domain_failure code must match the outcome")
            if self.decoy:
                raise ValueError("subject failures forbid decoy")
            if self.outcome == "compile_error" and failure.phase != "compile":
                raise ValueError("compile_error is always attributed to compile")
            if self.outcome == "runtime_error" and failure.phase == "compile":
                raise ValueError("runtime_error is initialize or execute only")
            if failure.phase in ("compile", "initialize"):
                if failure.case_idx is not None:
                    raise ValueError("compile/initialize failures have no case_idx")
            elif failure.case_idx is None or failure.case_idx >= self.total_cases:
                raise ValueError("execute failures require an in-range case_idx")
        return self


# ---------------------------------------------------------------------------
# Compile (spec section 6, v2.1)
# ---------------------------------------------------------------------------


class CompileOutcome(StrictModel):
    """Result of the v2.1 compile check (spec §6)."""

    outcome: Literal["compiled", "compile_error"]
    compiler_stage: Literal["frontend", "backend"] | None = None
    error_excerpt: Annotated[str, AfterValidator(utf8_len(400))] | None = None
    duration_ms: float | None = Field(default=None, ge=0)


# ---------------------------------------------------------------------------
# Performance (design section 4.3)
# ---------------------------------------------------------------------------


class InputShape(StrictModel):
    input_index: Annotated[int, Field(ge=0)]
    dimensions: list[Annotated[int, Field(ge=0)]]
    dtype: str | None = None


ReferenceLatencySamplesSource = Literal[
    "fresh_measurements", "cached_manifest", "historical_validated"
]


class PerformanceCase(StrictModel):
    baseline_cache_hit: bool | None = Field(default=None, exclude_if=lambda value: value is None)
    case_idx: Annotated[int, Field(ge=0)]
    input_shapes: list[InputShape]
    implementation_latency_ms: Annotated[float, Field(gt=0)]
    reference_latency_ms: Annotated[float, Field(gt=0)]
    speedup_vs_reference: Annotated[float, Field(gt=0)]
    measurement_method: MeasurementMethod | None = None
    # The eight reference-timing measurements behind reference_latency_ms.
    # Absent when no sample evidence exists (for example a legacy scalar);
    # present values are always real measurements, never fabricated.
    reference_latency_samples_ms: list[Annotated[float, Field(gt=0)]] | None = None
    reference_latency_samples_source: ReferenceLatencySamplesSource | None = None
    # Proof identity for historical_validated samples: the SHA256 of the
    # owner-adopted validated timing-reuse policy artifact that authorized
    # serving these historical measurements on this runtime.
    reference_timing_policy_sha256: Digest | None = None

    @model_validator(mode="after")
    def _reference_samples_match_source_and_median(self):
        samples = self.reference_latency_samples_ms
        if samples is None:
            if self.reference_latency_samples_source is not None:
                raise ValueError("reference_latency_samples_source requires samples")
            if self.reference_timing_policy_sha256 is not None:
                raise ValueError("reference_timing_policy_sha256 requires samples")
            return self
        if self.reference_latency_samples_source is None:
            raise ValueError("reference latency samples require a source label")
        if len(samples) != 8:
            raise ValueError("reference latency samples must be exactly eight measurements")
        if self.reference_latency_samples_source == "historical_validated":
            if self.reference_timing_policy_sha256 is None:
                raise ValueError("historical_validated samples require the policy digest")
        elif self.reference_timing_policy_sha256 is not None:
            raise ValueError(
                "reference_timing_policy_sha256 belongs to historical_validated samples"
            )
        median = statistics.median(samples)
        # The engine rounds the public scalar to four decimals.
        if not math.isclose(median, self.reference_latency_ms, rel_tol=0, abs_tol=5.01e-5):
            raise ValueError("reference_latency_ms must equal the median of its samples")
        return self


class UnmeasuredPerformanceCase(StrictModel):
    case_idx: Annotated[int, Field(ge=0)]
    input_shapes: list[InputShape]
    reason_code: Literal["UNSUPPORTED_CASE", "NO_STABLE_MEASUREMENT"]


class PerformanceResult(StrictModel):
    aggregation: Literal["geometric_mean_v1"]
    overall_speedup_vs_reference: Annotated[float, Field(gt=0)]
    selected_cases: Annotated[int, Field(ge=1)]
    measured_cases: Annotated[int, Field(ge=1)]
    coverage: Annotated[float, Field(ge=0, le=1)]
    measurement_method: MeasurementMethod | None = None
    per_case: list[PerformanceCase]
    unmeasured_cases: list[UnmeasuredPerformanceCase]

    @model_validator(mode="after")
    def _measurement_method_consistent(self):
        if self.measurement_method is not None:
            for case in self.per_case:
                if case.measurement_method != self.measurement_method:
                    raise ValueError(
                        "every measured case must share the overall measurement_method"
                    )
        return self

    @model_validator(mode="after")
    def _aggregation_arithmetic(self):
        measured = len(self.per_case)
        if self.measured_cases != measured:
            raise ValueError("measured_cases must equal len(per_case)")
        if self.selected_cases != measured + len(self.unmeasured_cases):
            raise ValueError("selected_cases must equal measured plus unmeasured")
        indices = [case.case_idx for case in self.per_case]
        indices += [case.case_idx for case in self.unmeasured_cases]
        if len(set(indices)) != len(indices):
            raise ValueError("case_idx must be unique across both case lists")
        expected_coverage = measured / self.selected_cases
        if not math.isclose(self.coverage, expected_coverage, rel_tol=1e-9):
            raise ValueError("coverage must equal measured_cases / selected_cases")
        expected = math.exp(
            sum(math.log(case.speedup_vs_reference) for case in self.per_case) / measured
        )
        if not math.isclose(self.overall_speedup_vs_reference, expected, rel_tol=1e-9):
            raise ValueError("overall speedup must be the geometric mean of per_case")
        return self


def _unverified_only(value):
    if value is not False:
        raise ValueError("correctness_verified must be false")
    return value


_Unverified = Annotated[Literal[False], BeforeValidator(_unverified_only)]


class SinglePerformanceCase(StrictModel):
    case_idx: Annotated[int, Field(ge=0)]
    input_shapes: list[InputShape]
    latency_ms: Annotated[float, Field(gt=0)]
    peak_memory_mb: Annotated[float, Field(ge=0)] | None = None


class SinglePerformanceResult(StrictModel):
    implementation: Literal["torch", "triton"]
    correctness_verified: _Unverified
    measurement_method: MeasurementMethod
    selected_cases: Annotated[int, Field(ge=1)]
    measured_cases: Annotated[int, Field(ge=1)]
    coverage: Annotated[float, Field(ge=0, le=1)]
    per_case: list[SinglePerformanceCase]
    unmeasured_cases: list[UnmeasuredPerformanceCase]

    @model_validator(mode="after")
    def _case_arithmetic(self):
        measured = len(self.per_case)
        if self.measured_cases != measured:
            raise ValueError("measured_cases must equal len(per_case)")
        if self.selected_cases != measured + len(self.unmeasured_cases):
            raise ValueError("selected_cases must equal measured plus unmeasured")
        indices = [case.case_idx for case in self.per_case]
        indices += [case.case_idx for case in self.unmeasured_cases]
        if len(set(indices)) != len(indices):
            raise ValueError("case_idx must be unique across both case lists")
        if not math.isclose(self.coverage, measured / self.selected_cases, rel_tol=1e-9):
            raise ValueError("coverage must equal measured_cases / selected_cases")
        return self


# ---------------------------------------------------------------------------
# Profile (design section 4.4)
# ---------------------------------------------------------------------------


class ProfileCounts(StrictModel):
    selected: Annotated[int, Field(ge=0)]
    complete: Annotated[int, Field(ge=0)]
    partial: Annotated[int, Field(ge=0)]
    failed: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def _selected_is_component_sum(self):
        if self.selected != self.complete + self.partial + self.failed:
            raise ValueError("selected must equal complete + partial + failed")
        return self


class OpKernelMetrics(StrictModel):
    """Per-kernel utilization ratios from the ``msprof op`` capture.

    All values are fractions in [0, 1]. Pipe-utilization ratios are captured
    as fractions by msprof; memory-bandwidth usage and L2 hit rates are
    captured as percentages and normalized to fractions. A metric the capture
    did not emit stays ``None``.
    """

    aic_cube_ratio: float | None = Field(default=None, ge=0, le=1)
    aic_mte2_ratio: float | None = Field(default=None, ge=0, le=1)
    aic_fixpipe_ratio: float | None = Field(default=None, ge=0, le=1)
    aiv_vec_ratio: float | None = Field(default=None, ge=0, le=1)
    aiv_mte2_ratio: float | None = Field(default=None, ge=0, le=1)
    aiv_mte3_ratio: float | None = Field(default=None, ge=0, le=1)
    gm_to_l1_usage: float | None = Field(default=None, ge=0, le=1)
    l0c_to_gm_usage: float | None = Field(default=None, ge=0, le=1)
    gm_to_ub_usage: float | None = Field(default=None, ge=0, le=1)
    ub_to_gm_usage: float | None = Field(default=None, ge=0, le=1)
    aic_l2_hit: float | None = Field(default=None, ge=0, le=1)
    aiv_l2_hit: float | None = Field(default=None, ge=0, le=1)


class OpKernel(StrictModel):
    op_name: str
    time_ms: Annotated[float, Field(ge=0)]
    percentage: Annotated[float, Field(ge=0)]
    metrics: OpKernelMetrics = Field(default_factory=OpKernelMetrics)


class OpProfileCase(StrictModel):
    case_idx: Annotated[int, Field(ge=0)]
    status: ProfileCaseStatus
    kernels: list[OpKernel]
    error: ErrorInfo | None = None

    @model_validator(mode="after")
    def _status_rules(self):
        if self.status == "complete" and self.error is not None:
            raise ValueError("complete cases must not carry an error")
        if self.status == "failed" and self.error is None:
            raise ValueError("failed cases must carry an error")
        if self.status == "partial":
            if self.error is None:
                raise ValueError("partial op cases must explain the gap via error")
            if not self.kernels:
                raise ValueError("partial cases must carry locally available data")
        if self.kernels:
            total = sum(kernel.percentage for kernel in self.kernels)
            if not 99.0 <= total <= 101.0:
                raise ValueError("kernel percentages must sum within [99.0, 101.0]")
        return self


class DeepRatios(StrictModel):
    mte2: float | None = Field(default=None, ge=0, le=1)
    cube: float | None = Field(default=None, ge=0, le=1)
    vec: float | None = Field(default=None, ge=0, le=1)
    fixp: float | None = Field(default=None, ge=0, le=1)
    mte3: float | None = Field(default=None, ge=0, le=1)
    scalar: float | None = Field(default=None, ge=0, le=1)


class DeepProfileCase(StrictModel):
    case_idx: Annotated[int, Field(ge=0)]
    status: ProfileCaseStatus
    target_kernel: str | None
    bound: Literal["MTE2", "CUBE", "VEC", "FIXP", "MTE3", "SCALAR", "NONE"] | None
    ratios: DeepRatios
    per_core_spread_ratio: float | None
    recommendations: list[str]
    missing_metrics: list[str]
    warnings: list[str]
    error: ErrorInfo | None = None

    @model_validator(mode="after")
    def _status_rules(self):
        if self.status == "complete" and self.error is not None:
            raise ValueError("complete cases must not carry an error")
        if self.status == "failed" and self.error is None:
            raise ValueError("failed cases must carry an error")
        if self.status == "partial" and not (
            self.missing_metrics or self.warnings or self.error is not None
        ):
            raise ValueError("partial cases must explain the gap")
        return self


def _check_profile_tallies(counts: ProfileCounts, cases: list) -> None:
    if counts.selected != len(cases):
        raise ValueError("counts.selected must equal len(cases)")
    for status in ("complete", "partial", "failed"):
        if getattr(counts, status) != sum(case.status == status for case in cases):
            raise ValueError(f"counts.{status} must tally case statuses")
    indices = [case.case_idx for case in cases]
    if len(set(indices)) != len(indices):
        raise ValueError("case_idx must be unique")


class OpProfileResult(StrictModel):
    implementation: Literal["torch", "triton"] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    correctness_verified: _Unverified | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    mode: Literal["op"]
    counts: ProfileCounts
    cases: list[OpProfileCase]

    @model_validator(mode="after")
    def _tallies(self):
        if (self.implementation is None) != (self.correctness_verified is None):
            raise ValueError("implementation and correctness_verified must appear together")
        _check_profile_tallies(self.counts, self.cases)
        return self


class DeepProfileResult(StrictModel):
    implementation: Literal["torch", "triton"] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    correctness_verified: _Unverified | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    mode: Literal["deep"]
    counts: ProfileCounts
    cases: list[DeepProfileCase]

    @model_validator(mode="after")
    def _tallies(self):
        if (self.implementation is None) != (self.correctness_verified is None):
            raise ValueError("implementation and correctness_verified must appear together")
        _check_profile_tallies(self.counts, self.cases)
        return self


ProfileResult = Annotated[
    OpProfileResult | DeepProfileResult, Field(discriminator="mode")
]


CorrectnessOutcome = Annotated[
    CheckSucceeded[CorrectnessResult] | CheckFailed | CheckSkipped | CheckTimedOut,
    Field(discriminator="status"),
]
PerformanceOutcome = Annotated[
    CheckSucceeded[PerformanceResult | SinglePerformanceResult]
    | CheckFailed | CheckSkipped | CheckTimedOut,
    Field(discriminator="status"),
]
ProfileOutcome = Annotated[
    CheckSucceeded[ProfileResult] | CheckFailed | CheckSkipped | CheckTimedOut,
    Field(discriminator="status"),
]
CompileCheckOutcome = Annotated[
    CheckSucceeded[CompileOutcome] | CheckFailed | CheckSkipped | CheckTimedOut,
    Field(discriminator="status"),
]


# ---------------------------------------------------------------------------
# Provenance, result, and resource (design sections 4.1 and 4.5)
# ---------------------------------------------------------------------------


class ExecutionEnvironment(StrictModel):
    hardware_model: str
    hardware_count: int
    driver_version: str
    runtime_version: str
    torch_version: str | None
    torch_npu_version: str | None
    triton_version: str | None
    toolchain_version: str


class Provenance(StrictModel):
    schema_version: SchemaVersion
    evaluator_build: str
    correctness_algorithm: str
    decoy_algorithm: str
    benchmark_algorithm: str
    profiler_algorithm: str
    grade_eligible: bool
    manifest_id: str | None
    problem_id: str | None
    execution_environment: ExecutionEnvironment | None

    @model_validator(mode="after")
    def _grade_eligibility(self):
        if (self.manifest_id is None) != (self.problem_id is None):
            raise ValueError("manifest_id and problem_id must move together")
        if self.grade_eligible and self.manifest_id is None:
            raise ValueError("grade_eligible requires manifest identity")
        return self


class EvaluationResult(StrictModel):
    correctness: CorrectnessOutcome | None
    performance: PerformanceOutcome | None
    profile: ProfileOutcome | None
    compile: CompileCheckOutcome | None = None
    provenance: Provenance

    @model_validator(mode="after")
    def _requested_checks_filled(self, info: ValidationInfo):
        context = info.context
        if isinstance(context, dict) and context.get("requested_checks") is not None:
            requested = context["requested_checks"]
            for check in Check.__args__:
                value = getattr(self, check)
                if check in requested and value is None:
                    raise ValueError(f"requested check {check} must have an outcome")
                if check not in requested and value is not None:
                    raise ValueError(f"unrequested check {check} must be null")
        return self


def dump_contract(model: StrictModel) -> bytes:
    """Serialize a durable contract model as canonical JSON, preserving null slots."""
    from ..serialization import canonical_json_bytes

    return canonical_json_bytes(model.model_dump(mode="json", exclude_none=False))


_TERMINAL_STATES = ("completed", "failed", "timed_out")


class EvaluationResource(StrictModel):
    schema_version: SchemaVersion
    evaluation_id: str = Field(min_length=1)
    revision: Annotated[int, Field(ge=1)]
    state: ResourceState
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    deadline_at: datetime
    request_digest: Digest
    reference_digest: Digest
    result: EvaluationResult | None

    @model_validator(mode="after")
    def _lifecycle_shape(self):
        if self.state in _TERMINAL_STATES:
            if self.result is None or self.finished_at is None:
                raise ValueError("terminal resources carry result and finished_at")
        else:
            if self.result is not None or self.finished_at is not None:
                raise ValueError("nonterminal resources carry neither result nor finished_at")
        if self.state == "queued":
            if self.started_at is not None:
                raise ValueError("queued resources have no started_at")
        elif self.state in ("running", "completed", "failed"):
            if self.started_at is None:
                raise ValueError("started resources require started_at")
        # timed_out may lack started_at: a queued deadline fires before any attempt.
        try:
            if self.updated_at < self.created_at:
                raise ValueError("updated_at must not precede created_at")
            if self.started_at is not None and self.started_at < self.created_at:
                raise ValueError("started_at must not precede created_at")
            if self.finished_at is not None:
                if self.finished_at < self.created_at:
                    raise ValueError("finished_at must not precede created_at")
                if self.started_at is not None and self.finished_at < self.started_at:
                    raise ValueError("finished_at must not precede started_at")
            if self.deadline_at <= self.created_at:
                raise ValueError("deadline_at must follow created_at")
        except TypeError as exc:
            raise ValueError("timestamps must share timezone awareness") from exc
        return self
