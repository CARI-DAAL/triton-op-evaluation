"""The only boundary between private engine JSON and the stable v2 contract.

Every private protocol model below is strict: unmodeled keys, scalar coercion,
and ordinary non-finite floats are rejected, so nothing the engine emits can
drift silently into stable output. The single documented exception is the
verify case ``metrics`` object, whose engine keys are category-dependent
diagnostics; only the explicitly projected subset is read and the rest is
dropped (never copied) at the boundary.

Shaping rules enforced here (design sections 4.2-4.5 and 5):

* divergence kinds are projected to the coarse stable taxonomy
  (``shape_mismatch`` / ``value_mismatch`` / ``non_finite_mismatch``); the
  engine's experimental names never leak;
* samples keep the engine's worst-k selection (validated) but are re-ranked
  into the contract order: finite ``abs(diff)`` descending, then sentinels
  lexically, ties broken by row-major index; at most eight points;
* verify/benchmark case indices are 1-based in the engine and 0-based in the
  contract; profile indices are already 0-based;
* only failures carrying the engine's private attribution object
  (``candidate_error``) may become a ``DomainFailure``; anything else fails
  closed as invalid engine output;
* performance is recomputed from per-case measurements (never the engine
  aggregate) under ``geometric_mean_v1``.
"""

import math
import re
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from ..api.contract import (
    CaseFailure,
    CaseMetrics,
    CorrectnessResult,
    DeepProfileCase,
    DeepProfileResult,
    DeepRatios,
    DomainFailure,
    ErrorInfo,
    FiniteOrSentinel,
    InputShape,
    MeasurementMethod,
    OpKernel,
    OpKernelMetrics,
    OpProfileCase,
    OpProfileResult,
    PerformanceCase,
    PerformanceResult,
    ProfileCounts,
    ProfileResult,
    SampleElement,
    StrictModel,
    UnmeasuredPerformanceCase,
)

MAX_SAMPLE_POINTS = 8
MAX_MESSAGE_BYTES = 512

# The engine's calibrated-experimental divergence names all project to the
# coarse stable ``value_mismatch`` (design section 4.2).
_EXPERIMENTAL_KINDS = {
    "sign_flip",
    "value_clamp",
    "constant_offset",
    "per_axis",
    "precision_loss",
    "scattered",
}

_PATH_TOKEN = re.compile(r"(?<!\S)/\S+")


class ShapingError(ValueError):
    """An engine result cannot satisfy the stable v2 contract."""


class UnattributedFailure(ShapingError):
    """A case failed in a way the engine protocol cannot attribute.

    The engine records non-accuracy case failures (framework-side crashes,
    device errors, output-flattening TypeErrors) as ``kind=None`` with no
    ``candidate_error``. That is not a protocol violation by the engine —
    the private protocol simply has no attribution for them — so the caller
    must project the check as a retryable measurement error instead of an
    invalid engine output. Carries one bounded, single-line diagnostic so
    the evaluation response remains debuggable.
    """

    def __init__(self, detail: str) -> None:
        super().__init__("correctness result carries an unattributed failure")
        # The detail flows into an ErrorInfo message (single line, no raw
        # paths, bounded UTF-8); the engine's error_msg is none of those.
        single_line = " ".join(detail.split())
        single_line = _PATH_TOKEN.sub("<path>", single_line)
        self.detail = single_line[:MAX_MESSAGE_BYTES]


_SHAPING_METRICS = None  # optional RedisMetrics; wired by the deployment


def note_shaping_failure(check: str) -> None:
    """Best-effort bounded counter at the exact shaping failure decision."""
    if _SHAPING_METRICS is None:
        return
    try:
        _SHAPING_METRICS.increment("shaping_failures_total", {"check": check})
    except ValueError:
        pass


def _parse(model, raw, name: str):
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        note_shaping_failure(name)
        raise ShapingError(f"{name} result does not match the engine protocol") from exc


def _checked(build, name: str):
    try:
        return build()
    except ValidationError as exc:
        note_shaping_failure(name)
        raise ShapingError(f"{name} result cannot satisfy the stable contract") from exc


def _redacted_message(value: str) -> str:
    """Attribution messages must be single-line, bounded, and free of raw
    tracebacks or absolute paths (the engine emits them pre-redacted)."""
    if "\r" in value or "\n" in value:
        raise ValueError("message must be a single line")
    if not 1 <= len(value.encode("utf-8")) <= MAX_MESSAGE_BYTES:
        raise ValueError("message must be 1..512 UTF-8 bytes")
    if "Traceback" in value or _PATH_TOKEN.search(value):
        raise ValueError("message must not contain tracebacks or absolute paths")
    return value


def _redact(text: object) -> str:
    """Best-effort redaction for free-text engine detail (profile case errors)."""
    line = " ".join(str(text).split())
    cut = line.find("Traceback")
    if cut >= 0:
        line = line[:cut].rstrip(" :")
    line = _PATH_TOKEN.sub("<path>", line)
    line = line.encode("utf-8", errors="ignore")[:MAX_MESSAGE_BYTES].decode(
        "utf-8", errors="ignore"
    )
    return line or "engine reported an error"


# ---------------------------------------------------------------------------
# Shared private pieces
# ---------------------------------------------------------------------------


class _TensorDesc(StrictModel):
    type: Literal["tensor"]
    shape: list[Annotated[int, Field(ge=0)]]
    dtype: str


class _ScalarDesc(StrictModel):
    type: Literal["scalar"]
    value: bool | int | float | str


_InputDesc = Annotated[_TensorDesc | _ScalarDesc, Field(discriminator="type")]


def _input_shapes(desc: list[_InputDesc]) -> list[InputShape]:
    """Structured tensor descriptors only; ``input_index`` is the position in
    the engine's input list. Compressed string shapes never reach here."""
    return [
        InputShape(input_index=index, dimensions=item.shape, dtype=item.dtype)
        for index, item in enumerate(desc)
        if isinstance(item, _TensorDesc)
    ]


# ---------------------------------------------------------------------------
# Correctness (verify.py private protocol)
# ---------------------------------------------------------------------------


class _SamplePoint(StrictModel):
    idx: list[int]
    candidate: FiniteOrSentinel
    reference: FiniteOrSentinel
    diff: FiniteOrSentinel


class _CandidateError(StrictModel):
    """Engine-private attribution object (verify.py ``_candidate_error``)."""

    phase: Literal["compile", "initialize", "execute", "execution_timeout"]
    message: Annotated[str, AfterValidator(_redacted_message)]


def _finite_or_none(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class _EngineCaseMetrics(StrictModel):
    # The engine's metrics object is a category-dependent diagnostics bag;
    # extras are dropped here, never carried into stable output. Optional
    # floats that arrive non-finite become null (design: "mere 仅在 engine 有
    # 有效 finite 值时返回"), so a BeforeValidator runs ahead of strict typing.
    model_config = ConfigDict(extra="ignore")

    max_abs_diff: Annotated[float | None, BeforeValidator(_finite_or_none)] = None
    matched_ratio: Annotated[float | None, BeforeValidator(_finite_or_none)] = None
    MERE: Annotated[float | None, BeforeValidator(_finite_or_none)] = None
    small_count: int | None = None
    normal_count: int | None = None
    atol: Annotated[float | None, BeforeValidator(_finite_or_none)] = None
    rtol: Annotated[float | None, BeforeValidator(_finite_or_none)] = None


def _engine_rank(point: _SamplePoint) -> tuple:
    """The engine's worst-first order: non-finite diffs rank worst, then
    descending abs(diff), ties broken by row-major index."""
    if isinstance(point.diff, str):
        return (float("-inf"), point.idx)
    return (-abs(point.diff), point.idx)


class _VerifyFailure(StrictModel):
    case_idx: Annotated[int, Field(ge=1)]
    input_desc: list[_InputDesc] | None = None
    error_type: str | None = None
    error_msg: str | None = None
    kind: str | None = None
    sample: list[_SamplePoint] = Field(default_factory=list, max_length=MAX_SAMPLE_POINTS)
    metrics: _EngineCaseMetrics | None = None
    candidate_error: _CandidateError | None = None

    @model_validator(mode="after")
    def _sample_is_engine_ordered(self):
        keys = [_engine_rank(point) for point in self.sample]
        if keys != sorted(keys):
            raise ValueError("sample is not in the engine's worst-first order")
        return self


class _VerifyResult(StrictModel):
    op_name: str
    total_cases: Annotated[int, Field(ge=1)]
    passed_cases: Annotated[int, Field(ge=0)]
    failed_cases: Annotated[int, Field(ge=0)]
    failures: list[_VerifyFailure]

    @model_validator(mode="after")
    def _counts_and_identity(self):
        if self.passed_cases + self.failed_cases != self.total_cases:
            raise ValueError("passed/failed counts contradict total_cases")
        indices = [failure.case_idx for failure in self.failures]
        if len(set(indices)) != len(indices):
            raise ValueError("duplicate failure case_idx")
        if any(index > self.total_cases for index in indices):
            raise ValueError("failure case_idx out of range")
        if len(self.failures) > self.failed_cases:
            raise ValueError("more failure entries than failed cases")
        return self


class _DecoyResult(StrictModel):
    decoy: Literal[True]
    total_cases: Annotated[int, Field(ge=1)]


def _coarse_kind(failure: _VerifyFailure) -> str:
    if failure.kind == "shape_mismatch":
        return "shape_mismatch"
    if failure.kind in _EXPERIMENTAL_KINDS:
        if any(
            isinstance(value, str)
            for point in failure.sample
            for value in (point.candidate, point.reference, point.diff)
        ):
            return "non_finite_mismatch"
        return "value_mismatch"
    raise ShapingError("correctness result carries an unknown divergence kind")


def _case_metrics(raw: _EngineCaseMetrics | None) -> CaseMetrics:
    if raw is None:
        return CaseMetrics()
    return CaseMetrics(
        max_abs_diff=raw.max_abs_diff,
        matched_ratio=raw.matched_ratio,
        mere=raw.MERE,
        small_count=raw.small_count,
        normal_count=raw.normal_count,
        atol=raw.atol,
        rtol=raw.rtol,
    )


def _stable_rank(point: SampleElement) -> tuple:
    """Contract sample order: finite abs(diff) descending, then sentinels
    lexically, ties broken by row-major index."""
    if isinstance(point.diff, str):
        return (1, point.diff, point.idx)
    return (0, -abs(point.diff), point.idx)


def _case_failure(failure: _VerifyFailure) -> CaseFailure:
    sample = [
        SampleElement(
            idx=point.idx,
            candidate=point.candidate,
            reference=point.reference,
            diff=point.diff,
        )
        for point in failure.sample
    ]
    sample.sort(key=_stable_rank)
    return CaseFailure(
        case_idx=failure.case_idx - 1,
        kind=_coarse_kind(failure),
        metrics=_case_metrics(failure.metrics),
        sample=sample,
    )


def _domain_failure(attributed: list[_VerifyFailure]) -> DomainFailure:
    error = attributed[0].candidate_error
    if error.phase == "compile":
        return DomainFailure(
            code="SUBJECT_COMPILE_ERROR",
            phase="compile",
            case_idx=None,
            message=error.message,
        )
    if error.phase == "execution_timeout":
        return DomainFailure(
            code="SUBJECT_EXECUTION_TIMEOUT",
            phase="execute",
            case_idx=attributed[0].case_idx - 1,
            message=error.message,
        )
    return DomainFailure(
        code="SUBJECT_RUNTIME_ERROR",
        phase=error.phase,
        case_idx=attributed[0].case_idx - 1 if error.phase == "execute" else None,
        message=error.message,
    )


def shape_correctness(raw: dict[str, object], decoy: bool) -> CorrectnessResult:
    if not isinstance(decoy, bool):
        raise ShapingError("correctness decoy flag must be a boolean")
    if decoy:
        payload = _parse(_DecoyResult, raw, "correctness")
        return _checked(
            lambda: CorrectnessResult(
                outcome="wrong",
                passed=False,
                decoy=True,
                passed_cases=payload.total_cases,
                total_cases=payload.total_cases,
                failures=[],
            ),
            "correctness",
        )

    result = _parse(_VerifyResult, raw, "correctness")
    failures = result.failures
    attributed = [f for f in failures if f.candidate_error is not None]
    mismatches = [f for f in failures if f.candidate_error is None]
    unattributed = [f for f in mismatches if f.kind is None]
    if unattributed:
        first = unattributed[0]
        detail = f"{first.error_type or 'unknown'}: {first.error_msg or 'no message'}"
        raise UnattributedFailure(detail)

    outcome = None
    domain_failure = None
    if attributed:
        phases = {f.candidate_error.phase for f in attributed}
        if "compile" in phases:
            if phases != {"compile"} or mismatches:
                raise ShapingError("compile attribution cannot mix with case results")
            outcome = "compile_error"
        else:
            if len(attributed) != 1 or attributed[0] is not failures[-1]:
                raise ShapingError("candidate attribution must terminate the case sequence")
            phase = attributed[0].candidate_error.phase
            outcome = "execution_timeout" if phase == "execution_timeout" else "runtime_error"
        domain_failure = _domain_failure(attributed)

    case_failures = _checked(
        lambda: [_case_failure(failure) for failure in mismatches], "correctness"
    )
    if outcome is None:
        outcome = "wrong" if case_failures else "correct"
    return _checked(
        lambda: CorrectnessResult(
            outcome=outcome,
            passed=outcome == "correct",
            decoy=False,
            passed_cases=result.passed_cases,
            total_cases=result.total_cases,
            failures=case_failures,
            domain_failure=domain_failure,
        ),
        "correctness",
    )


# ---------------------------------------------------------------------------
# Performance (benchmark.py private protocol)
# ---------------------------------------------------------------------------


class _ShapeMeasurement(StrictModel):
    avg_latency_ms: float
    peak_memory_mb: float
    latency_samples_ms: list[float] | None = None
    latency_samples_source: str | None = None
    latency_samples_proof_sha256: str | None = None


class _AggregateMeasurement(StrictModel):
    avg_latency_ms: float
    peak_memory_mb: float
    operators: dict[str, float]


class _PerShapeResult(StrictModel):
    case_idx: Annotated[int, Field(ge=1)]
    input_desc: list[_InputDesc]
    status: Literal["pass", "fail"]
    framework: _ShapeMeasurement | None
    implementation: _ShapeMeasurement | None
    speedup_vs_torch: float | None
    measurement_method: MeasurementMethod | None = None
    error_type: str | None
    error_msg: str | None

    @model_validator(mode="after")
    def _speedup_is_null_or_positive(self):
        if self.speedup_vs_torch is not None and self.speedup_vs_torch <= 0:
            raise ValueError("abnormal speedups must be normalized to null")
        if self.status == "fail" and (
            self.framework is not None or self.implementation is not None
        ):
            raise ValueError("failed cases must not carry measurements")
        return self


class _BenchmarkResult(StrictModel):
    op_name: str
    warmup: int
    repeats: int
    max_retries: int
    total_cases: Annotated[int, Field(ge=1)]
    passed_cases: Annotated[int, Field(ge=0)]
    failed_cases: Annotated[int, Field(ge=0)]
    nan_indices: list[int]
    inf_indices: list[int]
    zero_indices: list[int]
    negative_indices: list[int]
    none_indices: list[int]
    framework: _AggregateMeasurement | None
    implementation: _AggregateMeasurement | None
    speedup_vs_torch: float | None
    measurement_method: MeasurementMethod | None = None
    per_shape_results: list[_PerShapeResult]
    # One-based original identities of the cases this run measured; None means
    # the full contiguous task (legacy engine output).
    case_indices: list[Annotated[int, Field(ge=1)]] | None = None

    @model_validator(mode="after")
    def _counts_and_identity(self):
        if self.passed_cases + self.failed_cases != self.total_cases:
            raise ValueError("passed/failed counts contradict total_cases")
        if len(self.per_shape_results) != self.total_cases:
            raise ValueError("per_shape_results must cover every case")
        indices = [case.case_idx for case in self.per_shape_results]
        if len(set(indices)) != len(indices):
            raise ValueError("duplicate case_idx")
        abnormal = (
            self.nan_indices
            + self.inf_indices
            + self.zero_indices
            + self.negative_indices
            + self.none_indices
        )
        if len(set(abnormal)) != len(abnormal):
            raise ValueError("abnormal-speedup index lists overlap")
        if self.case_indices is not None:
            if len(self.case_indices) != self.total_cases:
                raise ValueError("case_indices must cover every selected case")
            if len(set(self.case_indices)) != len(self.case_indices):
                raise ValueError("case_indices must be unique")
            allowed = set(self.case_indices)
            if any(index not in allowed for index in indices):
                raise ValueError("case_idx outside the declared selection")
            if any(index not in allowed for index in abnormal):
                raise ValueError("abnormal-speedup index outside the declared selection")
        else:
            if any(index > self.total_cases for index in indices):
                raise ValueError("case_idx out of range")
            if any(index < 1 or index > self.total_cases for index in abnormal):
                raise ValueError("abnormal-speedup index out of range")
        by_index = {case.case_idx: case for case in self.per_shape_results}
        for index in abnormal:
            case = by_index[index]
            if case.status != "pass" or case.speedup_vs_torch is not None:
                raise ValueError("abnormal-speedup cases must pass with null speedup")
        return self


def shape_performance(raw: dict[str, object]) -> PerformanceResult:
    result = _parse(_BenchmarkResult, raw, "benchmark")
    per_case: list[PerformanceCase] = []
    unmeasured: list[UnmeasuredPerformanceCase] = []
    for case in result.per_shape_results:
        shapes = _input_shapes(case.input_desc)
        if case.status == "fail" or case.speedup_vs_torch is None:
            unmeasured.append(
                UnmeasuredPerformanceCase(
                    case_idx=case.case_idx - 1,
                    input_shapes=shapes,
                    reason_code="NO_STABLE_MEASUREMENT",
                )
            )
            continue
        if case.framework is None or case.implementation is None:
            raise ShapingError("benchmark passed case is missing a measurement")
        reference = case.framework.avg_latency_ms
        implementation = case.implementation.avg_latency_ms
        if not (reference > 0 and implementation > 0):
            raise ShapingError("benchmark result contains a non-positive latency")
        per_case.append(
            PerformanceCase(
                case_idx=case.case_idx - 1,
                input_shapes=shapes,
                implementation_latency_ms=implementation,
                reference_latency_ms=reference,
                speedup_vs_reference=reference / implementation,
                measurement_method=case.measurement_method,
                reference_latency_samples_ms=case.framework.latency_samples_ms,
                reference_latency_samples_source=case.framework.latency_samples_source,
                reference_timing_policy_sha256=case.framework.latency_samples_proof_sha256,
            )
        )
    if not per_case:
        raise ShapingError("benchmark result contains no valid measurement")
    selected = len(result.per_shape_results)
    overall = math.exp(
        sum(math.log(case.speedup_vs_reference) for case in per_case) / len(per_case)
    )
    return _checked(
        lambda: PerformanceResult(
            aggregation="geometric_mean_v1",
            overall_speedup_vs_reference=overall,
            selected_cases=selected,
            measured_cases=len(per_case),
            coverage=len(per_case) / selected,
            measurement_method=(result.measurement_method if per_case else None),
            per_case=per_case,
            unmeasured_cases=unmeasured,
        ),
        "benchmark",
    )


# ---------------------------------------------------------------------------
# Profile (profile_runner.py private protocol)
# ---------------------------------------------------------------------------


class _OpKernelRaw(StrictModel):
    name: Annotated[str, Field(min_length=1)]
    op_type: str
    duration_us: float | None
    metrics: dict[str, object]
    missing_metrics: list[str]


class _Recommendation(StrictModel):
    bound: str
    evidence: dict[str, object]
    recommendation: str
    source: str


class _BoundRatios(StrictModel):
    MTE2: float | None = Field(default=None, ge=0, le=1)
    CUBE: float | None = Field(default=None, ge=0, le=1)
    PUSHQ: float | None = Field(default=None, ge=0, le=1)
    FIXP: float | None = Field(default=None, ge=0, le=1)
    MTE3: float | None = Field(default=None, ge=0, le=1)
    SCALAR: float | None = Field(default=None, ge=0, le=1)
    SCALARLDST: float | None = Field(default=None, ge=0, le=1)


class _DeepBound(StrictModel):
    name: str
    ratios: _BoundRatios
    rule: str
    evidence: dict[str, object]


class _ProfileCase(StrictModel):
    status: Literal["complete", "partial", "failed"]
    target_kernel: str | None
    target_selection_metric: str | None
    kernels: list[_OpKernelRaw]
    raw_metrics: dict[str, object]
    bound: _DeepBound | None
    per_core: dict[str, object] | None
    recommendations: list[_Recommendation]
    missing_metrics: list[str]
    warnings: list[str]
    error: str | None
    case_idx: Annotated[int, Field(ge=0)]
    input_desc: list[_InputDesc]
    msprof_wallclock_sec: float


class _ProfileEnvelope(StrictModel):
    profile_mode: Literal["op", "deep"]
    requested_case_idx: int | None
    total_cases: Annotated[int, Field(ge=1)]
    selected_cases: Annotated[int, Field(ge=0)]
    completed_cases: Annotated[int, Field(ge=0)]
    partial_cases: Annotated[int, Field(ge=0)]
    failed_cases: Annotated[int, Field(ge=0)]
    remaining_case_indices: list[int]
    timed_out: bool
    summary_text: str
    warnings: list[str]
    wallclock_sec: float
    per_case_results: list[_ProfileCase]

    @model_validator(mode="after")
    def _counts_and_identity(self):
        if self.timed_out:
            raise ValueError("engine timeouts are check outcomes, not profile data")
        if self.remaining_case_indices:
            raise ValueError("a finished profile has no remaining cases")
        if self.selected_cases != self.completed_cases + self.partial_cases + (self.failed_cases):
            raise ValueError("selected count must equal the status sum")
        if self.selected_cases != len(self.per_case_results):
            raise ValueError("selected count must equal the number of case results")
        tallies = {"complete": 0, "partial": 0, "failed": 0}
        for case in self.per_case_results:
            tallies[case.status] += 1
        if (
            tallies["complete"] != self.completed_cases
            or tallies["partial"] != self.partial_cases
            or tallies["failed"] != self.failed_cases
        ):
            raise ValueError("status counts contradict the case results")
        indices = [case.case_idx for case in self.per_case_results]
        if len(set(indices)) != len(indices):
            raise ValueError("duplicate case_idx")
        if any(index >= self.total_cases for index in indices):
            raise ValueError("case_idx out of range")
        return self


_BOUND_NAMES = {
    "MTE2 BOUND": "MTE2",
    "CUBE BOUND": "CUBE",
    "VEC BOUND": "VEC",
    "FIXP BOUND": "FIXP",
    "MTE3 BOUND": "MTE3",
    "SCALAR BOUND": "SCALAR",
    "NO BOUND": "NONE",
}


def _case_error(code: str, detail: object) -> ErrorInfo:
    message = _redact(detail) if detail else "profile case did not complete"
    return ErrorInfo(
        code=code, phase="execution", check="profile", retryable=False, message=message
    )


# msprof pipe-utilization columns are fractions; Memory.csv / L2Cache.csv
# usage-rate and hit-rate columns are percentages. The contract exposes both
# as fractions in [0, 1]; anything unparsable or out of range stays None.
_PIPE_FRACTION_METRICS = (
    "aic_cube_ratio",
    "aic_mte2_ratio",
    "aic_fixpipe_ratio",
    "aiv_vec_ratio",
    "aiv_mte2_ratio",
    "aiv_mte3_ratio",
)
_PERCENT_FRACTION_METRICS = (
    "gm_to_l1_usage",
    "l0c_to_gm_usage",
    "gm_to_ub_usage",
    "ub_to_gm_usage",
    "aic_l2_hit",
    "aiv_l2_hit",
)


def _metric_fraction(value: object, *, scale: float) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    fraction = float(value) * scale
    if not 0.0 <= fraction <= 1.0:
        return None
    return fraction


def _kernel_metrics(kernel: _OpKernelRaw) -> OpKernelMetrics:
    raw = kernel.metrics
    fields: dict[str, float | None] = {
        name: _metric_fraction(raw.get(name), scale=1.0) for name in _PIPE_FRACTION_METRICS
    }
    fields.update(
        {
            name: _metric_fraction(raw.get(name), scale=1.0 / 100.0)
            for name in _PERCENT_FRACTION_METRICS
        }
    )
    return OpKernelMetrics(**fields)


def _op_kernels(case: _ProfileCase) -> list[OpKernel]:
    durations: list[float] = []
    for kernel in case.kernels:
        duration = kernel.duration_us
        if duration is None or not duration > 0:
            raise ShapingError("profile op case contains an invalid kernel duration")
        durations.append(duration)
    total = sum(durations)
    if case.kernels and not total > 0:
        raise ShapingError("profile op case has no measurable kernel time")
    return [
        OpKernel(
            op_name=kernel.name,
            time_ms=kernel.duration_us / 1000.0,
            percentage=kernel.duration_us / total * 100.0,
            metrics=_kernel_metrics(kernel),
        )
        for kernel in case.kernels
    ]


def _op_case(case: _ProfileCase) -> OpProfileCase:
    if case.bound is not None or case.per_core is not None:
        raise ShapingError("op profile case carries deep-only diagnostics")
    if case.status == "failed":
        if case.error is None:
            raise ShapingError("failed profile case must explain the failure")
        return OpProfileCase(
            case_idx=case.case_idx,
            status="failed",
            kernels=[],
            error=_case_error("PROFILE_CASE_FAILED", case.error),
        )
    kernels = _op_kernels(case)
    if case.status == "partial":
        detail = case.error or "; ".join([*case.warnings, *case.missing_metrics])
        return OpProfileCase(
            case_idx=case.case_idx,
            status="partial",
            kernels=kernels,
            error=_case_error("PROFILE_CASE_PARTIAL", detail),
        )
    return OpProfileCase(case_idx=case.case_idx, status="complete", kernels=kernels, error=None)


def _spread_ratio(per_core: dict[str, object]) -> float:
    value = per_core.get("spread_ratio")
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ShapingError("deep profile case has an invalid per-core spread ratio")
    return float(value)


def _deep_case(case: _ProfileCase) -> DeepProfileCase:
    if case.kernels:
        raise ShapingError("deep profile case carries op-only kernel timings")
    bound = None
    ratios = DeepRatios()
    if case.bound is not None:
        bound = _BOUND_NAMES.get(case.bound.name)
        if bound is None:
            raise ShapingError("deep profile case carries an unknown bound name")
        raw = case.bound.ratios
        scalar = [v for v in (raw.SCALAR, raw.SCALARLDST) if v is not None]
        ratios = DeepRatios(
            mte2=raw.MTE2,
            cube=raw.CUBE,
            vec=raw.PUSHQ,
            fixp=raw.FIXP,
            mte3=raw.MTE3,
            scalar=max(scalar) if scalar else None,
        )
    spread = _spread_ratio(case.per_core) if case.per_core is not None else None
    if case.status == "failed":
        if case.error is None:
            raise ShapingError("failed profile case must explain the failure")
        error = _case_error("PROFILE_CASE_FAILED", case.error)
    elif case.status == "partial":
        error = _case_error("PROFILE_CASE_PARTIAL", case.error) if case.error is not None else None
    else:
        if case.error is not None:
            raise ShapingError("complete profile case must not carry an error")
        error = None
    return DeepProfileCase(
        case_idx=case.case_idx,
        status=case.status,
        target_kernel=case.target_kernel,
        bound=bound,
        ratios=ratios,
        per_core_spread_ratio=spread,
        recommendations=[item.recommendation for item in case.recommendations],
        missing_metrics=list(case.missing_metrics),
        warnings=list(case.warnings),
        error=error,
    )


def shape_profile(raw: dict[str, object]) -> ProfileResult:
    envelope = _parse(_ProfileEnvelope, raw, "profile")
    counts = ProfileCounts(
        selected=envelope.selected_cases,
        complete=envelope.completed_cases,
        partial=envelope.partial_cases,
        failed=envelope.failed_cases,
    )
    if envelope.profile_mode == "op":
        return _checked(
            lambda: OpProfileResult(
                mode="op",
                counts=counts,
                cases=[_op_case(case) for case in envelope.per_case_results],
            ),
            "profile",
        )
    return _checked(
        lambda: DeepProfileResult(
            mode="deep",
            counts=counts,
            cases=[_deep_case(case) for case in envelope.per_case_results],
        ),
        "profile",
    )
