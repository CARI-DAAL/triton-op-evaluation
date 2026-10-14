"""Fixed-order v2 check orchestration with fenced progress and provenance.

Order and dependency rules (design section 5):

* compile obtains syntax/import evidence inside the leased attempt;
* correctness runs before the remaining on-card checks;
* performance runs only when the correctness check ``status == "succeeded"``
  *and* the subject ``data.outcome == "correct"`` — the derived ``passed``
  flag is never consulted; any other subject outcome skips performance with
  ``DEPENDENCY_NOT_PASSED``, and a failed/timeout correctness skips it with
  ``PRIOR_CHECK_FAILED``;
* profile is independent of the subject verdict (a wrong or decoy kernel is
  still profiled) and is skipped with ``PRIOR_CHECK_FAILED`` only when an
  earlier infrastructure failure made the worker environment untrustworthy;
* every requested check — including synthesized skips — is fenced through
  ``ProgressSink.begin``/``save``; a rejected write or a lost lease aborts the
  attempt via ``LeaseLostError`` and nothing further is produced;
* every requested slot is filled on every return path, and provenance is
  built from admission-pinned build/algorithm versions plus the actual
  worker environment. ``grade_eligible`` is the only service-side grade
  decision: a completed, manifest-bound, trusted grade run with complete
  reward-relevant provenance. It is never a score.
"""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from time import monotonic

from ..api.contract import (
    MAX_ERROR_MESSAGE_BYTES,
    Check,
    CheckFailed,
    CheckSkipped,
    CheckSucceeded,
    CheckTimedOut,
    CorrectnessResult,
    ErrorInfo,
    EvaluationRequest,
    EvaluationResult,
    ExecutionEnvironment,
    PerformanceResult,
    ProfileParams,
    ProfileResult,
    SinglePerformanceResult,
    SkipInfo,
)
from ..runtime.correctness_evidence import CorrectnessEvidenceError
from ..service.models import PinnedProvenance
from ..service.terminal_results import compile_outcome_from_precheck, complete_evaluation
from .base import (
    EngineCheckResult,
    LeaseLostError,
    ProgressSink,
)
from .benchmark import BenchmarkEvaluator
from .profile import ProfileEvaluator
from .shaping import (
    ShapingError,
    UnattributedFailure,
    shape_correctness,
    shape_performance,
    shape_profile_runs,
)
from .verify import VerifyEvaluator

_DEPENDENCY_MESSAGE = "correctness did not pass"
_PRIOR_FAILED_MESSAGE = "a prior requested check did not complete"

_CHECK_DATA_MODELS = {
    "correctness": CorrectnessResult,
    "performance": PerformanceResult | SinglePerformanceResult,
    "profile": ProfileResult,
}


def _is_decoy(raw: object) -> bool:
    return isinstance(raw, dict) and raw.get("decoy") is True


def _engine_outcome(check: Check, engine: EngineCheckResult, shape):
    """Map one engine invocation to a check outcome; subject verdicts come
    only from the shaping layer, never from process-level signals."""
    if engine.status == "timeout":
        return CheckTimedOut(status="timeout", error=engine.error)
    if engine.status == "failed":
        return CheckFailed(status="failed", error=engine.error)
    if engine.raw is None:
        return CheckFailed(
            status="failed",
            error=ErrorInfo(
                code="EVALUATOR_EXECUTION_FAILED",
                phase="execution",
                check=check,
                retryable=True,
                message="engine produced no usable result",
            ),
        )
    try:
        data = shape(engine.raw)
    except UnattributedFailure as error:
        # The engine honestly recorded a case failure it cannot attribute
        # (kind=None, no candidate_error). That is a measurement gap, not a
        # protocol violation: project it as retryable so a transient device
        # error recovers on resubmission instead of voiding the evaluation.
        # The prefix counts against ErrorInfo's 512-byte message cap: the
        # detail alone is bounded to 512 by UnattributedFailure, so compose
        # under the cap or ErrorInfo validation raises and the worker collapses
        # the whole evaluation into a protocol failure.
        prefix = "unattributed case failure: "
        message = prefix + error.detail[: MAX_ERROR_MESSAGE_BYTES - len(prefix)]
        return CheckFailed(
            status="failed",
            error=ErrorInfo(
                code="INTERNAL",
                phase="execution",
                check=check,
                retryable=True,
                message=message,
            ),
        )
    except ShapingError:
        return CheckFailed(
            status="failed",
            error=ErrorInfo(
                code="INVALID_ENGINE_OUTPUT",
                phase="shaping",
                check=check,
                retryable=False,
                message="engine output did not match the private protocol",
            ),
        )
    return CheckSucceeded[_CHECK_DATA_MODELS[check]](status="succeeded", data=data)


def _skipped(check: Check, code: str, depends_on: Check | None, message: str):
    return CheckSkipped(
        status="skipped",
        reason=SkipInfo(code=code, depends_on=depends_on, message=message),
    )


class EvaluationEvaluator:
    def __init__(
        self, *, verify: VerifyEvaluator, benchmark: BenchmarkEvaluator, profile: ProfileEvaluator
    ):
        self.verify = verify
        self.benchmark = benchmark
        self.profile = profile

    @staticmethod
    def _slot(check: Check, produce, lease_valid, progress: ProgressSink):
        if not progress.begin(check):
            raise LeaseLostError(f"begin({check}) was rejected by the fenced sink")
        if not lease_valid():
            raise LeaseLostError(f"lease lost before {check} execution")
        outcome = produce()
        if not lease_valid():
            raise LeaseLostError(f"lease lost during {check} execution")
        if not progress.save(check, outcome):
            raise LeaseLostError(f"save({check}) was rejected by the fenced sink")
        return outcome

    def run(
        self,
        request: EvaluationRequest,
        work_dir: Path,
        device_id: int | None,
        deadline_at: datetime,
        lease_valid: Callable[[], bool],
        execution_environment: ExecutionEnvironment | None,
        pinned: PinnedProvenance,
        progress: ProgressSink,
        *,
        precheck: Callable[[], str | None] | None = None,
        reuse_correctness: Callable[[], CheckSucceeded[CorrectnessResult]] | None = None,
    ) -> EvaluationResult:
        """Run correctness, conditional performance, then independent profile."""
        requested = list(request.checks)
        outcomes: dict[str, object] = {}
        infra_broken = False
        trace_engine = None
        trace_started = False

        if device_id is None and requested != ["compile"]:
            raise ValueError("CPU attempts can only execute the compile check")

        def compile_check():
            started = monotonic()
            verdict = precheck() if precheck is not None else None
            return compile_outcome_from_precheck(verdict, (monotonic() - started) * 1000)

        def correctness_check():
            if request.correctness_from is not None:
                try:
                    if reuse_correctness is None:
                        raise CorrectnessEvidenceError()
                    reused = reuse_correctness()
                    try:
                        reused = CheckSucceeded[CorrectnessResult].model_validate(reused)
                    except ValueError as error:
                        raise CorrectnessEvidenceError() from error
                    if (reused.data.outcome != "correct"
                            or reused.data.reused_from != request.correctness_from):
                        raise CorrectnessEvidenceError()
                    return reused
                except CorrectnessEvidenceError as error:
                    return CheckFailed(status="failed", error=ErrorInfo(
                        code=error.code, phase=error.phase, check="correctness",
                        retryable=False, message=str(error),
                    ))
            return _engine_outcome(
                "correctness",
                self.verify.run(request, work_dir, device_id, deadline_at, lease_valid),
                lambda raw: shape_correctness(raw, decoy=_is_decoy(raw)),
            )

        def performance_evidence(raw):
            if request.implementation == "comparison":
                result = shape_performance(raw)
                expected_method = (
                    "npu_event_v1" if request.performance.method == "event"
                    else "profiler_kernel_sum_v1"
                )
                if result.measurement_method != expected_method:
                    raise ShapingError("comparison timing has a different measurement method")
                correctness = outcomes.get("correctness")
                selection = request.performance.case_indices
                if selection is None and correctness is not None:
                    selection = list(range(correctness.data.total_cases))
                observed = {case.case_idx for case in result.per_case + result.unmeasured_cases}
                if selection is not None and observed != set(selection):
                    raise ShapingError("comparison timing has a different case selection")
                if correctness is not None and any(
                    index >= correctness.data.total_cases for index in observed
                ):
                    raise ShapingError("comparison timing exceeds the reference inventory")
                return result
            try:
                result = SinglePerformanceResult.model_validate(raw)
            except ValueError as error:
                raise ShapingError("invalid standalone performance evidence") from error
            method = request.performance.method
            expected_method = (
                "npu_event_v1" if method == "event" else "profiler_kernel_sum_v1"
            )
            if (result.implementation != request.implementation
                    or result.measurement_method != expected_method):
                raise ShapingError(
                    "standalone performance evidence has a different subject or method"
                )
            selection = request.performance.case_indices
            observed = {case.case_idx for case in result.per_case + result.unmeasured_cases}
            if selection is not None and observed != set(selection):
                raise ShapingError("standalone performance evidence has a different case selection")
            return result

        def profile_evidence(raw):
            result = shape_profile_runs(raw, request.profile or ProfileParams())
            if result.mode == "trace":
                expected = (
                    None if request.implementation == "comparison" else request.implementation
                )
                if result.implementation != expected:
                    raise ShapingError("trace evidence has a different implementation")
                correctness = outcomes.get("correctness")
                if (correctness is not None and correctness.status == "succeeded"
                        and result.total_cases != correctness.data.total_cases):
                    raise ShapingError("trace inventory differs from correctness evidence")
            if request.implementation != "comparison":
                result = type(result).model_validate({
                    **result.model_dump(mode="json"),
                    "implementation": request.implementation,
                    "correctness_verified": False,
                })
            return result

        if "compile" in requested:
            outcomes["compile"] = self._slot(
                "compile",
                compile_check,
                lease_valid,
                progress,
            )
            # Missing precheck evidence is independent of on-card execution.
            # It does not establish a broken worker environment or skip checks.

        if "correctness" in requested:
            correctness = self._slot(
                "correctness",
                correctness_check,
                lease_valid,
                progress,
            )
            outcomes["correctness"] = correctness
            infra_broken = correctness.status in ("failed", "timeout")

        if "performance" in requested:
            correctness = outcomes.get("correctness")
            if (
                not infra_broken
                and (
                    request.implementation != "comparison"
                    or (
                        correctness is not None and correctness.status == "succeeded"
                        and correctness.data.outcome == "correct"
                    )
                )
            ):
                if "profile" in requested and request.profile.mode == "trace":
                    if not progress.begin("performance") or not lease_valid():
                        raise LeaseLostError("lease lost before joint measurement")

                    def publish_performance(engine):
                        nonlocal trace_started
                        if "performance" in outcomes:
                            return outcomes["performance"].status == "succeeded"
                        outcome = _engine_outcome("performance", engine, performance_evidence)
                        if not lease_valid() or not progress.save("performance", outcome):
                            raise LeaseLostError("joint timing checkpoint was fenced out")
                        outcomes["performance"] = outcome
                        if outcome.status == "succeeded":
                            if not progress.begin("profile"):
                                raise LeaseLostError("joint trace begin was fenced out")
                            trace_started = True
                        return outcome.status == "succeeded"

                    performance_engine, trace_engine = self.benchmark.run_with_trace(
                        request, work_dir, device_id, deadline_at, lease_valid,
                        on_performance=publish_performance,
                    )
                    publish_performance(performance_engine)
                    performance = outcomes["performance"]
                else:
                    performance = self._slot(
                        "performance",
                        lambda: _engine_outcome(
                            "performance",
                            self.benchmark.run(
                                request, work_dir, device_id, deadline_at, lease_valid,
                            ),
                            performance_evidence,
                        ),
                        lease_valid,
                        progress,
                    )
                infra_broken = performance.status in ("failed", "timeout")
            else:
                performance = self._slot(
                    "performance",
                    lambda: _skipped(
                        "performance",
                        "PRIOR_CHECK_FAILED" if infra_broken else "DEPENDENCY_NOT_PASSED",
                        "correctness",
                        _PRIOR_FAILED_MESSAGE if infra_broken else _DEPENDENCY_MESSAGE,
                    ),
                    lease_valid,
                    progress,
                )
            outcomes["performance"] = performance

        if "profile" in requested:
            if infra_broken:
                profile = self._slot(
                    "profile",
                    lambda: _skipped("profile", "PRIOR_CHECK_FAILED", None, _PRIOR_FAILED_MESSAGE),
                    lease_valid,
                    progress,
                )
            elif trace_engine is not None:
                if not trace_started and not progress.begin("profile"):
                    raise LeaseLostError("joint trace begin was fenced out")
                if not lease_valid():
                    raise LeaseLostError("lease lost after joint trace")
                profile = _engine_outcome("profile", trace_engine, profile_evidence)
                if not progress.save("profile", profile):
                    raise LeaseLostError("joint trace save was fenced out")
            else:
                profile = self._slot(
                    "profile",
                    lambda: _engine_outcome(
                        "profile",
                        self.profile.run(
                            request,
                            work_dir,
                            device_id,
                            deadline_at,
                            lease_valid,
                        ),
                        profile_evidence,
                    ),
                    lease_valid,
                    progress,
                )
            outcomes["profile"] = profile

        return complete_evaluation(
            requested_checks=requested,
            outcomes=outcomes,
            pinned=pinned,
            execution_environment=execution_environment,
        )
