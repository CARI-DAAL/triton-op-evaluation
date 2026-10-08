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
from ..config import BENCHMARK_TIMEOUT_SECONDS, VERIFY_BUDGET_CAP_SECONDS
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
    def __init__(self, engine_dir: str, settings=None):
        self.engine_dir = engine_dir
        self.settings = settings

    def _verify_evaluator(self) -> VerifyEvaluator:
        cap = (
            self.settings.verify_budget_cap_seconds
            if self.settings is not None
            else VERIFY_BUDGET_CAP_SECONDS
        )
        policy = self.settings.reference_device_policy if self.settings is not None else "cpu-first"
        return VerifyEvaluator(
            self.engine_dir, budget_cap_seconds=cap, reference_device_policy=policy,
        )

    def _benchmark_evaluator(self) -> BenchmarkEvaluator:
        timeout = (
            self.settings.benchmark_timeout_seconds
            if self.settings is not None
            else BENCHMARK_TIMEOUT_SECONDS
        )
        return BenchmarkEvaluator(self.engine_dir, timeout_seconds=timeout)

    def _profile_evaluator(self) -> ProfileEvaluator:
        return ProfileEvaluator(self.engine_dir)

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
    ) -> EvaluationResult:
        """Run correctness, conditional performance, then independent profile."""
        requested = list(request.checks)
        outcomes: dict[str, object] = {}
        infra_broken = False

        if device_id is None and requested != ["compile"]:
            raise ValueError("CPU attempts can only execute the compile check")

        def compile_check():
            started = monotonic()
            verdict = precheck() if precheck is not None else None
            return compile_outcome_from_precheck(verdict, (monotonic() - started) * 1000)

        def performance_evidence(raw):
            if request.implementation == "comparison":
                return shape_performance(raw)
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
                lambda: _engine_outcome(
                    "correctness",
                    self._verify_evaluator().run(
                        request, work_dir, device_id, deadline_at, lease_valid
                    ),
                    lambda raw: shape_correctness(raw, decoy=_is_decoy(raw)),
                ),
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
                performance = self._slot(
                    "performance",
                    lambda: _engine_outcome(
                        "performance",
                        self._benchmark_evaluator().run(
                            request,
                            work_dir,
                            device_id,
                            deadline_at,
                            lease_valid,
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
            else:
                profile = self._slot(
                    "profile",
                    lambda: _engine_outcome(
                        "profile",
                        self._profile_evaluator().run(
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
