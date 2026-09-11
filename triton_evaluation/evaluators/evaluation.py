"""Fixed-order v2 check orchestration with fenced progress and provenance.

Order and dependency rules (design section 5):

* correctness always runs first when requested;
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

from ..api.contract import (
    Check,
    CheckFailed,
    CheckSkipped,
    CheckSucceeded,
    CheckTimedOut,
    CompileOutcome,
    CorrectnessResult,
    ErrorInfo,
    EvaluationRequest,
    EvaluationResult,
    ExecutionEnvironment,
    PerformanceResult,
    ProfileResult,
    Provenance,
    SkipInfo,
)
from ..config import BENCHMARK_TIMEOUT_SECONDS, VERIFY_BUDGET_CAP_SECONDS
from .base import (
    EngineCheckResult,
    LeaseLostError,
    PinnedProvenance,
    ProgressSink,
)
from .benchmark import BenchmarkEvaluator
from .profile import ProfileEvaluator
from .shaping import (
    ShapingError,
    shape_correctness,
    shape_performance,
    shape_profile,
)
from .verify import VerifyEvaluator

_DEPENDENCY_MESSAGE = "correctness did not pass"
_PRIOR_FAILED_MESSAGE = "a prior requested check did not complete"

_CHECK_DATA_MODELS = {
    "correctness": CorrectnessResult,
    "performance": PerformanceResult,
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


def _compile_outcome(correctness) -> CheckSucceeded[CompileOutcome]:
    """Assemble the v2.1 compile slot from the on-card subject verdict (spec §6).

    The compile check itself executes at the admission precheck — it never
    queues and never touches a card — so on-card it is derived: a subject
    compile_error attribution maps to outcome ``compile_error`` at the
    ``backend`` stage with the domain-failure excerpt; every other reachable
    subject verdict means the kernel compiled.
    """
    if (
        isinstance(correctness, CheckSucceeded)
        and getattr(correctness.data, "outcome", None) == "compile_error"
    ):
        failure = correctness.data.domain_failure
        return CheckSucceeded[CompileOutcome](
            status="succeeded",
            data=CompileOutcome(
                outcome="compile_error",
                compiler_stage="backend",
                error_excerpt=failure.message if failure is not None else None,
            ),
        )
    return CheckSucceeded[CompileOutcome](
        status="succeeded", data=CompileOutcome(outcome="compiled")
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
        return VerifyEvaluator(self.engine_dir, budget_cap_seconds=cap)

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
        device_id: int,
        deadline_at: datetime,
        lease_valid: Callable[[], bool],
        execution_environment: ExecutionEnvironment,
        pinned: PinnedProvenance,
        progress: ProgressSink,
    ) -> EvaluationResult:
        """Run correctness, conditional performance, then independent profile."""
        requested = list(request.checks)
        outcomes: dict[str, object] = {}
        infra_broken = False

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
            correctness = outcomes["correctness"]
            if (
                not infra_broken
                and correctness.status == "succeeded"
                and correctness.data.outcome == "correct"
            ):
                performance = self._slot(
                    "performance",
                    lambda: _engine_outcome(
                        "performance",
                        self._benchmark_evaluator().run(
                            request, work_dir, device_id, deadline_at,
                            lease_valid,
                        ),
                        shape_performance,
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
                    lambda: _skipped(
                        "profile", "PRIOR_CHECK_FAILED", None, _PRIOR_FAILED_MESSAGE
                    ),
                    lease_valid,
                    progress,
                )
            else:
                profile = self._slot(
                    "profile",
                    lambda: _engine_outcome(
                        "profile",
                        self._profile_evaluator().run(
                            request, work_dir, device_id, deadline_at,
                            lease_valid,
                        ),
                        shape_profile,
                    ),
                    lease_valid,
                    progress,
                )
            outcomes["profile"] = profile

        if "compile" in requested:
            outcomes["compile"] = _compile_outcome(outcomes.get("correctness"))

        return EvaluationResult.model_validate(
            {
                "compile": outcomes.get("compile"),
                "correctness": outcomes.get("correctness"),
                "performance": outcomes.get("performance"),
                "profile": outcomes.get("profile"),
                "provenance": _provenance(
                    pinned, execution_environment, outcomes, requested
                ),
            },
            context={"requested_checks": requested},
        )


def _provenance(
    pinned: PinnedProvenance,
    environment: ExecutionEnvironment,
    outcomes: dict[str, object],
    requested: list[str],
) -> Provenance:
    # ``completed`` means orchestration obtained an outcome for every requested
    # check without an infrastructure failure; subject failures still count.
    completed = all(
        outcomes[check].status not in ("failed", "timeout") for check in requested
    )
    grade_eligible = (
        pinned.purpose == "grade"
        and pinned.manifest_id is not None
        and pinned.problem_id is not None
        and completed
        and environment is not None
    )
    return Provenance(
        schema_version="2.0",
        evaluator_build=pinned.evaluator_build,
        correctness_algorithm=pinned.algorithms.correctness,
        decoy_algorithm=pinned.algorithms.decoy,
        benchmark_algorithm=pinned.algorithms.benchmark,
        profiler_algorithm=pinned.algorithms.profiler,
        grade_eligible=grade_eligible,
        manifest_id=pinned.manifest_id,
        problem_id=pinned.problem_id,
        execution_environment=environment,
    )
