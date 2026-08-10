"""Orchestrator for v2 ``/evaluations`` jobs.

Sequences the requested ``checks`` using the existing v1 sub-evaluators and
shapes the aggregate into one ``EvaluationPayload`` via the shaping layer.
Combination rule (spec §7):
  * correctness runs independently (VerifyEvaluator already does the decoy
    pre-check and returns result={"decoy": True, ...} — we reuse that verdict);
  * performance runs only when correctness was requested AND passed;
  * profile runs independently (a wrong kernel can still be profiled).

Sub-evaluators are obtained via overridable factories so tests inject stubs
without touching the real (torch-dependent) engine.
"""
from ..models import JobSpec, TaskStatus
from .base import EvaluatorResult, materialize
from .benchmark import BenchmarkEvaluator
from .profile import ProfileEvaluator
from .shaping import ShapingError, shape_correctness, shape_performance, shape_profile
from .verify import VerifyEvaluator

_STATUS_PRIORITY = {
    TaskStatus.SUCCEEDED.value: 0,
    TaskStatus.REJECTED.value: 1,
    TaskStatus.FAILED.value: 2,
    TaskStatus.TIMEOUT.value: 3,
}


def _worst_status(statuses: list[str]) -> str:
    return max(
        statuses or [TaskStatus.SUCCEEDED.value],
        key=_STATUS_PRIORITY.__getitem__,
    )


def _run_subcheck(name, evaluator, spec, verify_dir, device_id, statuses, errors):
    try:
        result = evaluator.run(spec, verify_dir, device_id)
    except Exception as exc:
        statuses.append(TaskStatus.FAILED.value)
        errors.append(f"{name} failed: {type(exc).__name__}: {exc}")
        return None
    if not isinstance(result, EvaluatorResult) or result.status not in _STATUS_PRIORITY:
        statuses.append(TaskStatus.FAILED.value)
        errors.append(f"{name} returned an invalid evaluator result")
        return None
    statuses.append(result.status)
    if result.error:
        errors.append(result.error)
    return result


class EvaluationEvaluator:
    def __init__(self, engine_dir: str):
        self.engine_dir = engine_dir

    def _verify_evaluator(self) -> VerifyEvaluator:
        return VerifyEvaluator(self.engine_dir)

    def _benchmark_evaluator(self) -> BenchmarkEvaluator:
        return BenchmarkEvaluator(self.engine_dir)

    def _profile_evaluator(self) -> ProfileEvaluator:
        return ProfileEvaluator(self.engine_dir)

    def run(self, spec: JobSpec, verify_dir: str, device_id: int) -> EvaluatorResult:
        checks = spec.checks or ["correctness"]
        materialize(spec, verify_dir)
        statuses: list[str] = []
        errors: list[str] = []

        verify_json = None
        decoy = False
        if "correctness" in checks:
            verify_result = _run_subcheck(
                "correctness",
                self._verify_evaluator(),
                spec,
                verify_dir,
                device_id,
                statuses,
                errors,
            )
            if verify_result is not None and verify_result.status == TaskStatus.SUCCEEDED.value:
                if verify_result.result is None:
                    statuses.append(TaskStatus.FAILED.value)
                    errors.append("correctness result is missing")
                elif not isinstance(verify_result.result, dict):
                    statuses.append(TaskStatus.FAILED.value)
                    errors.append("correctness result is invalid")
                else:
                    decoy = bool(verify_result.result.get("decoy"))
                    verify_json = None if decoy else verify_result.result

        correctness = None
        if "correctness" in checks:
            try:
                correctness = shape_correctness(verify_json, decoy=decoy)
            except Exception as exc:
                statuses.append(TaskStatus.FAILED.value)
                errors.append(f"correctness result is invalid: {exc}")
        correctness_passed = bool(correctness and correctness.get("passed"))

        performance = None
        if "performance" in checks and not decoy and correctness_passed:
            benchmark_result = _run_subcheck(
                "performance",
                self._benchmark_evaluator(),
                spec,
                verify_dir,
                device_id,
                statuses,
                errors,
            )
            if (
                benchmark_result is not None
                and benchmark_result.status == TaskStatus.SUCCEEDED.value
            ):
                try:
                    if benchmark_result.result is None:
                        raise ShapingError("benchmark result is missing")
                    performance = shape_performance(benchmark_result.result)
                except Exception as exc:
                    statuses.append(TaskStatus.FAILED.value)
                    errors.append(str(exc))

        profile = None
        if "profile" in checks:
            profile_result = _run_subcheck(
                "profile",
                self._profile_evaluator(),
                spec,
                verify_dir,
                device_id,
                statuses,
                errors,
            )
            if (
                profile_result is not None
                and profile_result.status == TaskStatus.SUCCEEDED.value
                and profile_result.result is None
            ):
                statuses.append(TaskStatus.FAILED.value)
                errors.append("profile result is missing")
            elif profile_result is not None and profile_result.result is not None:
                try:
                    profile = shape_profile(profile_result.result)
                except Exception as exc:
                    statuses.append(TaskStatus.FAILED.value)
                    errors.append(str(exc))

        payload = {
            "correctness": correctness,
            "performance": performance,
            "profile": profile,
        }
        return EvaluatorResult(
            _worst_status(statuses),
            result=payload,
            error="; ".join(errors) or None,
        )
