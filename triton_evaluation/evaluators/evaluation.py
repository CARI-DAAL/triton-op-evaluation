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
from .shaping import shape_payload
from .verify import VerifyEvaluator


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

        verify_json = None
        decoy = False
        if "correctness" in checks:
            v = self._verify_evaluator().run(spec, verify_dir, device_id)
            terminal = (
                TaskStatus.TIMEOUT.value, TaskStatus.FAILED.value, TaskStatus.REJECTED.value
            )
            if v.status in terminal:
                return EvaluatorResult(v.status, error=v.error)
            # VerifyEvaluator.run returns the verify_result dict (incl. validate key),
            # or {"decoy": True, ...} when the AST pre-check flagged a regression.
            decoy = bool(v.result and v.result.get("decoy"))
            verify_json = None if decoy else v.result

        def _all_passed(d: dict) -> bool:
            return d.get("total_cases", 0) > 0 and d.get("passed_cases") == d.get("total_cases")

        correctness_passed = bool(verify_json and _all_passed(verify_json))

        perf_json = None
        if "performance" in checks and not decoy and correctness_passed:
            p = self._benchmark_evaluator().run(spec, verify_dir, device_id)
            # A flaky/rejected benchmark must not fail the evaluation (spec §7: omit).
            if p.status == TaskStatus.SUCCEEDED.value:
                perf_json = p.result

        profile_json = None
        if "profile" in checks and not decoy:
            pr = self._profile_evaluator().run(spec, verify_dir, device_id)
            if pr.status == TaskStatus.SUCCEEDED.value and pr.result is not None:
                profile_json = pr.result

        payload = shape_payload(verify_json, perf_json, profile_json,
                                checks=checks, decoy=decoy)
        return EvaluatorResult(TaskStatus.SUCCEEDED.value, result=payload)
