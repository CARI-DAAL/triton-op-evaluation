"""Performance check runner: engine benchmark in the correctness work dir.

benchmark.py's L1 gate reads the verify_result.json written by the
correctness check, so the orchestrator only invokes this after a succeeded
and correct subject outcome. A gate rejection (exit 2), a timeout, or missing
output is an infrastructure failure, never a subject verdict.
Single-implementation timing uses its standalone runner and bypasses all
reference baseline production, caching, and comparison-result stamping.
"""

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from ..api.contract import EvaluationRequest, PerformanceParams
from .base import (
    EngineCheckResult,
    LeaseLostError,
    engine_failure,
    engine_timeout,
    input_generation_failure,
    log_engine_crash,
    materialize,
    read_json,
    remaining_seconds,
    run_script,
    safe_module_stem,
)
from .baseline import prepare_baseline


class _PerformancePhaseTimeout(TimeoutError):
    """Interrupt the shared runner while it is still producing timing."""


class _TraceStopped(RuntimeError):
    """Timing was published, but its shaped outcome does not permit trace."""


class BenchmarkEvaluator:
    def __init__(
        self, engine_dir: str, timeout_seconds: int = 600, baseline_cache=None, *, runner=None
    ):
        self.runner = runner
        self.engine_dir = engine_dir
        self.timeout_seconds = timeout_seconds
        self.baseline_cache = baseline_cache

    def _command(self, request, work_dir, params):
        single = request.implementation != "comparison"
        command = [
            sys.executable,
            str(Path(self.engine_dir) / ("single_benchmark.py" if single else "benchmark.py")),
            "--op_name", safe_module_stem(request),
            "--verify_dir", str(work_dir),
            "--triton_impl_name", request.triton_impl_name,
            "--warmup", str(params.warmup),
            "--repeats", str(params.repeats),
            "--measurement-method", params.method,
        ]
        if single:
            command += ["--implementation", request.implementation]
        if params.case_indices is not None:
            command += ["--case-indices", ",".join(str(i + 1) for i in params.case_indices)]
        if not single:
            command += ["--disable-reference-cache"]
        return command

    @staticmethod
    def _correctness_options(request, work_dir):
        if not getattr(request, "correctness_from", None):
            return []
        options = ["--correctness-evidence", str(Path(work_dir) / "correctness_evidence.json")]
        if request.non_compute:
            options.append("--correctness-non-compute")
        return options

    @staticmethod
    def _gate_failure(request):
        if getattr(request, "correctness_from", None):
            return engine_failure(
                "performance", "correctness evidence handoff is invalid",
                code="CORRECTNESS_EVIDENCE_INVALID", phase="request", retryable=False,
            )
        return engine_failure("performance", "benchmark verify gate rejected the run")

    def run(
        self,
        request: EvaluationRequest,
        work_dir: Path,
        device_id: int,
        deadline_at,
        lease_valid=None,
    ) -> EngineCheckResult:
        deadline_at = min(
            deadline_at, datetime.now(timezone.utc) + timedelta(seconds=self.timeout_seconds)
        )
        if lease_valid is not None and not lease_valid():
            raise LeaseLostError("benchmark attempt lost its lease")
        remaining = remaining_seconds(deadline_at)
        if remaining <= 0:
            return engine_timeout("performance", "resource deadline expired before performance ran")
        single = request.implementation != "comparison"
        if single or request.correctness_from is not None:
            materialize(request, work_dir)
        params = request.performance or PerformanceParams()
        if params.case_indices is not None and not params.case_indices:
            return engine_failure("performance", "performance case selection is empty")
        out_path = Path(work_dir) / "perf_result.json"
        cmd = self._command(request, work_dir, params)
        try:
            baseline = prepare_baseline(
                request, cache=self.baseline_cache, engine_dir=self.engine_dir,
                runner=self.runner or run_script, device_id=device_id,
                deadline_at=deadline_at, lease_valid=lease_valid,
            )
        except TimeoutError:
            return engine_timeout("performance", "baseline measurement exceeded the deadline")
        cmd += baseline.command_options()
        cmd += [*self._correctness_options(request, work_dir), "--output", str(out_path)]
        outcome = (self.runner or run_script)(
            cmd,
            str(work_dir),
            device_id,
            deadline_at,
            lease_valid,
            budget_seconds=max(
                1, min(self.timeout_seconds, math.ceil(remaining_seconds(deadline_at)))
            ),
        )
        if outcome.outcome == "fenced" or (lease_valid is not None and not lease_valid()):
            raise LeaseLostError("benchmark engine lost its lease")
        if outcome.timed_out or remaining_seconds(deadline_at) <= 0:
            return engine_timeout("performance", "performance exceeded the resource deadline")
        if outcome.exit_code == 2 and not single:
            return self._gate_failure(request)
        result = read_json(out_path, root=work_dir)
        if outcome.exit_code == 0 and isinstance(result, dict):
            if single:
                return EngineCheckResult(status="succeeded", raw=result, error=None)
            try:
                baseline.bind(result)
            except (ValueError, TypeError, KeyError) as error:
                return engine_failure(
                    "performance", f"invalid benchmark baseline evidence: {error}"
                )
            return EngineCheckResult(status="succeeded", raw=result, error=None)
        refused = input_generation_failure("performance", outcome)
        if refused is not None:
            return refused
        log_engine_crash("benchmark", outcome, result_file_exists=isinstance(result, dict))
        return engine_failure(
            "performance",
            "benchmark engine produced no usable result"
            f" (engine exit={outcome.exit_code})",
            outcome=outcome,
        )

    def run_with_trace(
        self,
        request: EvaluationRequest,
        work_dir: Path,
        device_id: int,
        deadline_at,
        lease_valid=None,
        *,
        on_performance=None,
    ) -> tuple[EngineCheckResult, EngineCheckResult]:
        """One candidate process; preserve and publish timing before trace exits.

        The reference-only cache producer remains a separate, isolated process.
        The callback executes through the runner's fenced polling boundary and
        receives the same result returned to the orchestrator exactly once.
        A literal False callback result stops later capture without fencing.
        """
        if request.profile is None or request.profile.mode != "trace":
            raise ValueError("run_with_trace requires an explicit trace profile")
        performance_deadline = min(
            deadline_at, datetime.now(timezone.utc) + timedelta(seconds=self.timeout_seconds)
        )
        if lease_valid is not None and not lease_valid():
            raise LeaseLostError("joint measurement attempt lost its lease")
        if remaining_seconds(deadline_at) <= 0:
            return (
                engine_timeout("performance", "resource deadline expired before performance ran"),
                engine_timeout("profile", "resource deadline expired before profile ran"),
            )
        params = request.performance or PerformanceParams()
        if params.case_indices == [] or request.profile.case_indices == []:
            return (
                engine_failure("performance", "joint measurement case selection is empty"),
                engine_failure("profile", "joint measurement case selection is empty"),
            )
        single = request.implementation != "comparison"
        if single or request.correctness_from is not None:
            materialize(request, work_dir)
        cmd = self._command(request, work_dir, params)

        def timing_timeout(message):
            return (
                engine_timeout("performance", message),
                engine_failure(
                    "profile", "trace did not run because the shared timing phase timed out",
                    code="SHARED_TIMING_TIMEOUT", retryable=False,
                ),
            )

        try:
            baseline = prepare_baseline(
                request, cache=self.baseline_cache, engine_dir=self.engine_dir,
                runner=self.runner or run_script, device_id=device_id,
                deadline_at=performance_deadline, lease_valid=lease_valid,
            )
        except TimeoutError:
            return timing_timeout("baseline measurement exceeded the deadline")
        cmd += baseline.command_options()
        # Only the candidate command becomes joint. The producer above must
        # never import a candidate or execute a requested trace.
        cmd[1] = str(Path(self.engine_dir) / "joint_measurement.py")
        suffix = uuid4().hex
        perf_path = Path(work_dir) / f"perf_trace_{suffix}.json"
        trace_path = Path(work_dir) / f"trace_result_{suffix}.json"
        cmd += [
            *self._correctness_options(request, work_dir),
            "--output", str(perf_path), *self._trace_options(request, trace_path),
        ]
        completed = None

        def checkpoint():
            nonlocal completed
            if lease_valid is not None and not lease_valid():
                return False
            if completed is None:
                raw = read_json(perf_path, root=work_dir)
                if isinstance(raw, dict):
                    try:
                        baseline.bind(raw)
                        completed = EngineCheckResult(status="succeeded", raw=raw, error=None)
                    except (ValueError, TypeError, KeyError) as error:
                        completed = engine_failure(
                            "performance", f"invalid benchmark baseline evidence: {error}"
                        )
                    if on_performance is not None:
                        if on_performance(completed) is False:
                            raise _TraceStopped("performance checkpoint stopped trace continuation")
                elif remaining_seconds(performance_deadline) <= 0:
                    raise _PerformancePhaseTimeout("performance exceeded its phase budget")
            return True

        try:
            outcome = (self.runner or run_script)(
                cmd, str(work_dir), device_id, deadline_at, checkpoint,
                budget_seconds=max(1, math.ceil(remaining_seconds(deadline_at))),
            )
            if outcome.outcome == "fenced" or not checkpoint():
                raise LeaseLostError("joint measurement engine lost its lease")
        except _PerformancePhaseTimeout:
            return timing_timeout("performance exceeded its phase budget")
        except _TraceStopped:
            return completed, engine_failure(
                "profile", "trace stopped after unsuccessful performance evidence",
                code="PRIOR_CHECK_FAILED", retryable=False,
            )
        expired = outcome.timed_out or remaining_seconds(deadline_at) <= 0
        if completed is None:
            if expired:
                completed = engine_timeout("performance", "performance exceeded the deadline")
            elif outcome.exit_code == 2 and not single:
                completed = self._gate_failure(request)
            else:
                completed = input_generation_failure("performance", outcome) or engine_failure(
                    "performance", "joint engine produced no usable performance result"
                )
        if expired:
            trace = engine_timeout("profile", "profile exceeded the resource deadline")
        else:
            raw = read_json(trace_path, root=work_dir)
            if isinstance(raw, dict):
                trace = EngineCheckResult(status="succeeded", raw=raw, error=None)
            else:
                trace = input_generation_failure("profile", outcome) or engine_failure(
                    "profile", "joint engine produced no usable trace result"
                )
        return completed, trace

    @staticmethod
    def _trace_options(request, trace_path):
        params = request.profile
        options = [
            "--trace-output", str(trace_path),
            "--max-trace-kernels", str(params.max_kernels_per_case),
        ]
        if params.case_indices is not None:
            options += [
                "--trace-case-indices", ",".join(str(i + 1) for i in params.case_indices)
            ]
        return options

    def run_trace(
        self,
        request: EvaluationRequest,
        work_dir: Path,
        device_id: int,
        deadline_at,
        lease_valid=None,
    ) -> EngineCheckResult:
        """Independent trace evidence, without timing or a correctness gate."""
        if request.profile is None or request.profile.mode != "trace":
            raise ValueError("run_trace requires an explicit trace profile")
        if lease_valid is not None and not lease_valid():
            raise LeaseLostError("trace attempt lost its lease")
        if remaining_seconds(deadline_at) <= 0:
            return engine_timeout("profile", "resource deadline expired before profile ran")
        if request.profile.case_indices == []:
            return engine_failure("profile", "profile case selection is empty")
        materialize(request, work_dir)
        path = Path(work_dir) / f"trace_result_{uuid4().hex}.json"
        # Defaults are a fixed standalone trace protocol. Joint requests reuse
        # their explicit performance warmup/repeats for the shared capture.
        params = PerformanceParams(method="profiler", warmup=5, repeats=20)
        cmd = self._command(request, work_dir, params)
        cmd[1] = str(Path(self.engine_dir) / "joint_measurement.py")
        cmd += ["--trace-only", *self._trace_options(request, path)]
        outcome = (self.runner or run_script)(
            cmd, str(work_dir), device_id, deadline_at, lease_valid,
            budget_seconds=max(1, math.ceil(remaining_seconds(deadline_at))),
        )
        if outcome.outcome == "fenced" or (lease_valid is not None and not lease_valid()):
            raise LeaseLostError("trace engine lost its lease")
        if outcome.timed_out or remaining_seconds(deadline_at) <= 0:
            return engine_timeout("profile", "profile exceeded the resource deadline")
        raw = read_json(path, root=work_dir)
        if outcome.exit_code == 0 and isinstance(raw, dict):
            return EngineCheckResult(status="succeeded", raw=raw, error=None)
        return input_generation_failure("profile", outcome) or engine_failure(
            "profile", "trace engine produced no usable result"
        )
