"""Performance check runner: engine benchmark in the correctness work dir.

benchmark.py's L1 gate reads the verify_result.json written by the
correctness check, so the orchestrator only invokes this after a succeeded
and correct subject outcome. A gate rejection (exit 2), a timeout, or missing
output is an infrastructure failure, never a subject verdict.
"""
import sys
from pathlib import Path

from ..api.contract import EvaluationRequest, PerformanceParams
from .base import (
    EngineCheckResult,
    LeaseLostError,
    engine_failure,
    engine_timeout,
    log_engine_crash,
    read_json,
    remaining_seconds,
    run_script,
    safe_module_stem,
)


class BenchmarkEvaluator:
    def __init__(self, engine_dir: str, timeout_seconds: int = 600):
        self.engine_dir = engine_dir
        self.timeout_seconds = timeout_seconds

    def run(
        self,
        request: EvaluationRequest,
        work_dir: Path,
        device_id: int,
        deadline_at,
        lease_valid=None,
    ) -> EngineCheckResult:
        remaining = remaining_seconds(deadline_at)
        if remaining <= 0:
            return engine_timeout(
                "performance", "resource deadline expired before performance ran"
            )
        params = request.performance or PerformanceParams()
        effective_method = params.method or (
            "event" if request.schema_version == "2.1" else "profiler"
        )
        out_path = Path(work_dir) / "perf_result.json"
        cmd = [
            sys.executable,
            str(Path(self.engine_dir) / "benchmark.py"),
            "--op_name",
            safe_module_stem(request),
            "--verify_dir",
            str(work_dir),
            "--triton_impl_name",
            request.triton_impl_name,
            "--warmup",
            str(params.warmup),
            "--repeats",
            str(params.repeats),
            "--measurement-method",
            effective_method,
            "--output",
            str(out_path),
        ]
        outcome = run_script(
            cmd,
            str(work_dir),
            device_id,
            deadline_at,
            lease_valid,
            budget_seconds=max(1, min(self.timeout_seconds, int(remaining))),
        )
        if outcome.outcome == "fenced":
            raise LeaseLostError("benchmark engine lost its lease")
        if outcome.timed_out or remaining_seconds(deadline_at) <= 0:
            return engine_timeout(
                "performance", "performance exceeded the resource deadline"
            )
        if outcome.exit_code == 2:
            return engine_failure(
                "performance", "benchmark verify gate rejected the run"
            )
        result = read_json(str(out_path))
        if outcome.exit_code == 0 and isinstance(result, dict):
            return EngineCheckResult(status="succeeded", raw=result, error=None)
        log_engine_crash(
            "benchmark", outcome, result_file_exists=isinstance(result, dict)
        )
        return engine_failure(
            "performance",
            f"benchmark engine produced no usable result (engine exit="
            f"{outcome.exit_code})",
        )
