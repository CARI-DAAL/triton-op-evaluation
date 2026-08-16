"""Performance check runner: engine benchmark in the correctness work dir.

benchmark.py's L1 gate reads the verify_result.json written by the
correctness check, so the orchestrator only invokes this after a succeeded
and correct subject outcome. A gate rejection (exit 2), a timeout, or missing
output is an infrastructure failure, never a subject verdict.
"""
import sys
from pathlib import Path

from ..api.contract import EvaluationRequest, PerformanceParams
from ..config import BENCHMARK_TIMEOUT_SECONDS
from .base import (
    EngineCheckResult,
    engine_failure,
    engine_timeout,
    read_json,
    remaining_seconds,
    run_script,
)


class BenchmarkEvaluator:
    def __init__(self, engine_dir: str):
        self.engine_dir = engine_dir

    def run(
        self,
        request: EvaluationRequest,
        work_dir: Path,
        device_id: int,
        deadline_at,
    ) -> EngineCheckResult:
        remaining = remaining_seconds(deadline_at)
        if remaining <= 0:
            return engine_timeout(
                "performance", "resource deadline expired before performance ran"
            )
        params = request.performance or PerformanceParams()
        out_path = Path(work_dir) / "perf_result.json"
        cmd = [
            sys.executable,
            str(Path(self.engine_dir) / "benchmark.py"),
            "--op_name",
            request.op_name,
            "--verify_dir",
            str(work_dir),
            "--triton_impl_name",
            request.triton_impl_name,
            "--warmup",
            str(params.warmup),
            "--repeats",
            str(params.repeats),
            "--output",
            str(out_path),
        ]
        outcome = run_script(
            cmd,
            str(work_dir),
            device_id,
            timeout=max(1, min(BENCHMARK_TIMEOUT_SECONDS, int(remaining))),
        )
        if outcome.timed_out:
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
        return engine_failure(
            "performance", "benchmark engine produced no usable result"
        )
