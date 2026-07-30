import sys
from pathlib import Path

from ..config import BENCHMARK_TIMEOUT_SECONDS
from ..models import JobSpec, TaskStatus
from .base import EvaluatorResult, materialize, read_json, run_script, tail


class BenchmarkEvaluator:
    """Runs benchmark.py in a verify_dir. Used internally by VerifyEvaluator once
    verify has passed and written verify_result.json — benchmark.py's L1 gate reads
    that file in-place, so no seeding is needed."""

    def __init__(self, engine_dir: str):
        self.engine_dir = engine_dir

    def _script_path(self) -> str:
        return str(Path(self.engine_dir) / "benchmark.py")

    def run(self, spec: JobSpec, verify_dir: str, device_id: int) -> EvaluatorResult:
        materialize(spec, verify_dir)

        out_path = str(Path(verify_dir) / "perf_result.json")
        cmd = [
            sys.executable,
            self._script_path(),
            "--op_name",
            spec.op_name,
            "--verify_dir",
            verify_dir,
            "--triton_impl_name",
            spec.triton_impl_name,
            "--warmup",
            str(spec.warmup),
            "--repeats",
            str(spec.repeats),
            "--output",
            out_path,
        ]

        outcome = run_script(
            cmd,
            verify_dir,
            device_id,
            timeout=BENCHMARK_TIMEOUT_SECONDS,
        )
        if outcome.timed_out:
            return EvaluatorResult(TaskStatus.TIMEOUT.value, error="benchmark timed out")
        if outcome.exit_code == 2:
            return EvaluatorResult(TaskStatus.REJECTED.value, error=tail(outcome.stderr))

        result = read_json(out_path)
        if outcome.exit_code == 0 and result is not None:
            return EvaluatorResult(TaskStatus.SUCCEEDED.value, result=result)
        return EvaluatorResult(TaskStatus.FAILED.value, error=tail(outcome.stderr))
