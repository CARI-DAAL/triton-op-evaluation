import json
import sys
from pathlib import Path

from ..models import JobSpec, TaskStatus
from .base import EvaluatorResult, materialize, run_script, read_json, tail


def verify_json_name(triton_impl_name: str) -> str:
    """Mirror the skill's resolve_verify_json_name (we do not import the skill)."""
    if triton_impl_name == "triton_ascend_impl":
        return "verify_result.json"
    suffix = triton_impl_name[len("triton_"):] if triton_impl_name.startswith("triton_") else triton_impl_name
    return f"verify_result_{suffix}.json"


class BenchmarkEvaluator:
    name = "benchmark"
    needs_bound_card = True

    def __init__(self, scripts_dir: str):
        self.scripts_dir = scripts_dir

    def _script_path(self) -> str:
        return str(Path(self.scripts_dir) / "benchmark.py")

    def run(self, spec: JobSpec, verify_dir: str, device_id: int) -> EvaluatorResult:
        materialize(spec, verify_dir)

        # Seed the verify JSON so benchmark.py's L1 gate can pass.
        if spec.verify_result_seed is not None:
            seed_path = Path(verify_dir) / verify_json_name(spec.triton_impl_name)
            seed_path.write_text(json.dumps(spec.verify_result_seed))

        out_path = str(Path(verify_dir) / "perf_result.json")
        cmd = [
            sys.executable, self._script_path(),
            "--op_name", spec.op_name,
            "--verify_dir", verify_dir,
            "--triton_impl_name", spec.triton_impl_name,
            "--warmup", str(spec.warmup),
            "--repeats", str(spec.repeats),
            "--output", out_path,
        ]
        if spec.verify_not_required:
            cmd.append("--verify_not_required")

        outcome = run_script(cmd, verify_dir, device_id, timeout=spec.timeout)
        if outcome.timed_out:
            return EvaluatorResult(TaskStatus.TIMEOUT.value, error="benchmark timed out")
        if outcome.exit_code == 2:
            return EvaluatorResult(TaskStatus.REJECTED.value, error=tail(outcome.stderr))

        result = read_json(out_path)
        if outcome.exit_code == 0 and result is not None:
            return EvaluatorResult(TaskStatus.SUCCEEDED.value, result=result)
        return EvaluatorResult(TaskStatus.FAILED.value, error=tail(outcome.stderr))
