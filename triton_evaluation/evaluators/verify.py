import sys
from pathlib import Path

from ..models import JobSpec, TaskStatus
from .base import EvaluatorResult, materialize, run_script, read_json, tail


class VerifyEvaluator:
    name = "verify"
    needs_bound_card = True

    def __init__(self, engine_dir: str):
        self.engine_dir = engine_dir

    def _script_path(self) -> str:
        return str(Path(self.engine_dir) / "verify.py")

    def run(self, spec: JobSpec, verify_dir: str, device_id: int) -> EvaluatorResult:
        materialize(spec, verify_dir)
        out_path = str(Path(verify_dir) / "verify_result.json")
        cmd = [
            sys.executable, self._script_path(),
            "--op_name", spec.op_name,
            "--verify_dir", verify_dir,
            "--triton_impl_name", spec.triton_impl_name,
            "--timeout", str(spec.timeout),
            "--output", out_path,
        ]
        if spec.non_compute:
            cmd.append("--non-compute")

        outcome = run_script(cmd, verify_dir, device_id, timeout=spec.timeout + 60)
        if outcome.timed_out:
            return EvaluatorResult(TaskStatus.TIMEOUT.value, error="verify timed out")

        result = read_json(out_path)
        if result is None:
            # JSON absent → the evaluation itself did not complete
            return EvaluatorResult(TaskStatus.FAILED.value, error=tail(outcome.stderr))
        # JSON present → job ran; correctness verdict is inside result (passed vs total)
        return EvaluatorResult(TaskStatus.SUCCEEDED.value, result=result)
