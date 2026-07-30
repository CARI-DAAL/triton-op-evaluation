import json
import sys
from pathlib import Path

from ..models import JobSpec, TaskStatus
from .base import EvaluatorResult, materialize, read_json, run_script, tail
from .benchmark import BenchmarkEvaluator


class VerifyEvaluator:
    def __init__(self, engine_dir: str):
        self.engine_dir = engine_dir

    def _script_path(self) -> str:
        return str(Path(self.engine_dir) / "verify.py")

    def _benchmark_evaluator(self) -> BenchmarkEvaluator:
        return BenchmarkEvaluator(self.engine_dir)

    def run(self, spec: JobSpec, verify_dir: str, device_id: int) -> EvaluatorResult:
        materialize(spec, verify_dir)

        # Step 0: AST decoy pre-check (validate_triton_impl.py), matching the skill's
        # mandatory Step 0. Regression → SUCCEEDED with a decoy result; verify.py not run.
        impl_file = str(Path(verify_dir) / f"{spec.op_name}_{spec.triton_impl_name}.py")
        val = run_script(
            [
                sys.executable,
                str(Path(self.engine_dir) / "validate_triton_impl.py"),
                impl_file,
                "--json",
            ],
            verify_dir,
            device_id,
            timeout=60,
        )
        parsed = None
        try:
            parsed = json.loads(val.stdout) if val.stdout.strip() else None
        except ValueError:
            parsed = None
        if parsed is not None and parsed.get("valid") is False:
            return EvaluatorResult(TaskStatus.SUCCEEDED.value, result={"decoy": True, **parsed})

        # Clean (or unparseable — let verify.py surface a real error): run verify.
        out_path = str(Path(verify_dir) / "verify_result.json")
        cmd = [
            sys.executable, self._script_path(),
            "--op_name", spec.op_name, "--verify_dir", verify_dir,
            "--triton_impl_name", spec.triton_impl_name,
            "--timeout", str(spec.timeout), "--output", out_path,
        ]
        if spec.non_compute:
            cmd.append("--non-compute")
        outcome = run_script(cmd, verify_dir, device_id, timeout=spec.timeout + 60)
        if outcome.timed_out:
            return EvaluatorResult(TaskStatus.TIMEOUT.value, error="verify timed out")
        result = read_json(out_path)
        if result is None:
            return EvaluatorResult(TaskStatus.FAILED.value, error=tail(outcome.stderr))
        result["validate"] = {"valid": True}

        # Once verify passes, measure speed in the same job on the same bound card.
        # benchmark.py's L1 gate reads verify_result.json already written here — no seeding.
        # A benchmark that gets rejected or blows up must not fail the verify job: we
        # simply omit the benchmark sub-object and keep the passing verify verdict.
        if spec.measure_speed and result.get("total_cases", 0) > 0 and (
            result.get("passed_cases") == result.get("total_cases")
        ):
            bench = self._benchmark_evaluator().run(spec, verify_dir, device_id)
            if bench.status == TaskStatus.SUCCEEDED.value and bench.result is not None:
                result["benchmark"] = bench.result

        return EvaluatorResult(TaskStatus.SUCCEEDED.value, result=result)
