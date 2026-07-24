import os
import shutil
import sys
from pathlib import Path

from ..models import JobSpec, TaskStatus
from .base import EvaluatorResult, materialize, read_json, run_script, tail


class ProfileEvaluator:
    name = "profile"
    needs_bound_card = True

    def __init__(self, engine_dir: str):
        self.engine_dir = engine_dir

    def _script_path(self) -> str:
        return str(Path(self.engine_dir) / "profile.py")

    @staticmethod
    def _msprof_bin() -> str:
        return os.environ.get("MSPROF_BIN", "msprof")

    @staticmethod
    def _validate_pass_timeout() -> None:
        raw = os.environ.get("MSPROF_TIMEOUT_SEC", "300")
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError("MSPROF_TIMEOUT_SEC must be a positive integer") from exc
        if value <= 0:
            raise ValueError("MSPROF_TIMEOUT_SEC must be a positive integer")

    def run(self, spec: JobSpec, verify_dir: str, device_id: int) -> EvaluatorResult:
        msprof_bin = self._msprof_bin()
        if shutil.which(msprof_bin) is None:
            return EvaluatorResult(
                TaskStatus.FAILED.value,
                error=f"MSPROF_BIN is not executable: {msprof_bin!r}",
            )

        try:
            self._validate_pass_timeout()
        except ValueError as exc:
            return EvaluatorResult(TaskStatus.FAILED.value, error=str(exc))

        materialize(spec, verify_dir)
        output = str(Path(verify_dir) / "profile_result.json")
        command = [
            sys.executable,
            self._script_path(),
            "--op_name",
            spec.op_name,
            "--verify_dir",
            verify_dir,
            "--triton_impl_name",
            spec.triton_impl_name,
            "--profile-mode",
            spec.profile_mode,
            "--timeout",
            str(spec.timeout),
            "--output",
            output,
        ]
        if spec.case_idx is not None:
            command.extend(["--case-idx", str(spec.case_idx)])
        if spec.profile_launch_count is not None:
            command.extend(["--launch-count", str(spec.profile_launch_count)])

        outcome = run_script(
            command,
            verify_dir,
            device_id,
            timeout=spec.timeout + 60,
        )
        result = read_json(output)
        if outcome.timed_out or outcome.exit_code == 124:
            if result is not None:
                result["timed_out"] = True
            return EvaluatorResult(
                TaskStatus.TIMEOUT.value,
                result=result,
                error="profile timed out",
            )
        if outcome.exit_code == 0 and result is not None:
            return EvaluatorResult(TaskStatus.SUCCEEDED.value, result=result)
        return EvaluatorResult(
            TaskStatus.FAILED.value,
            result=result,
            error=tail(outcome.stderr) or "profile engine failed without stderr",
        )
