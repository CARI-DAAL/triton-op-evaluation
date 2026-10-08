"""Profile check runner: engine profile_runner for op/deep case profiles.

Profile is independent of the correctness verdict. The engine profiles one
explicit ``--case-idx`` per run, so a multi-index selection is executed as
sequential engine runs in case order. Raw envelopes are collected unchanged;
the shaping boundary validates their identities before combining evidence.
Any timeout or failure aborts the whole check as an infrastructure outcome.
"""
import sys
from pathlib import Path

from ..api.contract import EvaluationRequest, ProfileParams
from .base import (
    EngineCheckResult,
    LeaseLostError,
    engine_failure,
    engine_timeout,
    materialize,
    read_json,
    remaining_seconds,
    run_script,
    safe_module_stem,
)

# The engine requires --timeout within [10, 86400].
_MIN_ENGINE_TIMEOUT_SECONDS = 10
_MAX_ENGINE_TIMEOUT_SECONDS = 86400


class ProfileEvaluator:
    def __init__(self, engine_dir: str, *, runner=None):
        self.runner = runner
        self.engine_dir = engine_dir

    def run(
        self,
        request: EvaluationRequest,
        work_dir: Path,
        device_id: int,
        deadline_at,
        lease_valid=None,
    ) -> EngineCheckResult:
        materialize(request, work_dir)
        params = request.profile or ProfileParams()
        indices = params.case_indices
        if indices is not None and not indices:
            return engine_failure("profile", "profile case selection is empty")
        if indices is None:
            result = self._run_once(
                request, params, work_dir, device_id, deadline_at, None, lease_valid
            )
            if result.status != "succeeded":
                return result
            raw = result.raw
        else:
            envelopes = []
            for index in indices:
                result = self._run_once(
                    request, params, work_dir, device_id, deadline_at, index,
                    lease_valid,
                )
                if result.status != "succeeded":
                    return result
                envelopes.append(result.raw)
            raw = {"profile_runs": envelopes}
        return EngineCheckResult(status="succeeded", raw=raw, error=None)

    def _run_once(
        self,
        request: EvaluationRequest,
        params: ProfileParams,
        work_dir: Path,
        device_id: int,
        deadline_at,
        case_idx: int | None,
        lease_valid=None,
    ) -> EngineCheckResult:
        remaining = remaining_seconds(deadline_at)
        if remaining <= 0:
            return engine_timeout(
                "profile", "resource deadline expired before profile ran"
            )
        budget = max(
            _MIN_ENGINE_TIMEOUT_SECONDS,
            min(int(remaining), _MAX_ENGINE_TIMEOUT_SECONDS),
        )
        suffix = "profile_result.json" if case_idx is None else (
            f"profile_result_{case_idx}.json"
        )
        out_path = Path(work_dir) / suffix
        cmd = [
            sys.executable,
            str(Path(self.engine_dir) / "profile_runner.py"),
            "--op_name",
            safe_module_stem(request),
            "--verify_dir",
            str(work_dir),
            "--triton_impl_name",
            request.triton_impl_name,
            "--implementation",
            request.implementation,
            "--profile-mode",
            params.mode,
            "--timeout",
            str(budget),
            "--output",
            str(out_path),
        ]
        if case_idx is not None:
            cmd.extend(["--case-idx", str(case_idx)])
        outcome = (self.runner or run_script)(
            cmd, str(work_dir), device_id, deadline_at, lease_valid,
            budget_seconds=budget + 60,
        )
        if outcome.outcome == "fenced":
            raise LeaseLostError("profile engine lost its lease")
        if outcome.timed_out or outcome.exit_code == 124:
            return engine_timeout("profile", "profile exceeded the resource deadline")
        if remaining_seconds(deadline_at) <= 0:
            return engine_timeout("profile", "profile exceeded the resource deadline")
        result = read_json(out_path, root=work_dir)
        if outcome.exit_code == 0 and isinstance(result, dict):
            return EngineCheckResult(status="succeeded", raw=result, error=None)
        return engine_failure("profile", "profile engine produced no usable result")
