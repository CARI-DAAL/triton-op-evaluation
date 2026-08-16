"""Correctness check runner: decoy pre-check, engine verify, decoy case count.

The engine subprocess decides subject outcomes; this module only decides
whether the engine produced trustworthy private JSON. Outer subprocess
timeouts, missing output, and ambiguous exits are infrastructure failures and
never receive candidate attribution (design section 5).
"""
import json
import re
import sys
from pathlib import Path

from ..api.contract import EvaluationRequest
from .base import (
    EngineCheckResult,
    engine_failure,
    engine_timeout,
    materialize,
    read_json,
    remaining_seconds,
    run_script,
)

# The decoy pre-check and reference case counting are short fixed budgets; the
# verify run itself is bounded by the resource deadline. The per-subject
# execution budget forwarded to the engine is capped so a hanging candidate
# forward is attributed long before the outer kill.
_DECOY_CHECK_TIMEOUT_SECONDS = 60
_PER_SUBJECT_BUDGET_CAP_SECONDS = 900
_OUTER_GRACE_SECONDS = 30

_COUNT_SNIPPET = (
    "import importlib.util, json, sys\n"
    "spec = importlib.util.spec_from_file_location('reference_module', sys.argv[1])\n"
    "module = importlib.util.module_from_spec(spec)\n"
    "spec.loader.exec_module(module)\n"
    "if hasattr(module, 'get_input_groups'):\n"
    "    total = len(module.get_input_groups())\n"
    "elif hasattr(module, 'get_inputs'):\n"
    "    total = 1\n"
    "else:\n"
    "    raise SystemExit('reference module provides no input provider')\n"
    "print(json.dumps({'total_cases': total}))\n"
)


def _extract_validate_json(text: str) -> dict | None:
    """Find the validator's JSON result inside formatted engine log output."""
    decoder = json.JSONDecoder()
    found = None
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except ValueError:
            continue
        if isinstance(value, dict) and "valid" in value:
            found = value
    return found


class VerifyEvaluator:
    def __init__(self, engine_dir: str):
        self.engine_dir = engine_dir

    def run(
        self,
        request: EvaluationRequest,
        work_dir: Path,
        device_id: int,
        deadline_at,
    ) -> EngineCheckResult:
        materialize(request, work_dir)
        if remaining_seconds(deadline_at) <= 0:
            return engine_timeout(
                "correctness", "resource deadline expired before correctness ran"
            )

        verdict = self._decoy_verdict(request, work_dir, device_id, deadline_at)
        if isinstance(verdict, EngineCheckResult):
            return verdict
        if verdict:
            return self._decoy_result(request, work_dir, device_id, deadline_at)
        return self._run_verify(request, work_dir, device_id, deadline_at)

    def _decoy_verdict(
        self, request: EvaluationRequest, work_dir: Path, device_id: int, deadline_at
    ):
        """True for a proven decoy, False/None to proceed with engine verify."""
        remaining = remaining_seconds(deadline_at)
        impl_file = Path(work_dir) / f"{request.op_name}_{request.triton_impl_name}.py"
        outcome = run_script(
            [
                sys.executable,
                str(Path(self.engine_dir) / "validate_triton_impl.py"),
                str(impl_file),
                "--json",
            ],
            str(work_dir),
            device_id,
            timeout=max(1, min(_DECOY_CHECK_TIMEOUT_SECONDS, int(remaining))),
        )
        if outcome.timed_out:
            return engine_failure("correctness", "decoy pre-check timed out")
        parsed = _extract_validate_json(outcome.stdout)
        if parsed is None:
            return None  # let verify.py surface a real error
        return parsed.get("valid") is False

    def _reference_case_count(
        self, request: EvaluationRequest, work_dir: Path, device_id: int, deadline_at
    ):
        """Resolve the trusted reference input count without running the candidate."""
        remaining = remaining_seconds(deadline_at)
        if remaining <= 0:
            return engine_timeout(
                "correctness", "resource deadline expired before case counting"
            )
        reference = Path(work_dir) / f"{request.op_name}_torch.py"
        outcome = run_script(
            [sys.executable, "-c", _COUNT_SNIPPET, str(reference)],
            str(work_dir),
            device_id,
            timeout=max(1, min(_DECOY_CHECK_TIMEOUT_SECONDS, int(remaining))),
        )
        if outcome.timed_out:
            return engine_failure("correctness", "reference case counting timed out")
        try:
            line = outcome.stdout.strip().splitlines()[-1]
            payload = json.loads(line)
            count = payload["total_cases"]
        except (IndexError, ValueError, KeyError, TypeError):
            return engine_failure(
                "correctness", "reference input count resolution failed"
            )
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            return engine_failure(
                "correctness", "reference input count resolution failed"
            )
        return count

    def _decoy_result(
        self, request: EvaluationRequest, work_dir: Path, device_id: int, deadline_at
    ) -> EngineCheckResult:
        count = self._reference_case_count(request, work_dir, device_id, deadline_at)
        if isinstance(count, EngineCheckResult):
            return count
        return EngineCheckResult(
            status="succeeded", raw={"decoy": True, "total_cases": count}, error=None
        )

    def _run_verify(
        self, request: EvaluationRequest, work_dir: Path, device_id: int, deadline_at
    ) -> EngineCheckResult:
        remaining = remaining_seconds(deadline_at)
        if remaining <= 0:
            return engine_timeout(
                "correctness", "resource deadline expired before correctness ran"
            )
        budget = max(1, min(int(remaining), _PER_SUBJECT_BUDGET_CAP_SECONDS))
        out_path = Path(work_dir) / "verify_result.json"
        cmd = [
            sys.executable,
            str(Path(self.engine_dir) / "verify.py"),
            "--op_name",
            request.op_name,
            "--verify_dir",
            str(work_dir),
            "--triton_impl_name",
            request.triton_impl_name,
            "--timeout",
            str(budget),
            "--output",
            str(out_path),
        ]
        if request.non_compute:
            cmd.append("--non-compute")
        outcome = run_script(
            cmd,
            str(work_dir),
            device_id,
            timeout=max(1, int(remaining)) + _OUTER_GRACE_SECONDS,
        )
        if outcome.timed_out:
            # The outer kill is ambiguous: never a candidate execution timeout.
            return engine_timeout(
                "correctness", "correctness exceeded the resource deadline"
            )
        result = read_json(str(out_path))
        if not isinstance(result, dict):
            return engine_failure(
                "correctness", "verify engine produced no usable result"
            )
        return EngineCheckResult(status="succeeded", raw=result, error=None)
