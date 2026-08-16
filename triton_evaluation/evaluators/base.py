"""Shared plumbing for the v2 evaluator layer.

``EngineCheckResult`` is what one engine invocation returns to the
orchestrator: either the private engine JSON (``raw``) or an infrastructure
``ErrorInfo`` — never a candidate-domain verdict, which only the shaping
layer may derive from attributed private output. ``ProgressSink`` is the
fenced durable-progress boundary: every requested check is begun and saved
through it, and a rejected call must abort the attempt via ``LeaseLostError``.
"""
import json
import os
import signal
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

from ..api.contract import Check, CheckOutcome, ErrorInfo, EvaluationRequest, Purpose
from ..models import AlgorithmVersions


class LeaseLostError(RuntimeError):
    """The attempt lost its lease or a fenced progress write was rejected."""


@dataclass(frozen=True)
class EngineCheckResult:
    status: Literal["succeeded", "failed", "timeout"]
    raw: dict[str, object] | None = None
    error: ErrorInfo | None = None


@dataclass(frozen=True)
class PinnedProvenance:
    """Provenance facts pinned at admission: build, algorithms, purpose, binding."""

    evaluator_build: str
    algorithms: AlgorithmVersions
    purpose: Purpose
    manifest_id: str | None
    problem_id: str | None

    @classmethod
    def for_request(
        cls,
        request: EvaluationRequest,
        *,
        evaluator_build: str,
        algorithms: AlgorithmVersions,
    ) -> "PinnedProvenance":
        context = request.grade_context
        return cls(
            evaluator_build=evaluator_build,
            algorithms=algorithms,
            purpose=request.purpose,
            manifest_id=context.manifest_id if context is not None else None,
            problem_id=context.problem_id if context is not None else None,
        )


class ProgressSink(Protocol):
    """Fenced check progress; implementations return False when fenced out."""

    def begin(self, check: Check) -> bool: ...

    def save(self, check: Check, outcome: CheckOutcome) -> bool: ...


@dataclass
class ScriptOutcome:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool


def engine_failure(
    check: Check,
    message: str,
    *,
    code: str = "EVALUATOR_EXECUTION_FAILED",
    phase: str = "execution",
    retryable: bool = True,
) -> EngineCheckResult:
    """Infrastructure failure of one engine invocation (never a subject verdict)."""
    return EngineCheckResult(
        status="failed",
        raw=None,
        error=ErrorInfo(
            code=code, phase=phase, check=check, retryable=retryable, message=message
        ),
    )


def engine_timeout(check: Check, message: str) -> EngineCheckResult:
    """The check's engine invocation ran out of time without attribution."""
    return EngineCheckResult(
        status="timeout",
        raw=None,
        error=ErrorInfo(
            code="DEADLINE_EXCEEDED",
            phase="deadline",
            check=check,
            retryable=False,
            message=message,
        ),
    )


def remaining_seconds(deadline_at: datetime) -> float:
    return (deadline_at - datetime.now(timezone.utc)).total_seconds()


def materialize(request: EvaluationRequest, work_dir: Path) -> None:
    d = Path(work_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{request.op_name}_torch.py").write_text(
        request.reference_code, encoding="utf-8"
    )
    (d / f"{request.op_name}_{request.triton_impl_name}.py").write_text(
        request.kernel_code, encoding="utf-8"
    )


def run_script(cmd: list, cwd: str, device_id: int, timeout: int) -> ScriptOutcome:
    env = dict(os.environ)
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(device_id)
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        return ScriptOutcome(proc.returncode, out, err, False)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        out, err = proc.communicate()
        return ScriptOutcome(-1, out or "", err or "", True)


def read_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
