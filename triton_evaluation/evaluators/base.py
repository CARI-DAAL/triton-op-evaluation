"""Shared plumbing for the v2 evaluator layer.

``EngineCheckResult`` is what one engine invocation returns to the
orchestrator: either the private engine JSON (``raw``) or an infrastructure
``ErrorInfo`` — never a candidate-domain verdict, which only the shaping
layer may derive from attributed private output. ``ProgressSink`` is the
fenced durable-progress boundary: every requested check is begun and saved
through it, and a rejected call must abort the attempt via ``LeaseLostError``.
"""
import json
import logging
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

from ..api.contract import Check, CheckOutcome, ErrorInfo, EvaluationRequest
from ..api.contract_artifacts import canonical_json_bytes, sha256_hex
from .sandbox import isolate_command

logger = logging.getLogger(__name__)


class LeaseLostError(RuntimeError):
    """The attempt lost its lease or a fenced progress write was rejected."""


@dataclass(frozen=True)
class EngineCheckResult:
    status: Literal["succeeded", "failed", "timeout"]
    raw: dict[str, object] | None = None
    error: ErrorInfo | None = None



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
    outcome: Literal["exited", "budget", "resource_deadline", "fenced"] = "exited"


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


def safe_module_stem(request: EvaluationRequest) -> str:
    """Internal module stem: ``eval_`` plus the first 24 request-digest hex chars.

    The public ``op_name`` is never interpolated into filesystem paths or
    import arguments; the digest-derived stem is the only name crossing into
    engine invocations.
    """
    digest = sha256_hex(
        canonical_json_bytes(request.model_dump(mode="json", exclude_none=False))
    )
    return f"eval_{digest[:24]}"


def materialize(request: EvaluationRequest, work_dir: Path) -> str:
    """Write the reference and implementation modules; return the safe stem."""
    stem = safe_module_stem(request)
    d = Path(work_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stem}_torch.py").write_text(request.reference_code, encoding="utf-8")
    (d / f"{stem}_{request.triton_impl_name}.py").write_text(
        request.kernel_code, encoding="utf-8"
    )
    return stem


_SCRIPT_POLL_SECONDS = 0.05


def run_script(
    cmd: list,
    cwd,
    device_id: int | None,
    deadline_at=None,
    lease_valid=None,
    *,
    budget_seconds: float | None = None,
) -> ScriptOutcome:
    """Run one engine invocation with bounded polling.

    The process group is killed when the absolute resource deadline expires
    (``resource_deadline``), when the attempt lease is lost (``fenced``), or
    when the engine-level ``budget_seconds`` elapses (``budget``). None of
    those is a candidate execution timeout. Output goes to temporary files so
    a chatty engine can never deadlock on a full pipe.
    """
    if lease_valid is not None and not lease_valid():
        return ScriptOutcome(-1, "", "", True, "fenced")
    if deadline_at is not None and remaining_seconds(deadline_at) <= 0:
        return ScriptOutcome(-1, "", "", True, "resource_deadline")
    env = dict(os.environ)
    # A CPU precheck must not inherit the parent's NPU assignment.
    env["ASCEND_RT_VISIBLE_DEVICES"] = "" if device_id is None else str(device_id)
    cmd, env = isolate_command(cmd, cwd, device_id, env)
    with tempfile.TemporaryFile(mode="w+") as out_file, tempfile.TemporaryFile(
        mode="w+"
    ) as err_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=out_file,
            stderr=err_file,
            start_new_session=True,
        )
        try:
            started = time.monotonic()
            while True:
                if proc.poll() is not None:
                    break
                if deadline_at is not None and remaining_seconds(deadline_at) <= 0:
                    _kill_group(proc)
                    return _collected(proc, out_file, err_file, "resource_deadline")
                if lease_valid is not None and not lease_valid():
                    _kill_group(proc)
                    return _collected(proc, out_file, err_file, "fenced")
                if (
                    budget_seconds is not None
                    and time.monotonic() - started >= budget_seconds
                ):
                    _kill_group(proc)
                    return _collected(proc, out_file, err_file, "budget")
                time.sleep(_SCRIPT_POLL_SECONDS)
            return _collected(proc, out_file, err_file, "exited")
        finally:
            # The entrypoint can exit while candidate-created children remain.
            # Its process group belongs to this invocation on every exit path.
            _kill_group(proc)


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        # start_new_session fixes the group ID to the entrypoint PID, including
        # after that entrypoint has exited and getpgid(pid) would fail.
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def _collected(proc, out_file, err_file, outcome: str) -> ScriptOutcome:
    proc.wait()
    out_file.seek(0)
    err_file.seek(0)
    return ScriptOutcome(
        exit_code=proc.returncode,
        stdout=out_file.read(),
        stderr=err_file.read(),
        timed_out=outcome != "exited",
        outcome=outcome,
    )


def log_engine_crash(
    check: str,
    outcome: "ScriptOutcome",
    *,
    result_file_exists: bool,
    log_path: Path | None = None,
) -> None:
    """Retain first-failure evidence: one appended record per engine death.

    Captures the exit code, run outcome, result-file presence, and the tail of
    engine stdout/stderr so an intermittent failure is diagnosable after the
    fact (the per-attempt work dir is cleaned up by the worker). Best-effort:
    a logging failure never changes the evaluation outcome.
    """
    destination = log_path or Path("logs") / "engine_crashes.log"
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "a", encoding="utf-8") as handle:
            handle.write(
                f"===== {datetime.now(timezone.utc).isoformat()} "
                f"{check} crash exit={outcome.exit_code} "
                f"outcome={outcome.outcome} "
                f"result_file_exists={result_file_exists}\n"
            )
            handle.write(f"--- stderr tail ---\n{outcome.stderr[-6000:]}\n")
            handle.write(f"--- stdout tail ---\n{outcome.stdout[-6000:]}\n")
    except OSError:
        logger.warning("failed to retain engine crash evidence")


def read_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
