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
import stat
import subprocess
import tempfile
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol
from uuid import uuid4

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
    modules = {
        f"{stem}_torch.py": request.reference_code,
        f"{stem}_{request.triton_impl_name}.py": request.kernel_code,
    }
    with _directory_descriptor(Path(work_dir), create=True) as directory_fd:
        for name, source in modules.items():
            temporary = f".{name}.{uuid4().hex}.tmp"
            fd = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=directory_fd,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    stream.write(source)
                # Replace the directory entry itself, never an existing target.
                # A later check can restage modules after candidate execution.
                os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
            finally:
                try:
                    os.unlink(temporary, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
    return stem


@contextmanager
def _directory_descriptor(path: Path, *, create: bool = False):
    """Anchor host staging and collection without traversing directory aliases."""
    if ".." in path.parts:
        raise ValueError("directory path must not contain parent traversal")
    path = path.absolute()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    with ExitStack() as stack:
        fd = os.open(path.anchor, flags)
        stack.callback(os.close, fd)
        for component in path.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, mode=0o700, dir_fd=fd)
                child = os.open(component, flags, dir_fd=fd)
            stack.callback(os.close, child)
            fd = child
        yield fd


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


def read_json(path: str | Path, *, root: str | Path) -> dict | None:
    """Read a regular result beneath the trusted attempt without following links.

    Check lexical containment, then walk every directory with descriptor-relative
    O_NOFOLLOW opens. An attacker replacing a checked parent with a symlink cannot
    redirect a later open. The attempt root and its ancestors cannot be symlinks
    either. Invalid or absent output keeps the existing unusable-result semantics.
    """
    try:
        path, root = Path(path), Path(root)
        if ".." in path.parts or ".." in root.parts:
            return None
        path, root = path.absolute(), root.absolute()
        relative = path.relative_to(root)
        if not relative.parts:
            return None
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        with _directory_descriptor(root) as directory_fd, ExitStack() as stack:
            for component in relative.parts[:-1]:
                directory_fd = os.open(component, directory_flags, dir_fd=directory_fd)
                stack.callback(os.close, directory_fd)
            file_fd = os.open(
                relative.name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                dir_fd=directory_fd,
            )
            with os.fdopen(file_fd, encoding="utf-8") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    return None
                return json.load(stream)
    except Exception:
        return None
