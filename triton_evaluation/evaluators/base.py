import json
import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from ..models import JobSpec


@dataclass
class EvaluatorResult:
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None


@dataclass
class ScriptOutcome:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool


class Evaluator(Protocol):
    name: str
    needs_bound_card: bool

    def run(self, spec: JobSpec, verify_dir: str, device_id: int) -> "EvaluatorResult": ...


def materialize(spec: JobSpec, verify_dir: str) -> None:
    d = Path(verify_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{spec.op_name}_torch.py").write_text(spec.reference_code)
    (d / f"{spec.op_name}_{spec.triton_impl_name}.py").write_text(spec.kernel_code)


def run_script(cmd: list, cwd: str, device_id: int, timeout: int) -> ScriptOutcome:
    env = dict(os.environ)
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(device_id)
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        return ScriptOutcome(proc.returncode, out, err, False)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        out, err = proc.communicate()
        return ScriptOutcome(-1, out or "", err or "", True)


def read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def tail(text: str, limit: int = 2000) -> str:
    if text is None:
        return ""
    return text if len(text) <= limit else text[-limit:]
