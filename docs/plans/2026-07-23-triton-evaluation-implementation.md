# triton-evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package the `triton-op-verifier` skill's verify/benchmark scripts into a standalone FastAPI + Redis evaluator service with one worker pinned per NPU card.

**Architecture:** FastAPI receives async eval requests and pushes jobs onto a Redis list; one worker process per NPU card blocks on the queue, materializes the code to a temp dir, subprocess-calls the skill script (`ASCEND_RT_VISIBLE_DEVICES` pinned), parses the skill's JSON, and writes the result back to Redis. Clients poll `GET /tasks/{id}`. Evaluators are pluggable (verify, benchmark, later profile).

**Tech Stack:** Python 3.10+, FastAPI, uvicorn, redis-py (sync client), pydantic v2, pytest, fakeredis, httpx (TestClient).

## Global Constraints

- Subprocess-call the skill scripts by path; NEVER `import` the skill or kernelgym. (path from `Settings.skill_scripts_dir`)
- `result` field passes the skill's native JSON through verbatim — do not reinvent the result schema.
- One card = one worker; the worker pins its card via `ASCEND_RT_VISIBLE_DEVICES` and each subprocess call re-sets it in its env.
- Task status reflects only whether the *evaluation ran*, never the correctness/perf verdict (which lives inside `result`).
- verify: JSON present → `SUCCEEDED` regardless of exit 0 vs 1; JSON absent → `FAILED`.
- benchmark: exit 2 → `REJECTED`; exit 0 + JSON → `SUCCEEDED`; else `FAILED`.
- benchmark request MUST carry `verify_task_id` OR `verify_not_required=true`, else HTTP 422.
- Task records carry a 24h TTL (`Settings.task_ttl_seconds`).
- Importable package: `triton_evaluation` (underscore). Repo dir: `triton-evaluation` (hyphen).
- All commands run from repo root `triton-evaluation/`.

---

## File Structure

```
triton-evaluation/
├── pyproject.toml
├── triton_evaluation/
│   ├── __init__.py
│   ├── config.py            # Settings (Task 1)
│   ├── models.py            # enums, JobSpec, Job, API models (Task 2)
│   ├── broker.py            # Redis wrapper (Task 3)
│   ├── evaluators/
│   │   ├── __init__.py      # build_registry (Task 6)
│   │   ├── base.py          # Evaluator proto, materialize, run_script (Task 4)
│   │   ├── verify.py        # VerifyEvaluator (Task 5)
│   │   └── benchmark.py     # BenchmarkEvaluator (Task 6)
│   ├── worker.py            # Worker loop + gate resolution + reaper (Task 7)
│   └── api.py               # FastAPI app (Task 8)
├── run_api.py               # entrypoint (Task 9)
├── run_worker.py            # entrypoint (Task 9)
├── docs/{specs,plans}/
└── tests/
    ├── conftest.py
    ├── fixtures/            # canned verify_result.json / perf_result.json + fake skill scripts
    ├── test_models.py
    ├── test_broker.py
    ├── test_evaluators.py
    ├── test_worker.py
    └── test_api.py
```

---

## Task 1: Project scaffold + config

**Files:**
- Create: `pyproject.toml`
- Create: `triton_evaluation/__init__.py` (empty)
- Create: `triton_evaluation/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Settings` frozen dataclass with fields `redis_url: str`, `skill_scripts_dir: str`, `queue_key: str`, `task_key_prefix: str`, `task_ttl_seconds: int`, `dequeue_block_seconds: int`, `reaper_grace_seconds: int`; classmethod `Settings.from_env() -> Settings`.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "triton-evaluation"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.110",
    "uvicorn>=0.27",
    "redis>=5.0",
    "pydantic>=2.6",
]

[project.optional-dependencies]
test = ["pytest>=8.0", "fakeredis>=2.21", "httpx>=0.27"]

[tool.setuptools.packages.find]
include = ["triton_evaluation*"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```

- [ ] **Step 2: Create empty `triton_evaluation/__init__.py`**

```python
```

- [ ] **Step 3: Write the failing test** — `tests/test_config.py`

```python
import os
from triton_evaluation.config import Settings


def test_defaults():
    s = Settings()
    assert s.queue_key == "eval:queue"
    assert s.task_key_prefix == "eval:task:"
    assert s.task_ttl_seconds == 24 * 3600


def test_from_env_override(monkeypatch):
    monkeypatch.setenv("TRITON_EVAL_REDIS_URL", "redis://x:1/2")
    monkeypatch.setenv("TRITON_EVAL_SKILL_SCRIPTS_DIR", "/skills/scripts")
    s = Settings.from_env()
    assert s.redis_url == "redis://x:1/2"
    assert s.skill_scripts_dir == "/skills/scripts"
```

- [ ] **Step 4: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'triton_evaluation.config'`

- [ ] **Step 5: Implement `triton_evaluation/config.py`**

```python
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    redis_url: str = "redis://localhost:6379/0"
    skill_scripts_dir: str = ""
    queue_key: str = "eval:queue"
    task_key_prefix: str = "eval:task:"
    task_ttl_seconds: int = 24 * 3600
    dequeue_block_seconds: int = 5
    reaper_grace_seconds: int = 120

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            redis_url=os.environ.get("TRITON_EVAL_REDIS_URL", cls.redis_url),
            skill_scripts_dir=os.environ.get("TRITON_EVAL_SKILL_SCRIPTS_DIR", cls.skill_scripts_dir),
            task_ttl_seconds=int(os.environ.get("TRITON_EVAL_TASK_TTL", cls.task_ttl_seconds)),
            dequeue_block_seconds=int(os.environ.get("TRITON_EVAL_DEQUEUE_BLOCK", cls.dequeue_block_seconds)),
            reaper_grace_seconds=int(os.environ.get("TRITON_EVAL_REAPER_GRACE", cls.reaper_grace_seconds)),
        )
```

Note: `cls.redis_url` reads the dataclass field default on the class — valid for frozen dataclasses.

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml triton_evaluation/__init__.py triton_evaluation/config.py tests/test_config.py
git commit -m "feat: project scaffold + Settings config"
```

---

## Task 2: Data models

**Files:**
- Create: `triton_evaluation/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class TaskStatus(str, Enum)`: `QUEUED RUNNING SUCCEEDED FAILED TIMEOUT REJECTED`
  - `class JobType(str, Enum)`: `VERIFY="verify"`, `BENCHMARK="benchmark"`
  - `@dataclass JobSpec` fields: `op_name, reference_code, kernel_code, triton_impl_name="triton_ascend_impl", non_compute=False, timeout=900, warmup=5, repeats=50, verify_task_id: Optional[str]=None, verify_not_required=False, verify_result_seed: Optional[dict]=None`
  - `@dataclass Job` fields `task_id: str, type: str, spec: JobSpec`; methods `to_json() -> str`, `from_json(str) -> Job` (staticmethod)
  - pydantic models `VerifyRequest`, `BenchmarkRequest` (with gate validator), `SubmitResponse`, `TaskResponse`

- [ ] **Step 1: Write the failing test** — `tests/test_models.py`

```python
import pytest
from pydantic import ValidationError
from triton_evaluation.models import (
    TaskStatus, JobType, JobSpec, Job, VerifyRequest, BenchmarkRequest,
)


def test_job_json_roundtrip():
    spec = JobSpec(op_name="softmax", reference_code="R", kernel_code="K", non_compute=True)
    job = Job(task_id="t1", type=JobType.VERIFY.value, spec=spec)
    back = Job.from_json(job.to_json())
    assert back.task_id == "t1"
    assert back.type == "verify"
    assert back.spec.op_name == "softmax"
    assert back.spec.non_compute is True
    assert back.spec.triton_impl_name == "triton_ascend_impl"


def test_benchmark_request_requires_gate():
    with pytest.raises(ValidationError):
        BenchmarkRequest(op_name="s", reference_code="R", kernel_code="K")


def test_benchmark_request_accepts_verify_task_id():
    r = BenchmarkRequest(op_name="s", reference_code="R", kernel_code="K", verify_task_id="v1")
    assert r.verify_task_id == "v1"


def test_verify_request_defaults():
    r = VerifyRequest(op_name="s", reference_code="R", kernel_code="K")
    assert r.timeout == 900
    assert r.non_compute is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'triton_evaluation.models'`

- [ ] **Step 3: Implement `triton_evaluation/models.py`**

```python
import json
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class TaskStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    REJECTED = "REJECTED"


class JobType(str, Enum):
    VERIFY = "verify"
    BENCHMARK = "benchmark"


@dataclass
class JobSpec:
    op_name: str
    reference_code: str
    kernel_code: str
    triton_impl_name: str = "triton_ascend_impl"
    # verify
    non_compute: bool = False
    timeout: int = 900
    # benchmark
    warmup: int = 5
    repeats: int = 50
    verify_task_id: Optional[str] = None
    verify_not_required: bool = False
    verify_result_seed: Optional[dict] = None  # injected by worker for the L1 gate


@dataclass
class Job:
    task_id: str
    type: str
    spec: JobSpec

    def to_json(self) -> str:
        return json.dumps({"task_id": self.task_id, "type": self.type, "spec": asdict(self.spec)})

    @staticmethod
    def from_json(payload: str) -> "Job":
        d = json.loads(payload)
        return Job(task_id=d["task_id"], type=d["type"], spec=JobSpec(**d["spec"]))


class VerifyRequest(BaseModel):
    op_name: str
    reference_code: str
    kernel_code: str
    non_compute: bool = False
    triton_impl_name: str = "triton_ascend_impl"
    timeout: int = Field(default=900, ge=10, le=3600)


class BenchmarkRequest(BaseModel):
    op_name: str
    reference_code: str
    kernel_code: str
    triton_impl_name: str = "triton_ascend_impl"
    warmup: int = Field(default=5, ge=0, le=100)
    repeats: int = Field(default=50, ge=1, le=1000)
    timeout: int = Field(default=1800, ge=10, le=3600)
    verify_task_id: Optional[str] = None
    verify_not_required: bool = False

    @model_validator(mode="after")
    def _require_gate(self):
        if not self.verify_task_id and not self.verify_not_required:
            raise ValueError("benchmark requires verify_task_id or verify_not_required=true")
        return self


class SubmitResponse(BaseModel):
    task_id: str


class TaskResponse(BaseModel):
    task_id: str
    type: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None
    device_id: Optional[int] = None
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_models.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add triton_evaluation/models.py tests/test_models.py
git commit -m "feat: job/task data models + API request/response schemas"
```

---

## Task 3: Redis broker

**Files:**
- Create: `triton_evaluation/broker.py`
- Test: `tests/test_broker.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Consumes: `Settings` (Task 1), `Job`, `JobSpec`, `TaskStatus`, `JobType` (Task 2).
- Produces: `class Broker` with `__init__(self, client, settings)`, classmethod `from_settings(settings)`, and methods:
  - `create_task(self, job: Job) -> None` — writes QUEUED record + LPUSH job
  - `dequeue(self) -> Optional[Job]` — BRPOP with block timeout
  - `mark_running(self, task_id: str, device_id: int) -> None`
  - `store_result(self, task_id: str, status: str, result: Optional[dict]=None, error: Optional[str]=None) -> None`
  - `get_task(self, task_id: str) -> Optional[dict]` — parsed record (result→dict, times→float, device_id→int)
  - `iter_running_tasks(self) -> list[dict]` — for reaper
  - `ping(self) -> None`

- [ ] **Step 1: Create `tests/conftest.py`**

```python
import pytest
import fakeredis

from triton_evaluation.config import Settings
from triton_evaluation.broker import Broker


@pytest.fixture
def settings(tmp_path):
    return Settings(skill_scripts_dir=str(tmp_path), dequeue_block_seconds=1)


@pytest.fixture
def broker(settings):
    client = fakeredis.FakeStrictRedis(decode_responses=True)
    return Broker(client, settings)
```

- [ ] **Step 2: Write the failing test** — `tests/test_broker.py`

```python
from triton_evaluation.models import Job, JobSpec, JobType, TaskStatus


def _job(task_id="t1", type_="verify"):
    spec = JobSpec(op_name="softmax", reference_code="R", kernel_code="K")
    return Job(task_id=task_id, type=type_, spec=spec)


def test_create_then_get(broker):
    broker.create_task(_job())
    rec = broker.get_task("t1")
    assert rec["task_id"] == "t1"
    assert rec["type"] == "verify"
    assert rec["status"] == "QUEUED"
    assert isinstance(rec["created_at"], float)


def test_get_missing_returns_none(broker):
    assert broker.get_task("nope") is None


def test_enqueue_dequeue_roundtrip(broker):
    broker.create_task(_job("t2"))
    job = broker.dequeue()
    assert job.task_id == "t2"
    assert job.spec.op_name == "softmax"


def test_dequeue_empty_returns_none(broker):
    assert broker.dequeue() is None


def test_store_result_parses_json(broker):
    broker.create_task(_job("t3"))
    broker.store_result("t3", TaskStatus.SUCCEEDED.value, result={"passed_cases": 3, "total_cases": 3})
    rec = broker.get_task("t3")
    assert rec["status"] == "SUCCEEDED"
    assert rec["result"]["passed_cases"] == 3


def test_mark_running_sets_device(broker):
    broker.create_task(_job("t4"))
    broker.mark_running("t4", 5)
    rec = broker.get_task("t4")
    assert rec["status"] == "RUNNING"
    assert rec["device_id"] == 5
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_broker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'triton_evaluation.broker'`

- [ ] **Step 4: Implement `triton_evaluation/broker.py`**

```python
import json
import time
from typing import Optional

import redis

from .config import Settings
from .models import Job, TaskStatus


def _parse_task(raw: dict) -> dict:
    def _f(v):
        return float(v) if v not in (None, "") else None

    return {
        "task_id": raw.get("task_id"),
        "type": raw.get("type"),
        "status": raw.get("status"),
        "result": json.loads(raw["result"]) if raw.get("result") else None,
        "error": raw.get("error") or None,
        "device_id": int(raw["device_id"]) if raw.get("device_id") not in (None, "") else None,
        "created_at": _f(raw.get("created_at")),
        "started_at": _f(raw.get("started_at")),
        "finished_at": _f(raw.get("finished_at")),
    }


class Broker:
    def __init__(self, client: "redis.Redis", settings: Settings):
        self.r = client
        self.s = settings

    @classmethod
    def from_settings(cls, settings: Settings) -> "Broker":
        return cls(redis.from_url(settings.redis_url, decode_responses=True), settings)

    def _task_key(self, task_id: str) -> str:
        return f"{self.s.task_key_prefix}{task_id}"

    def ping(self) -> None:
        self.r.ping()

    def create_task(self, job: Job) -> None:
        key = self._task_key(job.task_id)
        self.r.hset(key, mapping={
            "task_id": job.task_id,
            "type": job.type,
            "status": TaskStatus.QUEUED.value,
            "created_at": repr(time.time()),
        })
        self.r.expire(key, self.s.task_ttl_seconds)
        self.r.lpush(self.s.queue_key, job.to_json())

    def dequeue(self) -> Optional[Job]:
        item = self.r.brpop(self.s.queue_key, timeout=self.s.dequeue_block_seconds)
        if item is None:
            return None
        _, payload = item
        return Job.from_json(payload)

    def mark_running(self, task_id: str, device_id: int) -> None:
        key = self._task_key(task_id)
        self.r.hset(key, mapping={
            "status": TaskStatus.RUNNING.value,
            "started_at": repr(time.time()),
            "device_id": str(device_id),
        })
        self.r.expire(key, self.s.task_ttl_seconds)

    def store_result(self, task_id: str, status: str,
                     result: Optional[dict] = None, error: Optional[str] = None) -> None:
        key = self._task_key(task_id)
        mapping = {"status": status, "finished_at": repr(time.time())}
        if result is not None:
            mapping["result"] = json.dumps(result)
        if error is not None:
            mapping["error"] = error
        self.r.hset(key, mapping=mapping)
        self.r.expire(key, self.s.task_ttl_seconds)

    def get_task(self, task_id: str) -> Optional[dict]:
        key = self._task_key(task_id)
        if not self.r.exists(key):
            return None
        return _parse_task(self.r.hgetall(key))

    def iter_running_tasks(self) -> list:
        out = []
        for key in self.r.scan_iter(match=f"{self.s.task_key_prefix}*"):
            raw = self.r.hgetall(key)
            if raw.get("status") == TaskStatus.RUNNING.value:
                out.append(_parse_task(raw))
        return out
```

Note: `time.time()` stored via `repr()` to preserve full float precision; `_parse_task` reads it back with `float()`.

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_broker.py -v`
Expected: PASS (6 passed)

- [ ] **Step 6: Commit**

```bash
git add triton_evaluation/broker.py tests/test_broker.py tests/conftest.py
git commit -m "feat: Redis broker (queue + task records)"
```

---

## Task 4: Evaluator base (materialize + subprocess runner)

**Files:**
- Create: `triton_evaluation/evaluators/__init__.py` (empty for now)
- Create: `triton_evaluation/evaluators/base.py`
- Test: `tests/test_evaluators.py` (base portion)

**Interfaces:**
- Consumes: `JobSpec` (Task 2).
- Produces:
  - `@dataclass EvaluatorResult` fields `status: str, result: Optional[dict]=None, error: Optional[str]=None`
  - `@dataclass ScriptOutcome` fields `exit_code: int, stdout: str, stderr: str, timed_out: bool`
  - `materialize(spec: JobSpec, verify_dir: str) -> None` — writes `{op}_torch.py` (reference_code) and `{op}_{triton_impl_name}.py` (kernel_code)
  - `run_script(cmd: list[str], cwd: str, device_id: int, timeout: int) -> ScriptOutcome` — Popen in new session, kills process group on timeout, injects `ASCEND_RT_VISIBLE_DEVICES`
  - `read_json(path: str) -> Optional[dict]`
  - `tail(text: str, limit: int=2000) -> str`

- [ ] **Step 1: Create empty `triton_evaluation/evaluators/__init__.py`**

```python
```

- [ ] **Step 2: Write the failing test** — `tests/test_evaluators.py`

```python
import sys
from pathlib import Path

from triton_evaluation.models import JobSpec
from triton_evaluation.evaluators.base import materialize, run_script, read_json, tail


def test_materialize_writes_named_files(tmp_path):
    spec = JobSpec(op_name="softmax", reference_code="REF", kernel_code="IMPL")
    materialize(spec, str(tmp_path))
    assert (tmp_path / "softmax_torch.py").read_text() == "REF"
    assert (tmp_path / "softmax_triton_ascend_impl.py").read_text() == "IMPL"


def test_run_script_captures_exit_and_env(tmp_path):
    script = tmp_path / "echo_env.py"
    script.write_text(
        "import os, sys\n"
        "sys.stdout.write(os.environ.get('ASCEND_RT_VISIBLE_DEVICES',''))\n"
        "sys.exit(7)\n"
    )
    outcome = run_script([sys.executable, str(script)], str(tmp_path), device_id=3, timeout=30)
    assert outcome.exit_code == 7
    assert outcome.stdout == "3"
    assert outcome.timed_out is False


def test_run_script_times_out(tmp_path):
    script = tmp_path / "sleep.py"
    script.write_text("import time\ntime.sleep(30)\n")
    outcome = run_script([sys.executable, str(script)], str(tmp_path), device_id=0, timeout=1)
    assert outcome.timed_out is True


def test_read_json_missing_returns_none(tmp_path):
    assert read_json(str(tmp_path / "nope.json")) is None


def test_tail_truncates():
    assert tail("x" * 5000, limit=100).endswith("x" * 50)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_evaluators.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'triton_evaluation.evaluators.base'`

- [ ] **Step 4: Implement `triton_evaluation/evaluators/base.py`**

```python
import json
import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_evaluators.py -v`
Expected: PASS (5 passed)

- [ ] **Step 6: Commit**

```bash
git add triton_evaluation/evaluators/__init__.py triton_evaluation/evaluators/base.py tests/test_evaluators.py
git commit -m "feat: evaluator base — materialize + process-group subprocess runner"
```

---

## Task 5: VerifyEvaluator

**Files:**
- Create: `triton_evaluation/evaluators/verify.py`
- Create: `tests/fixtures/fake_verify.py` (a stand-in for the skill's verify.py)
- Test: append to `tests/test_evaluators.py`

**Interfaces:**
- Consumes: `EvaluatorResult`, `materialize`, `run_script`, `read_json`, `tail` (Task 4); `JobSpec`, `TaskStatus` (Task 2).
- Produces: `class VerifyEvaluator` with attrs `name="verify"`, `needs_bound_card=True`, `__init__(self, scripts_dir: str)`, `run(self, spec: JobSpec, verify_dir: str, device_id: int) -> EvaluatorResult`. It calls `<scripts_dir>/verify.py` and writes `verify_result.json` in `verify_dir`.

- [ ] **Step 1: Create the fake skill script** — `tests/fixtures/fake_verify.py`

This emulates the skill's `verify.py` CLI contract: writes a `verify_result.json` and exits per an env-controlled code.

```python
import argparse
import json
import os
import sys

p = argparse.ArgumentParser()
p.add_argument("--op_name", required=True)
p.add_argument("--verify_dir", default=".")
p.add_argument("--triton_impl_name", default="triton_ascend_impl")
p.add_argument("--timeout", type=int, default=900)
p.add_argument("--output", default=None)
p.add_argument("--non-compute", action="store_true")
args = p.parse_args()

mode = os.environ.get("FAKE_VERIFY_MODE", "pass")  # pass | fail | crash
out = args.output or os.path.join(args.verify_dir, "verify_result.json")

if mode == "crash":
    sys.stderr.write("boom\n")
    sys.exit(1)

result = {
    "op_name": args.op_name,
    "total_cases": 2,
    "passed_cases": 2 if mode == "pass" else 1,
    "failed_cases": 0 if mode == "pass" else 1,
    "failures": [] if mode == "pass" else [{"case_idx": 2, "error_type": "AccuracyError"}],
}
with open(out, "w") as f:
    json.dump(result, f)
sys.exit(0 if mode == "pass" else 1)
```

- [ ] **Step 2: Write the failing test** — append to `tests/test_evaluators.py`

```python
import os
import pytest
from triton_evaluation.evaluators.verify import VerifyEvaluator

FIXTURES = Path(__file__).parent / "fixtures"


class _VerifyHarness(VerifyEvaluator):
    """Point verify.py at the fake script instead of the real skill script."""
    def _script_path(self):
        return str(FIXTURES / "fake_verify.py")


def _spec():
    return JobSpec(op_name="softmax", reference_code="R", kernel_code="K")


def test_verify_pass_is_succeeded(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_VERIFY_MODE", "pass")
    res = _VerifyHarness("unused").run(_spec(), str(tmp_path), device_id=1)
    assert res.status == "SUCCEEDED"
    assert res.result["passed_cases"] == 2


def test_verify_fail_is_still_succeeded(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_VERIFY_MODE", "fail")
    res = _VerifyHarness("unused").run(_spec(), str(tmp_path), device_id=1)
    assert res.status == "SUCCEEDED"
    assert res.result["passed_cases"] == 1  # wrong kernel, but job ran fine


def test_verify_crash_no_json_is_failed(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_VERIFY_MODE", "crash")
    res = _VerifyHarness("unused").run(_spec(), str(tmp_path), device_id=1)
    assert res.status == "FAILED"
    assert "boom" in res.error
```

Note: the `_VerifyHarness._script_path()` override requires `VerifyEvaluator.run()` to call `self._script_path()` to locate the script. Implement it that way in Step 4.

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_evaluators.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'triton_evaluation.evaluators.verify'`

- [ ] **Step 4: Implement `triton_evaluation/evaluators/verify.py`**

```python
import sys
from pathlib import Path

from ..models import JobSpec, TaskStatus
from .base import EvaluatorResult, materialize, run_script, read_json, tail


class VerifyEvaluator:
    name = "verify"
    needs_bound_card = True

    def __init__(self, scripts_dir: str):
        self.scripts_dir = scripts_dir

    def _script_path(self) -> str:
        return str(Path(self.scripts_dir) / "verify.py")

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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_evaluators.py -v`
Expected: PASS (all evaluator tests pass)

- [ ] **Step 6: Commit**

```bash
git add triton_evaluation/evaluators/verify.py tests/fixtures/fake_verify.py tests/test_evaluators.py
git commit -m "feat: VerifyEvaluator wrapping skill verify.py"
```

---

## Task 6: BenchmarkEvaluator + registry

**Files:**
- Create: `triton_evaluation/evaluators/benchmark.py`
- Modify: `triton_evaluation/evaluators/__init__.py` (add `build_registry`)
- Create: `tests/fixtures/fake_benchmark.py`
- Test: append to `tests/test_evaluators.py`

**Interfaces:**
- Consumes: base helpers (Task 4); `JobSpec`, `TaskStatus` (Task 2).
- Produces:
  - `class BenchmarkEvaluator` with `name="benchmark"`, `needs_bound_card=True`, `__init__(self, scripts_dir)`, `run(self, spec, verify_dir, device_id) -> EvaluatorResult`. Seeds the verify JSON from `spec.verify_result_seed` before running.
  - module function `verify_json_name(triton_impl_name: str) -> str` (mirrors skill's `resolve_verify_json_name`)
  - `build_registry(scripts_dir: str) -> dict[str, object]` returning `{"verify": VerifyEvaluator(...), "benchmark": BenchmarkEvaluator(...)}`

- [ ] **Step 1: Create the fake skill script** — `tests/fixtures/fake_benchmark.py`

Emulates the skill's `benchmark.py`: enforces the L1 gate (verify_result.json must exist and be all-passed unless `--verify_not_required`), then writes `perf_result.json`.

```python
import argparse
import json
import os
import sys

p = argparse.ArgumentParser()
p.add_argument("--op_name", required=True)
p.add_argument("--verify_dir", default=".")
p.add_argument("--triton_impl_name", default="triton_ascend_impl")
p.add_argument("--warmup", type=int, default=5)
p.add_argument("--repeats", type=int, default=50)
p.add_argument("--output", default=None)
p.add_argument("--verify_not_required", action="store_true")
args = p.parse_args()


def _verify_json_name(name):
    if name == "triton_ascend_impl":
        return "verify_result.json"
    suffix = name[len("triton_"):] if name.startswith("triton_") else name
    return "verify_result_%s.json" % suffix


if not args.verify_not_required:
    vpath = os.path.join(args.verify_dir, _verify_json_name(args.triton_impl_name))
    try:
        with open(vpath) as f:
            v = json.load(f)
    except Exception:
        sys.stderr.write("L1 gate: verify_result missing\n")
        sys.exit(2)
    if v.get("total_cases", 0) == 0 or v.get("passed_cases") != v.get("total_cases"):
        sys.stderr.write("L1 gate: verify not all passed\n")
        sys.exit(2)

out = args.output or os.path.join(args.verify_dir, "perf_result.json")
with open(out, "w") as f:
    json.dump({"op_name": args.op_name, "speedup_vs_torch": 2.0,
               "total_cases": 1, "passed_cases": 1, "failed_cases": 0}, f)
sys.exit(0)
```

- [ ] **Step 2: Write the failing test** — append to `tests/test_evaluators.py`

```python
from triton_evaluation.evaluators.benchmark import BenchmarkEvaluator, verify_json_name
from triton_evaluation.evaluators import build_registry


class _BenchHarness(BenchmarkEvaluator):
    def _script_path(self):
        return str(FIXTURES / "fake_benchmark.py")


def test_verify_json_name_mapping():
    assert verify_json_name("triton_ascend_impl") == "verify_result.json"
    assert verify_json_name("triton_optimized") == "verify_result_optimized.json"


def test_benchmark_with_seed_passes_gate(tmp_path):
    spec = _spec()
    spec.verify_result_seed = {"total_cases": 2, "passed_cases": 2}
    res = _BenchHarness("unused").run(spec, str(tmp_path), device_id=2)
    assert res.status == "SUCCEEDED"
    assert res.result["speedup_vs_torch"] == 2.0


def test_benchmark_without_seed_is_rejected(tmp_path):
    spec = _spec()  # no seed, no verify_not_required
    res = _BenchHarness("unused").run(spec, str(tmp_path), device_id=2)
    assert res.status == "REJECTED"


def test_benchmark_verify_not_required_skips_gate(tmp_path):
    spec = _spec()
    spec.verify_not_required = True
    res = _BenchHarness("unused").run(spec, str(tmp_path), device_id=2)
    assert res.status == "SUCCEEDED"


def test_registry_has_both(tmp_path):
    reg = build_registry(str(tmp_path))
    assert reg["verify"].name == "verify"
    assert reg["benchmark"].name == "benchmark"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_evaluators.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'triton_evaluation.evaluators.benchmark'`

- [ ] **Step 4: Implement `triton_evaluation/evaluators/benchmark.py`**

```python
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
```

- [ ] **Step 5: Implement `build_registry` in `triton_evaluation/evaluators/__init__.py`**

```python
from .verify import VerifyEvaluator
from .benchmark import BenchmarkEvaluator


def build_registry(scripts_dir: str) -> dict:
    return {
        "verify": VerifyEvaluator(scripts_dir),
        "benchmark": BenchmarkEvaluator(scripts_dir),
    }
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_evaluators.py -v`
Expected: PASS (all evaluator tests pass)

- [ ] **Step 7: Commit**

```bash
git add triton_evaluation/evaluators/benchmark.py triton_evaluation/evaluators/__init__.py tests/fixtures/fake_benchmark.py tests/test_evaluators.py
git commit -m "feat: BenchmarkEvaluator with gate seeding + evaluator registry"
```

---

## Task 7: Worker loop + gate resolution + reaper

**Files:**
- Create: `triton_evaluation/worker.py`
- Test: `tests/test_worker.py`

**Interfaces:**
- Consumes: `Broker` (Task 3), `build_registry`/evaluators (Task 6), `JobType`, `TaskStatus` (Task 2), `Settings` (Task 1).
- Produces:
  - `class Worker` with `__init__(self, broker, registry, device_id, settings)`, `run_once(self) -> bool` (returns True if a job was processed, False if idle), `serve_forever(self)`.
  - `reap_stale_tasks(broker, settings) -> int` — marks RUNNING tasks older than `started_at + grace` as `FAILED(worker_lost)`; returns count reaped.
- Behavior: on a benchmark job with `spec.verify_task_id`, fetch that task's stored `result` from the broker and assign it to `spec.verify_result_seed` before dispatch. Each job runs in a fresh `tempfile.mkdtemp` dir, removed in `finally`.

- [ ] **Step 1: Write the failing test** — `tests/test_worker.py`

```python
import time

from triton_evaluation.models import Job, JobSpec, JobType, TaskStatus
from triton_evaluation.worker import Worker, reap_stale_tasks
from triton_evaluation.evaluators.base import EvaluatorResult


class FakeEvaluator:
    name = "fake"
    needs_bound_card = True

    def __init__(self):
        self.seen = []

    def run(self, spec, verify_dir, device_id):
        self.seen.append((spec, verify_dir, device_id))
        return EvaluatorResult(TaskStatus.SUCCEEDED.value, result={"ok": True})


def _spec():
    return JobSpec(op_name="s", reference_code="R", kernel_code="K")


def test_run_once_dispatches_and_stores(broker, settings):
    ev = FakeEvaluator()
    broker.create_task(Job("t1", JobType.VERIFY.value, _spec()))
    w = Worker(broker, {"verify": ev}, device_id=4, settings=settings)
    assert w.run_once() is True
    rec = broker.get_task("t1")
    assert rec["status"] == "SUCCEEDED"
    assert rec["result"] == {"ok": True}
    assert rec["device_id"] == 4
    assert ev.seen[0][2] == 4  # device_id passed through


def test_run_once_idle_returns_false(broker, settings):
    w = Worker(broker, {}, device_id=0, settings=settings)
    assert w.run_once() is False


def test_benchmark_gate_resolution_injects_seed(broker, settings):
    # prior verify task with a stored result
    broker.create_task(Job("v1", JobType.VERIFY.value, _spec()))
    broker.store_result("v1", TaskStatus.SUCCEEDED.value,
                        result={"total_cases": 2, "passed_cases": 2})
    ev = FakeEvaluator()
    bspec = _spec()
    bspec.verify_task_id = "v1"
    broker.create_task(Job("b1", JobType.BENCHMARK.value, bspec))
    w = Worker(broker, {"benchmark": ev}, device_id=0, settings=settings)
    w.run_once()
    seen_spec = ev.seen[0][0]
    assert seen_spec.verify_result_seed == {"total_cases": 2, "passed_cases": 2}


def test_evaluator_exception_marks_failed(broker, settings):
    class Boom:
        name = "verify"
        needs_bound_card = True
        def run(self, *a):
            raise RuntimeError("kaboom")

    broker.create_task(Job("t2", JobType.VERIFY.value, _spec()))
    w = Worker(broker, {"verify": Boom()}, device_id=0, settings=settings)
    w.run_once()
    rec = broker.get_task("t2")
    assert rec["status"] == "FAILED"
    assert "kaboom" in rec["error"]


def test_reaper_marks_stale_running(broker, settings):
    broker.create_task(Job("t3", JobType.VERIFY.value, _spec()))
    broker.mark_running("t3", 0)
    # backdate started_at beyond the grace window
    broker.r.hset(broker._task_key("t3"),
                  "started_at", repr(time.time() - settings.reaper_grace_seconds - 10))
    n = reap_stale_tasks(broker, settings)
    assert n == 1
    assert broker.get_task("t3")["status"] == "FAILED"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_worker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'triton_evaluation.worker'`

- [ ] **Step 3: Implement `triton_evaluation/worker.py`**

```python
import shutil
import tempfile
import time

from .broker import Broker
from .config import Settings
from .models import JobType, TaskStatus


class Worker:
    def __init__(self, broker: Broker, registry: dict, device_id: int, settings: Settings):
        self.broker = broker
        self.registry = registry
        self.device_id = device_id
        self.settings = settings

    def run_once(self) -> bool:
        job = self.broker.dequeue()
        if job is None:
            return False
        self.broker.mark_running(job.task_id, self.device_id)
        try:
            evaluator = self.registry[job.type]
            spec = job.spec
            if job.type == JobType.BENCHMARK.value and spec.verify_task_id:
                prior = self.broker.get_task(spec.verify_task_id)
                spec.verify_result_seed = (prior or {}).get("result")
            verify_dir = tempfile.mkdtemp(prefix=f"eval_{job.task_id}_")
            try:
                res = evaluator.run(spec, verify_dir, self.device_id)
            finally:
                shutil.rmtree(verify_dir, ignore_errors=True)
            self.broker.store_result(job.task_id, res.status, res.result, res.error)
        except Exception as e:  # dispatch/setup failure — never crash the worker
            self.broker.store_result(job.task_id, TaskStatus.FAILED.value, error=repr(e))
        return True

    def serve_forever(self) -> None:
        while True:
            processed = self.run_once()
            if not processed:
                reap_stale_tasks(self.broker, self.settings)


def reap_stale_tasks(broker: Broker, settings: Settings) -> int:
    now = time.time()
    reaped = 0
    for rec in broker.iter_running_tasks():
        started = rec.get("started_at")
        if started is None:
            continue
        if now - started > settings.reaper_grace_seconds:
            broker.store_result(rec["task_id"], TaskStatus.FAILED.value, error="worker_lost")
            reaped += 1
    return reaped
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_worker.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add triton_evaluation/worker.py tests/test_worker.py
git commit -m "feat: worker loop, benchmark gate resolution, stale-task reaper"
```

---

## Task 8: FastAPI app

**Files:**
- Create: `triton_evaluation/api.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `Broker` (Task 3); `VerifyRequest`, `BenchmarkRequest`, `SubmitResponse`, `TaskResponse`, `Job`, `JobSpec`, `JobType` (Task 2).
- Produces: `create_app(broker: Broker) -> FastAPI` with routes `POST /verify`, `POST /benchmark`, `GET /tasks/{task_id}`, `GET /healthz`.

- [ ] **Step 1: Write the failing test** — `tests/test_api.py`

```python
import fakeredis
from fastapi.testclient import TestClient

from triton_evaluation.broker import Broker
from triton_evaluation.api import create_app


def _client(settings):
    client = fakeredis.FakeStrictRedis(decode_responses=True)
    broker = Broker(client, settings)
    return TestClient(create_app(broker)), broker


def test_post_verify_enqueues(settings):
    client, broker = _client(settings)
    r = client.post("/verify", json={"op_name": "s", "reference_code": "R", "kernel_code": "K"})
    assert r.status_code == 200
    task_id = r.json()["task_id"]
    rec = broker.get_task(task_id)
    assert rec["status"] == "QUEUED"
    assert rec["type"] == "verify"
    assert broker.dequeue().task_id == task_id


def test_post_benchmark_without_gate_is_422(settings):
    client, _ = _client(settings)
    r = client.post("/benchmark", json={"op_name": "s", "reference_code": "R", "kernel_code": "K"})
    assert r.status_code == 422


def test_post_benchmark_with_verify_task_id_ok(settings):
    client, broker = _client(settings)
    r = client.post("/benchmark", json={
        "op_name": "s", "reference_code": "R", "kernel_code": "K", "verify_task_id": "v1",
    })
    assert r.status_code == 200
    job = broker.dequeue()
    assert job.type == "benchmark"
    assert job.spec.verify_task_id == "v1"


def test_get_task_unknown_is_404(settings):
    client, _ = _client(settings)
    assert client.get("/tasks/nope").status_code == 404


def test_get_task_returns_record(settings):
    client, broker = _client(settings)
    task_id = client.post("/verify", json={"op_name": "s", "reference_code": "R", "kernel_code": "K"}).json()["task_id"]
    r = client.get(f"/tasks/{task_id}")
    assert r.status_code == 200
    assert r.json()["status"] == "QUEUED"


def test_healthz(settings):
    client, _ = _client(settings)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["redis"] == "ok"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'triton_evaluation.api'`

- [ ] **Step 3: Implement `triton_evaluation/api.py`**

```python
import uuid

from fastapi import FastAPI, HTTPException

from .broker import Broker
from .models import (
    VerifyRequest, BenchmarkRequest, SubmitResponse, TaskResponse,
    Job, JobSpec, JobType,
)


def create_app(broker: Broker) -> FastAPI:
    app = FastAPI(title="triton-evaluation")

    @app.post("/verify", response_model=SubmitResponse)
    def submit_verify(req: VerifyRequest):
        task_id = uuid.uuid4().hex
        spec = JobSpec(
            op_name=req.op_name, reference_code=req.reference_code, kernel_code=req.kernel_code,
            triton_impl_name=req.triton_impl_name, non_compute=req.non_compute, timeout=req.timeout,
        )
        broker.create_task(Job(task_id, JobType.VERIFY.value, spec))
        return SubmitResponse(task_id=task_id)

    @app.post("/benchmark", response_model=SubmitResponse)
    def submit_benchmark(req: BenchmarkRequest):
        task_id = uuid.uuid4().hex
        spec = JobSpec(
            op_name=req.op_name, reference_code=req.reference_code, kernel_code=req.kernel_code,
            triton_impl_name=req.triton_impl_name, warmup=req.warmup, repeats=req.repeats,
            timeout=req.timeout, verify_task_id=req.verify_task_id,
            verify_not_required=req.verify_not_required,
        )
        broker.create_task(Job(task_id, JobType.BENCHMARK.value, spec))
        return SubmitResponse(task_id=task_id)

    @app.get("/tasks/{task_id}", response_model=TaskResponse)
    def get_task(task_id: str):
        rec = broker.get_task(task_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="task not found")
        return TaskResponse(**rec)

    @app.get("/healthz")
    def healthz():
        try:
            broker.ping()
            return {"redis": "ok"}
        except Exception:
            return {"redis": "down"}

    return app
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_api.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add triton_evaluation/api.py tests/test_api.py
git commit -m "feat: FastAPI app — /verify /benchmark /tasks /healthz"
```

---

## Task 9: Entrypoints + README

**Files:**
- Create: `run_api.py`
- Create: `run_worker.py`
- Create: `README.md`
- Test: `tests/test_entrypoints.py`

**Interfaces:**
- Consumes: `Settings`, `Broker`, `create_app`, `build_registry`, `Worker`.
- Produces: `run_worker.build_worker(card: int) -> Worker` (sets `ASCEND_RT_VISIBLE_DEVICES` and constructs the Worker) — factored out so it is importable/testable without launching the loop.

- [ ] **Step 1: Write the failing test** — `tests/test_entrypoints.py`

```python
import os
import importlib


def test_build_worker_sets_visible_devices(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_EVAL_REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("TRITON_EVAL_SKILL_SCRIPTS_DIR", str(tmp_path))
    run_worker = importlib.import_module("run_worker")
    w = run_worker.build_worker(card=3)
    assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "3"
    assert w.device_id == 3
    assert "verify" in w.registry and "benchmark" in w.registry
```

Note: `build_worker` constructs a real `Broker` (no connection is made until a command runs), so this test does not require a live Redis.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_entrypoints.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'run_worker'`

- [ ] **Step 3: Implement `run_worker.py`**

```python
import argparse
import os

from triton_evaluation.config import Settings
from triton_evaluation.broker import Broker
from triton_evaluation.evaluators import build_registry
from triton_evaluation.worker import Worker


def build_worker(card: int) -> Worker:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(card)
    settings = Settings.from_env()
    broker = Broker.from_settings(settings)
    registry = build_registry(settings.skill_scripts_dir)
    return Worker(broker, registry, device_id=card, settings=settings)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", type=int, required=True, help="Physical NPU card id to pin this worker to")
    args = ap.parse_args()
    build_worker(args.card).serve_forever()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement `run_api.py`**

```python
import uvicorn

from triton_evaluation.config import Settings
from triton_evaluation.broker import Broker
from triton_evaluation.api import create_app

app = create_app(Broker.from_settings(Settings.from_env()))


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Implement `README.md`**

```markdown
# triton-evaluation

Standalone FastAPI + Redis evaluator service wrapping the `triton-op-verifier` skill.

## Run

```bash
export TRITON_EVAL_REDIS_URL=redis://localhost:6379/0
export TRITON_EVAL_SKILL_SCRIPTS_DIR=/abs/path/to/cannbot-skills/ops/triton-op-verifier/scripts

# API
python run_api.py

# One worker per physical NPU card (pins ASCEND_RT_VISIBLE_DEVICES)
python run_worker.py --card 0 &
python run_worker.py --card 1 &
# ... one per card. Supervise with systemd/docker; restart on crash.
```

## Endpoints

- `POST /verify` → `{task_id}`
- `POST /benchmark` (requires `verify_task_id` or `verify_not_required`) → `{task_id}`
- `GET /tasks/{task_id}` → status + skill result JSON
- `GET /healthz`

## Test

```bash
pip install -e ".[test]"
python -m pytest -v
```
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_entrypoints.py -v`
Expected: PASS (1 passed)

- [ ] **Step 7: Run the full suite**

Run: `python -m pytest -v`
Expected: PASS (all tasks' tests green)

- [ ] **Step 8: Commit**

```bash
git add run_api.py run_worker.py README.md tests/test_entrypoints.py
git commit -m "feat: api/worker entrypoints + README"
```

---

## Self-Review Notes

- **Spec coverage:** §3 modules → Tasks 1-8; §4 API → Task 8; §5 Redis layout → Task 3; §6 gate → Tasks 6 (seed) + 7 (resolution) + 2 (422); §7 pluggable evaluators → Tasks 4-6 (`build_registry`, `needs_bound_card`); §8 status mapping → Tasks 5-6; §9 lifecycle/reaper → Tasks 7, 9; §10 testing → every task; §11 layout → File Structure.
- **Deferred (spec §12, non-v1):** reliable queue (`BRPOPLPUSH`), profile evaluator, multi-host, auth — intentionally out of scope; `needs_bound_card` attribute is the only forward hook included.
- **Type consistency:** `Broker`, `EvaluatorResult`, `JobSpec.verify_result_seed`, `build_registry`, `verify_json_name` names are used identically across Tasks 3-9.
- **Real-skill integration note:** the fixtures `fake_verify.py` / `fake_benchmark.py` reproduce the skill CLI contract for CPU tests. The `_script_path()` seam lets the real evaluators point at `Settings.skill_scripts_dir`; the on-NPU integration test (spec §10) is the only step requiring real hardware and is not encoded as a unit task.
