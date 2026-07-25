# Profile Endpoint（msprof）Implementation Plan v2

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone asynchronous `POST /profile` endpoint that profiles every input case by default, optionally profiles one zero-based `case_idx`, and returns trustworthy `msprof op` or raw `msprof` diagnostics without changing verify or benchmark semantics.

**Architecture:** `JobType.PROFILE` is dispatched through the existing shared Redis queue to whichever configured single-card worker becomes free first; once claimed, the complete profile job remains on that card. `ProfileEvaluator` invokes a stdlib-only `engine/profile.py` orchestrator, which delegates light capture to `engine/msprof_op.py` and deep 7+1 capture/analysis to `engine/msprof_deep.py`.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic v2, Redis/fakeredis, pytest, stdlib `subprocess`/`csv`/`sqlite3`, Ascend CANN `msprof op` and `msprof`.

## Global Constraints

- The approved design is `docs/specs/2026-07-25-profile-endpoint-msprof-design.md`; this plan must not weaken it.
- Public evaluation endpoints are only `POST /verify` and `POST /profile`. Do not restore `POST /benchmark` or add `JobType.BENCHMARK`.
- Benchmark remains an internal `/verify?measure_speed=true` stage on the same card as that verify job.
- Profile and verify share the existing Redis queue. A profile job has no affinity to a previous verify job; any free configured worker may claim it.
- “Free card” means that card's service worker is idle. v1 does not poll
  `npu-smi` or coordinate with processes outside this service, so configured
  cards must be dedicated to the service.
- After a worker claims a profile job, all selected cases, warmups, and deep metric passes stay on that worker's card.
- `case_idx=None` means all cases and is the default. An integer selects one zero-based case.
- `profile_mode` is `Literal["op", "deep"]`; `profile_launch_count` is valid only for op mode.
- Profile is independent of correctness and must not read or require `verify_result.json`.
- Profile task timeout defaults to 3600 seconds and accepts 10–86400 seconds. A single profiler process is capped by `MSPROF_TIMEOUT_SEC` (default 300) and the job's remaining budget.
- Missing profiler metrics remain unknown. Never convert a missing metric to zero.
- Deep metrics from different complete `Op Name` values must never be merged.
- Bound rules and priority come from `cannbot-skills/ops/ops-profiling/references/msprof-guide.md`; PUSHQ must not be substituted with `aiv_vec_ratio`.
- Flow provenance is explicit: op mode follows
  `cannbot-skills/ops/ops-profiling/references/msprof-op-guide.md`; deep
  collection follows `scripts/msprof_profile_run.sh`; cross-metric and
  per-core analysis follows `scripts/msprof_perf_summary.py`,
  `references/csv_fields_reference.md`, and `references/msprof-guide.md`.
- Deep mode performs three non-profiled warmups, seven task-based metric runs in the documented order, then one sample-based run.
- Every generated runner calls `enforce_conv_fp32()` before the first warmup or kernel launch.
- New engine modules use only the Python standard library at module import time. `torch` and `torch_npu` are imported only by the generated runner subprocess.
- Ported CANN/cannbot code retains the CANN Open Software License v2 header and a source-provenance module comment.
- Every new engine module starts with this exact repository license header:

```python
#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
```

- Do not hardcode 24/48 core counts. Per-core output reports observed IDs and labels run-wide data as `scope="profile_run"`.
- A normal orchestration with failed kernel cases is `SUCCEEDED` with per-case errors. Infrastructure/input failure is `FAILED`. Deadline exhaustion is `TIMEOUT` with the latest valid snapshot.
- Use TDD. Run the named failing test before each implementation step and commit after every task.
- Real NPU completion cannot be claimed until the final `msprof` smoke passes in a CANN-enabled environment.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `triton_evaluation/models.py` | Profile request/job fields and validation | Modify |
| `triton_evaluation/api/routes.py` | `POST /profile` submission | Modify |
| `triton_evaluation/evaluators/profile.py` | Engine subprocess adapter and status mapping | Create |
| `triton_evaluation/evaluators/__init__.py` | Register `"profile"` only | Modify |
| `triton_evaluation/engine/profile.py` | Runner source, discovery, multi-case orchestration, deadline, atomic snapshots, CLI | Create |
| `triton_evaluation/engine/msprof_op.py` | Shared process helper, kernel extraction, op-mode capture/parser | Create |
| `triton_evaluation/engine/msprof_deep.py` | Deep capture, stable target merge, Bound, per-core, recommendations | Create |
| `tests/fixtures/fake_profile.py` | ProfileEvaluator engine double | Create |
| `tests/fixtures/fake_msprof.py` | Executable profiler double that emits CSV/SQLite | Create |
| `tests/test_models.py` | Profile model validation/round-trip | Modify |
| `tests/test_api.py` | Profile enqueue and HTTP 422 | Modify |
| `tests/test_evaluators.py` | ProfileEvaluator and registry behavior | Modify |
| `tests/test_worker.py` | Any free worker may claim profile; bound card is passed through | Modify |
| `tests/test_entrypoints.py` | Worker registry contains verify/profile, not benchmark | Modify |
| `tests/test_profile_op.py` | op parser, CLI, process, recommendations | Create |
| `tests/test_profile_runner.py` | Runner discovery, case execution, HF32 ordering | Create |
| `tests/test_profile_deep.py` | 7+1 capture, stable merge, Bound, SQLite | Create |
| `tests/test_profile_main.py` | Full/single orchestration, snapshots, error and timeout semantics | Create |
| `tests/test_profile_fake_msprof.py` | Backend subprocess integration without NPU | Create |
| `tests/msprof_smoke_kernels.py` | Known-good single/multi-case Triton fixture | Create |
| `tests/test_profile_npu_integration.py` | Opt-in real NPU smoke | Create |
| `.env.example` | msprof environment | Modify |
| `README.md` | Public endpoint and examples | Modify |
| `triton_evaluation/config.py` | Correct bundled-engine description | Modify |
| `triton_evaluation/engine/__init__.py` | Correct engine capability description | Modify |

---

### Task 1: Profile data model and `POST /profile`

**Files:**
- Modify: `triton_evaluation/models.py`
- Modify: `triton_evaluation/api/routes.py`
- Modify: `tests/test_models.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Produces: `JobType.PROFILE == "profile"`.
- Produces: `JobSpec.profile_mode`, `case_idx`, and `profile_launch_count`.
- Produces: `ProfileRequest` with conditional Pydantic validation.
- Produces: `POST /profile -> SubmitResponse`.

- [ ] **Step 1: Write failing model tests**

Add `ProfileRequest` to the import in `tests/test_models.py`, then append:

```python
import pytest
from pydantic import ValidationError


def test_profile_job_roundtrip_carries_profile_fields():
    spec = JobSpec(
        op_name="softmax",
        reference_code="R",
        kernel_code="K",
        timeout=7200,
        profile_mode="deep",
        case_idx=2,
    )
    job = Job(task_id="p1", type=JobType.PROFILE.value, spec=spec)
    back = Job.from_json(job.to_json())
    assert back.type == "profile"
    assert back.spec.timeout == 7200
    assert back.spec.profile_mode == "deep"
    assert back.spec.case_idx == 2
    assert back.spec.profile_launch_count is None


def test_profile_request_defaults_to_all_cases_in_op_mode():
    req = ProfileRequest(op_name="softmax", reference_code="R", kernel_code="K")
    assert req.profile_mode == "op"
    assert req.case_idx is None
    assert req.profile_launch_count is None
    assert req.timeout == 3600


@pytest.mark.parametrize("mode", ["bad", "", "OP"])
def test_profile_request_rejects_unknown_mode(mode):
    with pytest.raises(ValidationError):
        ProfileRequest(
            op_name="softmax",
            reference_code="R",
            kernel_code="K",
            profile_mode=mode,
        )


def test_profile_request_rejects_launch_count_in_deep_mode():
    with pytest.raises(ValidationError, match="profile_launch_count is only valid"):
        ProfileRequest(
            op_name="softmax",
            reference_code="R",
            kernel_code="K",
            profile_mode="deep",
            profile_launch_count=2,
        )
```

- [ ] **Step 2: Run the model tests and verify RED**

Run:

```bash
python -m pytest tests/test_models.py -q
```

Expected: collection fails because `ProfileRequest` and `JobType.PROFILE` do not exist.

- [ ] **Step 3: Implement the model contract**

In `triton_evaluation/models.py`, extend imports and definitions exactly as follows:

```python
from typing import Literal

from pydantic import BaseModel, Field, model_validator


ProfileMode = Literal["op", "deep"]


class JobType(str, Enum):
    VERIFY = "verify"
    PROFILE = "profile"
```

Append these fields to `JobSpec` after `repeats`:

```python
    # standalone profile job
    profile_mode: ProfileMode = "op"
    case_idx: int | None = None
    profile_launch_count: int | None = None
```

Add this model after `VerifyRequest`:

```python
class ProfileRequest(BaseModel):
    op_name: str
    reference_code: str
    kernel_code: str
    triton_impl_name: str = "triton_ascend_impl"
    profile_mode: ProfileMode = "op"
    case_idx: int | None = Field(default=None, ge=0)
    profile_launch_count: int | None = Field(default=None, ge=1, le=100)
    timeout: int = Field(default=3600, ge=10, le=86400)

    @model_validator(mode="after")
    def validate_mode_specific_fields(self):
        if self.profile_mode == "deep" and self.profile_launch_count is not None:
            raise ValueError("profile_launch_count is only valid in op mode")
        return self
```

- [ ] **Step 4: Verify the model tests are GREEN**

Run:

```bash
python -m pytest tests/test_models.py -q
```

Expected: all model tests pass.

- [ ] **Step 5: Write failing API tests**

Add `ProfileRequest` handling expectations to `tests/test_api.py`:

```python
def test_post_profile_defaults_to_all_cases(settings):
    client, broker = _client(settings)
    response = client.post(
        "/profile",
        json={"op_name": "softmax", "reference_code": "R", "kernel_code": "K"},
    )
    assert response.status_code == 200
    job = broker.dequeue()
    assert job.type == "profile"
    assert job.spec.profile_mode == "op"
    assert job.spec.case_idx is None
    assert job.spec.timeout == 3600


def test_post_profile_accepts_one_deep_case(settings):
    client, broker = _client(settings)
    response = client.post(
        "/profile",
        json={
            "op_name": "softmax",
            "reference_code": "R",
            "kernel_code": "K",
            "profile_mode": "deep",
            "case_idx": 3,
            "timeout": 7200,
        },
    )
    assert response.status_code == 200
    job = broker.dequeue()
    assert job.spec.profile_mode == "deep"
    assert job.spec.case_idx == 3
    assert job.spec.timeout == 7200


def test_post_profile_rejects_deep_launch_count(settings):
    client, broker = _client(settings)
    response = client.post(
        "/profile",
        json={
            "op_name": "softmax",
            "reference_code": "R",
            "kernel_code": "K",
            "profile_mode": "deep",
            "profile_launch_count": 2,
        },
    )
    assert response.status_code == 422
    assert broker.dequeue() is None
```

- [ ] **Step 6: Run the API tests and verify RED**

Run:

```bash
python -m pytest tests/test_api.py -q
```

Expected: `/profile` returns 404.

- [ ] **Step 7: Implement the route**

Import `ProfileRequest` in `triton_evaluation/api/routes.py`, then add:

```python
@router.post("/profile", response_model=SubmitResponse)
def submit_profile(req: ProfileRequest, broker: Broker = Depends(get_broker)):
    task_id = uuid.uuid4().hex
    spec = JobSpec(
        op_name=req.op_name,
        reference_code=req.reference_code,
        kernel_code=req.kernel_code,
        triton_impl_name=req.triton_impl_name,
        timeout=req.timeout,
        profile_mode=req.profile_mode,
        case_idx=req.case_idx,
        profile_launch_count=req.profile_launch_count,
    )
    broker.create_task(Job(task_id, JobType.PROFILE.value, spec))
    return SubmitResponse(task_id=task_id)
```

- [ ] **Step 8: Verify Task 1**

Run:

```bash
python -m pytest tests/test_models.py tests/test_api.py -q
```

Expected: all tests pass, including the existing assertion that `/benchmark` returns 404.

- [ ] **Step 9: Commit**

```bash
git add triton_evaluation/models.py triton_evaluation/api/routes.py tests/test_models.py tests/test_api.py
git commit -m "feat(api): add standalone profile job"
```

---

### Task 2: `ProfileEvaluator`, registry, and shared-queue scheduling

**Files:**
- Create: `triton_evaluation/evaluators/profile.py`
- Create: `tests/fixtures/fake_profile.py`
- Modify: `triton_evaluation/evaluators/__init__.py`
- Modify: `tests/test_evaluators.py`
- Modify: `tests/test_worker.py`
- Modify: `tests/test_entrypoints.py`

**Interfaces:**
- Consumes: `JobSpec` profile fields from Task 1.
- Produces: `ProfileEvaluator.run(spec, verify_dir, device_id) -> EvaluatorResult`.
- Produces: registry keys exactly `{"verify", "profile"}`.
- Maps engine exit `124` or outer wall-clock expiry to `TIMEOUT`, preserving a valid JSON snapshot.

- [ ] **Step 1: Create the fake engine fixture**

Create `tests/fixtures/fake_profile.py`:

```python
import argparse
import json
import os


parser = argparse.ArgumentParser()
parser.add_argument("--op_name", required=True)
parser.add_argument("--verify_dir", required=True)
parser.add_argument("--triton_impl_name", required=True)
parser.add_argument("--profile-mode", required=True)
parser.add_argument("--case-idx", type=int)
parser.add_argument("--launch-count", type=int)
parser.add_argument("--timeout", type=int, required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

payload = {
    "profile_mode": args.profile_mode,
    "requested_case_idx": args.case_idx,
    "total_cases": 2,
    "selected_cases": 1 if args.case_idx is not None else 2,
    "completed_cases": 1 if args.case_idx is not None else 2,
    "partial_cases": 0,
    "failed_cases": 0,
    "remaining_case_indices": [],
    "timed_out": False,
    "summary_text": "fake profile",
    "warnings": [],
    "wallclock_sec": 0.01,
    "per_case_results": [],
    "visible_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
}
with open(args.output, "w", encoding="utf-8") as handle:
    if os.environ.get("FAKE_PROFILE_BAD_JSON") == "1":
        handle.write("{")
    else:
        json.dump(payload, handle)

raise SystemExit(int(os.environ.get("FAKE_PROFILE_EXIT", "0")))
```

- [ ] **Step 2: Write failing evaluator and registry tests**

Add these imports and tests to `tests/test_evaluators.py`:

```python
from triton_evaluation.evaluators import profile as profile_mod
from triton_evaluation.evaluators.profile import ProfileEvaluator


class _ProfileHarness(ProfileEvaluator):
    def _script_path(self):
        return str(FIXTURES / "fake_profile.py")


def _profile_spec(**overrides):
    values = {
        "op_name": "softmax",
        "reference_code": "R",
        "kernel_code": "K",
        "timeout": 10,
        "profile_mode": "op",
        "case_idx": None,
        "profile_launch_count": None,
    }
    values.update(overrides)
    return JobSpec(**values)


def test_profile_evaluator_returns_direct_result(tmp_path, monkeypatch):
    monkeypatch.setenv("MSPROF_BIN", sys.executable)
    result = _ProfileHarness("unused").run(_profile_spec(), str(tmp_path), device_id=5)
    assert result.status == "SUCCEEDED"
    assert result.result["profile_mode"] == "op"
    assert result.result["selected_cases"] == 2
    assert result.result["visible_device"] == "5"


def test_profile_evaluator_passes_single_case_and_launch_count(tmp_path, monkeypatch):
    monkeypatch.setenv("MSPROF_BIN", sys.executable)
    result = _ProfileHarness("unused").run(
        _profile_spec(case_idx=1, profile_launch_count=3),
        str(tmp_path),
        device_id=2,
    )
    assert result.status == "SUCCEEDED"
    assert result.result["requested_case_idx"] == 1
    assert result.result["selected_cases"] == 1


def test_profile_evaluator_missing_msprof_is_failed(tmp_path, monkeypatch):
    monkeypatch.setenv("MSPROF_BIN", str(tmp_path / "missing-msprof"))
    result = _ProfileHarness("unused").run(_profile_spec(), str(tmp_path), device_id=0)
    assert result.status == "FAILED"
    assert "MSPROF_BIN is not executable" in result.error


def test_profile_evaluator_rejects_invalid_pass_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("MSPROF_BIN", sys.executable)
    monkeypatch.setenv("MSPROF_TIMEOUT_SEC", "zero")
    result = _ProfileHarness("unused").run(_profile_spec(), str(tmp_path), device_id=0)
    assert result.status == "FAILED"
    assert "MSPROF_TIMEOUT_SEC must be a positive integer" in result.error


def test_profile_evaluator_nonzero_exit_keeps_valid_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("MSPROF_BIN", sys.executable)
    monkeypatch.setenv("FAKE_PROFILE_EXIT", "2")
    result = _ProfileHarness("unused").run(_profile_spec(), str(tmp_path), device_id=0)
    assert result.status == "FAILED"
    assert result.result["summary_text"] == "fake profile"


def test_profile_evaluator_rejects_invalid_result_json(tmp_path, monkeypatch):
    monkeypatch.setenv("MSPROF_BIN", sys.executable)
    monkeypatch.setenv("FAKE_PROFILE_BAD_JSON", "1")
    result = _ProfileHarness("unused").run(_profile_spec(), str(tmp_path), device_id=0)
    assert result.status == "FAILED"
    assert result.result is None


def test_profile_timeout_keeps_last_snapshot(tmp_path, monkeypatch):
    snapshot = {
        "profile_mode": "deep",
        "timed_out": False,
        "per_case_results": [{"case_idx": 0, "status": "complete"}],
    }
    monkeypatch.setenv("MSPROF_BIN", sys.executable)
    monkeypatch.setattr(
        profile_mod,
        "run_script",
        lambda *args, **kwargs: ScriptOutcome(124, "", "deadline", False),
    )
    monkeypatch.setattr(profile_mod, "read_json", lambda path: dict(snapshot))
    result = ProfileEvaluator("unused").run(
        _profile_spec(profile_mode="deep"),
        str(tmp_path),
        device_id=0,
    )
    assert result.status == "TIMEOUT"
    assert result.result["timed_out"] is True
    assert result.result["per_case_results"][0]["case_idx"] == 0


def test_profile_timeout_without_snapshot_is_still_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("MSPROF_BIN", sys.executable)
    monkeypatch.setattr(
        profile_mod,
        "run_script",
        lambda *args, **kwargs: ScriptOutcome(-1, "", "", True),
    )
    monkeypatch.setattr(profile_mod, "read_json", lambda path: None)
    result = ProfileEvaluator("unused").run(
        _profile_spec(profile_mode="deep"),
        str(tmp_path),
        device_id=0,
    )
    assert result.status == "TIMEOUT"
    assert result.result is None


def test_registry_has_verify_and_profile_only(tmp_path):
    registry = build_registry(str(tmp_path))
    assert set(registry) == {"verify", "profile"}
    assert registry["profile"].needs_bound_card is True
```

Replace the old `test_registry_has_only_verify` instead of keeping two contradictory registry tests.

- [ ] **Step 3: Run the evaluator tests and verify RED**

Run:

```bash
python -m pytest tests/test_evaluators.py -q
```

Expected: import fails because `evaluators/profile.py` does not exist.

- [ ] **Step 4: Implement `ProfileEvaluator`**

Create `triton_evaluation/evaluators/profile.py`:

```python
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
```

Update `triton_evaluation/evaluators/__init__.py`:

```python
from .profile import ProfileEvaluator
from .verify import VerifyEvaluator


def build_registry(engine_dir: str) -> dict:
    return {
        "verify": VerifyEvaluator(engine_dir),
        "profile": ProfileEvaluator(engine_dir),
    }
```

- [ ] **Step 5: Verify evaluator tests are GREEN**

Run:

```bash
python -m pytest tests/test_evaluators.py -q
```

Expected: all evaluator tests pass.

- [ ] **Step 6: Write the worker scheduling regression**

Append to `tests/test_worker.py`:

```python
def test_profile_job_runs_on_whichever_worker_dequeues_it(broker, settings):
    evaluator = FakeEvaluator()
    spec = JobSpec(
        op_name="softmax",
        reference_code="R",
        kernel_code="K",
        profile_mode="deep",
    )
    broker.create_task(Job("profile-1", JobType.PROFILE.value, spec))
    worker = Worker(broker, {"profile": evaluator}, device_id=7, settings=settings)
    assert worker.run_once() is True
    record = broker.get_task("profile-1")
    assert record["device_id"] == 7
    assert evaluator.seen[0][2] == 7
```

Update `tests/test_entrypoints.py`:

```python
    assert set(w.registry) == {"verify", "profile"}
    assert "benchmark" not in w.registry
```

- [ ] **Step 7: Verify queue/entrypoint tests**

Run:

```bash
python -m pytest tests/test_worker.py tests/test_entrypoints.py -q
```

Expected: all tests pass. No worker implementation change is required because the queue is already shared and workers already pass their bound `device_id`.

- [ ] **Step 8: Commit**

```bash
git add triton_evaluation/evaluators/profile.py triton_evaluation/evaluators/__init__.py tests/fixtures/fake_profile.py tests/test_evaluators.py tests/test_worker.py tests/test_entrypoints.py
git commit -m "feat(evaluator): dispatch profile jobs on free workers"
```

---

### Task 3: op-mode process helper, kernel targeting, and parser

**Files:**
- Create: `triton_evaluation/engine/msprof_op.py`
- Create: `tests/test_profile_op.py`

**Interfaces:**
- Produces: `CommandOutcome`.
- Produces: `run_command(cmd, cwd, env, timeout)`.
- Produces: `remaining_pass_timeout(deadline)`.
- Produces: `extract_triton_kernel_names(source)`.
- Produces: `parse_op_output(output_dir, kernel_names)`.
- Produces: `profile_op_case(app_cmd, kernel_names, output_dir, launch_count, cwd, env, deadline)`.

- [ ] **Step 1: Write parser and targeting tests**

Create `tests/test_profile_op.py`:

```python
import csv
import sys
from pathlib import Path

from triton_evaluation.engine import msprof_op


KERNEL_SOURCE = """
import triton

@triton.jit
def first_kernel(x):
    return x

@triton.jit()
def second_kernel(x):
    return x
"""


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_op_dir(root: Path, name: str, duration: float, include_l2: bool = True):
    op_dir = root / "OPPROF_fake" / name / "0"
    _write_csv(
        op_dir / "OpBasicInfo.csv",
        ["Op Name", "Op Type", "Task Duration(us)", "Block Dim"],
        [{"Op Name": name, "Op Type": "vector", "Task Duration(us)": duration, "Block Dim": 8}],
    )
    _write_csv(
        op_dir / "PipeUtilization.csv",
        ["aiv_vec_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio"],
        [{"aiv_vec_ratio": 0.6, "aiv_mte2_ratio": 0.3, "aiv_mte3_ratio": 0.1}],
    )
    _write_csv(
        op_dir / "Memory.csv",
        ["GM_to_UB_bw_usage_rate(%)", "UB_to_GM_bw_usage_rate(%)"],
        [{"GM_to_UB_bw_usage_rate(%)": 45, "UB_to_GM_bw_usage_rate(%)": 20}],
    )
    if include_l2:
        _write_csv(
            op_dir / "L2Cache.csv",
            ["aiv_total_hit_rate(%)"],
            [{"aiv_total_hit_rate(%)": 91}],
        )


def test_extract_triton_kernel_names_is_ordered_and_unique():
    assert msprof_op.extract_triton_kernel_names(KERNEL_SOURCE) == [
        "first_kernel",
        "second_kernel",
    ]


def test_parse_op_output_selects_slowest_matching_custom_kernel(tmp_path):
    _write_op_dir(tmp_path, "first_kernel_hash", 3.0)
    _write_op_dir(tmp_path, "second_kernel_hash", 8.0, include_l2=False)
    _write_op_dir(tmp_path, "unrelated_torch_op", 99.0)
    parsed = msprof_op.parse_op_output(
        str(tmp_path),
        ["first_kernel", "second_kernel"],
    )
    assert parsed["target_kernel"] == "second_kernel_hash"
    assert [kernel["name"] for kernel in parsed["kernels"]] == [
        "second_kernel_hash",
        "first_kernel_hash",
    ]
    assert parsed["status"] == "partial"
    assert "L2Cache.csv" in " ".join(parsed["warnings"])
    assert parsed["missing_metrics"] == ["L2Cache.csv"]


def test_missing_custom_kernel_is_failed(tmp_path):
    _write_op_dir(tmp_path, "unrelated_torch_op", 99.0)
    parsed = msprof_op.parse_op_output(str(tmp_path), ["first_kernel"])
    assert parsed["status"] == "failed"
    assert parsed["target_kernel"] is None


def test_run_command_timeout_terminates_the_process(tmp_path):
    script = tmp_path / "sleep.py"
    script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    outcome = msprof_op.run_command(
        [sys.executable, str(script)],
        cwd=str(tmp_path),
        env={},
        timeout=1,
    )
    assert outcome.timed_out is True
    assert outcome.exit_code == -1


def test_malformed_profiler_directory_does_not_hide_good_kernel(tmp_path):
    _write_op_dir(tmp_path, "first_kernel_hash", 3.0)
    bad = tmp_path / "OPPROF_fake" / "broken" / "0"
    bad.mkdir(parents=True)
    (bad / "OpBasicInfo.csv").write_bytes(b"\xff\xfe")
    parsed = msprof_op.parse_op_output(str(tmp_path), ["first_kernel"])
    assert parsed["status"] == "partial"
    assert parsed["target_kernel"] == "first_kernel_hash"
    assert "cannot parse" in " ".join(parsed["warnings"])
```

- [ ] **Step 2: Run parser tests and verify RED**

Run:

```bash
python -m pytest tests/test_profile_op.py -q
```

Expected: import fails because `engine/msprof_op.py` does not exist.

- [ ] **Step 3: Implement process, extraction, and parsing**

Create `triton_evaluation/engine/msprof_op.py` with the same CANN v2 header used by `_common_utils.py`, followed by:

```python
"""Service-adapted msprof-op capture.

Source flow: cannbot-skills/ops/ops-profiling/references/msprof-op-guide.md.
The service adaptation adds a bounded process runner, Triton kernel-name
filtering, recursive OpBasicInfo discovery, and structured JSON-ready output.
"""

from __future__ import annotations

import csv
import glob
import math
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Any


_JIT_DEF_RE = re.compile(
    r"^[ \t]*@(?:triton\.)?jit\b[^\n]*\n"
    r"(?:^[ \t]*@[^\n]*\n)*"
    r"^[ \t]*def\s+([A-Za-z_]\w*)\s*\(",
    re.MULTILINE,
)


@dataclass
class CommandOutcome:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


def run_command(
    cmd: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    timeout: int,
) -> CommandOutcome:
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return CommandOutcome(process.returncode, stdout or "", stderr or "")
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        stdout, stderr = process.communicate()
        return CommandOutcome(-1, stdout or "", stderr or "", timed_out=True)


def remaining_pass_timeout(deadline: float) -> int:
    raw = os.environ.get("MSPROF_TIMEOUT_SEC", "300")
    try:
        configured = int(raw)
    except ValueError as exc:
        raise ValueError("MSPROF_TIMEOUT_SEC must be a positive integer") from exc
    if configured <= 0:
        raise ValueError("MSPROF_TIMEOUT_SEC must be a positive integer")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("profile job deadline exceeded")
    return max(1, min(configured, math.ceil(remaining)))


def extract_triton_kernel_names(source: str) -> list[str]:
    return list(dict.fromkeys(match.group(1) for match in _JIT_DEF_RE.finditer(source)))


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.upper() in {"NA", "N/A"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _read_rows(path: str) -> list[dict[str, str]]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _column_mean(rows: list[dict[str, str]], name: str) -> float | None:
    values = [_to_float(row.get(name)) for row in rows]
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _matches_custom(name: str, kernel_names: list[str]) -> bool:
    return any(name == kernel or name.startswith(kernel) for kernel in kernel_names)


def _parse_kernel_dir(op_dir: str) -> tuple[dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    missing_metrics: list[str] = []
    basic = _read_rows(os.path.join(op_dir, "OpBasicInfo.csv"))
    if not basic:
        return None, ["missing OpBasicInfo.csv"]
    row = basic[0]
    name = (row.get("Op Name") or "").strip()
    if not name:
        return None, ["OpBasicInfo.csv has no Op Name"]

    pipe_path = os.path.join(op_dir, "PipeUtilization.csv")
    memory_path = os.path.join(op_dir, "Memory.csv")
    l2_path = os.path.join(op_dir, "L2Cache.csv")
    for path in (pipe_path, memory_path, l2_path):
        if not os.path.isfile(path):
            missing = os.path.basename(path)
            missing_metrics.append(missing)
            warnings.append(f"{name}: missing {missing}")

    pipe = _read_rows(pipe_path)
    memory = _read_rows(memory_path)
    l2 = _read_rows(l2_path)
    metrics = {
        "task_duration_us": _to_float(row.get("Task Duration(us)")),
        "block_dim": _to_float(row.get("Block Dim")),
        "aic_cube_ratio": _column_mean(pipe, "aic_cube_ratio"),
        "aic_mte2_ratio": _column_mean(pipe, "aic_mte2_ratio"),
        "aic_fixpipe_ratio": _column_mean(pipe, "aic_fixpipe_ratio"),
        "aiv_vec_ratio": _column_mean(pipe, "aiv_vec_ratio"),
        "aiv_mte2_ratio": _column_mean(pipe, "aiv_mte2_ratio"),
        "aiv_mte3_ratio": _column_mean(pipe, "aiv_mte3_ratio"),
        "gm_to_l1_usage": _column_mean(memory, "GM_to_L1_bw_usage_rate(%)"),
        "l0c_to_gm_usage": _column_mean(memory, "L0C_to_GM_bw_usage_rate(%)"),
        "gm_to_ub_usage": _column_mean(memory, "GM_to_UB_bw_usage_rate(%)"),
        "ub_to_gm_usage": _column_mean(memory, "UB_to_GM_bw_usage_rate(%)"),
        "aic_l2_hit": _column_mean(l2, "aic_total_hit_rate(%)"),
        "aiv_l2_hit": _column_mean(l2, "aiv_total_hit_rate(%)"),
    }
    if metrics["task_duration_us"] is None:
        missing_metrics.append("OpBasicInfo.Task Duration(us)")
        warnings.append(f"{name}: Task Duration(us) is missing or invalid")
    return {
        "name": name,
        "op_type": (row.get("Op Type") or "").strip(),
        "duration_us": metrics["task_duration_us"],
        "metrics": metrics,
        "missing_metrics": missing_metrics,
    }, warnings


def _op_recommendations(target: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = target["metrics"]
    compute = metrics.get("aiv_vec_ratio")
    load = metrics.get("aiv_mte2_ratio")
    recommendations: list[dict[str, Any]] = []
    if compute is not None and load is not None and load > compute:
        recommendations.append({
            "bound": "op-mode memory-pipe dominant",
            "evidence": {"aiv_mte2_ratio": load, "aiv_vec_ratio": compute},
            "recommendation": "Reduce GM round trips or fuse adjacent elementwise work.",
            "source": "msprof-op-guide.md",
        })
    if load is not None and load < 0.60:
        recommendations.append({
            "bound": "op-mode low MTE2 activity",
            "evidence": {"aiv_mte2_ratio": load},
            "recommendation": "Evaluate multibuffer=True and re-profile; keep it only if measured latency improves.",
            "source": "msprof-op-guide.md",
        })
    return recommendations


def parse_op_output(output_dir: str, kernel_names: list[str]) -> dict[str, Any]:
    kernels: list[dict[str, Any]] = []
    warnings: list[str] = []
    basic_files = sorted(glob.glob(
        os.path.join(output_dir, "**", "OpBasicInfo.csv"),
        recursive=True,
    ))
    for basic_file in basic_files:
        try:
            parsed, local_warnings = _parse_kernel_dir(os.path.dirname(basic_file))
        except (OSError, UnicodeError, csv.Error) as exc:
            warnings.append(
                f"{basic_file}: cannot parse profiler CSV: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        if parsed is not None and _matches_custom(parsed["name"], kernel_names):
            kernels.append(parsed)
            warnings.extend(local_warnings)
    kernels.sort(
        key=lambda item: (
            item["duration_us"] is not None,
            item["duration_us"] or 0.0,
        ),
        reverse=True,
    )
    selectable = [item for item in kernels if item["duration_us"] is not None]
    if not selectable:
        return {
            "status": "failed",
            "target_kernel": None,
            "target_selection_metric": "OpBasicInfo",
            "kernels": [],
            "raw_metrics": {},
            "bound": None,
            "per_core": None,
            "recommendations": [],
            "missing_metrics": [],
            "warnings": warnings,
            "error": "no matching custom Triton kernel with a valid duration was parsed",
        }
    target = selectable[0]
    return {
        "status": "partial" if warnings else "complete",
        "target_kernel": target["name"],
        "target_selection_metric": "OpBasicInfo",
        "kernels": kernels,
        "raw_metrics": target["metrics"],
        "bound": None,
        "per_core": None,
        "recommendations": _op_recommendations(target),
        "missing_metrics": target["missing_metrics"],
        "warnings": warnings,
        "error": None,
    }
```

- [ ] **Step 4: Verify parser tests are GREEN**

Run:

```bash
python -m pytest tests/test_profile_op.py -q
```

Expected: all current op tests pass.

- [ ] **Step 5: Write the failing capture-command test**

Append:

```python
def test_profile_op_case_builds_equals_joined_cli(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, *, cwd, env, timeout):
        seen["cmd"] = cmd
        seen["timeout"] = timeout
        _write_op_dir(Path(cmd[2].split("=", 1)[1]), "first_kernel_hash", 4.0)
        return msprof_op.CommandOutcome(0, "", "")

    monkeypatch.setattr(msprof_op, "run_command", fake_run)
    result = msprof_op.profile_op_case(
        app_cmd=["python", "runner.py", "--case-idx", "0"],
        kernel_names=["first_kernel"],
        output_dir=str(tmp_path / "capture"),
        launch_count=3,
        cwd=str(tmp_path),
        env={},
        deadline=msprof_op.time.monotonic() + 30,
    )
    assert seen["cmd"][:4] == [
        "msprof",
        "op",
        f"--output={tmp_path / 'capture'}",
        "--launch-count=3",
    ]
    assert "--kernel-name=first_kernel" in seen["cmd"]
    assert result["status"] == "complete"


def test_profile_op_case_pass_timeout_fails_only_that_case(tmp_path, monkeypatch):
    monkeypatch.setattr(
        msprof_op,
        "run_command",
        lambda *args, **kwargs: msprof_op.CommandOutcome(
            -1, "", "", timed_out=True
        ),
    )
    result = msprof_op.profile_op_case(
        app_cmd=["python", "runner.py", "--case-idx", "0"],
        kernel_names=["first_kernel"],
        output_dir=str(tmp_path / "capture"),
        launch_count=None,
        cwd=str(tmp_path),
        env={},
        deadline=msprof_op.time.monotonic() + 30,
    )
    assert result["status"] == "failed"
    assert "per-process timeout" in result["error"]
```

- [ ] **Step 6: Implement `profile_op_case`**

Append to `msprof_op.py`:

```python
def profile_op_case(
    *,
    app_cmd: list[str],
    kernel_names: list[str],
    output_dir: str,
    launch_count: int | None,
    cwd: str,
    env: dict[str, str],
    deadline: float,
) -> dict[str, Any]:
    msprof_bin = os.environ.get("MSPROF_BIN", "msprof")
    command = [msprof_bin, "op", f"--output={output_dir}"]
    if launch_count is not None:
        command.append(f"--launch-count={launch_count}")
    command.append(f"--kernel-name={'|'.join(kernel_names)}")
    command.extend(app_cmd)
    outcome = run_command(
        command,
        cwd=cwd,
        env=env,
        timeout=remaining_pass_timeout(deadline),
    )
    if time.monotonic() >= deadline:
        raise TimeoutError("profile job deadline exceeded")
    if outcome.timed_out:
        return {
            "status": "failed",
            "target_kernel": None,
            "target_selection_metric": "OpBasicInfo",
            "kernels": [],
            "raw_metrics": {},
            "bound": None,
            "per_core": None,
            "recommendations": [],
            "missing_metrics": [],
            "warnings": [],
            "error": "msprof op exceeded its per-process timeout",
        }
    if outcome.exit_code != 0:
        return {
            "status": "failed",
            "target_kernel": None,
            "target_selection_metric": "OpBasicInfo",
            "kernels": [],
            "raw_metrics": {},
            "bound": None,
            "per_core": None,
            "recommendations": [],
            "missing_metrics": [],
            "warnings": [],
            "error": f"msprof op exit={outcome.exit_code}; stderr-tail={outcome.stderr[-1000:]}",
        }
    return parse_op_output(output_dir, kernel_names)
```

- [ ] **Step 7: Verify Task 3 and commit**

Run:

```bash
python -m pytest tests/test_profile_op.py -q
python -m py_compile triton_evaluation/engine/msprof_op.py
```

Expected: all tests pass and compilation succeeds.

```bash
git add triton_evaluation/engine/msprof_op.py tests/test_profile_op.py
git commit -m "feat(profile): add msprof op backend"
```

---

### Task 4: Generated runner and deterministic case discovery

**Files:**
- Create: `triton_evaluation/engine/profile.py`
- Create: `tests/test_profile_runner.py`

**Interfaces:**
- Consumes: `CommandOutcome`, `run_command`, `remaining_pass_timeout` from Task 3.
- Produces: `build_runner_source() -> str`.
- Produces: `write_runner(directory) -> str`.
- Produces: `discover_cases(runner_path, engine_dir, verify_dir, op_name, impl_name, python_bin, env, deadline) -> list[dict]`.
- Runner CLI: `--describe-cases` or `--case-idx=N`.

- [ ] **Step 1: Write failing runner tests**

Create `tests/test_profile_runner.py`:

```python
import os
import sys
from pathlib import Path

import pytest

from triton_evaluation.engine import profile


def _write_fake_runtime(root: Path):
    (root / "torch.py").write_text(
        "class Tensor:\n"
        "    def __init__(self, value=None): self.value = value; self.device = None\n"
        "    def to(self, device): self.device = device; return self\n"
        "def manual_seed(value): pass\n"
        "class _Npu:\n"
        "    def manual_seed_all(self, value): pass\n"
        "    def synchronize(self): pass\n"
        "npu = _Npu()\n"
        "class no_grad:\n"
        "    def __enter__(self): return self\n"
        "    def __exit__(self, *args): return False\n",
        encoding="utf-8",
    )
    (root / "_common_utils.py").write_text(
        "import os\n"
        "def enforce_conv_fp32():\n"
        "    marker = os.environ.get('HF32_MARKER')\n"
        "    if marker:\n"
        "        open(marker, 'w').write('off')\n"
        "def describe_input(inputs):\n"
        "    def safe(value):\n"
        "        return value if isinstance(value, (int, float, str, bool)) else repr(value)\n"
        "    return [{'type': 'scalar', 'value': safe(value)} for value in inputs]\n",
        encoding="utf-8",
    )


def _write_task(root: Path):
    (root / "sample_torch.py").write_text(
        "def get_init_inputs(): return []\n"
        "def get_input_groups():\n"
        "    yield [11]\n"
        "    yield [22]\n",
        encoding="utf-8",
    )
    (root / "sample_triton_ascend_impl.py").write_text(
        "import os\n"
        "class ModelNew:\n"
        "    def __init__(self): pass\n"
        "    def to(self, device): return self\n"
        "    def __call__(self, value):\n"
        "        marker = os.environ['KERNEL_MARKER']\n"
        "        assert open(os.environ['HF32_MARKER']).read() == 'off'\n"
        "        open(marker, 'w').write(str(value))\n",
        encoding="utf-8",
    )


def test_runner_discovers_generator_cases_without_running_kernel(tmp_path):
    _write_fake_runtime(tmp_path)
    _write_task(tmp_path)
    runner = profile.write_runner(str(tmp_path))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path)
    cases = profile.discover_cases(
        runner_path=runner,
        engine_dir=str(tmp_path),
        verify_dir=str(tmp_path),
        op_name="sample",
        impl_name="triton_ascend_impl",
        python_bin=sys.executable,
        env=env,
        deadline=profile.time.monotonic() + 30,
    )
    assert [case["case_idx"] for case in cases] == [0, 1]
    assert cases[1]["input_desc"][0]["value"] == 22


def test_runner_falls_back_to_single_get_inputs_case(tmp_path):
    _write_fake_runtime(tmp_path)
    _write_task(tmp_path)
    (tmp_path / "sample_torch.py").write_text(
        "def get_init_inputs(): return []\n"
        "def get_inputs(): return [33]\n",
        encoding="utf-8",
    )
    runner = profile.write_runner(str(tmp_path))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path)
    cases = profile.discover_cases(
        runner_path=runner,
        engine_dir=str(tmp_path),
        verify_dir=str(tmp_path),
        op_name="sample",
        impl_name="triton_ascend_impl",
        python_bin=sys.executable,
        env=env,
        deadline=profile.time.monotonic() + 30,
    )
    assert cases == [
        {"case_idx": 0, "input_desc": [{"type": "scalar", "value": 33}]}
    ]


def test_runner_accepts_tuple_case_groups(tmp_path):
    _write_fake_runtime(tmp_path)
    _write_task(tmp_path)
    (tmp_path / "sample_torch.py").write_text(
        "def get_init_inputs(): return []\n"
        "def get_input_groups(): return ([11], (22,))\n",
        encoding="utf-8",
    )
    runner = profile.write_runner(str(tmp_path))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path)
    cases = profile.discover_cases(
        runner_path=runner,
        engine_dir=str(tmp_path),
        verify_dir=str(tmp_path),
        op_name="sample",
        impl_name="triton_ascend_impl",
        python_bin=sys.executable,
        env=env,
        deadline=profile.time.monotonic() + 30,
    )
    assert [case["case_idx"] for case in cases] == [0, 1]


def test_runner_rejects_empty_input_groups(tmp_path):
    _write_fake_runtime(tmp_path)
    _write_task(tmp_path)
    (tmp_path / "sample_torch.py").write_text(
        "def get_init_inputs(): return []\n"
        "def get_input_groups(): return iter(())\n",
        encoding="utf-8",
    )
    runner = profile.write_runner(str(tmp_path))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path)
    with pytest.raises(RuntimeError, match="case discovery failed"):
        profile.discover_cases(
            runner_path=runner,
            engine_dir=str(tmp_path),
            verify_dir=str(tmp_path),
            op_name="sample",
            impl_name="triton_ascend_impl",
            python_bin=sys.executable,
            env=env,
            deadline=profile.time.monotonic() + 30,
        )


def test_runner_moves_nested_tensor_inputs_to_local_npu(tmp_path):
    _write_fake_runtime(tmp_path)
    (tmp_path / "sample_torch.py").write_text(
        "import torch\n"
        "def get_init_inputs(): return []\n"
        "def get_input_groups():\n"
        "    return [[{'x': torch.Tensor(7)}, (torch.Tensor(8),)]]\n",
        encoding="utf-8",
    )
    (tmp_path / "sample_triton_ascend_impl.py").write_text(
        "import os\n"
        "class ModelNew:\n"
        "    def __init__(self): pass\n"
        "    def to(self, device): return self\n"
        "    def __call__(self, mapping, pair):\n"
        "        assert mapping['x'].device == 'npu:0'\n"
        "        assert pair[0].device == 'npu:0'\n"
        "        open(os.environ['KERNEL_MARKER'], 'w').write('nested-ok')\n",
        encoding="utf-8",
    )
    runner = profile.write_runner(str(tmp_path))
    hf32_marker = tmp_path / "hf32.txt"
    kernel_marker = tmp_path / "kernel.txt"
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(tmp_path),
        "HF32_MARKER": str(hf32_marker),
        "KERNEL_MARKER": str(kernel_marker),
    })
    outcome = profile.run_command(
        [
            sys.executable,
            runner,
            "--engine-dir",
            str(tmp_path),
            "--verify-dir",
            str(tmp_path),
            "--op-name",
            "sample",
            "--triton-impl-name",
            "triton_ascend_impl",
            "--case-idx",
            "0",
        ],
        cwd=str(tmp_path),
        env=env,
        timeout=30,
    )
    assert outcome.exit_code == 0, outcome.stderr
    assert kernel_marker.read_text() == "nested-ok"


def test_runner_disables_hf32_before_selected_case_launch(tmp_path):
    _write_fake_runtime(tmp_path)
    _write_task(tmp_path)
    runner = profile.write_runner(str(tmp_path))
    hf32_marker = tmp_path / "hf32.txt"
    kernel_marker = tmp_path / "kernel.txt"
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(tmp_path),
        "HF32_MARKER": str(hf32_marker),
        "KERNEL_MARKER": str(kernel_marker),
    })
    outcome = profile.run_command(
        [
            sys.executable,
            runner,
            "--engine-dir",
            str(tmp_path),
            "--verify-dir",
            str(tmp_path),
            "--op-name",
            "sample",
            "--triton-impl-name",
            "triton_ascend_impl",
            "--case-idx",
            "1",
        ],
        cwd=str(tmp_path),
        env=env,
        timeout=30,
    )
    assert outcome.exit_code == 0, outcome.stderr
    assert hf32_marker.read_text() == "off"
    assert kernel_marker.read_text() == "22"
```

- [ ] **Step 2: Run runner tests and verify RED**

Run:

```bash
python -m pytest tests/test_profile_runner.py -q
```

Expected: import fails because `engine/profile.py` does not exist.

- [ ] **Step 3: Implement runner source and discovery**

Create `triton_evaluation/engine/profile.py` with the CANN v2 header and:

```python
"""Profile CLI and multi-case orchestrator.

This module remains stdlib-only. Torch/NPU imports live in the generated
runner source and execute in a child process pinned by the service worker.
"""

from __future__ import annotations

import json
import os
import textwrap
import time
from pathlib import Path

try:
    from .msprof_op import remaining_pass_timeout, run_command
except ImportError:  # direct execution: python engine/profile.py
    from msprof_op import remaining_pass_timeout, run_command


def build_runner_source() -> str:
    return textwrap.dedent(
        r'''
        import argparse
        import importlib.util
        import json
        import os
        import sys


        def load_module(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"cannot load module: {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module


        def seed(torch):
            torch.manual_seed(0)
            if hasattr(torch, "npu"):
                torch.npu.manual_seed_all(0)


        def resolve_groups(reference, torch):
            seed(torch)
            if hasattr(reference, "get_input_groups"):
                groups = list(reference.get_input_groups())
                if not groups:
                    raise RuntimeError("get_input_groups() returned no cases")
            elif hasattr(reference, "get_inputs"):
                groups = [reference.get_inputs()]
            else:
                raise RuntimeError("reference defines neither get_input_groups nor get_inputs")
            normalized = []
            for index, group in enumerate(groups):
                if not isinstance(group, (list, tuple)):
                    raise TypeError(f"case {index} must be a list or tuple of arguments")
                normalized.append(list(group))
            return normalized


        def to_npu(value, torch):
            if isinstance(value, torch.Tensor):
                return value.to("npu:0")
            if isinstance(value, list):
                return [to_npu(item, torch) for item in value]
            if isinstance(value, tuple):
                return tuple(to_npu(item, torch) for item in value)
            if isinstance(value, dict):
                return {key: to_npu(item, torch) for key, item in value.items()}
            return value


        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--engine-dir", required=True)
            parser.add_argument("--verify-dir", required=True)
            parser.add_argument("--op-name", required=True)
            parser.add_argument("--triton-impl-name", required=True)
            mode = parser.add_mutually_exclusive_group(required=True)
            mode.add_argument("--describe-cases", action="store_true")
            mode.add_argument("--case-idx", type=int)
            args = parser.parse_args()

            sys.path.insert(0, args.engine_dir)
            sys.path.insert(0, args.verify_dir)
            import torch
            from _common_utils import describe_input, enforce_conv_fp32

            reference = load_module(
                f"{args.op_name}_torch",
                os.path.join(args.verify_dir, f"{args.op_name}_torch.py"),
            )
            groups = resolve_groups(reference, torch)
            if args.describe_cases:
                print(json.dumps({
                    "total_cases": len(groups),
                    "cases": [
                        {"case_idx": index, "input_desc": describe_input(group)}
                        for index, group in enumerate(groups)
                    ],
                }))
                return

            index = args.case_idx
            if index is None or index < 0 or index >= len(groups):
                raise IndexError(f"case_idx {index} out of range for {len(groups)} cases")
            enforce_conv_fp32()
            implementation = load_module(
                f"{args.op_name}_{args.triton_impl_name}",
                os.path.join(
                    args.verify_dir,
                    f"{args.op_name}_{args.triton_impl_name}.py",
                ),
            )
            seed(torch)
            model = implementation.ModelNew(*reference.get_init_inputs()).to("npu:0")
            inputs = [to_npu(value, torch) for value in groups[index]]
            with torch.no_grad():
                model(*inputs)
            if hasattr(torch, "npu"):
                torch.npu.synchronize()
            print(json.dumps({"case_idx": index, "input_desc": describe_input(groups[index])}))


        if __name__ == "__main__":
            main()
        '''
    )


def write_runner(directory: str) -> str:
    path = Path(directory) / "profile_runner.py"
    path.write_text(build_runner_source(), encoding="utf-8")
    return str(path)


def _last_json_object(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("runner produced no JSON object")


def discover_cases(
    *,
    runner_path: str,
    engine_dir: str,
    verify_dir: str,
    op_name: str,
    impl_name: str,
    python_bin: str,
    env: dict[str, str],
    deadline: float,
) -> list[dict]:
    command = [
        python_bin,
        runner_path,
        "--engine-dir",
        engine_dir,
        "--verify-dir",
        verify_dir,
        "--op-name",
        op_name,
        "--triton-impl-name",
        impl_name,
        "--describe-cases",
    ]
    outcome = run_command(
        command,
        cwd=verify_dir,
        env=env,
        timeout=remaining_pass_timeout(deadline),
    )
    if time.monotonic() >= deadline:
        raise TimeoutError("profile job deadline exceeded")
    if outcome.timed_out:
        raise RuntimeError("case discovery exceeded its per-process timeout")
    if outcome.exit_code != 0:
        raise RuntimeError(f"case discovery failed: {outcome.stderr[-1000:]}")
    payload = _last_json_object(outcome.stdout)
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("case discovery returned no cases")
    return cases
```

- [ ] **Step 4: Verify Task 4 and commit**

Run:

```bash
python -m pytest tests/test_profile_runner.py -q
python -m py_compile triton_evaluation/engine/profile.py
```

Expected: both runner tests pass and compilation succeeds.

```bash
git add triton_evaluation/engine/profile.py tests/test_profile_runner.py
git commit -m "feat(profile): add deterministic case runner"
```

---

### Task 5: Deep 7+1 capture and exact target merge

**Files:**
- Create: `triton_evaluation/engine/msprof_deep.py`
- Create: `tests/test_profile_deep.py`

**Interfaces:**
- Consumes: `CommandOutcome`, `run_command`, and `remaining_pass_timeout`.
- Produces: `METRIC_GROUPS`.
- Produces: `select_and_merge_metrics(metric_dirs, kernel_names)`.
- Produces: `profile_deep_case(app_cmd, kernel_names, output_dir, cwd, env, deadline)`.

- [ ] **Step 1: Write failing stable-target tests**

Create `tests/test_profile_deep.py`:

```python
import csv
from pathlib import Path

from triton_evaluation.engine import msprof_deep


def _summary(path: Path, rows: list[dict]):
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_metric_merge_keeps_one_complete_op_name(tmp_path):
    pipe = tmp_path / "PipeUtilization"
    memory = tmp_path / "Memory"
    _summary(
        pipe / "op_summary.csv",
        [
            {"Op Name": "kernel_a_hash", "Task Duration(us)": 9, "aiv_mte2_ratio": 0.4},
            {"Op Name": "kernel_b_hash", "Task Duration(us)": 5, "aiv_mte2_ratio": 0.9},
        ],
    )
    _summary(
        memory / "op_summary.csv",
        [
            {"Op Name": "kernel_a_hash", "Task Duration(us)": 4, "GM_to_UB_bw_usage_rate(%)": 31},
            {"Op Name": "kernel_b_hash", "Task Duration(us)": 20, "GM_to_UB_bw_usage_rate(%)": 99},
        ],
    )
    merged = msprof_deep.select_and_merge_metrics(
        {"PipeUtilization": str(pipe), "Memory": str(memory)},
        ["kernel_a", "kernel_b"],
    )
    assert merged["target_kernel"] == "kernel_a_hash"
    assert merged["raw_metrics"]["GM_to_UB_bw_usage_rate(%)"] == "31"
    assert merged["evidence"]["Memory"]["op_name"] == "kernel_a_hash"


def test_missing_metric_stays_missing_not_zero(tmp_path):
    pipe = tmp_path / "PipeUtilization"
    _summary(
        pipe / "op_summary.csv",
        [{"Op Name": "kernel_a_hash", "Task Duration(us)": 9, "aiv_mte2_ratio": 0.4}],
    )
    merged = msprof_deep.select_and_merge_metrics(
        {"PipeUtilization": str(pipe)},
        ["kernel_a"],
    )
    assert "Memory" in merged["missing_metrics"]
    assert "GM_to_UB_bw_usage_rate(%)" not in merged["raw_metrics"]
```

- [ ] **Step 2: Run deep merge tests and verify RED**

Run:

```bash
python -m pytest tests/test_profile_deep.py -q
```

Expected: import fails because `engine/msprof_deep.py` does not exist.

- [ ] **Step 3: Implement CSV discovery and exact-name merge**

Create `triton_evaluation/engine/msprof_deep.py` with the CANN v2 header and:

```python
"""Raw msprof deep capture and diagnosis.

Source flow: cannbot-skills/ops/ops-profiling/scripts/msprof_profile_run.sh
and scripts/msprof_perf_summary.py, with rules from references/msprof-guide.md.
Missing data is explicit and rows are merged only after one complete target
Op Name is selected.
"""

from __future__ import annotations

import csv
import glob
import os
import time
from typing import Any

try:
    from .msprof_op import CommandOutcome, remaining_pass_timeout, run_command
except ImportError:  # direct execution through engine/profile.py
    from msprof_op import CommandOutcome, remaining_pass_timeout, run_command


METRIC_GROUPS = (
    "PipeUtilization",
    "ArithmeticUtilization",
    "Memory",
    "MemoryL0",
    "MemoryUB",
    "L2Cache",
    "ResourceConflictRatio",
)


def _read_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summary_file(directory: str) -> str | None:
    candidates = sorted(glob.glob(
        os.path.join(directory, "**", "op_summary_*.csv"),
        recursive=True,
    ))
    if not candidates:
        candidates = sorted(glob.glob(
            os.path.join(directory, "**", "op_summary.csv"),
            recursive=True,
        ))
    return candidates[-1] if candidates else None


def _float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.upper() in {"NA", "N/A"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _custom_name(name: str, kernel_names: list[str]) -> bool:
    return any(name == kernel or name.startswith(kernel) for kernel in kernel_names)


def _pick_longest(rows: list[dict[str, str]]) -> dict[str, str] | None:
    timed = [
        (duration, row)
        for row in rows
        if (duration := _float(row.get("Task Duration(us)"))) is not None
    ]
    if not timed:
        return None
    return max(timed, key=lambda item: item[0])[1]


def select_and_merge_metrics(
    metric_dirs: dict[str, str],
    kernel_names: list[str],
) -> dict[str, Any]:
    rows_by_metric: dict[str, list[dict[str, str]]] = {}
    missing = [metric for metric in METRIC_GROUPS if metric not in metric_dirs]
    parse_warnings: list[str] = []
    for metric, directory in metric_dirs.items():
        summary = _summary_file(directory)
        if summary is None:
            missing.append(metric)
            continue
        try:
            rows_by_metric[metric] = _read_rows(summary)
        except (OSError, UnicodeError, csv.Error) as exc:
            missing.append(metric)
            parse_warnings.append(
                f"{metric}: cannot parse {summary}: {type(exc).__name__}: {exc}"
            )

    target_metric = None
    target_row = None
    for metric in METRIC_GROUPS:
        candidates = [
            row for row in rows_by_metric.get(metric, [])
            if _custom_name((row.get("Op Name") or "").strip(), kernel_names)
        ]
        chosen = _pick_longest(candidates)
        if chosen is not None:
            target_metric = metric
            target_row = chosen
            break
    if target_row is None:
        return {
            "target_kernel": None,
            "target_selection_metric": None,
            "raw_metrics": {},
            "missing_metrics": sorted(set(missing)),
            "evidence": {},
            "warnings": parse_warnings,
            "error": "no matching custom Triton kernel in deep op_summary",
        }

    target_name = (target_row.get("Op Name") or "").strip()
    merged: dict[str, str] = {}
    evidence: dict[str, dict[str, Any]] = {}
    for metric in METRIC_GROUPS:
        rows = [
            row for row in rows_by_metric.get(metric, [])
            if (row.get("Op Name") or "").strip() == target_name
        ]
        chosen = _pick_longest(rows)
        if chosen is None:
            missing.append(metric)
            continue
        evidence[metric] = {
            "op_name": target_name,
            "candidate_rows": len(rows),
            "selected_duration_us": _float(chosen.get("Task Duration(us)")),
        }
        for key, value in chosen.items():
            if key and value not in {None, "", "NA", "N/A"} and key not in merged:
                merged[key] = value
    return {
        "target_kernel": target_name,
        "target_selection_metric": target_metric,
        "raw_metrics": merged,
        "missing_metrics": sorted(set(missing)),
        "evidence": evidence,
        "warnings": parse_warnings,
        "error": None,
    }
```

- [ ] **Step 4: Verify stable merge tests**

Run:

```bash
python -m pytest tests/test_profile_deep.py -q
```

Expected: both merge tests pass.

- [ ] **Step 5: Write failing 3+7+1 command test**

Append:

```python
def test_deep_capture_runs_three_warmups_then_seven_plus_sample(tmp_path, monkeypatch):
    commands = []

    def fake_run(cmd, *, cwd, env, timeout):
        commands.append(cmd)
        return msprof_deep.CommandOutcome(0, "", "")

    monkeypatch.setattr(msprof_deep, "run_command", fake_run)
    monkeypatch.setattr(
        msprof_deep,
        "select_and_merge_metrics",
        lambda dirs, names: {
            "target_kernel": "kernel_a_hash",
            "target_selection_metric": "PipeUtilization",
            "raw_metrics": {},
            "missing_metrics": [],
            "evidence": {},
            "warnings": [],
            "error": None,
        },
    )
    result = msprof_deep.profile_deep_case(
        app_cmd=["python", "runner.py", "--case-idx", "0"],
        kernel_names=["kernel_a"],
        output_dir=str(tmp_path),
        cwd=str(tmp_path),
        env={},
        deadline=msprof_deep.time.monotonic() + 60,
    )
    assert commands[:3] == [["python", "runner.py", "--case-idx", "0"]] * 3
    profiled = commands[3:]
    assert len(profiled) == 8
    assert [next(arg for arg in cmd if arg.startswith("--aic-metrics=")) for cmd in profiled[:7]] == [
        f"--aic-metrics={metric}" for metric in msprof_deep.METRIC_GROUPS
    ]
    assert all("--aic-mode=task-based" in cmd for cmd in profiled[:7])
    assert "--aic-mode=sample-based" in profiled[-1]
    assert result["target_kernel"] == "kernel_a_hash"


def test_deep_metric_pass_timeout_is_partial_not_job_timeout(tmp_path, monkeypatch):
    calls = 0

    def fake_run(cmd, *, cwd, env, timeout):
        nonlocal calls
        calls += 1
        if calls == 4:
            return msprof_deep.CommandOutcome(-1, "", "", timed_out=True)
        return msprof_deep.CommandOutcome(0, "", "")

    monkeypatch.setattr(msprof_deep, "run_command", fake_run)
    monkeypatch.setattr(
        msprof_deep,
        "select_and_merge_metrics",
        lambda dirs, names: {
            "target_kernel": "kernel_a_hash",
            "target_selection_metric": "ArithmeticUtilization",
            "raw_metrics": {},
            "missing_metrics": ["PipeUtilization"],
            "evidence": {},
            "warnings": [],
            "error": None,
        },
    )
    result = msprof_deep.profile_deep_case(
        app_cmd=["python", "runner.py", "--case-idx", "0"],
        kernel_names=["kernel_a"],
        output_dir=str(tmp_path),
        cwd=str(tmp_path),
        env={},
        deadline=msprof_deep.time.monotonic() + 60,
    )
    assert calls == 11
    assert result["status"] == "partial"
    assert "per-process timeout" in " ".join(result["warnings"])
```

`CommandOutcome` and `time` are already imported by the Step 3 module block.

- [ ] **Step 6: Implement deep capture**

Append:

```python
def _task_command(
    msprof_bin: str,
    output: str,
    metric: str,
    app_cmd: list[str],
) -> list[str]:
    return [
        msprof_bin,
        f"--output={output}",
        "--ai-core=on",
        "--aic-mode=task-based",
        f"--aic-metrics={metric}",
        "--task-time=on",
        "--ascendcl=on",
        *app_cmd,
    ]


def profile_deep_case(
    *,
    app_cmd: list[str],
    kernel_names: list[str],
    output_dir: str,
    cwd: str,
    env: dict[str, str],
    deadline: float,
) -> dict[str, Any]:
    msprof_bin = os.environ.get("MSPROF_BIN", "msprof")
    warnings: list[str] = []
    for warmup_index in range(3):
        outcome = run_command(
            app_cmd,
            cwd=cwd,
            env=env,
            timeout=remaining_pass_timeout(deadline),
        )
        if time.monotonic() >= deadline:
            raise TimeoutError("profile job deadline exceeded")
        if outcome.timed_out:
            return {
                "status": "failed",
                "target_kernel": None,
                "target_selection_metric": None,
                "kernels": [],
                "raw_metrics": {},
                "bound": None,
                "per_core": None,
                "recommendations": [],
                "missing_metrics": list(METRIC_GROUPS),
                "warnings": warnings,
                "error": (
                    f"deep warmup {warmup_index + 1} exceeded "
                    "its per-process timeout"
                ),
            }
        if outcome.exit_code != 0:
            return {
                "status": "failed",
                "target_kernel": None,
                "target_selection_metric": None,
                "kernels": [],
                "raw_metrics": {},
                "bound": None,
                "per_core": None,
                "recommendations": [],
                "missing_metrics": list(METRIC_GROUPS),
                "warnings": warnings,
                "error": f"warmup exit={outcome.exit_code}; stderr-tail={outcome.stderr[-1000:]}",
            }

    metric_dirs: dict[str, str] = {}
    for metric in METRIC_GROUPS:
        metric_dir = os.path.join(output_dir, f"PROF_{metric}")
        os.makedirs(metric_dir, exist_ok=True)
        outcome = run_command(
            _task_command(msprof_bin, metric_dir, metric, app_cmd),
            cwd=cwd,
            env=env,
            timeout=remaining_pass_timeout(deadline),
        )
        if time.monotonic() >= deadline:
            raise TimeoutError("profile job deadline exceeded")
        if outcome.timed_out:
            warnings.append(f"{metric}: exceeded its per-process timeout")
        elif outcome.exit_code == 0:
            metric_dirs[metric] = metric_dir
        else:
            warnings.append(
                f"{metric}: exit={outcome.exit_code}; stderr-tail={outcome.stderr[-1000:]}"
            )

    sample_dir = os.path.join(output_dir, "PROF_Sample")
    os.makedirs(sample_dir, exist_ok=True)
    sample_command = [
        msprof_bin,
        f"--output={sample_dir}",
        "--ai-core=on",
        "--aic-metrics=PipeUtilization",
        "--aic-mode=sample-based",
        "--aic-freq=100",
        "--task-time=on",
        "--ascendcl=on",
        *app_cmd,
    ]
    sample = run_command(
        sample_command,
        cwd=cwd,
        env=env,
        timeout=remaining_pass_timeout(deadline),
    )
    if time.monotonic() >= deadline:
        raise TimeoutError("profile job deadline exceeded")
    sample_ok = False
    if sample.timed_out:
        warnings.append("sample-based: exceeded its per-process timeout")
    elif sample.exit_code == 0:
        sample_ok = True
    else:
        warnings.append(
            f"sample-based: exit={sample.exit_code}; stderr-tail={sample.stderr[-1000:]}"
        )

    merged = select_and_merge_metrics(metric_dirs, kernel_names)
    warnings.extend(merged["warnings"])
    failed = merged["target_kernel"] is None
    partial = bool(warnings or merged["missing_metrics"])
    return {
        "status": "failed" if failed else ("partial" if partial else "complete"),
        "target_kernel": merged["target_kernel"],
        "target_selection_metric": merged["target_selection_metric"],
        "kernels": [],
        "raw_metrics": merged["raw_metrics"],
        "bound": None,
        "per_core": None,
        "recommendations": [],
        "missing_metrics": merged["missing_metrics"],
        "warnings": warnings,
        "error": merged["error"],
        "_sample_dir": sample_dir if sample_ok else None,
        "_merge_evidence": merged["evidence"],
    }
```

- [ ] **Step 7: Verify Task 5 and commit**

Run:

```bash
python -m pytest tests/test_profile_deep.py -q
python -m py_compile triton_evaluation/engine/msprof_deep.py
```

Expected: all deep capture/merge tests pass.

```bash
git add triton_evaluation/engine/msprof_deep.py tests/test_profile_deep.py
git commit -m "feat(profile): add deep msprof capture"
```

---

### Task 6: Deep Bound, per-core scope, and recommendations

**Files:**
- Modify: `triton_evaluation/engine/msprof_deep.py`
- Modify: `tests/test_profile_deep.py`

**Interfaces:**
- Produces: `canonical_pipe_ratios(raw) -> dict[str, float | None]`.
- Produces: `missing_canonical_pipe_fields(raw) -> list[str]`.
- Produces: `classify_bound(raw) -> dict | None`.
- Produces: `load_per_core_cycles(sample_dir)`.
- Produces: `per_core_balance(raw, sample_dir)`.
- Produces: `recommendations_for(bound)`.

- [ ] **Step 1: Write failing Bound tests**

Add `import pytest` and `import sqlite3` with the test imports, then append:

```python
def test_bound_priority_and_exact_thresholds():
    raw = {
        "aiv_mte2_ratio": "0.81",
        "aic_cube_ratio": "0.95",
        "pushq_ratio": "0.90",
        "aic_fixpipe_ratio": "0.10",
        "aiv_mte3_ratio": "0.10",
        "aiv_scalar_ratio": "0.10",
        "scalarldst_ratio": "0.10",
    }
    bound = msprof_deep.classify_bound(raw)
    assert bound["name"] == "MTE2 BOUND"
    assert bound["rule"] == "MTE2 > 0.80"


def test_vec_uses_pushq_not_aiv_vec_ratio():
    raw = {
        "aiv_mte2_ratio": "0.10",
        "aic_cube_ratio": "0.10",
        "aiv_vec_ratio": "0.99",
        "pushq_ratio": "0.20",
        "aic_fixpipe_ratio": "0.10",
        "aiv_mte3_ratio": "0.10",
        "aiv_scalar_ratio": "0.10",
        "scalarldst_ratio": "0.10",
    }
    assert msprof_deep.classify_bound(raw)["name"] == "NO BOUND"


def test_missing_prior_metric_makes_bound_unknown():
    raw = {
        "aic_cube_ratio": "0.95",
        "pushq_ratio": "0.10",
        "aic_fixpipe_ratio": "0.10",
        "aiv_mte3_ratio": "0.10",
        "aiv_scalar_ratio": "0.10",
        "scalarldst_ratio": "0.10",
    }
    assert msprof_deep.classify_bound(raw) is None
    assert "PipeUtilization.MTE2" in msprof_deep.missing_canonical_pipe_fields(raw)


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("aic_cube_ratio", "CUBE BOUND"),
        ("pushq_ratio", "VEC BOUND"),
        ("aic_fixpipe_ratio", "FIXP BOUND"),
        ("aiv_mte3_ratio", "MTE3 BOUND"),
        ("aiv_scalar_ratio", "SCALAR BOUND"),
        ("scalarldst_ratio", "SCALAR BOUND"),
    ],
)
def test_each_bound_rule_is_reachable(field, expected):
    raw = {
        "aiv_mte2_ratio": "0.10",
        "aic_cube_ratio": "0.10",
        "pushq_ratio": "0.10",
        "aic_fixpipe_ratio": "0.10",
        "aiv_mte3_ratio": "0.10",
        "aiv_scalar_ratio": "0.10",
        "scalarldst_ratio": "0.10",
    }
    raw[field] = "0.81"
    assert msprof_deep.classify_bound(raw)["name"] == expected


@pytest.mark.parametrize(
    "field",
    [
        "pushq_ratio",
        "aic_fixpipe_ratio",
        "aiv_mte3_ratio",
        "aiv_scalar_ratio",
        "scalarldst_ratio",
    ],
)
def test_eighty_percent_is_not_above_threshold(field):
    raw = {
        "aiv_mte2_ratio": "0.10",
        "aic_cube_ratio": "0.10",
        "pushq_ratio": "0.10",
        "aic_fixpipe_ratio": "0.10",
        "aiv_mte3_ratio": "0.10",
        "aiv_scalar_ratio": "0.10",
        "scalarldst_ratio": "0.10",
    }
    raw[field] = "0.80"
    assert msprof_deep.classify_bound(raw)["name"] == "NO BOUND"


@pytest.mark.parametrize("field", ["aiv_mte2_ratio", "aic_cube_ratio"])
def test_seventy_percent_is_not_above_largest_threshold(field):
    raw = {
        "aiv_mte2_ratio": "0.10",
        "aic_cube_ratio": "0.10",
        "pushq_ratio": "0.10",
        "aic_fixpipe_ratio": "0.10",
        "aiv_mte3_ratio": "0.10",
        "aiv_scalar_ratio": "0.10",
        "scalarldst_ratio": "0.10",
    }
    raw[field] = "0.70"
    assert msprof_deep.classify_bound(raw)["name"] == "NO BOUND"
```

- [ ] **Step 2: Run Bound tests and verify RED**

Run:

```bash
python -m pytest tests/test_profile_deep.py -q
```

Expected: fails because `classify_bound` is undefined.

- [ ] **Step 3: Implement tri-state Bound classification**

Append:

```python
PIPE_ALIASES = {
    "MTE2": ("aic_mte2_ratio", "aiv_mte2_ratio", "mte2_ratio"),
    "CUBE": ("aic_cube_ratio", "aic_mac_ratio", "cube_ratio"),
    "PUSHQ": ("pushq_ratio", "aiv_pushq_ratio"),
    "FIXP": ("aic_fixpipe_ratio", "fixpipe_ratio"),
    "MTE3": ("aic_mte3_ratio", "aiv_mte3_ratio", "mte3_ratio"),
    "SCALAR": ("aic_scalar_ratio", "aiv_scalar_ratio", "scalar_ratio"),
    "SCALARLDST": ("scalarldst_ratio", "aic_scalarldst_ratio", "aiv_scalarldst_ratio"),
}


def _max_alias(raw: dict[str, Any], aliases: tuple[str, ...]) -> float | None:
    values = [_float(raw.get(alias)) for alias in aliases]
    present = [value for value in values if value is not None]
    return max(present) if present else None


def canonical_pipe_ratios(raw: dict[str, Any]) -> dict[str, float | None]:
    return {name: _max_alias(raw, aliases) for name, aliases in PIPE_ALIASES.items()}


def missing_canonical_pipe_fields(raw: dict[str, Any]) -> list[str]:
    return [
        f"PipeUtilization.{name}"
        for name, value in canonical_pipe_ratios(raw).items()
        if value is None
    ]


def classify_bound(raw: dict[str, Any]) -> dict[str, Any] | None:
    ratios = canonical_pipe_ratios(raw)
    comparison = [ratios[name] for name in PIPE_ALIASES]

    def largest(name: str) -> bool | None:
        if any(value is None for value in comparison):
            return None
        value = ratios[name]
        return value == max(comparison) and value > 0.70

    rules: list[tuple[str, str, bool | None]] = [
        (
            "MTE2 BOUND",
            "MTE2 > 0.80",
            None if ratios["MTE2"] is None else ratios["MTE2"] > 0.80,
        ),
        ("MTE2 BOUND", "MTE2 is largest and > 0.70", largest("MTE2")),
        (
            "CUBE BOUND",
            "CUBE > 0.80",
            None if ratios["CUBE"] is None else ratios["CUBE"] > 0.80,
        ),
        ("CUBE BOUND", "CUBE is largest and > 0.70", largest("CUBE")),
        (
            "VEC BOUND",
            "PUSHQ > 0.80",
            None if ratios["PUSHQ"] is None else ratios["PUSHQ"] > 0.80,
        ),
        (
            "FIXP BOUND",
            "FIXP > 0.80",
            None if ratios["FIXP"] is None else ratios["FIXP"] > 0.80,
        ),
        (
            "MTE3 BOUND",
            "MTE3 > 0.80",
            None if ratios["MTE3"] is None else ratios["MTE3"] > 0.80,
        ),
        (
            "SCALAR BOUND",
            "SCALAR > 0.80 or SCALARLDST > 0.80",
            None
            if ratios["SCALAR"] is None or ratios["SCALARLDST"] is None
            else ratios["SCALAR"] > 0.80 or ratios["SCALARLDST"] > 0.80,
        ),
    ]
    for name, rule, decision in rules:
        if decision is None:
            return None
        if decision:
            return {"name": name, "ratios": ratios, "rule": rule, "evidence": dict(raw)}
    return {"name": "NO BOUND", "ratios": ratios, "rule": "no rule matched", "evidence": dict(raw)}
```

- [ ] **Step 4: Verify Bound tests are GREEN**

Run:

```bash
python -m pytest tests/test_profile_deep.py -q
```

Expected: all Bound tests pass.

- [ ] **Step 5: Write failing per-core tests**

Append:

```python
def test_per_core_reports_observed_ids_and_profile_run_scope(tmp_path):
    db_dir = tmp_path / "PROF_fake" / "device_0" / "sqlite"
    db_dir.mkdir(parents=True)
    database = db_dir / "aicore.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE AICoreOriginalData(coreid INTEGER, task_cyc INTEGER)")
    connection.executemany(
        "INSERT INTO AICoreOriginalData VALUES (?, ?)",
        [(2, 100), (2, 50), (9, 200)],
    )
    connection.commit()
    connection.close()
    result = msprof_deep.per_core_balance(
        {"aicore_time(us)": "20"},
        str(tmp_path),
    )
    assert result["scope"] == "profile_run"
    assert result["core_ids"] == [2, 9]
    assert result["active_cores"] == 2
    assert "%" not in result["verdict"]
    assert result["spread_ratio"] == 0.25


def test_per_core_missing_database_is_none(tmp_path):
    assert msprof_deep.per_core_balance({"aicore_time(us)": "20"}, str(tmp_path)) is None


def test_per_core_unknown_schema_is_none(tmp_path):
    db_dir = tmp_path / "PROF_fake" / "device_0" / "sqlite"
    db_dir.mkdir(parents=True)
    connection = sqlite3.connect(db_dir / "aicore.db")
    connection.execute("CREATE TABLE DifferentData(core INTEGER, cycles INTEGER)")
    connection.commit()
    connection.close()
    assert msprof_deep.per_core_balance(
        {"aicore_time(us)": "20"},
        str(tmp_path),
    ) is None
```

- [ ] **Step 6: Implement per-core and recommendations**

Add `import sqlite3` and `import statistics` to `msprof_deep.py`, then append:

```python
def load_per_core_cycles(sample_dir: str) -> list[tuple[int, int]]:
    candidates = sorted(glob.glob(
        os.path.join(sample_dir, "**", "device_0", "sqlite", "aicore.db"),
        recursive=True,
    ))
    if not candidates:
        return []
    connection = None
    try:
        connection = sqlite3.connect(candidates[-1])
        rows = list(connection.execute(
            "SELECT coreid, SUM(task_cyc) "
            "FROM AICoreOriginalData WHERE task_cyc > 0 "
            "GROUP BY coreid ORDER BY coreid"
        ))
    except sqlite3.Error:
        return []
    finally:
        if connection is not None:
            connection.close()
    return [
        (int(core_id), int(cycles))
        for core_id, cycles in rows
        if core_id is not None and cycles is not None
    ]


def per_core_balance(raw: dict[str, Any], sample_dir: str) -> dict[str, Any] | None:
    rows = load_per_core_cycles(sample_dir)
    aicore_time = _float(raw.get("aicore_time(us)"))
    if not rows or aicore_time is None or aicore_time <= 0:
        return None
    max_cycles = max(cycles for _, cycles in rows)
    if max_cycles <= 0:
        return None
    ns_per_cycle = aicore_time * 1000.0 / max_cycles
    per_core = [
        {"core_id": core_id, "time_us": cycles * ns_per_cycle / 1000.0}
        for core_id, cycles in rows
    ]
    times = [entry["time_us"] for entry in per_core]
    maximum = max(times)
    minimum = min(times)
    spread = (maximum - minimum) / maximum if maximum else 0.0
    if spread < 0.10:
        verdict = "balanced"
    elif spread < 0.30:
        verdict = "warning"
    else:
        verdict = "severely_imbalanced"
    return {
        "scope": "profile_run",
        "warning": "SQLite rows are run-wide and are not claimed as target-only.",
        "core_ids": [entry["core_id"] for entry in per_core],
        "active_cores": len(per_core),
        "frequency_ghz": 1.0 / ns_per_cycle,
        "min_us": minimum,
        "avg_us": statistics.mean(times),
        "max_us": maximum,
        "spread_ratio": spread,
        "balanced_under_10pct": spread < 0.10,
        "verdict": verdict,
        "per_core": per_core,
    }


_RECOMMENDATIONS = {
    "MTE2 BOUND": "Reduce GM loads, improve contiguous access, and evaluate buffering overlap.",
    "CUBE BOUND": "Increase useful Cube work per launch and validate tiling with measured latency.",
    "VEC BOUND": "Reduce vector issue pressure or fuse redundant vector passes.",
    "FIXP BOUND": "Check output alignment and FixPipe writeback configuration.",
    "MTE3 BOUND": "Reduce or coalesce GM stores and avoid unnecessary intermediate outputs.",
    "SCALAR BOUND": "Reduce scalar address/control work and shorten serial dependency chains.",
}


def recommendations_for(bound: dict[str, Any] | None) -> list[dict[str, Any]]:
    if bound is None or bound["name"] == "NO BOUND":
        return []
    return [{
        "bound": bound["name"],
        "evidence": {"ratios": bound["ratios"], "rule": bound["rule"]},
        "recommendation": _RECOMMENDATIONS[bound["name"]],
        "source": "ops-profiling/references/optimization_quickref.md",
    }]
```

In `profile_deep_case`, replace the block from
`merged = select_and_merge_metrics(...)` through its return statement with:

```python
    merged = select_and_merge_metrics(metric_dirs, kernel_names)
    warnings.extend(merged["warnings"])
    missing_metrics = sorted(set(
        merged["missing_metrics"]
        + missing_canonical_pipe_fields(merged["raw_metrics"])
    ))
    bound = classify_bound(merged["raw_metrics"])
    per_core = (
        per_core_balance(merged["raw_metrics"], sample_dir)
        if sample_ok
        else None
    )
    if bound is None and merged["target_kernel"] is not None:
        warnings.append("insufficient canonical pipe fields for Bound classification")
    if sample_ok and per_core is None:
        warnings.append("sample-based output could not produce reliable per-core data")
    failed = merged["target_kernel"] is None
    partial = bool(warnings or missing_metrics)
    raw_metrics = dict(merged["raw_metrics"])
    raw_metrics["_merge_evidence"] = merged["evidence"]
    return {
        "status": "failed" if failed else ("partial" if partial else "complete"),
        "target_kernel": merged["target_kernel"],
        "target_selection_metric": merged["target_selection_metric"],
        "kernels": [],
        "raw_metrics": raw_metrics,
        "bound": bound,
        "per_core": per_core,
        "recommendations": recommendations_for(bound),
        "missing_metrics": missing_metrics,
        "warnings": warnings,
        "error": merged["error"],
    }
```

- [ ] **Step 7: Verify Task 6 and commit**

Run:

```bash
python -m pytest tests/test_profile_deep.py -q
python -m py_compile triton_evaluation/engine/msprof_deep.py
```

Expected: all deep analysis tests pass.

```bash
git add triton_evaluation/engine/msprof_deep.py tests/test_profile_deep.py
git commit -m "feat(profile): classify deep bottlenecks"
```

---

### Task 7: Multi-case orchestration, atomic snapshots, and CLI

**Files:**
- Modify: `triton_evaluation/engine/profile.py`
- Create: `tests/test_profile_main.py`

**Interfaces:**
- Produces: `ProfileConfig`.
- Produces: `validate_profile_config(config)`.
- Produces: `select_case_indices(cases, requested)`.
- Produces: `run_profile(config, ...) -> tuple[result, exit_code]`.
- Produces: CLI exit 0 for completed orchestration, 2 for input/infrastructure failure, 124 for timeout.

- [ ] **Step 1: Write failing orchestration tests**

Create `tests/test_profile_main.py`:

```python
import json
import sys
from pathlib import Path

import pytest

from triton_evaluation.engine import profile


def _config(tmp_path, **overrides):
    (tmp_path / "sample_triton_ascend_impl.py").write_text(
        "import triton\n"
        "@triton.jit\n"
        "def sample_kernel(x):\n"
        "    return x\n",
        encoding="utf-8",
    )
    values = {
        "op_name": "sample",
        "verify_dir": str(tmp_path),
        "triton_impl_name": "triton_ascend_impl",
        "profile_mode": "op",
        "case_idx": None,
        "launch_count": None,
        "timeout": 60,
        "output": str(tmp_path / "profile_result.json"),
        "engine_dir": str(tmp_path),
        "python_bin": sys.executable,
    }
    values.update(overrides)
    return profile.ProfileConfig(**values)


def _complete(case_idx):
    return {
        "status": "complete",
        "target_kernel": f"kernel_{case_idx}",
        "target_selection_metric": "OpBasicInfo",
        "kernels": [],
        "raw_metrics": {},
        "bound": None,
        "per_core": None,
        "recommendations": [],
        "missing_metrics": [],
        "warnings": [],
        "error": None,
    }


def test_default_profiles_all_cases_and_snapshots_each(tmp_path, monkeypatch):
    monkeypatch.setattr(profile, "write_runner", lambda directory: "runner.py")
    discovered = [
        {"case_idx": 0, "input_desc": [{"shape": [16]}]},
        {"case_idx": 1, "input_desc": [{"shape": [32]}]},
    ]
    snapshots = []
    result, exit_code = profile.run_profile(
        _config(tmp_path),
        discover=lambda **kwargs: discovered,
        op_backend=lambda **kwargs: _complete(int(kwargs["app_cmd"][-1])),
        deep_backend=None,
        snapshot=lambda path, payload: snapshots.append(json.loads(json.dumps(payload))),
    )
    assert exit_code == 0
    assert [item["case_idx"] for item in result["per_case_results"]] == [0, 1]
    assert result["selected_cases"] == 2
    assert result["completed_cases"] == 2
    assert len(snapshots) == 3
    assert (
        result["completed_cases"]
        + result["partial_cases"]
        + result["failed_cases"]
        + len(result["remaining_case_indices"])
        == result["selected_cases"]
    )


def test_single_case_selection_is_zero_based(tmp_path, monkeypatch):
    monkeypatch.setattr(profile, "write_runner", lambda directory: "runner.py")
    discovered = [
        {"case_idx": 0, "input_desc": []},
        {"case_idx": 1, "input_desc": []},
    ]
    seen = []

    def backend(**kwargs):
        seen.append(kwargs["app_cmd"])
        return _complete(1)

    result, exit_code = profile.run_profile(
        _config(tmp_path, case_idx=1),
        discover=lambda **kwargs: discovered,
        op_backend=backend,
        deep_backend=None,
        snapshot=lambda path, payload: None,
    )
    assert exit_code == 0
    assert result["requested_case_idx"] == 1
    assert result["selected_cases"] == 1
    assert seen[0][-2:] == ["--case-idx", "1"]


def test_case_failure_does_not_stop_full_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(profile, "write_runner", lambda directory: "runner.py")
    discovered = [
        {"case_idx": 0, "input_desc": []},
        {"case_idx": 1, "input_desc": []},
    ]

    def backend(**kwargs):
        index = int(kwargs["app_cmd"][-1])
        if index == 0:
            value = _complete(index)
            value.update({"status": "failed", "target_kernel": None, "error": "kernel crash"})
            return value
        return _complete(index)

    result, exit_code = profile.run_profile(
        _config(tmp_path),
        discover=lambda **kwargs: discovered,
        op_backend=backend,
        deep_backend=None,
        snapshot=lambda path, payload: None,
    )
    assert exit_code == 0
    assert result["failed_cases"] == 1
    assert result["completed_cases"] == 1


def test_deadline_returns_timeout_with_remaining_indices(tmp_path, monkeypatch):
    monkeypatch.setattr(profile, "write_runner", lambda directory: "runner.py")
    discovered = [
        {"case_idx": 0, "input_desc": []},
        {"case_idx": 1, "input_desc": []},
    ]

    def backend(**kwargs):
        raise TimeoutError("metric timed out")

    result, exit_code = profile.run_profile(
        _config(tmp_path),
        discover=lambda **kwargs: discovered,
        op_backend=backend,
        deep_backend=None,
        snapshot=lambda path, payload: None,
    )
    assert exit_code == 124
    assert result["timed_out"] is True
    assert result["remaining_case_indices"] == [0, 1]
    assert (
        result["completed_cases"]
        + result["partial_cases"]
        + result["failed_cases"]
        + len(result["remaining_case_indices"])
        == result["selected_cases"]
    )


def test_out_of_range_case_is_input_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(profile, "write_runner", lambda directory: "runner.py")
    with pytest.raises(IndexError, match="out of range"):
        profile.run_profile(
            _config(tmp_path, case_idx=4),
            discover=lambda **kwargs: [{"case_idx": 0, "input_desc": []}],
            op_backend=lambda **kwargs: _complete(0),
            deep_backend=None,
            snapshot=lambda path, payload: None,
        )


def test_invalid_pass_timeout_fails_before_case_discovery(tmp_path, monkeypatch):
    monkeypatch.setenv("MSPROF_TIMEOUT_SEC", "0")
    with pytest.raises(ValueError, match="positive integer"):
        profile.run_profile(
            _config(tmp_path),
            discover=lambda **kwargs: pytest.fail("discovery must not run"),
            op_backend=lambda **kwargs: _complete(0),
            snapshot=lambda path, payload: None,
        )


def test_deep_mode_rejects_launch_count_before_discovery(tmp_path):
    with pytest.raises(ValueError, match="only valid in op mode"):
        profile.run_profile(
            _config(tmp_path, profile_mode="deep", launch_count=2),
            discover=lambda **kwargs: pytest.fail("discovery must not run"),
            snapshot=lambda path, payload: None,
        )


def test_case_capture_directories_are_cleaned(tmp_path, monkeypatch):
    monkeypatch.setattr(profile, "write_runner", lambda directory: "runner.py")
    capture_dirs = []

    def backend(**kwargs):
        capture_dirs.append(Path(kwargs["output_dir"]))
        return _complete(0)

    result, exit_code = profile.run_profile(
        _config(tmp_path),
        discover=lambda **kwargs: [{"case_idx": 0, "input_desc": []}],
        op_backend=backend,
        snapshot=lambda path, payload: None,
    )
    assert exit_code == 0
    assert result["completed_cases"] == 1
    assert capture_dirs and all(not path.exists() for path in capture_dirs)


def test_atomic_snapshot_leaves_no_temporary_file(tmp_path):
    output = tmp_path / "profile_result.json"
    profile.atomic_write_json(str(output), {"selected_cases": 1})
    assert json.loads(output.read_text()) == {"selected_cases": 1}
    assert not (tmp_path / ".profile_result.json.tmp").exists()
```

- [ ] **Step 2: Run orchestration tests and verify RED**

Run:

```bash
python -m pytest tests/test_profile_main.py -q
```

Expected: fails because `ProfileConfig` and `run_profile` are undefined.

- [ ] **Step 3: Implement result helpers and orchestration**

Add the stdlib imports to `profile.py`:

```python
import argparse
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Callable
```

Replace the Task 4 compatibility import block with:

```python
try:
    from .msprof_deep import profile_deep_case
    from .msprof_op import (
        extract_triton_kernel_names,
        profile_op_case,
        remaining_pass_timeout,
        run_command,
    )
except ImportError:  # direct execution: python engine/profile.py
    from msprof_deep import profile_deep_case
    from msprof_op import (
        extract_triton_kernel_names,
        profile_op_case,
        remaining_pass_timeout,
        run_command,
    )
```

Append:

```python
@dataclass
class ProfileConfig:
    op_name: str
    verify_dir: str
    triton_impl_name: str
    profile_mode: str
    case_idx: int | None
    launch_count: int | None
    timeout: int
    output: str
    engine_dir: str
    python_bin: str


def validate_profile_config(config: ProfileConfig) -> None:
    if config.profile_mode not in {"op", "deep"}:
        raise ValueError("profile_mode must be 'op' or 'deep'")
    if config.case_idx is not None and config.case_idx < 0:
        raise ValueError("case_idx must be non-negative")
    if config.launch_count is not None and not 1 <= config.launch_count <= 100:
        raise ValueError("launch_count must be between 1 and 100")
    if config.profile_mode == "deep" and config.launch_count is not None:
        raise ValueError("launch_count is only valid in op mode")
    if not 10 <= config.timeout <= 86400:
        raise ValueError("timeout must be between 10 and 86400 seconds")


def atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    target = Path(path)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, target)


def select_case_indices(cases: list[dict], requested: int | None) -> list[int]:
    if requested is None:
        return [int(case["case_idx"]) for case in cases]
    if requested < 0 or requested >= len(cases):
        raise IndexError(f"case_idx {requested} out of range for {len(cases)} cases")
    return [requested]


def _empty_result(config: ProfileConfig, total_cases: int, selected: list[int]) -> dict:
    return {
        "profile_mode": config.profile_mode,
        "requested_case_idx": config.case_idx,
        "total_cases": total_cases,
        "selected_cases": len(selected),
        "completed_cases": 0,
        "partial_cases": 0,
        "failed_cases": 0,
        "remaining_case_indices": list(selected),
        "timed_out": False,
        "summary_text": "",
        "warnings": [],
        "wallclock_sec": 0.0,
        "per_case_results": [],
    }


def _refresh_counts(result: dict[str, Any]) -> None:
    statuses = [case["status"] for case in result["per_case_results"]]
    result["completed_cases"] = statuses.count("complete")
    result["partial_cases"] = statuses.count("partial")
    result["failed_cases"] = statuses.count("failed")
    result["summary_text"] = (
        f"profile mode={result['profile_mode']}: "
        f"complete={result['completed_cases']}, "
        f"partial={result['partial_cases']}, "
        f"failed={result['failed_cases']}, "
        f"remaining={len(result['remaining_case_indices'])}"
    )


def run_profile(
    config: ProfileConfig,
    *,
    discover: Callable[..., list[dict]] | None = None,
    op_backend: Callable[..., dict] | None = None,
    deep_backend: Callable[..., dict] | None = None,
    snapshot: Callable[[str, dict], None] = atomic_write_json,
) -> tuple[dict[str, Any], int]:
    validate_profile_config(config)
    started = time.monotonic()
    deadline = started + config.timeout
    remaining_pass_timeout(deadline)
    discover_fn = discover if discover is not None else discover_cases
    op_backend_fn = op_backend if op_backend is not None else profile_op_case
    deep_backend_fn = (
        deep_backend if deep_backend is not None else profile_deep_case
    )
    runner_path = write_runner(config.verify_dir)
    env = dict(os.environ)
    cases = discover_fn(
        runner_path=runner_path,
        engine_dir=config.engine_dir,
        verify_dir=config.verify_dir,
        op_name=config.op_name,
        impl_name=config.triton_impl_name,
        python_bin=config.python_bin,
        env=env,
        deadline=deadline,
    )
    selected = select_case_indices(cases, config.case_idx)
    result = _empty_result(config, len(cases), selected)
    snapshot(config.output, result)

    kernel_path = Path(config.verify_dir) / (
        f"{config.op_name}_{config.triton_impl_name}.py"
    )
    kernel_names = extract_triton_kernel_names(
        kernel_path.read_text(encoding="utf-8")
    )
    if not kernel_names:
        raise RuntimeError("kernel source contains no @triton.jit/@jit function")

    by_index = {int(case["case_idx"]): case for case in cases}
    for index in selected:
        case_started = time.monotonic()
        app_cmd = [
            config.python_bin,
            runner_path,
            "--engine-dir",
            config.engine_dir,
            "--verify-dir",
            config.verify_dir,
            "--op-name",
            config.op_name,
            "--triton-impl-name",
            config.triton_impl_name,
            "--case-idx",
            str(index),
        ]
        try:
            with tempfile.TemporaryDirectory(
                prefix=f"profile_case_{index}_",
                dir=config.verify_dir,
            ) as capture_dir:
                if config.profile_mode == "op":
                    case_result = op_backend_fn(
                        app_cmd=app_cmd,
                        kernel_names=kernel_names,
                        output_dir=capture_dir,
                        launch_count=config.launch_count,
                        cwd=config.verify_dir,
                        env=env,
                        deadline=deadline,
                    )
                else:
                    case_result = deep_backend_fn(
                        app_cmd=app_cmd,
                        kernel_names=kernel_names,
                        output_dir=capture_dir,
                        cwd=config.verify_dir,
                        env=env,
                        deadline=deadline,
                    )
            if time.monotonic() >= deadline:
                raise TimeoutError("profile job deadline exceeded")
        except TimeoutError as exc:
            result["timed_out"] = True
            result["warnings"].append(str(exc))
            result["wallclock_sec"] = time.monotonic() - started
            _refresh_counts(result)
            snapshot(config.output, result)
            return result, 124
        except Exception as exc:
            case_result = {
                "status": "failed",
                "target_kernel": None,
                "target_selection_metric": None,
                "kernels": [],
                "raw_metrics": {},
                "bound": None,
                "per_core": None,
                "recommendations": [],
                "missing_metrics": [],
                "warnings": [],
                "error": f"{type(exc).__name__}: {exc}",
            }

        case_result = dict(case_result)
        case_result["case_idx"] = index
        case_result["input_desc"] = by_index[index]["input_desc"]
        case_result["msprof_wallclock_sec"] = time.monotonic() - case_started
        result["per_case_results"].append(case_result)
        result["remaining_case_indices"].remove(index)
        result["wallclock_sec"] = time.monotonic() - started
        _refresh_counts(result)
        snapshot(config.output, result)

    return result, 0
```

- [ ] **Step 4: Verify orchestration tests**

Run:

```bash
python -m pytest tests/test_profile_main.py -q
```

Expected: all orchestration tests pass.

- [ ] **Step 5: Write failing CLI mapping tests**

Append:

```python
def test_main_writes_input_error_and_returns_two(tmp_path, monkeypatch):
    config = _config(tmp_path, case_idx=9)
    monkeypatch.setattr(profile, "_config_from_args", lambda: config)
    monkeypatch.setattr(
        profile,
        "discover_cases",
        lambda **kwargs: [{"case_idx": 0, "input_desc": []}],
    )
    assert profile.main() == 2
    payload = json.loads(Path(config.output).read_text())
    assert payload["error_type"] == "IndexError"
```

- [ ] **Step 6: Implement CLI and error mapping**

Append:

```python
def _config_from_args() -> ProfileConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op_name", required=True)
    parser.add_argument("--verify_dir", required=True)
    parser.add_argument("--triton_impl_name", required=True)
    parser.add_argument("--profile-mode", choices=("op", "deep"), required=True)
    parser.add_argument("--case-idx", type=int)
    parser.add_argument("--launch-count", type=int)
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    return ProfileConfig(
        op_name=args.op_name,
        verify_dir=os.path.abspath(args.verify_dir),
        triton_impl_name=args.triton_impl_name,
        profile_mode=args.profile_mode,
        case_idx=args.case_idx,
        launch_count=args.launch_count,
        timeout=args.timeout,
        output=os.path.abspath(args.output),
        engine_dir=os.path.dirname(os.path.abspath(__file__)),
        python_bin=os.environ.get("MSPROF_PROFILE_PYTHON") or sys.executable,
    )


def main() -> int:
    config = _config_from_args()
    try:
        result, exit_code = run_profile(config)
        atomic_write_json(config.output, result)
        return exit_code
    except Exception as exc:
        atomic_write_json(config.output, {
            "profile_mode": config.profile_mode,
            "requested_case_idx": config.case_idx,
            "timed_out": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        print(f"profile failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

The `None` defaults are resolved inside `run_profile`, as shown in Step 3. This
keeps CLI lookup patchable in tests while op-mode calls never invoke the deep
backend.

- [ ] **Step 7: Verify Task 7 and commit**

Run:

```bash
python -m pytest tests/test_profile_main.py tests/test_profile_runner.py -q
python -m py_compile triton_evaluation/engine/profile.py
```

Expected: all tests pass and compilation succeeds.

```bash
git add triton_evaluation/engine/profile.py tests/test_profile_main.py
git commit -m "feat(profile): orchestrate all input cases"
```

---

### Task 8: Fake-msprof subprocess integration

**Files:**
- Create: `tests/fixtures/fake_msprof.py`
- Create: `tests/test_profile_fake_msprof.py`

**Interfaces:**
- Exercises the real subprocess, filesystem discovery, CSV parser, 7+1 sequence, and SQLite parser without torch or NPU.

- [ ] **Step 1: Create an executable fake profiler**

Create `tests/fixtures/fake_msprof.py`:

```python
#!/usr/bin/env python3
import csv
import os
import sqlite3
import sys
import time
from pathlib import Path


def option(name):
    prefix = f"--{name}="
    return next(arg.split("=", 1)[1] for arg in sys.argv if arg.startswith(prefix))


def write_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


with open(os.environ["FAKE_MSPROF_LOG"], "a", encoding="utf-8") as log:
    log.write(" ".join(sys.argv[1:]) + "\n")

is_op = len(sys.argv) > 1 and sys.argv[1] == "op"
output = Path(option("output"))
sleep_seconds = float(os.environ.get("FAKE_MSPROF_SLEEP_SEC", "0"))
if sleep_seconds:
    time.sleep(sleep_seconds)
if is_op:
    if os.environ.get("FAKE_MSPROF_FAIL_OP") == "1":
        raise SystemExit(9)
    op_dir = output / "OPPROF_fake" / "add_kernel_hash" / "0"
    write_csv(op_dir / "OpBasicInfo.csv", {
        "Op Name": "add_kernel_hash",
        "Op Type": "vector",
        "Task Duration(us)": "6",
        "Block Dim": "8",
    })
    write_csv(op_dir / "PipeUtilization.csv", {
        "aiv_vec_ratio": "0.6",
        "aiv_mte2_ratio": "0.3",
        "aiv_mte3_ratio": "0.1",
    })
    write_csv(op_dir / "Memory.csv", {
        "GM_to_UB_bw_usage_rate(%)": "42",
        "UB_to_GM_bw_usage_rate(%)": "21",
    })
    write_csv(op_dir / "L2Cache.csv", {"aiv_total_hit_rate(%)": "90"})
    raise SystemExit(0)

metric = option("aic-metrics")
if "--aic-mode=sample-based" in sys.argv:
    if os.environ.get("FAKE_MSPROF_FAIL_SAMPLE") == "1":
        raise SystemExit(8)
    db_dir = output / "PROF_fake" / "device_0" / "sqlite"
    db_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_dir / "aicore.db")
    connection.execute("CREATE TABLE AICoreOriginalData(coreid INTEGER, task_cyc INTEGER)")
    connection.executemany(
        "INSERT INTO AICoreOriginalData VALUES (?, ?)",
        [(0, 100), (1, 90)],
    )
    connection.commit()
    connection.close()
    raise SystemExit(0)

if os.environ.get("FAKE_MSPROF_FAIL_METRIC") == metric:
    raise SystemExit(7)

row = {
    "Op Name": "add_kernel_hash",
    "Task Duration(us)": "6",
    "aicore_time(us)": "10",
}
if metric == "PipeUtilization":
    row.update({
        "aiv_mte2_ratio": "0.85",
        "aic_cube_ratio": "0.10",
        "pushq_ratio": "0.10",
        "aic_fixpipe_ratio": "0.10",
        "aiv_mte3_ratio": "0.10",
        "aiv_scalar_ratio": "0.10",
        "scalarldst_ratio": "0.10",
    })
write_csv(
    output / "PROF_fake" / "mindstudio_profiler_output" / "op_summary_0.csv",
    row,
)
```

Mark it executable in the implementation step:

```bash
chmod +x tests/fixtures/fake_msprof.py
```

- [ ] **Step 2: Write backend integration tests**

Create `tests/test_profile_fake_msprof.py`:

```python
import os
import sys
import time
from pathlib import Path

from triton_evaluation.engine import msprof_deep, msprof_op


FAKE = Path(__file__).parent / "fixtures" / "fake_msprof.py"


def _environment(tmp_path, monkeypatch):
    log = tmp_path / "calls.log"
    monkeypatch.setenv("MSPROF_BIN", str(FAKE))
    monkeypatch.setenv("FAKE_MSPROF_LOG", str(log))
    return dict(os.environ), log


def test_op_backend_with_fake_executable(tmp_path, monkeypatch):
    env, log = _environment(tmp_path, monkeypatch)
    result = msprof_op.profile_op_case(
        app_cmd=[sys.executable, "-c", "pass"],
        kernel_names=["add_kernel"],
        output_dir=str(tmp_path / "op"),
        launch_count=None,
        cwd=str(tmp_path),
        env=env,
        deadline=time.monotonic() + 30,
    )
    assert result["status"] == "complete"
    assert result["target_kernel"] == "add_kernel_hash"
    assert "--kernel-name=add_kernel" in log.read_text()


def test_op_backend_nonzero_exit_is_failed_case(tmp_path, monkeypatch):
    env, _ = _environment(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_MSPROF_FAIL_OP", "1")
    env["FAKE_MSPROF_FAIL_OP"] = "1"
    result = msprof_op.profile_op_case(
        app_cmd=[sys.executable, "-c", "pass"],
        kernel_names=["add_kernel"],
        output_dir=str(tmp_path / "op"),
        launch_count=None,
        cwd=str(tmp_path),
        env=env,
        deadline=time.monotonic() + 30,
    )
    assert result["status"] == "failed"
    assert "exit=9" in result["error"]


def test_op_backend_pass_timeout_is_failed_case(tmp_path, monkeypatch):
    env, _ = _environment(tmp_path, monkeypatch)
    monkeypatch.setenv("MSPROF_TIMEOUT_SEC", "1")
    monkeypatch.setenv("FAKE_MSPROF_SLEEP_SEC", "2")
    env.update({"MSPROF_TIMEOUT_SEC": "1", "FAKE_MSPROF_SLEEP_SEC": "2"})
    result = msprof_op.profile_op_case(
        app_cmd=[sys.executable, "-c", "pass"],
        kernel_names=["add_kernel"],
        output_dir=str(tmp_path / "op"),
        launch_count=None,
        cwd=str(tmp_path),
        env=env,
        deadline=time.monotonic() + 30,
    )
    assert result["status"] == "failed"
    assert "per-process timeout" in result["error"]


def test_deep_backend_with_fake_executable(tmp_path, monkeypatch):
    env, log = _environment(tmp_path, monkeypatch)
    result = msprof_deep.profile_deep_case(
        app_cmd=[sys.executable, "-c", "pass"],
        kernel_names=["add_kernel"],
        output_dir=str(tmp_path / "deep"),
        cwd=str(tmp_path),
        env=env,
        deadline=time.monotonic() + 60,
    )
    calls = log.read_text().splitlines()
    assert len(calls) == 8
    assert result["target_kernel"] == "add_kernel_hash"
    assert result["bound"]["name"] == "MTE2 BOUND"
    assert result["per_core"]["core_ids"] == [0, 1]


def test_deep_metric_failure_continues_as_partial(tmp_path, monkeypatch):
    env, log = _environment(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_MSPROF_FAIL_METRIC", "Memory")
    env["FAKE_MSPROF_FAIL_METRIC"] = "Memory"
    result = msprof_deep.profile_deep_case(
        app_cmd=[sys.executable, "-c", "pass"],
        kernel_names=["add_kernel"],
        output_dir=str(tmp_path / "deep"),
        cwd=str(tmp_path),
        env=env,
        deadline=time.monotonic() + 60,
    )
    assert len(log.read_text().splitlines()) == 8
    assert result["status"] == "partial"
    assert "Memory" in result["missing_metrics"]
    assert result["target_kernel"] == "add_kernel_hash"


def test_deep_sample_failure_keeps_task_metrics(tmp_path, monkeypatch):
    env, _ = _environment(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_MSPROF_FAIL_SAMPLE", "1")
    env["FAKE_MSPROF_FAIL_SAMPLE"] = "1"
    result = msprof_deep.profile_deep_case(
        app_cmd=[sys.executable, "-c", "pass"],
        kernel_names=["add_kernel"],
        output_dir=str(tmp_path / "deep"),
        cwd=str(tmp_path),
        env=env,
        deadline=time.monotonic() + 60,
    )
    assert result["status"] == "partial"
    assert result["target_kernel"] == "add_kernel_hash"
    assert result["per_core"] is None
```

Warmups execute the app directly and therefore are not written to the fake-msprof log; the log contains exactly seven task-based calls and one sample-based call.

- [ ] **Step 3: Run integration tests**

Run:

```bash
python -m pytest tests/test_profile_fake_msprof.py -q
```

Expected: both tests pass.

- [ ] **Step 4: Run all profile tests and commit**

Run:

```bash
python -m pytest tests/test_profile_op.py tests/test_profile_runner.py tests/test_profile_deep.py tests/test_profile_main.py tests/test_profile_fake_msprof.py -q
```

Expected: all profile tests pass.

```bash
git add tests/fixtures/fake_msprof.py tests/test_profile_fake_msprof.py
git commit -m "test(profile): exercise fake msprof subprocesses"
```

---

### Task 9: Documentation, environment, and full regression

**Files:**
- Modify: `README.md`
- Modify: `.env.example`
- Modify: `triton_evaluation/config.py`
- Modify: `triton_evaluation/engine/__init__.py`

**Interfaces:**
- Documents only `/verify` and `/profile`.
- Documents any-free-card scheduling and in-job fixed-card behavior.
- Documents `MSPROF_BIN`, `MSPROF_PROFILE_PYTHON`, `MSPROF_TIMEOUT_SEC`.

- [ ] **Step 1: Update README endpoint documentation**

Replace the opening evaluator sentence with:

```markdown
The vendored evaluation engine provides correctness verification, optional
in-job benchmarking, and standalone msprof diagnostics. Jobs run as isolated
subprocesses on workers pinned one-per-NPU-card.
```

Replace the endpoint list with:

```markdown
- `POST /verify` → `{task_id}`. Pass `measure_speed: true` to run the internal
  benchmark after every verify case passes; performance is returned under
  `result.benchmark`.
- `POST /profile` → `{task_id}`. `profile_mode` is `"op"` (default) or
  `"deep"`. Omit `case_idx` to profile every input case, or pass a zero-based
  index to profile one case. Profile does not require a verify task.
- `GET /tasks/{task_id}` → task status and direct evaluator result JSON.
- `GET /healthz` → Redis health.

All verify/profile jobs share one Redis queue. Whichever configured card worker
becomes free first may claim a profile job. Once claimed, all cases and metric
passes in that job stay on that card. “Free” is worker state, not an
`npu-smi` probe; cards configured for this service are expected to be
dedicated.
```

Add this complete request fixture and the two endpoint calls:

```bash
python - <<'PY' >/tmp/triton-profile-request.json
import json

reference_code = """import torch

class Model(torch.nn.Module):
    def forward(self, x, y):
        return x + y

def get_init_inputs():
    return []

def get_input_groups():
    return [
        [torch.randn(4096), torch.randn(4096)],
        [torch.randn(8192), torch.randn(8192)],
    ]
"""

kernel_code = """import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x, y, out, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    value = tl.load(x + offsets, mask=mask) + tl.load(y + offsets, mask=mask)
    tl.store(out + offsets, value, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        n = x.numel()
        add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
        return out
"""

print(json.dumps({
    "op_name": "add",
    "reference_code": reference_code,
    "kernel_code": kernel_code,
    "profile_mode": "op",
}))
PY

TASK_ID=$(
  curl -sS -X POST http://127.0.0.1:10907/profile \
    -H 'Content-Type: application/json' \
    --data-binary @/tmp/triton-profile-request.json |
  python -c 'import json,sys; print(json.load(sys.stdin)["task_id"])'
)
curl -sS "http://127.0.0.1:10907/tasks/${TASK_ID}"

python - <<'PY'
import json

path = "/tmp/triton-profile-request.json"
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)
payload.update({"profile_mode": "deep", "case_idx": 1, "timeout": 7200})
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle)
PY

curl -X POST http://127.0.0.1:10907/profile \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/triton-profile-request.json
```

- [ ] **Step 2: Update environment documentation**

Append to `.env.example`:

```dotenv
# --- Consumed by ProfileEvaluator / engine/profile.py ---

# msprof executable. Use an absolute path when CANN's bin directory is not on PATH.
MSPROF_BIN=msprof

# Python used by msprof to execute the generated kernel runner.
# Leave unset to use the worker's sys.executable.
# MSPROF_PROFILE_PYTHON=/path/to/npu/env/bin/python

# Maximum seconds for one runner/msprof subprocess. The total job timeout still wins.
MSPROF_TIMEOUT_SEC=300
```

Change the stale comments in `config.py` and `engine/__init__.py` from
`verify / benchmark / validate` to
`verify / internal benchmark / profile / validate`.

- [ ] **Step 3: Run static checks and the full suite**

Run:

```bash
python -m py_compile \
  triton_evaluation/models.py \
  triton_evaluation/api/routes.py \
  triton_evaluation/evaluators/profile.py \
  triton_evaluation/engine/profile.py \
  triton_evaluation/engine/msprof_op.py \
  triton_evaluation/engine/msprof_deep.py
python -m ruff check triton_evaluation tests
python -m pytest -q
```

Expected:

- `py_compile` exits 0.
- Ruff exits 0. Engine files are excluded by the existing project configuration.
- The complete pytest suite passes; the pre-existing Starlette deprecation warning may remain.

- [ ] **Step 4: Verify public route and registry invariants**

Run:

```bash
python -m pytest \
  tests/test_api.py::test_benchmark_route_is_gone \
  tests/test_evaluators.py::test_registry_has_verify_and_profile_only \
  tests/test_worker.py::test_profile_job_runs_on_whichever_worker_dequeues_it \
  -q
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add README.md .env.example triton_evaluation/config.py triton_evaluation/engine/__init__.py
git commit -m "docs: document profile endpoint and msprof environment"
```

---

### Task 10: Opt-in real NPU smoke and acceptance

**Files:**
- Create: `tests/msprof_smoke_kernels.py`
- Create: `tests/test_profile_npu_integration.py`

**Interfaces:**
- Uses complete known-good Triton-Ascend add source.
- Skips unless `TRITON_EVAL_RUN_NPU_PROFILE_TESTS=1`.
- Validates op single-case, op all-case, deep single-case, HF32, and target name.

- [ ] **Step 1: Add a complete two-case known-good fixture**

Create `tests/msprof_smoke_kernels.py`:

```python
REFERENCE_CODE = """
import torch

class Model(torch.nn.Module):
    def forward(self, x, y):
        return x + y

def get_init_inputs():
    return []

def get_input_groups():
    return [
        [torch.randn(4096), torch.randn(4096)],
        [torch.randn(8192), torch.randn(8192)],
    ]
"""


KERNEL_CODE = """
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, x, y):
        output = torch.empty_like(x)
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
        return output
"""
```

- [ ] **Step 2: Add the opt-in integration test**

Create `tests/test_profile_npu_integration.py`:

```python
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent
sys.path.insert(0, str(TESTS_DIR))
from msprof_smoke_kernels import KERNEL_CODE, REFERENCE_CODE  # noqa: E402


pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_EVAL_RUN_NPU_PROFILE_TESTS") != "1",
    reason="set TRITON_EVAL_RUN_NPU_PROFILE_TESTS=1 on an NPU host",
)


def _materialize(tmp_path):
    (tmp_path / "add_torch.py").write_text(REFERENCE_CODE, encoding="utf-8")
    (tmp_path / "add_triton_ascend_impl.py").write_text(KERNEL_CODE, encoding="utf-8")


def _run(tmp_path, mode, case_idx):
    output = tmp_path / f"{mode}_{case_idx}.json"
    engine = Path(__file__).parents[1] / "triton_evaluation" / "engine" / "profile.py"
    command = [
        sys.executable,
        str(engine),
        "--op_name",
        "add",
        "--verify_dir",
        str(tmp_path),
        "--triton_impl_name",
        "triton_ascend_impl",
        "--profile-mode",
        mode,
        "--timeout",
        "7200",
        "--output",
        str(output),
    ]
    if case_idx is not None:
        command.extend(["--case-idx", str(case_idx)])
    completed = subprocess.run(command, text=True, capture_output=True, timeout=7260)
    assert completed.returncode == 0, completed.stderr[-2000:]
    return json.loads(output.read_text())


def test_real_msprof_op_all_and_single_case(tmp_path):
    assert shutil.which(os.environ.get("MSPROF_BIN", "msprof"))
    _materialize(tmp_path)
    all_cases = _run(tmp_path, "op", None)
    assert all_cases["selected_cases"] == 2
    assert all(case["target_kernel"].startswith("add_kernel") for case in all_cases["per_case_results"])
    one_case = _run(tmp_path, "op", 1)
    assert one_case["selected_cases"] == 1
    assert one_case["per_case_results"][0]["case_idx"] == 1


def test_real_msprof_deep_single_case(tmp_path):
    assert shutil.which(os.environ.get("MSPROF_BIN", "msprof"))
    _materialize(tmp_path)
    result = _run(tmp_path, "deep", 0)
    case = result["per_case_results"][0]
    assert case["target_kernel"].startswith("add_kernel")
    assert case["status"] in {"complete", "partial"}
    assert case["error"] is None
```

The explicit `TESTS_DIR` insertion keeps the fixture import independent of
whether `tests/` is installed as a Python package.

- [ ] **Step 3: Verify the default CPU suite skips NPU tests**

Run:

```bash
python -m pytest tests/test_profile_npu_integration.py -q
```

Expected: 2 skipped.

- [ ] **Step 4: Commit the smoke fixture**

```bash
git add tests/msprof_smoke_kernels.py tests/test_profile_npu_integration.py
git commit -m "test(profile): add opt-in NPU msprof smoke"
```

- [ ] **Step 5: Run real op smoke on the NPU host**

Preflight:

```bash
command -v msprof
python -c "import torch, torch_npu, triton; print(torch.npu.is_available())"
```

Expected: an absolute msprof path and `True`.

Run:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
TRITON_EVAL_RUN_NPU_PROFILE_TESTS=1 \
python -m pytest \
  tests/test_profile_npu_integration.py::test_real_msprof_op_all_and_single_case \
  -v
```

Expected: 1 passed.

- [ ] **Step 6: Run real deep smoke**

Run:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
TRITON_EVAL_RUN_NPU_PROFILE_TESTS=1 \
python -m pytest \
  tests/test_profile_npu_integration.py::test_real_msprof_deep_single_case \
  -v
```

Expected: 1 passed. Confirm no residual profiler process:

```bash
pgrep -af "msprof|profile_runner.py"
```

Expected: no process from this test remains.

- [ ] **Step 7: Final verification**

Run:

```bash
python -m ruff check triton_evaluation tests
python -m pytest -q
git status --short
```

Expected:

- Ruff passes.
- Full pytest passes with the NPU tests skipped unless explicitly enabled.
- The worktree contains no uncommitted implementation changes.
- Real NPU completion is reported only if Steps 5 and 6 passed.

---

## Plan Self-Review Checklist

- Task 1 covers the API model, all-case default, zero-based single-case selection, and mode validation.
- Task 2 covers direct result storage, timeout snapshot retention, registry shape, and any-free-worker scheduling.
- Tasks 3–4 cover stdlib-only imports, exact op CLI, generated runner discovery, nested input movement, deterministic inputs, and HF32 ordering.
- Tasks 5–6 cover 3+7+1 deep capture, stable complete-name merge, missing-as-unknown, canonical Bound priority, run-wide per-core scope, and deterministic recommendations.
- Task 7 covers full/single orchestration, independent case failures, atomic snapshots, remaining indices, and exit codes.
- Task 8 covers real subprocess/CSV/SQLite behavior without NPU.
- Task 9 covers public documentation, environment variables, stale wording, and the full regression suite.
- Task 10 contains complete known-good source and separates skipped CPU validation from real NPU acceptance.
- No task adds a public benchmark route, verify gate, hardcoded core count, cross-kernel merge, or `result.profile` wrapper.
