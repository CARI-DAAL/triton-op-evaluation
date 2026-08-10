# Synchronous evaluation API v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish API v2 as one synchronous `POST /evaluations` endpoint with a terminal typed
response, composable checks, typed performance/profile projections, Redis-backed per-session
in-flight limiting, and no public v1/polling routes.

**Architecture:** The API validates the request and `X-Session-ID`, acquires a Redis lease,
enqueues one internal `evaluation` job, and asynchronously polls the Broker until that job is
terminal or the request-wide timeout expires. Workers continue to run the existing evaluator
orchestrator and release the lease only after storing a terminal record. The engine remains
private: `evaluators/shaping.py` projects its JSON into pydantic contract structures.

**Tech Stack:** Python 3.10+, FastAPI, pydantic v2, redis/fakeredis, pytest/httpx; pure-stdlib
divergence code under the vendored engine.

## Global Constraints

- Work only in `.claude/worktrees/feat-evaluation-api-v2` on
  `worktree-feat-evaluation-api-v2`.
- The approved spec is `docs/specs/2026-08-06-evaluation-api-v2-design.md` at commit `44f23c0`.
- `timeout` is the total synchronous API wait from request arrival, including queue time and all
  checks. API timeout does not cancel the job.
- Public routes after migration are only `POST /evaluations` and `GET /healthz`.
- `EvaluationResult` has terminal statuses only and has no `revision` field.
- `performance` requires correctness in the same request and only runs after correctness passes.
- profile is independent: wrong or decoy correctness must not skip it.
- `X-Session-ID` is required and one unfinished backend job per session is enforced in Redis.
- `benchmark.py` and `profile_runner.py` are not modified.
- Engine correctness changes remain additive; no accuracy algorithm or existing JSON field changes.
- CI must remain torch-free/numpy-free outside the existing fake-module engine tests.
- Ruff line length is 100; `triton_evaluation/engine/` remains excluded.
- Use `python3 -m pytest` because this environment has no `python` command.
- Follow red-green-refactor for every behavior change and make a focused commit after each task.

## Current Checkpoint and Cleanup

The branch already contains Tasks 1–6 of the superseded async plan: base contract, divergence
classifier, additive verify emission, initial shaping, JobSpec plumbing, and the orchestrator.
Keep the divergence work. Revise the contract/shaping/orchestrator in place.

Before Task 1, remove only the interrupted uncommitted async route attempt created after
`ebd49bd`: restore `triton_evaluation/api/routes.py` to its `ebd49bd` content with `apply_patch`
and delete the untracked `tests/test_api_v2.py` with `apply_patch`. Do not touch the untracked old
plan files.

## File Structure

| File | Responsibility |
|---|---|
| `triton_evaluation/api/contract.py` | Stable sync request/result and fully typed submodels |
| `triton_evaluation/evaluators/shaping.py` | Engine JSON → stable correctness/performance/profile projections |
| `triton_evaluation/evaluators/evaluation.py` | Check sequencing, dependency rule, terminal aggregation |
| `triton_evaluation/broker.py` | Task persistence plus atomic session lease acquire/release |
| `triton_evaluation/models.py` | Internal job fields, including `checks` and `session_id` |
| `triton_evaluation/worker.py` | Worker dispatch, deadline budget, terminal lease release |
| `triton_evaluation/api/routes.py` | Sole sync evaluation route, internal poll, health route |
| `triton_evaluation/config.py` | API poll interval and session lease key prefix |
| `tests/test_contract.py` | Stable wire-model behavior |
| `tests/test_shaping.py` | All engine-to-contract projections |
| `tests/test_evaluation_evaluator.py` | Combination and terminal-status rules |
| `tests/test_broker.py`, `tests/test_worker.py` | Lease lifecycle and internal plumbing |
| `tests/test_api_v2.py` | Header validation, sync wait, timeout, 429, route removal |
| `scripts/test_eval_service.py` | Synchronous v2 NPU smoke client |
| `README.md`, deployment examples | v2 request/response and safe bind instructions |

---

## Task 1: Replace the async-era wire contract

**Files:**
- Modify: `triton_evaluation/api/contract.py`
- Modify: `tests/test_contract.py`

**Interfaces:**
- Produces `TerminalStatus`, `EvaluationRequest`, `EvaluationResult`, `EvaluationPayload`,
  `PerShapeResult`, `PerformanceResult`, `OpProfile`, `DeepProfileSummary`, `ProfileData`, and
  `ProfileResult` exactly as named in the approved spec.
- Removes `CONTRACT_REVISION` and queued/running response states.

- [ ] **Step 1: Write failing terminal-contract tests**

Replace revision/in-progress assertions in `tests/test_contract.py` with literal expectations:

```python
def test_evaluation_result_is_terminal_and_has_no_revision():
    result = EvaluationResult(task_id="t", status="SUCCEEDED", result=EvaluationPayload())
    assert result.model_dump() == {
        "task_id": "t", "status": "SUCCEEDED", "error": None,
        "result": {"correctness": None, "performance": None, "profile": None},
    }
    with pytest.raises(ValidationError):
        EvaluationResult(task_id="t", status="QUEUED", result=EvaluationPayload())


def test_evaluation_request_rejects_duplicate_checks():
    with pytest.raises(ValidationError):
        EvaluationRequest(
            op_name="a", reference_code="R", kernel_code="K",
            checks=["correctness", "correctness"],
        )
```

Add typed performance/profile round trips with hand-written literals:

```python
def test_typed_performance_and_op_profile_roundtrip():
    payload = EvaluationPayload(
        performance=PerformanceResult(
            speedup_vs_torch=2.5,
            per_shape=[PerShapeResult(
                shape="[20120,512]", latency_ms=0.4,
                throughput=None, speedup_vs_torch=2.5,
            )],
        ),
        profile=ProfileResult(
            profile_state="complete",
            data=ProfileData(
                mode="op", selected_cases=1, completed_cases=1,
                partial_cases=0, failed_cases=0, timed_out=False,
                ops=[OpProfile(case_idx=0, op_name="add_kernel", time_ms=0.4,
                               percentage=100.0)],
            ),
        ),
    )
    back = EvaluationPayload.model_validate_json(payload.model_dump_json())
    assert back.performance.per_shape[0].latency_ms == 0.4
    assert back.profile.data.ops[0].percentage == 100.0
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python3 -m pytest tests/test_contract.py -q
```

Expected: failures for `CONTRACT_REVISION`, accepted `QUEUED`, missing typed profile classes, and
duplicate checks being accepted.

- [ ] **Step 3: Implement the minimal contract**

In `contract.py`:

```python
TerminalStatus = Literal["SUCCEEDED", "FAILED", "TIMEOUT", "REJECTED"]
ProfileState = Literal["complete", "partial"]
ProfileCaseState = Literal["complete", "partial", "failed"]
BoundKind = Literal[
    "MTE2 BOUND", "CUBE BOUND", "VEC BOUND", "FIXP BOUND",
    "MTE3 BOUND", "SCALAR BOUND", "NO BOUND",
]
```

Add the spec's typed classes. `ProfileData` validates mode-specific fields:

```python
@model_validator(mode="after")
def _mode_specific_payload(self):
    if self.mode == "op":
        self.ops = self.ops or []
        self.summaries = None
    else:
        self.summaries = self.summaries or []
        self.ops = None
    return self
```

Update `EvaluationRequest._params_only_with_matching_check` to reject duplicates before clearing
irrelevant parameter objects. Update `EvaluationResult.status` to `TerminalStatus` and delete
`revision`.

- [ ] **Step 4: Verify GREEN and lint**

```bash
python3 -m pytest tests/test_contract.py -q
ruff check triton_evaluation/api/contract.py tests/test_contract.py
```

- [ ] **Step 5: Commit**

```bash
git add triton_evaluation/api/contract.py tests/test_contract.py
git commit -m "feat(api): finalize synchronous v2 wire contract"
```

---

## Task 2: Project real benchmark/profile JSON into typed payloads

**Files:**
- Modify: `triton_evaluation/evaluators/shaping.py`
- Modify: `tests/test_shaping.py`

**Interfaces:**
- Produces `ShapingError(ValueError)`.
- Produces `shape_correctness(raw, *, decoy=False) -> dict | None`.
- Produces `shape_performance(raw) -> dict` and rejects incomplete benchmark results.
- Produces `shape_profile(raw) -> dict` for both op and deep modes.
- Keeps `shape_payload(...) -> dict` as the orchestrator-facing aggregate.

- [ ] **Step 1: Write failing performance projection tests**

Use a complete benchmark fixture matching `benchmark.result_to_dict`:

```python
def test_performance_projects_only_stable_per_shape_fields():
    raw = {
        "speedup_vs_torch": 2.5, "failed_cases": 0,
        "per_shape_results": [{
            "case_idx": 1,
            "input_desc": [{"type": "tensor", "shape": [20120, 512], "dtype": "torch.float32"}],
            "status": "pass",
            "framework": {"avg_latency_ms": 1.0, "peak_memory_mb": 2.0},
            "implementation": {"avg_latency_ms": 0.4, "peak_memory_mb": 1.0},
            "speedup_vs_torch": 2.5, "error_type": None, "error_msg": None,
        }],
    }
    assert shape_performance(raw) == {
        "speedup_vs_torch": 2.5,
        "per_shape": [{
            "shape": "[20120,512]", "latency_ms": 0.4,
            "throughput": None, "speedup_vs_torch": 2.5,
        }],
    }


@pytest.mark.parametrize("mutation", [
    {"failed_cases": 1},
    {"speedup_vs_torch": None},
    {"per_shape_results": []},
])
def test_performance_rejects_incomplete_engine_result(mutation):
    raw = _complete_perf_json()
    raw.update(mutation)
    with pytest.raises(ShapingError):
        shape_performance(raw)
```

- [ ] **Step 2: Write failing op/deep profile tests**

Op fixture must contain two kernels with `duration_us` 400 and 100; assert `time_ms` 0.4/0.1 and
percentages 80/20. Deep fixture must contain fixed `bound.ratios`, per-core spread/verdict, one
recommendation, missing metrics, warnings, and error; assert no `raw_metrics`, `evidence`, or other
engine-private key survives in the result.

- [ ] **Step 3: Run tests and verify RED**

```bash
python3 -m pytest tests/test_shaping.py -q
```

Expected: missing `shape_performance`, `shape_profile`, typed fields, and `ShapingError`.

- [ ] **Step 4: Implement projection helpers**

Use compact JSON for shape IDs:

```python
def _shape_id(input_desc: list[dict]) -> str:
    shapes = [item["shape"] for item in input_desc
              if item.get("type") == "tensor" and isinstance(item.get("shape"), list)]
    value = shapes[0] if len(shapes) == 1 else shapes
    return json.dumps(value, separators=(",", ":"))
```

`shape_performance` accepts only `failed_cases == 0`, finite numeric overall speedup, and at least
one passing per-shape item with numeric implementation latency and speedup. Otherwise raise
`ShapingError`.

For op profile, flatten every `per_case_results[].kernels[]`, calculate case-local duration totals,
and map to `OpProfile`. For deep profile, read only `bound.name/rule/ratios`,
`per_core.spread_ratio/verdict`, recommendation text, and fixed common fields. `_profile_state`
returns partial for timed out, failed/partial cases, or incomplete selection.

Update correctness mapping so numerical/shape failures set `error=None`, compile/runtime failures
keep the engine error, and `mean_abs_err` stays `None`.

- [ ] **Step 5: Verify GREEN and regression**

```bash
python3 -m pytest tests/test_shaping.py tests/test_contract.py -q
ruff check triton_evaluation/evaluators/shaping.py tests/test_shaping.py
```

- [ ] **Step 6: Commit**

```bash
git add triton_evaluation/evaluators/shaping.py tests/test_shaping.py
git commit -m "feat(evaluators): shape typed performance and profile results"
```

---

## Task 3: Make profile independent and aggregate evaluator terminal states

**Files:**
- Modify: `triton_evaluation/evaluators/evaluation.py`
- Modify: `tests/test_evaluation_evaluator.py`

**Interfaces:**
- Consumes `shape_payload` / `ShapingError` from Task 2.
- Produces one `EvaluatorResult` with a stable payload even when a requested sub-check fails.
- Terminal priority: `TIMEOUT > FAILED > REJECTED > SUCCEEDED`.

- [ ] **Step 1: Write failing combination tests**

Add behavior tests that observe the real orchestrator result, not mock call assertions:

```python
def test_decoy_still_returns_profile(tmp_path):
    result = _Harness(verify_mode="decoy").run(
        _spec(["correctness", "performance", "profile"]), str(tmp_path), 0,
    )
    assert result.result["correctness"]["decoy"] is True
    assert result.result["performance"] is None
    assert result.result["profile"]["profile_state"] == "complete"


def test_wrong_kernel_skips_performance_but_keeps_profile(tmp_path):
    result = _Harness(verify_mode="fail").run(
        _spec(["correctness", "performance", "profile"]), str(tmp_path), 0,
    )
    assert result.status == "SUCCEEDED"
    assert result.result["correctness"]["passed"] is False
    assert result.result["performance"] is None
    assert result.result["profile"] is not None


def test_profile_timeout_returns_timeout_with_partial_profile(tmp_path):
    result = _Harness(profile_status="TIMEOUT").run(
        _spec(["profile"]), str(tmp_path), 0,
    )
    assert result.status == "TIMEOUT"
    assert result.result["profile"]["profile_state"] == "partial"
```

Add a malformed benchmark fixture and assert overall `FAILED`, correctness retained, and
`performance is None`.

- [ ] **Step 2: Run tests and verify RED**

```bash
python3 -m pytest tests/test_evaluation_evaluator.py -q
```

Expected: decoy currently skips profile; terminal failures currently return early or are discarded;
raw profile fixtures no longer satisfy Task 2 shaping.

- [ ] **Step 3: Implement status aggregation**

Replace early returns with recorded sub-results. Always run requested profile unless process-level
materialization itself raises. Gate only performance. Shape available JSON, catch `ShapingError`
as a performance failure, and aggregate status with:

```python
_STATUS_PRIORITY = {"SUCCEEDED": 0, "REJECTED": 1, "FAILED": 2, "TIMEOUT": 3}


def _worst_status(statuses: list[str]) -> str:
    return max(statuses or [TaskStatus.SUCCEEDED.value], key=_STATUS_PRIORITY.__getitem__)
```

Join non-empty sub-errors with `"; "`. Return the shaped payload for all terminal states.

- [ ] **Step 4: Verify GREEN and evaluator regressions**

```bash
python3 -m pytest tests/test_evaluation_evaluator.py tests/test_evaluators.py -q
ruff check triton_evaluation/evaluators/evaluation.py tests/test_evaluation_evaluator.py
```

- [ ] **Step 5: Commit**

```bash
git add triton_evaluation/evaluators/evaluation.py tests/test_evaluation_evaluator.py
git commit -m "fix(evaluators): keep profile independent in v2 orchestration"
```

---

## Task 4: Add Redis session leases and release them from workers

**Files:**
- Modify: `triton_evaluation/config.py`
- Modify: `triton_evaluation/models.py`
- Modify: `triton_evaluation/broker.py`
- Modify: `triton_evaluation/worker.py`
- Modify: `tests/test_models.py`
- Modify: `tests/test_broker.py`
- Modify: `tests/test_worker.py`

**Interfaces:**
- Adds `Settings.session_key_prefix = "eval:session:"` and
  `Settings.api_poll_interval_seconds = 0.25` (`ge=0`, `le=5`).
- Adds `JobSpec.session_id: str | None = None`.
- Adds `Broker.acquire_session(session_id, task_id, ttl_seconds) -> bool`.
- Adds `Broker.release_session(session_id, task_id) -> bool` with owner comparison.
- Adds `compute_session_lease_ttl(spec, settings) -> int`.

- [ ] **Step 1: Write failing Broker lease tests**

```python
def test_session_lease_has_single_owner(broker):
    assert broker.acquire_session("session-a", "task-1", 60) is True
    assert broker.acquire_session("session-a", "task-2", 60) is False
    assert broker.release_session("session-a", "task-2") is False
    assert broker.release_session("session-a", "task-1") is True
    assert broker.acquire_session("session-a", "task-2", 60) is True


def test_sessions_are_independent(broker):
    assert broker.acquire_session("a", "task-a", 60) is True
    assert broker.acquire_session("b", "task-b", 60) is True
```

- [ ] **Step 2: Write failing worker lifecycle tests**

Round-trip `session_id` through Job JSON. In worker tests acquire a lease, run both successful and
raising evaluators, then assert the same session can be acquired by a new owner only after
`run_once` stores its terminal record.

- [ ] **Step 3: Run tests and verify RED**

```bash
python3 -m pytest tests/test_models.py tests/test_broker.py tests/test_worker.py -q
```

- [ ] **Step 4: Implement leases**

Acquire with Redis `SET key task_id NX EX ttl`. Release with a WATCH/MULTI owner check so an old
worker cannot delete a newer lease. Validate positive integer TTL.

Compute lease TTL as:

```python
def compute_session_lease_ttl(spec, settings) -> int:
    return math.ceil(
        compute_max_runtime(spec) + settings.reaper_grace_seconds + spec.timeout
    )
```

In `Worker.run_once`, set `terminal_stored = True` only after `store_result` succeeds. An outer
`finally` releases `job.spec.session_id` only when `terminal_stored` is true; if Redis cannot store
the terminal record, retain the lease until TTL rather than permitting duplicate work. Jobs with
`session_id=None` keep existing behavior.

- [ ] **Step 5: Verify GREEN and lint**

```bash
python3 -m pytest tests/test_models.py tests/test_broker.py tests/test_worker.py -q
ruff check triton_evaluation/config.py triton_evaluation/models.py \
  triton_evaluation/broker.py triton_evaluation/worker.py tests/test_broker.py tests/test_worker.py
```

- [ ] **Step 6: Commit**

```bash
git add triton_evaluation/config.py triton_evaluation/models.py triton_evaluation/broker.py \
  triton_evaluation/worker.py tests/test_models.py tests/test_broker.py tests/test_worker.py
git commit -m "feat(queue): enforce one in-flight evaluation per session"
```

---

## Task 5: Implement the synchronous `/evaluations` boundary

**Files:**
- Rewrite: `triton_evaluation/api/routes.py`
- Create: `tests/test_api_v2.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Public routes: `POST /evaluations`, `GET /healthz` only.
- `X-Session-ID` syntax: `^[A-Za-z0-9._:-]{1,128}$`.
- Performance without correctness: HTTP 400.
- Lease conflict: HTTP 429 and `Retry-After: 1`.
- Accepted job: wait for terminal record or return synthetic `TIMEOUT` at request deadline.

- [ ] **Step 1: Write failing request-boundary tests**

Tests with a real `Broker(fakeredis)` assert missing/invalid session returns 400,
performance-only returns 400, pre-acquired same-session lease returns 429, and `/verify`,
`/profile`, `/tasks/x`, and `/evaluations/x` return 404.

- [ ] **Step 2: Write failing sync terminal and timeout tests**

For terminal behavior, use a Broker test double that preserves real `create_task` but changes the
record to `SUCCEEDED` with a literal stable payload on its second `get_task`. Assert one POST returns
that `EvaluationResult` and enqueued JobSpec contains checks, parameters, and session ID.

For timeout behavior, monkeypatch route `time.monotonic` to advance beyond the request budget and
`asyncio.sleep` to an async no-op. Assert HTTP 200 with:

```python
{
    "task_id": ANY_STRING,
    "status": "TIMEOUT",
    "error": "evaluation request timed out",
    "result": {"correctness": None, "performance": None, "profile": None},
}
```

Then assert the task remains `QUEUED` and its session lease remains held.

- [ ] **Step 3: Run tests and verify RED**

```bash
python3 -m pytest tests/test_api_v2.py tests/test_api.py -q
```

Expected: no sync v2 route, legacy routes still 200, no header/lease validation.

- [ ] **Step 4: Implement route helpers**

Use these focused boundaries:

```python
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
TERMINAL = {"SUCCEEDED", "FAILED", "TIMEOUT", "REJECTED"}


def _validate_session_id(value: str | None) -> str:
    if value is None or SESSION_ID_RE.fullmatch(value) is None:
        raise HTTPException(status_code=400, detail="invalid or missing X-Session-ID")
    return value


async def _wait_for_result(broker, task_id, timeout, poll_interval):
    deadline = time.monotonic() + timeout
    while True:
        record = broker.get_task(task_id)
        if record and record.get("status") in TERMINAL:
            return _to_evaluation_result(record)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return EvaluationResult(
                task_id=task_id, status="TIMEOUT",
                error="evaluation request timed out", result=EvaluationPayload(),
            )
        await asyncio.sleep(min(poll_interval, remaining))
```

`submit_evaluation` validates the subset rule, creates `JobSpec(..., session_id=session_id)`,
acquires the lease before `create_task`, releases it only if task creation raises, and then awaits
`_wait_for_result`. Convert malformed terminal worker payloads to a contract-shaped `FAILED`
response with `error="invalid worker result"` rather than leaking a response-validation 500.

Delete v1 and public polling handlers/imports from `routes.py`.

- [ ] **Step 5: Verify GREEN and API regression**

```bash
python3 -m pytest tests/test_api_v2.py tests/test_api.py -q
ruff check triton_evaluation/api/routes.py tests/test_api_v2.py tests/test_api.py
```

- [ ] **Step 6: Commit**

```bash
git add triton_evaluation/api/routes.py tests/test_api_v2.py tests/test_api.py
git commit -m "feat(api): make evaluations synchronous and remove v1 routes"
```

---

## Task 6: Prove the full route → queue → worker → typed-response story

**Files:**
- Modify: `tests/test_api_v2.py`

**Interfaces:**
- Exercises a real Broker, real Worker, real EvaluationEvaluator orchestration, real shaping, and
  the real synchronous route; only external engine sub-evaluators are faked.

- [ ] **Step 1: Write the failing E2E test**

Start a worker thread before POST. Its registry contains an `EvaluationEvaluator` harness whose
verify returns one numeric failure and whose profile returns a complete typed raw op profile. POST
checks `correctness/performance/profile` with a valid session header. Assert the single response:

```python
assert body["status"] == "SUCCEEDED"
assert body["result"]["correctness"]["passed"] is False
assert body["result"]["correctness"]["failures"][0]["kind"] == "constant_offset"
assert body["result"]["performance"] is None
assert body["result"]["profile"]["profile_state"] == "complete"
```

Also assert the worker released the session lease after storing the result.

- [ ] **Step 2: Run test and verify RED if integration gaps remain**

```bash
python3 -m pytest tests/test_api_v2.py::test_end_to_end_sync_combination_contract -q
```

- [ ] **Step 3: Fix only integration wiring gaps**

Do not weaken the combination assertions. If the test already passes, no production change is
needed.

- [ ] **Step 4: Run the API/worker/evaluator slice**

```bash
python3 -m pytest tests/test_api_v2.py tests/test_worker.py \
  tests/test_evaluation_evaluator.py tests/test_shaping.py -q
```

- [ ] **Step 5: Commit**

```bash
git add tests/test_api_v2.py
git commit -m "test(api): cover synchronous v2 end to end"
```

---

## Task 7: Migrate docs, smoke client, and safe binding defaults

**Files:**
- Modify: `run_api.py`
- Modify: `scripts/start.sh`
- Modify: `docker/entrypoint.sh`
- Modify: `docker-compose.yml`
- Modify: `docker-compose.npu0-test.yml`
- Modify: `scripts/test_eval_service.py`
- Modify: `README.md`
- Modify: `.env.example`
- Modify: `docker.env.example`
- Modify: `docs/ascend-docker-deploy.md`
- Modify: `tests/test_deployment.py`
- Modify: `tests/test_entrypoints.py`
- Create: `tests/test_smoke_script.py`

**Interfaces:**
- Host execution defaults to `127.0.0.1`.
- Smoke client performs one synchronous POST with `X-Session-ID`, then inspects typed payload.
- Docker documentation distinguishes container listen address from host loopback publication.

- [ ] **Step 1: Write failing deployment/smoke behavior tests**

Update deployment expectations to require the host-facing default `127.0.0.1`. Extract the smoke
client response-verdict function and test literal success, correctness mismatch, transport failure,
and TIMEOUT terminal responses without an NPU.

- [ ] **Step 2: Run tests and verify RED**

```bash
python3 -m pytest tests/test_deployment.py tests/test_entrypoints.py \
  tests/test_smoke_script.py -q
```

Expected: old `0.0.0.0`, `/verify`, task polling, and v1 dict assumptions.

- [ ] **Step 3: Implement the migration**

Set the host/default bind in `run_api.py`, `scripts/start.sh`, `docker/entrypoint.sh`, both compose
files, and environment examples to `127.0.0.1`. Keep the current host-network NPU topology; the
loopback bind makes the published API host-local. Rewrite the smoke client to POST `/evaluations`
with:

```python
headers = {
    "Content-Type": "application/json",
    "X-Session-ID": args.session_id,
}
body = {
    "op_name": args.op,
    "reference_code": REFERENCE_CODE,
    "kernel_code": KERNEL_CODE,
    "checks": ["correctness", "performance"] if args.measure_speed else ["correctness"],
    "timeout": args.job_timeout,
}
```

Use HTTP read timeout `job_timeout + 60`; remove the poll loop and `--max-wait/--poll` options.
Update examples and response parsing to `result.correctness` / `result.performance` and document
that HTTP read timeout must not be retried automatically.

- [ ] **Step 4: Verify GREEN**

```bash
python3 -m pytest tests/test_deployment.py tests/test_entrypoints.py \
  tests/test_smoke_script.py -q
ruff check run_api.py scripts/test_eval_service.py tests/test_smoke_script.py
```

- [ ] **Step 5: Commit**

```bash
git add run_api.py scripts/start.sh docker/entrypoint.sh docker-compose.yml \
  docker-compose.npu0-test.yml scripts/test_eval_service.py README.md .env.example \
  docker.env.example docs/ascend-docker-deploy.md tests/test_deployment.py \
  tests/test_entrypoints.py tests/test_smoke_script.py
git commit -m "docs(api): migrate smoke workflow to synchronous evaluations"
```

---

## Task 8: Full verification and calibration handoff

**Files:**
- Modify only files required by a reproduced regression.
- Record calibration evidence in `docs/specs/2026-08-06-evaluation-api-v2-design.md` only after a
  real labeled NPU corpus is available and measured.

- [ ] **Step 1: Run complete static checks**

```bash
ruff check .
python3 -m compileall -q triton_evaluation tests scripts
```

- [ ] **Step 2: Run the full torch-free suite**

```bash
python3 -m pytest -q
```

- [ ] **Step 3: Inspect diff and contract exposure**

```bash
git diff origin/feat/evaluation-api-v2...HEAD --check
git status --short
```

Manually confirm OpenAPI exposes only `/evaluations` and `/healthz`, response schemas contain no
`revision`, and no response projection includes `reference_code`, input values, raw profile metrics,
or engine traceback for a classified numerical failure.

- [ ] **Step 4: Run on-device smoke when an NPU service is available**

```bash
python3 scripts/test_eval_service.py --url http://127.0.0.1:11907 \
  --session-id api-v2-smoke --job-timeout 600
```

The smoke must return one terminal typed response without task polling.

- [ ] **Step 5: Calibration gate**

Run the existing pure classifier against a labeled corpus of at least several dozen real failing
cases covering every retained kind. Compute per-kind precision and recall from literal expected vs
predicted labels. Ship thresholds only if every retained kind has precision > 90% and recall > 80%;
otherwise merge the failing label into `scattered`, update contract/tests, and rerun the full suite.
If no labeled tensor corpus or NPU is present in the workspace, report this external evidence gate
separately; do not claim calibrated thresholds.

- [ ] **Step 6: Commit any reproduced verification fixes**

Use one focused fix commit per reproduced failure; do not make a catch-all cleanup commit.

## Plan Self-Review

- Spec §1–§3: Tasks 4–7 cover sync submit/poll, overall timeout, session header/lease, route removal,
  and client migration.
- Spec §4: Tasks 1–2 cover terminal result and all typed submodels/projections.
- Spec §5: existing committed classifier/emission work is preserved; Task 8 retains the real-NPU
  calibration evidence gate.
- Spec §6: Tasks 2–3 cover shaping, performance dependency, independent profile, and terminal
  aggregation.
- Spec §7–§9: Tasks 4, 5, and 7 cover orphan lease lifecycle and safe deployment/client behavior.
- Spec §10: Tasks 1–8 include every contract, shaping, combination, removal, regression, smoke, and
  calibration acceptance item.
- No task reintroduces async consumer polling, v1 aliases, `revision`, untyped performance/profile
  passthrough, or cancellation.
